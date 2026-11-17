# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
GEMM + all-gather of its output over the same transports as bench_all_gather_methods.py, on the
torch allocator. Each rank computes ``C_r = A_r @ W.T`` (``M_local x N``); every rank ends with
all of ``C`` (``world * M_local x N``) in ``out``.

    PYTHONPATH=$PWD torchrun --nproc_per_node=8 benchmark/ops/matmul_all_gather_transports/bench_gemm_all_gather.py --csv out.csv

The GEMM config (block sizes, warps) is picked per shape by timing our GEMM alone on rank 0 and is
then used by every method built on it, so their tiles are bitwise identical. Methods:

- gemm / torch_gemm: the GEMM alone (ours / hipBLASLt), no communication. References.
- rccl_ag: the RCCL all-gather alone (the GEMM runs untimed first). Reference: with both,
  max(gemm, rccl_ag) is what perfect overlap would take.
- rccl / torch_rccl: the GEMM, then an in-place RCCL all-gather.
- torch_rccl_overlap (S): hipBLASLt in S row slices; each slice's RCCL all-gather runs on a second
  stream while the next slice computes. No custom kernels. S is chosen per shape: report the best.
- iris_ops: Iris's fused ``iris.ops.matmul_all_gather`` with its default config, as shipped. Ends
  with its own ``async_op=False`` barrier (host sync + distributed barrier).
- iris_cu: our GEMM's epilogue stores each tile into every peer from the CUs; the last tile signals
  each peer once.
- sdma_tile: the epilogue copies each tile to every peer on the device-initiated SDMA queues; the
  last tile signals each peer once.
- sdma_rows: the last tile of each row block copies the whole row block (linear) the same way.
- plan_tiles (T): a triggered plan (``iris.experimental.plan``), tile_chunks(tiles_per_chunk=T) in
  the GEMM's own dispatch order; the epilogue marks the tile's chunk ready. The host's part runs
  untimed between iterations; ``round_overhead_us`` is its median (``next_round()``, ``start()``,
  ``prepare_next()``; the blocking ``wait()`` that checks the round is not counted).
- plan_tiles_loop (T = CU count): the same, timing a whole stream-ordered loop iteration:
  ``start()``, GEMM, ``prepare_next()``, wait kernel, ``wait(blocking=False)``, ``next_round()``.
  This is what a loop calling the plan every step pays; ``round_overhead_us`` is the host time of
  that iteration.
- plan_rows: a triggered plan with one linear chunk per row block, in completion order.

Every communicating method ends when all of ``C`` has arrived here. Timing is iris.do_bench
(barriers and an L2 flush before every iteration). Between iterations (untimed) W is negated and
``out`` wiped, so data from an earlier iteration fails the final check (bitwise against the same
GEMM; iris_ops, whose GEMM differs, against hipBLASLt with a tolerance). ``gemm_us`` is the GEMM
kernel alone inside each method (events around its launch).
"""

import argparse
import csv
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemm_ag_kernels as gk  # noqa: E402

import iris  # noqa: E402
from iris.host.memory.allocators.torch_allocator import TorchAllocator  # noqa: E402
from iris.ops import matmul_all_gather  # noqa: E402
from iris.experimental.plan import State, TriggeredPlan, chunks, tile_chunks  # noqa: E402

DTYPE = torch.bfloat16
SHAPES = [
    "256x4096x4096",
    "1024x4096x4096",
    "1024x8192x8192",
    "2048x8192x8192",
    "4096x8192x4096",
    "4096x8192x8192",
    "4096x8192x16384",
    "8192x8192x8192",
]
# BLOCK_M, BLOCK_N, BLOCK_K, num_warps; GROUP_M=8 and the XCD remap throughout
CONFIGS = [
    (256, 256, 64, 8),
    (256, 128, 64, 8),
    (128, 256, 64, 8),
    (128, 128, 64, 4),
    (128, 64, 64, 4),
    (64, 128, 64, 4),
    (64, 64, 64, 4),
]
WAIT_BUDGET = 1 << 24
METHODS = [
    "gemm",
    "torch_gemm",
    "rccl_ag",
    "rccl",
    "torch_rccl",
    "torch_rccl_overlap",
    "iris_ops",
    "iris_cu",
    "sdma_tile",
    "sdma_rows",
    "plan_tiles",
    "plan_tiles_loop",
    "plan_rows",
]


def config(bm, bn, bk, warps, group_m=8):
    return dict(BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=group_m, NUM_XCDS=8, num_warps=warps, num_stages=2)


def pick_config(m_local, n, k, group_m):
    """Rank 0 times our GEMM alone in every config that fits; every rank uses its choice."""
    choice = [None]
    if dist.get_rank() == 0:
        dev = torch.device("cuda", torch.cuda.current_device())
        A = torch.randn(m_local, k, dtype=DTYPE, device=dev)
        W = torch.randn(n, k, dtype=DTYPE, device=dev)
        out = torch.empty(m_local, n, dtype=DTYPE, device=dev)
        d = torch.zeros(1, dtype=torch.int64, device=dev)
        best = None
        for bm, bn, bk, warps in CONFIGS:
            if m_local % bm or n % bn or k % bk:
                continue
            c = config(bm, bn, bk, warps, group_m)

            def run(c=c, bm=bm, bn=bn):
                gk.gemm_all_gather[((m_local // bm) * (n // bn),)](
                    A, W, out, m_local, n, k, A.stride(0), W.stride(0), out.stride(0), 0,
                    d, d, d, d, d, d, 0, d, 1, WORLD=1, MODE=gk.LOCAL,
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=c["GROUP_M"], NUM_XCDS=c["NUM_XCDS"],
                    num_warps=c["num_warps"], num_stages=c["num_stages"], waves_per_eu=0, matrix_instr_nonkdim=16,
                    kpack=1,
                )  # fmt: skip

            t = triton.testing.do_bench(run, warmup=10, rep=50)
            if best is None or t < best[0]:
                best = (t, c)
        choice[0] = best[1]
    dist.broadcast_object_list(choice, src=0)
    return choice[0]


class Problem:
    """Inputs, output and per-rank references for one shape."""

    def __init__(self, ctx, m_local, n, k, cfg):
        self.ctx, self.rank, self.world = ctx, ctx.get_rank(), ctx.get_num_ranks()
        self.m_local, self.n, self.k, self.cfg = m_local, n, k, cfg
        bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
        if m_local % bm or n % bn or k % bk:
            raise ValueError(f"{m_local}x{n}x{k} is not a multiple of the {bm}x{bn}x{bk} blocks")
        self.tiles_m, self.tiles_n = m_local // bm, n // bn
        self.num_tiles = self.tiles_m * self.tiles_n
        dev = torch.device("cuda", torch.cuda.current_device())
        self.A = self._a(self.rank, dev)
        self.W = torch.randn(n, k, dtype=DTYPE, device=dev, generator=torch.Generator(dev).manual_seed(12345))
        self.out = ctx.zeros(self.world * m_local, n, dtype=DTYPE)
        self.mine = self.out[self.rank * m_local : (self.rank + 1) * m_local]
        self.arrivals = ctx.zeros(self.world, dtype=torch.int64)
        self.row_counters = torch.zeros(self.tiles_m, dtype=torch.int64, device=dev)
        self.scratch = torch.zeros(64 * 16, dtype=torch.int64, device=dev)
        self.status = torch.zeros(self.world, dtype=torch.int32, device=dev)
        self.dummy = torch.zeros(1, dtype=torch.int64, device=dev)
        self.sign = 1
        self._expected = {}
        self.events = []

    def _a(self, r, dev):
        g = torch.Generator(dev).manual_seed(1000 + r)
        return torch.randn(self.m_local, self.k, dtype=DTYPE, device=dev, generator=g)

    def order(self):
        """Tile id of each dispatch step, from the kernel's own mapping."""
        c = self.cfg
        return [
            gk.tile_of.fn(i, self.tiles_m, self.tiles_n, c["GROUP_M"], c["NUM_XCDS"]) for i in range(self.num_tiles)
        ]

    def gemm(self, mode, rank=None, out=None, A=None, **kw):
        c = self.cfg
        out = self.out if out is None else out
        d = self.dummy
        args = dict(
            heap_bases=kw.get("heap_bases", d),
            ce_ctx=kw.get("ce_ctx", d),
            arrivals=kw.get("arrivals", d),
            scratch=kw.get("scratch", d),
            row_counters=kw.get("row_counters", d),
            handle=kw.get("handle", d),
            first_chunk=kw.get("first_chunk", 0),
            row_chunk=kw.get("row_chunk", d),
            TILES_PER_CHUNK=kw.get("tiles_per_chunk", 1),
        )
        A = self.A if A is None else A
        gk.gemm_all_gather[(self.num_tiles,)](
            A, self.W, out, self.m_local, self.n, self.k, A.stride(0), self.W.stride(0), out.stride(0),
            self.rank if rank is None else rank, **args,
            WORLD=self.world, MODE=mode, BLOCK_M=c["BLOCK_M"], BLOCK_N=c["BLOCK_N"], BLOCK_K=c["BLOCK_K"],
            GROUP_M=c["GROUP_M"], NUM_XCDS=c["NUM_XCDS"], num_warps=c["num_warps"], num_stages=c["num_stages"],
            waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1,
        )  # fmt: skip

    def torch_gemm(self, slices=1):
        ms = self.m_local // slices
        for s in range(slices):
            torch.matmul(self.A[s * ms : (s + 1) * ms], self.W.T, out=self.mine[s * ms : (s + 1) * ms])

    def timed(self, fn):
        """Run ``fn`` between two events, to time the GEMM kernel inside a method."""
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        self.events.append((start, end))

    def expected(self, kind):
        """
        All of C as each rank computes it, for the current sign of W: ``ours``, ``torch`` (one
        hipBLASLt call per rank) or ``torch_s<S>`` (S row slices).
        """
        if kind not in self._expected:
            dev = self.out.device
            ref = torch.empty_like(self.out)
            slices = int(kind[len("torch_s") :]) if kind.startswith("torch_s") else 1
            ms = self.m_local // slices
            for r in range(self.world):
                A = self._a(r, dev)
                if kind == "ours":
                    self.gemm(gk.LOCAL, rank=r, out=ref, A=A)
                else:
                    for s in range(slices):
                        rows = slice(r * self.m_local + s * ms, r * self.m_local + (s + 1) * ms)
                        torch.matmul(A[s * ms : (s + 1) * ms], self.W.T, out=ref[rows])
            torch.cuda.synchronize()
            self._expected[kind] = (self.sign, ref)
        sign, ref = self._expected[kind]
        return ref if sign == self.sign else -ref

    def next_iteration(self):
        """Untimed: flip W's sign and wipe ``out``, so an earlier iteration's data cannot pass."""
        self.W.neg_()
        self.sign = -self.sign
        self.out.fill_(float("nan"))

    def check(self, kind, gathered=True, approx=False):
        torch.cuda.synchronize()
        exp = self.expected(kind)
        got = self.out
        if not gathered:
            exp, got = exp[self.rank * self.m_local : (self.rank + 1) * self.m_local], self.mine
        if approx:
            # A different GEMM: bf16 rounding differences, but a stale (negated) or missing tile fails
            tol = 2e-2 * float(exp.abs().max())
            bad = ~((got.float() - exp.float()).abs() <= tol)
        else:
            bad = ~(got == exp)
        if bad.any():
            rows = bad.any(dim=1).nonzero()[:, 0] // self.m_local
            raise RuntimeError(
                f"{int(bad.sum())}/{got.numel()} elements wrong; source ranks {sorted(set(rows.tolist()))}"
            )


def check_status(status, what):
    if not bool(status.all()):
        raise RuntimeError(f"{what}: {int((status == 0).sum())} waits ran out of budget")


class Arm:
    """How to run one method: the timed ``run``, the untimed ``preamble``, ``finish``, and its check."""

    def __init__(self, run, preamble, finish=None, kind="ours", gathered=True, approx=False):
        self.run, self.preamble, self.finish = run, preamble, finish
        self.kind, self.gathered, self.approx = kind, gathered, approx


def counter_arm(p, mode, per_round, **kw):
    """A GEMM whose epilogue sends the tiles itself and bumps arrival counters, then a wait for them."""
    ctx = p.ctx
    p.arrivals.zero_()
    p.row_counters.zero_()
    ctx.barrier()
    iteration = [0]
    kw.update(heap_bases=ctx.get_heap_bases(), arrivals=p.arrivals)

    def run():
        iteration[0] += 1
        p.timed(lambda: p.gemm(mode, **kw))
        gk.wait_arrivals[(p.world,)](p.arrivals, iteration[0] * per_round, p.rank, p.status, BUDGET=WAIT_BUDGET)

    return Arm(run, p.next_iteration, lambda: check_status(p.status, "arrival wait"))


def plan_arm(p, schedule, mode, loop=False, **kw):
    """
    A GEMM marking a plan's chunks ready, then a wait kernel. With ``loop``, the timed region is a
    whole stream-ordered loop iteration (``start()`` ... ``next_round()``); otherwise the host's part
    is untimed.
    """
    ctx = p.ctx
    plan = TriggeredPlan(ctx, p.out, schedule, timeout=60, prepare_ahead=True)
    status = torch.zeros(plan.num_chunks, dtype=torch.int32, device=p.out.device)
    first = plan.chunk_id(p.rank, 0)
    handle = [None]
    overhead = []

    def launch():
        p.timed(lambda: p.gemm(mode, handle=handle[0], first_chunk=first, **kw))
        gk.wait_plan[(plan.num_chunks,)](handle[0], status, BUDGET=WAIT_BUDGET, num_warps=1)

    if loop:

        def run():
            t0 = time.perf_counter()
            handle[0] = plan.start()
            p.timed(lambda: p.gemm(mode, handle=handle[0], first_chunk=first, **kw))
            plan.prepare_next()
            gk.wait_plan[(plan.num_chunks,)](handle[0], status, BUDGET=WAIT_BUDGET, num_warps=1)
            plan.wait(blocking=False)
            plan.next_round()
            overhead.append(time.perf_counter() - t0)

        preamble = p.next_iteration
    else:
        run = launch

        def preamble():
            t = 0.0
            if plan.state is State.RUNNING:
                plan.wait()  # blocking: checks the round's marks after the GPU is done; not host prep
                t0 = time.perf_counter()
                plan.next_round()
                t = time.perf_counter() - t0
            p.next_iteration()
            t0 = time.perf_counter()
            handle[0] = plan.start()
            plan.prepare_next()
            overhead.append(t + time.perf_counter() - t0)

    def finish():
        try:
            if plan.state is State.RUNNING:
                plan.wait()
            check_status(status, "triggered plan")
        finally:
            plan.close()
        extra = {"chunks": len(schedule)}
        if overhead:
            extra["round_overhead_us"] = statistics.median(overhead) * 1e6
        return extra

    return Arm(run, preamble, finish)


def tile_schedule(p, tiles_per_chunk):
    c = p.cfg
    return tile_chunks(p.mine, (c["BLOCK_M"], c["BLOCK_N"]), tiles_per_chunk, order=p.order(), base=p.out)


def make_arm(method, p, param):
    ctx = p.ctx
    if method == "gemm":
        return Arm(lambda: p.timed(lambda: p.gemm(gk.LOCAL)), p.next_iteration, gathered=False)
    if method == "torch_gemm":
        return Arm(lambda: p.timed(p.torch_gemm), p.next_iteration, kind="torch", gathered=False)
    if method == "rccl_ag":

        def preamble():
            p.next_iteration()
            p.gemm(gk.LOCAL)

        return Arm(lambda: dist.all_gather_into_tensor(p.out, p.mine), preamble)
    if method == "rccl":

        def run():
            p.timed(lambda: p.gemm(gk.LOCAL))
            dist.all_gather_into_tensor(p.out, p.mine)

        return Arm(run, p.next_iteration)
    if method == "torch_rccl":

        def run():
            p.timed(p.torch_gemm)
            dist.all_gather_into_tensor(p.out, p.mine)

        return Arm(run, p.next_iteration, kind="torch")
    if method == "torch_rccl_overlap":
        slices, ms = param, p.m_local // param
        comm = torch.cuda.Stream()
        views = [
            [p.out[r * p.m_local + s * ms : r * p.m_local + (s + 1) * ms] for r in range(p.world)]
            for s in range(slices)
        ]

        def run():
            compute = torch.cuda.current_stream()
            for s in range(slices):
                torch.matmul(p.A[s * ms : (s + 1) * ms], p.W.T, out=views[s][p.rank])
                comm.wait_stream(compute)
                with torch.cuda.stream(comm):
                    dist.all_gather(views[s], views[s][p.rank])
            compute.wait_stream(comm)

        return Arm(run, p.next_iteration, kind=f"torch_s{slices}")
    if method == "iris_ops":
        return Arm(lambda: matmul_all_gather(ctx, p.out, p.A, p.W.T), p.next_iteration, kind="torch", approx=True)
    if method == "iris_cu":
        return counter_arm(p, gk.CU, 1, row_counters=p.row_counters)
    if method == "sdma_tile":
        return counter_arm(
            p, gk.SDMA_TILE, 1, ce_ctx=ctx.get_copy_engine_ctx(), scratch=p.scratch, row_counters=p.row_counters
        )
    if method == "sdma_rows":
        return counter_arm(p, gk.SDMA_ROWS, p.tiles_m, ce_ctx=ctx.get_copy_engine_ctx(), row_counters=p.row_counters)
    if method == "plan_tiles":
        return plan_arm(p, tile_schedule(p, param), gk.PLAN_TILES, tiles_per_chunk=param)
    if method == "plan_tiles_loop":
        return plan_arm(p, tile_schedule(p, param), gk.PLAN_TILES, loop=True, tiles_per_chunk=param)
    if method == "plan_rows":
        # Row block m completes with its last tile: send row blocks in that order
        last = np.zeros(p.tiles_m, dtype=np.int64)
        for step, tile in enumerate(p.order()):
            last[tile // p.tiles_n] = step
        order_m = np.argsort(last, kind="stable")
        rows = chunks(p.mine, p.tiles_m, base=p.out, writers=p.tiles_n)
        row_chunk = torch.empty(p.tiles_m, dtype=torch.int64, device=p.out.device)
        row_chunk[torch.as_tensor(order_m, device=p.out.device)] = torch.arange(p.tiles_m, device=p.out.device)
        return plan_arm(p, [rows[m] for m in order_m], gk.PLAN_ROWS, row_chunk=row_chunk)
    raise ValueError(method)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--shapes", nargs="+", default=SHAPES, help="M_local x N x K")
    parser.add_argument("--block", default="auto", help="BLOCK_M x BLOCK_N x BLOCK_K, or auto (per shape)")
    parser.add_argument("--group_m", type=int, default=8)
    parser.add_argument("--num_warps", type=int, default=8, help="with an explicit --block")
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=[],
        help="per shape, overriding --block: M_localxNxK=BLOCK_MxBLOCK_NxBLOCK_KwWARPS",
    )
    parser.add_argument("--tiles_per_chunk", nargs="+", type=int, default=None, help="plan_tiles; default: CUs, CUs/4")
    parser.add_argument("--slices", nargs="+", type=int, default=[2, 4, 8], help="torch_rccl_overlap")
    parser.add_argument("--n_warmup", type=int, default=5)
    parser.add_argument("--n_repeat", type=int, default=20)
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    world = dist.get_world_size()
    shapes = [tuple(int(x) for x in s.split("x")) for s in args.shapes]
    # The heap does not reuse freed memory: room for every shape's output, plus plan gates and counters
    heap = sum(world * m * n * DTYPE.itemsize for m, n, _ in shapes) + (64 << 20)
    ctx = iris.iris(heap, engines=("sdma",))
    if not isinstance(ctx.heap.allocator, TorchAllocator):
        raise RuntimeError("run on the torch allocator (unset IRIS_ALLOCATOR)")
    rank = ctx.get_rank()
    cus = torch.cuda.get_device_properties(local_rank).multi_processor_count
    tpcs = args.tiles_per_chunk or [cus, cus // 4]

    rows_out = []
    if rank == 0:
        gpu = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        print(f"{world} ranks, {gpu}, G{args.group_m}, median of {args.n_repeat}")
        head = f"{'shape':<17} {'blocks':<13} {'method':<18} {'param':>5} {'time_us':>9} {'gemm_us':>9} {'TFLOPs':>7}"
        print(head + f" {'exposed_us':>10}  extra", flush=True)
    for m_local, n, k in shapes:
        pinned = dict(b.split("=") for b in args.blocks).get(f"{m_local}x{n}x{k}")
        if pinned:
            dims, warps = pinned.split("w")
            cfg = config(*(int(x) for x in dims.split("x")), int(warps), args.group_m)
        elif args.block == "auto":
            cfg = pick_config(m_local, n, k, args.group_m)
        else:
            bm, bn, bk = (int(x) for x in args.block.split("x"))
            cfg = config(bm, bn, bk, args.num_warps, args.group_m)
        blocks = f"{cfg['BLOCK_M']}x{cfg['BLOCK_N']}x{cfg['BLOCK_K']}w{cfg['num_warps']}"
        p = Problem(ctx, m_local, n, k, cfg)
        ref_us = {}
        for method in args.methods:
            if method in ("plan_tiles",):
                params = tpcs
            elif method == "plan_tiles_loop":
                params = [cus]
            elif method == "torch_rccl_overlap":
                params = [s for s in args.slices if m_local % s == 0]
            else:
                params = [0]
            for param in params:
                try:
                    arm = make_arm(method, p, param)
                    p.events = []
                    times = iris.do_bench(
                        arm.run, barrier_fn=ctx.barrier, preamble_fn=arm.preamble, n_warmup=args.n_warmup,
                        n_repeat=args.n_repeat, return_mode="all",
                    )  # fmt: skip
                    extra = (arm.finish() if arm.finish else None) or {}
                    ctx.barrier()
                    p.check(arm.kind, arm.gathered, arm.approx)
                    ctx.barrier()
                except Exception as e:  # noqa: BLE001
                    if method != "iris_ops":
                        raise
                    # Fails the same way on every rank (a compile error), so the ranks stay in step
                    if rank == 0:
                        print(f"{m_local}x{n}x{k:<9} {method}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                    ctx.barrier()
                    continue
                us = statistics.median(times) * 1e3
                events = p.events[-args.n_repeat :]
                gemm_phase = statistics.median(a.elapsed_time(b) for a, b in events) * 1e3 if events else None
                if method in ("gemm", "torch_gemm"):
                    ref_us[method] = us
                base = ref_us.get("torch_gemm" if method.startswith("torch") else "gemm")
                comm = method not in ("gemm", "torch_gemm", "rccl_ag", "iris_ops")
                exposed = us - base if comm and base is not None else None
                tflops = 2 * m_local * n * k / (us * 1e-6) / 1e12
                row = dict(m_local=m_local, n=n, k=k, ranks=world, blocks=blocks, method=method, param=param or "",
                           time_us=us, gemm_phase_us="" if gemm_phase is None else gemm_phase, gemm_tflops=tflops,
                           exposed_us="" if exposed is None else exposed, **extra)  # fmt: skip
                rows_out.append(row)
                if rank == 0:
                    shape = f"{m_local}x{n}x{k}"
                    tail = " ".join(f"{a}={b:.0f}" for a, b in extra.items())
                    ex = f"{exposed:10.1f}" if exposed is not None else f"{'':>10}"
                    gp = f"{gemm_phase:9.1f}" if gemm_phase is not None else f"{'':>9}"
                    line = f"{shape:<17} {blocks:<13} {method:<18} {param or '':>5} {us:9.1f} {gp} {tflops:7.0f} {ex}"
                    print(f"{line}  {tail}", flush=True)
        del p
        ctx.barrier()
    if rank == 0 and args.csv:
        fields = ["m_local", "n", "k", "ranks", "blocks", "method", "param", "time_us", "gemm_phase_us",
                  "gemm_tflops", "exposed_us", "round_overhead_us", "chunks"]  # fmt: skip
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows_out)
    ctx.barrier()
    del ctx
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
