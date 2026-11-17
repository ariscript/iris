# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
GEMM + all-gather kernels for bench_gemm_all_gather.py.

Each rank computes ``C_r = A_r @ W.T`` (``A_r`` is ``M_local x K``, ``W`` is ``N x K``) into rows
``[r * M_local, (r + 1) * M_local)`` of ``out`` (``world * M_local x N``, on the symmetric heap), and
every rank ends with all of ``out``. One program computes one tile (the grid is the tile count), in
the order ``tile_of`` gives; the epilogue ``MODE`` decides how the tile reaches the peers:

- LOCAL: store locally only (the GEMM alone, or followed by RCCL).
- CU: also store the tile into every peer's ``out`` from the CUs; the last tile signals each peer once.
- SDMA_TILE: copy the tile to every peer on the device-initiated SDMA queues (2D sub-window copy);
  the last tile then adds 1 to each peer's arrival counter for this rank, on the same queue.
- SDMA_ROWS: the last tile of a row block copies the whole row block (one linear copy) to every
  peer, then adds 1 to the peer's arrival counter.
- PLAN_TILES / PLAN_ROWS: mark the tile's chunk of a triggered plan ready (tile_chunks, or one
  chunk per row block); the copies were prepared on the host.
"""

import triton
import triton.language as tl
from xio import sdma_ep

import iris
from iris.device import sdma_utils
from iris.mem.triton.context import __translate
from iris.mem.utils import wait_cnt
from iris.experimental.plan import TriggeredView

LOCAL = tl.constexpr(0)
CU = tl.constexpr(1)
SDMA_TILE = tl.constexpr(2)
SDMA_ROWS = tl.constexpr(3)
PLAN_TILES = tl.constexpr(4)
PLAN_ROWS = tl.constexpr(5)


@triton.jit
def tile_of(step, tiles_m, tiles_n, GROUP_M: tl.constexpr, NUM_XCDS: tl.constexpr):
    """
    Tile id (row-major) of the program dispatched ``step``-th. Programs go to XCDs round-robin;
    each XCD gets a contiguous range of grouped tile ids, so its L2 sees neighbouring tiles.
    Plain integer arithmetic, so the host runs it as ``tile_of.fn``.
    """
    num_tiles = tiles_m * tiles_n
    per_xcd = num_tiles // NUM_XCDS
    t = step
    if step < per_xcd * NUM_XCDS:
        t = (step % NUM_XCDS) * per_xcd + step // NUM_XCDS
    in_group = GROUP_M * tiles_n
    first_m = (t // in_group) * GROUP_M
    size_m = min(tiles_m - first_m, GROUP_M)
    pid_m = first_m + (t % in_group) % size_m
    pid_n = (t % in_group) // size_m
    return pid_m * tiles_n + pid_n


@triton.jit
def _queue(ce_ctx, peer):
    q = ce_ctx + sdma_ep.QUEUE_DEVICE_CTX_SIZE * peer
    return (
        tl.load(q + 0).to(tl.pointer_type(tl.uint32)),
        tl.load(q + 1).to(tl.pointer_type(tl.uint64)),
        tl.load(q + 2).to(tl.pointer_type(tl.uint64)),
        tl.load(q + 3).to(tl.pointer_type(tl.uint64)),
        tl.load(q + 4).to(tl.pointer_type(tl.uint64)),
        tl.load(q + 5).to(tl.pointer_type(tl.uint64)),
    )


@triton.jit
def sdma_copy_linear(ce_ctx, peer, src, dst, nbytes):
    """One COPY_LINEAR on this rank's device-initiated queue to ``peer`` (scalar addresses)."""
    queue, read_ptr, write_ptr, doorbell, cached, committed = _queue(ce_ctx, peer)
    size = sdma_ep.COPY_LINEAR_COMMAND_BYTES
    base, offset = sdma_utils.acquire_fadd(queue, read_ptr, write_ptr, doorbell, cached, committed, size)
    sdma_utils.place_nop_packet(queue, base, offset)
    sdma_utils.place_copy_packet(queue, base + offset, tl.cast(nbytes, tl.uint32), src.to(tl.uint64), dst.to(tl.uint64))
    sdma_utils.submit(write_ptr, doorbell, committed, base, base + offset + size)


@triton.jit
def sdma_copy_tile(ce_ctx, peer, src_base, dst_base, width_bytes, height, pitch_bytes, x_bytes, y):
    """One sub-window copy of a ``height x width_bytes`` tile at ``(x_bytes, y)``, same place in both buffers."""
    queue, read_ptr, write_ptr, doorbell, cached, committed = _queue(ce_ctx, peer)
    size = sdma_ep.COPY_LINEAR_SUB_WINDOW_COMMAND_BYTES
    base, offset = sdma_utils.acquire_fadd(queue, read_ptr, write_ptr, doorbell, cached, committed, size)
    sdma_utils.place_nop_packet(queue, base, offset)
    sdma_utils.place_sub_window_copy_packet(
        queue,
        base + offset,
        src_base.to(tl.uint64),
        dst_base.to(tl.uint64),
        tile_width=tl.cast(width_bytes, tl.uint32),
        tile_height=tl.cast(height, tl.uint32),
        src_buffer_pitch=tl.cast(pitch_bytes, tl.uint32),
        dst_buffer_pitch=tl.cast(pitch_bytes, tl.uint32),
        src_x=tl.cast(x_bytes, tl.uint32),
        src_y=tl.cast(y, tl.uint32),
        dst_x=tl.cast(x_bytes, tl.uint32),
        dst_y=tl.cast(y, tl.uint32),
    )
    sdma_utils.submit(write_ptr, doorbell, committed, base, base + offset + size)


@triton.jit
def gemm_all_gather(
    A,
    W,
    out,
    M_local,
    N,
    K,
    stride_am,
    stride_wn,
    stride_om,
    rank,
    heap_bases,
    ce_ctx,
    arrivals,
    scratch,
    row_counters,
    handle,
    first_chunk,
    row_chunk,
    TILES_PER_CHUNK,
    WORLD: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    STRIP_M: tl.constexpr = 32,
):
    step = tl.program_id(0)
    tiles_m = tl.cdiv(M_local, BLOCK_M)
    tiles_n = tl.cdiv(N, BLOCK_N)
    tile = tile_of(step, tiles_m, tiles_n, GROUP_M, NUM_XCDS)
    pid_m = tile // tiles_n
    pid_n = tile % tiles_n
    tl.assume(stride_am > 0)
    tl.assume(stride_wn > 0)
    tl.assume(stride_om > 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M_local
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_M), BLOCK_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptr = A + rm[:, None] * stride_am + rk[None, :]
    w_ptr = W + rk[:, None] + rn[None, :] * stride_wn  # W.T as a K x N block (K contiguous)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):  # K % BLOCK_K == 0 (checked on the host)
        a = tl.load(tl.multiple_of(a_ptr, (1, 16)))
        w = tl.load(tl.multiple_of(w_ptr, (16, 1)))
        acc += tl.dot(a, w)
        a_ptr += BLOCK_K
        w_ptr += BLOCK_K
    c = acc.to(out.type.element_ty)

    # M_local % BLOCK_M == 0 and N % BLOCK_N == 0 (checked on the host): no masks
    row0 = rank * M_local + pid_m * BLOCK_M
    om = row0 + tl.arange(0, BLOCK_M)
    on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(out + om[:, None] * stride_om + on[None, :], c)

    if MODE == CU:
        # Stream the tile from L2 to each peer in strips: storing the register tile to 3+ places needs
        # a layout conversion through 128 KiB of LDS (or spills, in a loop)
        wait_cnt()
        tl.debug_barrier()
        for j in range(1, WORLD):
            peer = (rank + j) % WORLD
            dst = __translate(out, rank, peer, heap_bases)
            for r in range(0, BLOCK_M, STRIP_M):
                so = (row0 + r + tl.arange(0, STRIP_M))[:, None] * stride_om + on[None, :]
                tl.store(dst + so, tl.load(out + so))
        # Every wave's remote stores done, then a release on the ticket (row_counters[0]); the last
        # tile acquires every other tile's and releases one arrival signal to each peer
        wait_cnt()
        tl.debug_barrier()
        ticket = tl.atomic_add(row_counters, 1, sem="acq_rel", scope="sys")
        if ticket % (tiles_m * tiles_n) == tiles_m * tiles_n - 1:
            for q in tl.static_range(1, WORLD):
                peer = (rank + q) % WORLD
                iris.atomic_add(arrivals + rank, 1, rank, peer, heap_bases, sem="release", scope="sys")
    elif MODE == SDMA_TILE:
        # The copy engine reads behind L2: write the tile back before queueing its copies
        wait_cnt()
        tl.debug_barrier()
        tl.atomic_add(scratch + (step % 64) * 16, 1, sem="release", scope="sys")
        elem = out.type.element_ty.primitive_bitwidth // 8
        for j in tl.static_range(1, WORLD):
            peer = (rank + j) % WORLD
            dst = __translate(out, rank, peer, heap_bases)
            sdma_copy_tile(
                ce_ctx, peer, out, dst, BLOCK_N * elem, BLOCK_M, stride_om * elem, pid_n * BLOCK_N * elem, row0
            )
        # Each queue runs in order and every tile queued its copies before taking a ticket, so one
        # arrival signal per peer from the last tile follows them all (row_counters[0] is the ticket)
        ticket = tl.atomic_add(row_counters, 1, sem="acq_rel", scope="gpu")
        if ticket % (tiles_m * tiles_n) == tiles_m * tiles_n - 1:
            for j in tl.static_range(1, WORLD):
                peer = (rank + j) % WORLD
                iris.atomic_add(
                    arrivals + rank, 1, rank, peer, heap_bases, copy_engine_ctx=ce_ctx, use_copy_engine=True
                )
    elif MODE == SDMA_ROWS:
        wait_cnt()
        tl.debug_barrier()
        ticket = tl.atomic_add(row_counters + pid_m, 1, sem="release", scope="sys")
        if ticket % tiles_n == tiles_n - 1:
            elem = out.type.element_ty.primitive_bitwidth // 8
            src = out + row0 * stride_om
            for j in tl.static_range(1, WORLD):
                peer = (rank + j) % WORLD
                dst = __translate(src, rank, peer, heap_bases)
                sdma_copy_linear(ce_ctx, peer, src, dst, BLOCK_M * stride_om * elem)
                iris.atomic_add(
                    arrivals + rank, 1, rank, peer, heap_bases, copy_engine_ctx=ce_ctx, use_copy_engine=True
                )
    elif MODE == PLAN_TILES:
        TriggeredView.initialize(handle).mark_ready(first_chunk + step // TILES_PER_CHUNK)
    elif MODE == PLAN_ROWS:
        TriggeredView.initialize(handle).mark_ready(first_chunk + tl.load(row_chunk + pid_m))


@triton.jit
def wait_arrivals(arrivals, target, rank, status, BUDGET: tl.constexpr):
    """Program s waits until rank s's arrival counter here reaches ``target``."""
    s = tl.program_id(0)
    ok = s == rank
    i = 0
    while (ok == 0) & (i < BUDGET):
        ok = tl.atomic_add(arrivals + s, 0, sem="acquire", scope="sys") >= target
        i += 1
    tl.store(status + s, ok.to(tl.int32))


@triton.jit
def wait_plan(handle, status, BUDGET: tl.constexpr):
    """Program q waits for chunk q to arrive here (or, for this rank's own, to be marked ready)."""
    cv = TriggeredView.initialize(handle)
    q = tl.program_id(0)
    ok = cv.arrived(q)
    i = 0
    while (ok == 0) & (i < BUDGET):
        ok = cv.arrived(q)
        i += 1
    tl.store(status + q, ok.to(tl.int32))
