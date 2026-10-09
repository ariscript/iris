# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon device side of triggered SDMA plans (see ``iris.triggered``).

Same view and gate layout as ``iris.mem.triton.triggered``; only the kernel language differs.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from iris.mem.triton.triggered import (
    G_ARRIVAL,
    G_COMPLETION,
    G_COUNTER,
    G_READY,
    GATE_WORDS,
    H_EPOCH,
    H_GATES,
    H_RANK,
    HEADER_WORDS,
    R_CONTRIBUTORS,
    R_OWNER,
    R_ROLE,
    RECORD_WORDS,
    ROLE_CONSUME,
    ROLE_PUBLISH,
)
from iris.mem.utils import wait_cnt


@aggregate
class TriggeredView:
    """
    Gluon device handle for one epoch of a triggered plan.

    Usage::

        @gluon.jit
        def producer(out, view, ...):
            tv = TriggeredView.initialize(view)
            ...                          # write this program's part of batch b
            tv.publish(b)                # whole CTA, after its writes

        @gluon.jit
        def consumer(out, view, ...):
            tv = TriggeredView.initialize(view)
            tv.wait(b)                   # batch b's data is now visible
            x = gl.load(out + ...)
    """

    view: gl.tensor
    gates: gl.tensor
    epoch: gl.tensor
    rank: gl.tensor

    @gluon.constexpr_function
    def __init__(self, view, gates, epoch, rank):
        self.view = view
        self.gates = gates
        self.epoch = epoch
        self.rank = rank

    @staticmethod
    @gluon.jit
    def initialize(view):
        """Read the plan's view tensor (from ``TriggeredPlan.begin_device_epoch()``)."""
        gates = gl.cast(gl.load(view + H_GATES), gl.pointer_type(gl.int64))
        return TriggeredView(view, gates, gl.load(view + H_EPOCH), gl.load(view + H_RANK))

    @gluon.jit
    def _record(self, batch, word):
        return gl.load(self.view + HEADER_WORDS + batch * RECORD_WORDS + word)

    @gluon.jit
    def can_publish(self, batch):
        return (self._record(batch, R_ROLE) & ROLE_PUBLISH) != 0

    @gluon.jit
    def can_consume(self, batch):
        return (self._record(batch, R_ROLE) & ROLE_CONSUME) != 0

    @gluon.jit
    def publish(self, batch):
        """
        Contribute to batch ``batch``; the last of its contributors opens the gate.

        Call from the whole CTA, after the CTA's writes to the batch: outside divergent branches
        and outside ``gl.warp_specialize`` partitions.
        """
        # Copy engines read behind L2: every wave's stores must land before the release writes back
        wait_cnt()
        gl.barrier()
        gate = self.gates + batch * GATE_WORDS
        contributors = self._record(batch, R_CONTRIBUTORS)
        ticket = gl.atomic_add(gate + G_COUNTER, 1, sem="acq_rel", scope="sys")
        if ticket % contributors == contributors - 1:
            gl.atomic_xchg(gate + G_READY, self.epoch, sem="release", scope="sys")

    @gluon.jit
    def arrived(self, batch):
        """
        True once batch ``batch`` is visible here this epoch (published, for this rank's own).

        Call uniformly across the CTA: the result reaches the other waves through a CTA barrier, so
        every wave must make the same number of calls (true of ``wait`` and ``reusable`` too).
        """
        gate = self.gates + batch * GATE_WORDS
        word = gl.where(self._record(batch, R_OWNER) == self.rank, G_READY, G_ARRIVAL)
        return gl.atomic_add(gate + word, 0, sem="acquire", scope="sys") >= self.epoch

    @gluon.jit
    def wait(self, batch):
        """Spin until ``arrived(batch)``."""
        while self.arrived(batch) == 0:
            pass

    @gluon.jit
    def reusable(self, batch):
        """True once batch ``batch`` is complete this epoch (set by ``end_device_epoch()``)."""
        gate = self.gates + batch * GATE_WORDS
        return gl.atomic_add(gate + G_COMPLETION, 0, sem="acquire", scope="sys") >= self.epoch
