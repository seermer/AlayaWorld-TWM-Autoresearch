"""Gradient-accumulation schedule for the rollout trainer.

Kept as a pure function so the schedule can be tested without a GPU, a dataset or
a distributed process group.
"""
from __future__ import annotations


def accum_flags(micro_index: int, accum_steps: int) -> tuple[bool, bool, float]:
    """Return (zero_grad_now, step_now, loss_scale) for one micro-batch.

    `micro_index` counts micro-batches from zero. With accum_steps=1 every
    micro-batch both zeroes and steps at scale 1.0, which is exactly the behaviour
    the trainer had before accumulation existed.
    """
    if accum_steps < 1:
        raise ValueError(f"grad_accum_steps must be >= 1, got {accum_steps}")
    position = micro_index % accum_steps
    return position == 0, position == accum_steps - 1, 1.0 / accum_steps
