"""The accumulation schedule, isolated from the trainer so it can be tested at all.

Getting this wrong is silent: _sync_grads_outside_fsdp does all_reduce(SUM) then
div_(world_size), so calling it on a non-final micro-step divides already-reduced
gradients a second time and quietly scales the update down.
"""
import pytest

from alaya.trainer.grad_accum import accum_flags


def test_accum_steps_one_is_the_existing_behaviour():
    for micro in range(5):
        assert accum_flags(micro, 1) == (True, True, 1.0)


def test_accum_steps_four_zeroes_first_and_steps_last():
    flags = [accum_flags(i, 4) for i in range(8)]
    assert [f[0] for f in flags] == [True, False, False, False, True, False, False, False]
    assert [f[1] for f in flags] == [False, False, False, True, False, False, False, True]


def test_loss_scale_is_the_reciprocal_so_windows_average():
    assert accum_flags(0, 4)[2] == pytest.approx(0.25)
    assert accum_flags(3, 4)[2] == pytest.approx(0.25)


def test_zero_and_step_never_collide_for_accum_above_one():
    for micro in range(12):
        first, last, _ = accum_flags(micro, 3)
        assert not (first and last)


def test_rejects_a_non_positive_window():
    with pytest.raises(ValueError):
        accum_flags(0, 0)
