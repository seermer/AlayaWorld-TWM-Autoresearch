"""A case's rollout noise comes from its own seed, 42 without one, plus the prompt variant."""
from types import SimpleNamespace

import torch

from alaya.trainer.rollout_trainer import RolloutTrainer


def _draw(metadata, variant=0):
    trainer = SimpleNamespace(dist=SimpleNamespace(device="cpu"))
    return torch.randn(4, generator=RolloutTrainer._validation_noise_generator(trainer, metadata, variant))


def test_noise_follows_the_case_seed_and_the_variant():
    assert torch.equal(_draw({}), _draw({"wbench_seed": 42}))
    assert torch.equal(_draw({"wbench_case_id": "1"}), _draw({"wbench_case_id": "2"}))
    assert not torch.equal(_draw({"wbench_seed": 7}), _draw({}))
    assert not torch.equal(_draw({}, 0), _draw({}, 1))
