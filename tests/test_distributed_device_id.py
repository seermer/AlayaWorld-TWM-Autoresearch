"""init_distributed must pass device_id to init_process_group.

Without it torch cannot infer the rank->GPU mapping: every rank logs
"No device id is provided ..." plus a ProcessGroupNCCL warning that explicitly
flags a possible hang if the mapping is ever wrong. Verified on real hardware --
with device_id absent a 2-rank all_reduce raises that UserWarning on both ranks;
with it present the same collective is clean.
"""
import datetime
import torch
import torch.distributed as dist

from alaya.utils import distributed as D


def _fake_env(monkeypatch, world_size=4, local_rank=2):
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setenv("RANK", str(local_rank))


def test_init_distributed_passes_device_id_on_cuda(monkeypatch):
    _fake_env(monkeypatch)
    captured = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _i: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "get_rank", lambda: 2)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: captured.update(kw))

    state = D.init_distributed()

    assert captured["backend"] == "nccl"
    assert captured["device_id"] == torch.device("cuda", 2), (
        "init_process_group must receive device_id so NCCL binds the rank to its GPU"
    )
    assert isinstance(captured["timeout"], datetime.timedelta)
    assert state.device == torch.device("cuda", 2)


def test_init_distributed_sends_no_device_id_on_cpu(monkeypatch):
    """gloo has no device to bind; device_id must stay None rather than a cpu device."""
    _fake_env(monkeypatch)
    captured = {}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "get_rank", lambda: 2)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(dist, "init_process_group", lambda **kw: captured.update(kw))

    D.init_distributed()

    assert captured["backend"] == "gloo"
    assert captured["device_id"] is None
