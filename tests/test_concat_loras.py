"""A concatenated adapter must apply exactly the sum of its parts, in bf16, without
touching the base weights (adding a small delta into a bf16 base rounds most of it away)."""
import pytest
import safetensors.torch as st
import torch
import torch.nn as nn

from alaya.model.lora import LoRAForwardManager
from scripts.tools.concat_loras import concat_loras

KEY = "diffusion_model.transformer_blocks.0.attn1.to_q.lora_{}.weight"


def _lora(path, rank, scale, seed, module="0.attn1.to_q"):
    g = torch.Generator().manual_seed(seed)
    prefix = f"diffusion_model.transformer_blocks.{module}.lora_"
    st.save_file({prefix + "A.weight": (torch.randn(rank, 8, generator=g) * scale).to(torch.bfloat16),
                  prefix + "B.weight": (torch.randn(8, rank, generator=g) * scale).to(torch.bfloat16)},
                 str(path))
    return st.load_file(str(path))


def test_concatenated_delta_is_the_exact_sum(tmp_path):
    big = _lora(tmp_path / "a.safetensors", rank=4, scale=1.0, seed=0)
    small = _lora(tmp_path / "b.safetensors", rank=2, scale=1e-3, seed=1)
    rank = concat_loras([tmp_path / "a.safetensors", tmp_path / "b.safetensors"], tmp_path / "out")
    assert rank == 6
    out = st.load_file(str(tmp_path / "out" / "lora.safetensors"))
    a, b = out[KEY.format("A")], out[KEY.format("B")]
    assert a.dtype == b.dtype == torch.bfloat16
    expected = sum(s[KEY.format("B")].double() @ s[KEY.format("A")].double() for s in (big, small))
    assert torch.equal(b.double() @ a.double(), expected)


def test_concatenated_adapter_loads_and_runs_on_a_bf16_base(tmp_path):
    big = _lora(tmp_path / "a.safetensors", rank=4, scale=1.0, seed=0)
    small = _lora(tmp_path / "b.safetensors", rank=2, scale=1e-3, seed=1)
    concat_loras([tmp_path / "a.safetensors", tmp_path / "b.safetensors"], tmp_path / "out")

    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Module()])
    model.blocks[0].attn1 = nn.Module()
    model.blocks[0].attn1.to_q = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    manager = LoRAForwardManager(trainable=False)
    manager.init_for_training(model, ["to_q"], rank=6, alpha=6, dtype=torch.bfloat16,
                              device=torch.device("cpu"))
    manager.register_hooks(model)
    assert manager.load(str(tmp_path / "out" / "lora.safetensors")) == 2

    linear = model.blocks[0].attn1.to_q
    x = torch.randn(3, 8).to(torch.bfloat16)
    base = linear(x)
    manager.enable()
    got = linear(x)
    want = base + sum(x @ s[KEY.format("A")].T @ s[KEY.format("B")].T for s in (big, small))
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=1e-3)


def test_different_module_sets_are_refused(tmp_path):
    _lora(tmp_path / "a.safetensors", rank=4, scale=1.0, seed=0)
    _lora(tmp_path / "b.safetensors", rank=2, scale=1.0, seed=1, module="1.attn1.to_q")
    with pytest.raises(ValueError, match="different modules"):
        concat_loras([tmp_path / "a.safetensors", tmp_path / "b.safetensors"], tmp_path / "out")
