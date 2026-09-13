"""The LoRA merge arithmetic and key mapping.

A wrong scaling or a silently-unmatched key produces a checkpoint that loads
cleanly and generates subtly wrong video, so both are asserted here.
"""
import subprocess
import sys
from pathlib import Path

import pytest
import safetensors.torch as st
import torch


def _tiny_base(path: Path):
    torch.save({"blocks.0.attn1.to_q.weight": torch.zeros(8, 8),
                "blocks.1.attn1.to_k.weight": torch.zeros(8, 8),
                "patchify_proj.weight": torch.ones(8, 8)}, path)


def _tiny_lora(path: Path, rank: int = 2):
    a = torch.full((rank, 8), 0.5)
    b = torch.full((8, rank), 2.0)
    st.save_file({
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight": a,
        "diffusion_model.transformer_blocks.0.attn1.to_q.lora_B.weight": b,
    }, str(path))


def test_merge_into_a_pt_base_writes_transformer_pt(tmp_path):
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    _tiny_lora(ckpt / "lora.safetensors")
    torch.save({"dummy": torch.zeros(1)}, ckpt / "history_encoder.pt")
    base = tmp_path / "transformer.pt"
    _tiny_base(base)
    out = tmp_path / "merged"

    result = subprocess.run(
        [sys.executable, "scripts/tools/merge_lora_for_rollout.py",
         "--ckpt_dir", str(ckpt), "--base_transformer", str(base),
         "--output", str(out), "--lora_rank", "2", "--lora_alpha", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    merged = torch.load(out / "transformer.pt", map_location="cpu")
    assert (out / "history_encoder.pt").exists()
    # delta = B @ A * (alpha/rank) = 2.0 * 0.5 * 2 * (2/2) = 2.0 in every cell
    expected = torch.full((8, 8), 2.0)
    torch.testing.assert_close(merged["blocks.0.attn1.to_q.weight"], expected)
    # an untouched key must survive byte for byte
    torch.testing.assert_close(merged["patchify_proj.weight"], torch.ones(8, 8))


def test_merge_fails_loudly_on_an_unmatched_lora_key(tmp_path):
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    st.save_file({
        "diffusion_model.transformer_blocks.99.attn1.to_q.lora_A.weight": torch.zeros(2, 8),
        "diffusion_model.transformer_blocks.99.attn1.to_q.lora_B.weight": torch.zeros(8, 2),
    }, str(ckpt / "lora.safetensors"))
    base = tmp_path / "transformer.pt"
    _tiny_base(base)

    result = subprocess.run(
        [sys.executable, "scripts/tools/merge_lora_for_rollout.py",
         "--ckpt_dir", str(ckpt), "--base_transformer", str(base),
         "--output", str(tmp_path / "merged"), "--lora_rank", "2", "--lora_alpha", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "blocks.99" in (result.stderr + result.stdout)


def test_merge_rejects_a_wrong_lora_rank(tmp_path):
    """--lora_rank 4 against a rank-2 LoRA must not silently mis-scale the delta.

    (B @ A) is [out, in] no matter what rank actually is, so the shape assert on
    the merged weight cannot catch this -- only an explicit rank check can.
    """
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    _tiny_lora(ckpt / "lora.safetensors", rank=2)
    base = tmp_path / "transformer.pt"
    _tiny_base(base)

    result = subprocess.run(
        [sys.executable, "scripts/tools/merge_lora_for_rollout.py",
         "--ckpt_dir", str(ckpt), "--base_transformer", str(base),
         "--output", str(tmp_path / "merged"), "--lora_rank", "4", "--lora_alpha", "4"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    combined = result.stderr + result.stdout
    assert "4" in combined and "2" in combined


def test_merge_into_a_safetensors_base_writes_diffusion_pytorch_model(tmp_path):
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    _tiny_lora(ckpt / "lora.safetensors")
    torch.save({"dummy": torch.zeros(1)}, ckpt / "history_encoder.pt")
    base = tmp_path / "base.safetensors"
    st.save_file({"blocks.0.attn1.to_q.weight": torch.zeros(8, 8),
                  "blocks.1.attn1.to_k.weight": torch.zeros(8, 8),
                  "patchify_proj.weight": torch.ones(8, 8)}, str(base))
    out = tmp_path / "merged"

    result = subprocess.run(
        [sys.executable, "scripts/tools/merge_lora_for_rollout.py",
         "--ckpt_dir", str(ckpt), "--base_transformer", str(base),
         "--output", str(out), "--lora_rank", "2", "--lora_alpha", "2"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    out_file = out / "diffusion_pytorch_model.safetensors"
    assert out_file.exists()
    merged = st.load_file(str(out_file))
    assert (out / "history_encoder.pt").exists()
    # delta = B @ A * (alpha/rank) = 2.0 * 0.5 * 2 * (2/2) = 2.0 in every cell
    expected = torch.full((8, 8), 2.0)
    torch.testing.assert_close(merged["blocks.0.attn1.to_q.weight"], expected)
    # an untouched key must survive byte for byte
    torch.testing.assert_close(merged["patchify_proj.weight"], torch.ones(8, 8))
