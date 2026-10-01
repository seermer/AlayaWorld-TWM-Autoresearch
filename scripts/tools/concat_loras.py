"""
Concatenate LoRA adapters into one adapter whose delta is exactly the sum of theirs.

Usage:
    python scripts/tools/concat_loras.py \
        --loras <a>/lora.safetensors <b>/lora.safetensors --output <dir>

Writes <dir>/lora.safetensors with rank = sum of the input ranks, loadable through
paths.dmd_resume with lora.rank set to that sum. Per module:
    A = cat(A_1, A_2, ...) over rows, B = cat(B_1, B_2, ...) over columns,
so B @ A = sum_i B_i @ A_i. The adapters are applied as stored (the runtime hook
adds x @ A.T @ B.T with no alpha/rank factor), and the base weights are untouched.
"""
import argparse
from pathlib import Path

import safetensors.torch as st
import torch


def concat_loras(lora_paths: list[Path], out_dir: Path) -> int:
    """Write out_dir/lora.safetensors; returns the combined rank."""
    states = [st.load_file(str(path), device="cpu") for path in lora_paths]
    keys = set(states[0])
    for path, state in zip(lora_paths[1:], states[1:]):
        if set(state) != keys:
            differing = sorted(keys ^ set(state))
            raise ValueError(f"{path} targets different modules than {lora_paths[0]}: "
                             f"{len(differing)} keys differ, e.g. {differing[:3]}")
    out = {}
    for key in sorted(keys):
        if key.endswith(".lora_A.weight"):
            out[key] = torch.cat([state[key] for state in states], dim=0)
        elif key.endswith(".lora_B.weight"):
            out[key] = torch.cat([state[key] for state in states], dim=1)
        else:
            raise ValueError(f"unexpected lora key: {key}")
    ranks = {out[key].shape[0] for key in out if key.endswith(".lora_A.weight")}
    if len(ranks) != 1:
        raise ValueError(f"modules ended up with different ranks: {sorted(ranks)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    st.save_file(out, str(out_dir / "lora.safetensors"))
    return ranks.pop()


def main():
    p = argparse.ArgumentParser(description="Concatenate LoRA adapters into one exact-sum adapter")
    p.add_argument("--loras", required=True, nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    rank = concat_loras(args.loras, args.output)
    print(f"[Concat] {len(args.loras)} adapters -> {args.output / 'lora.safetensors'} (rank {rank})")


if __name__ == "__main__":
    main()
