"""Pre-encode every WBench prompt into runtime.text_embed_cache_dir.

Gemma-3-12B is 24GB in bf16, so on a 24GB card it cannot sit next to the DiT.
The WBench prompt set is finite and fully enumerable ahead of time (one base
caption plus one accumulated prompt-schedule entry per turn, per case), so we
encode it once here — with the text encoder spread over as many GPUs as it needs
— and the generation run then serves every prompt from the on-disk cache with
ALAYA_SKIP_TEXT_ENCODER=1.

    python -m scripts.tools.precache_wbench_text_embeds --config configs/wbench_full.yaml

Re-runs are incremental: prompts already on disk are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="the same yaml the generation run uses")
    parser.add_argument("--mode", default=None, help="validation mode name (default: the single wbench mode)")
    parser.add_argument("--cache-dir", default=None, help="override runtime.text_embed_cache_dir")
    parser.add_argument("--device-map", default="auto", help='accelerate device_map for Gemma ("auto", "cpu", ...)')
    parser.add_argument("--dry-run", action="store_true", help="only enumerate and report, do not load Gemma")
    args = parser.parse_args()

    from alaya.config.loader import load_config
    from alaya.data.dataloader import build_validation_dataset
    from alaya.data.text_embed_cache import cache_path, disk_put

    cfg = load_config(args.config)
    modes = cfg.validation.modes
    mode_name = args.mode or next(
        (n for n, m in modes.items() if str(m.dataset.source) == "wbench_navi"), None
    )
    if mode_name is None:
        raise SystemExit("no validation mode with dataset.source=wbench_navi in this config")
    mode_cfg = modes[mode_name]

    cache_dir = args.cache_dir or cfg.runtime.text_embed_cache_dir
    if not cache_dir:
        raise SystemExit("no cache dir: set runtime.text_embed_cache_dir or pass --cache-dir")
    os.makedirs(cache_dir, exist_ok=True)

    # frames=1: only the prompt fields matter here, so skip building the pixel tensors.
    dataset = build_validation_dataset(cfg, mode_cfg, min_frames=1, max_frames=1)
    print(f"[Precache] mode={mode_name} cases={len(dataset)} cache_dir={cache_dir}", flush=True)

    prompts: list[str] = []
    seen: set[str] = set()
    # Classifier-free guidance is always on in validation (_validation_cfg_scale maps
    # cfg_scale<=1 to 3.0), so the negative prompt is encoded once per sample too.
    for text in [cfg.validation.negative_prompt]:
        if text:
            seen.add(str(text))
            prompts.append(str(text))
    for i in range(len(dataset)):
        sample = dataset[i]
        meta = sample["metadata"]
        for text in [sample["caption"], *(meta.get("wbench_prompt_schedule") or [])]:
            text = str(text)
            if text and text not in seen:
                seen.add(text)
                prompts.append(text)
    missing = [p for p in prompts if not os.path.exists(cache_path(cache_dir, p))]
    print(
        f"[Precache] {len(prompts)} distinct prompts, {len(prompts) - len(missing)} already cached, "
        f"{len(missing)} to encode",
        flush=True,
    )
    if args.dry_run or not missing:
        return

    from alaya.model.loader import load_text_encoder

    os.environ["ALAYA_GEMMA_DEVICE_MAP"] = args.device_map
    dtype = torch.bfloat16
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    text_encoder, encode_text = load_text_encoder(
        cfg.paths.effective_transformer, cfg.paths.gemma, device=device, dtype=dtype
    )

    t0 = time.time()
    for n, prompt in enumerate(missing, 1):
        with torch.no_grad():
            output = encode_text(text_encoder, [prompt])
        ctx = output[0][0] if isinstance(output, list) and output[0].dim() == 3 else output[0]
        disk_put(cache_dir, prompt, ctx)
        if n % 25 == 0 or n == len(missing):
            rate = n / max(time.time() - t0, 1e-6)
            print(
                f"[Precache] {n}/{len(missing)} ({100 * n / len(missing):.1f}%) "
                f"{rate:.2f} it/s ETA {(len(missing) - n) / max(rate, 1e-6) / 60:.1f}min",
                flush=True,
            )
    print(f"[Precache] done in {(time.time() - t0) / 60:.1f}min -> {cache_dir}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
