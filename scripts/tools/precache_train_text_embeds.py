"""Pre-encode every training caption into runtime.text_embed_cache_dir.

Gemma-3-12B is 24GB in bf16 and cannot sit next to the sharded DiT on a 24GB card,
so low-compute training runs with ALAYA_SKIP_TEXT_ENCODER=1 and serves prompts from
disk. A cache miss then raises rather than silently mis-encoding, so the cache has
to be complete before training starts.

That is only sound when the prompt set is finite, which is a property of the config
rather than of this script -- so the finiteness is asserted here instead of assumed.

    CUDA_VISIBLE_DEVICES=0,1 ALAYA_GEMMA_MAX_MEMORY="0=13GiB,1=13GiB" \
      python scripts/tools/precache_train_text_embeds.py \
          --config configs/stage2b_arsft_lowcompute.yaml --device-map auto

Re-runs are incremental: prompts already on disk are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _assert_finite_prompt_set(cfg) -> None:
    """Refuse to run when the config can draw a prompt this pass cannot enumerate.

    Training reads from the cache with the encoder unloaded, so anything drawn at
    random from a field that is not enumerated here becomes a hard failure hours
    into a run. Better to fail now, naming the key.
    """
    from fastvideo.dataset.t2v_datasets import MultiSourceVideoDataset

    if float(getattr(cfg.data, "abstract_caption_prob", 0.0)) != 0.0:
        raise SystemExit(
            "data.abstract_caption_prob must be 0.0: it draws from an abstract_caption "
            "field that this pass does not enumerate"
        )
    from alaya.data.dataloader import _train_sources

    for name, sources, _weight, _use_cache in _train_sources(cfg):
        for source in sources:
            source_cfg = MultiSourceVideoDataset.SOURCE_CONFIGS.get(source)
            if source_cfg is None:
                raise SystemExit(f"unknown data source: {source!r}")
            # Segment captions are enumerable (every segment prompt is cached); what is
            # not is concatenating neighbouring segments into a new string at read time.
            concat_prob = float(os.environ.get("LTX_SEGMENT_CONCAT_PROB", "0") or 0)
            if source_cfg.get("use_segment_caption", False) and concat_prob > 0:
                raise SystemExit(
                    f"source {name!r} uses segment captions with LTX_SEGMENT_CONCAT_PROB={concat_prob}, "
                    "which joins segments into strings this pass cannot enumerate; unset it"
                )


# Labels RolloutTrainer._validation_prompt_for_label resolves to the sample's own caption,
# which the dataset enumeration already covers; anything else is used as the prompt itself.
_SCHEDULE_KEYWORDS = {"magic", "event", "caption", "base", "raw_caption", "raw"}


def _schedule_prompts(cfg) -> list[str]:
    """Literal prompts from validation.modes.*.prompt_schedule, in first-seen order."""
    out: list[str] = []
    for mode_cfg in cfg.validation.modes.values():
        for label in getattr(mode_cfg, "prompt_schedule", None) or []:
            text = str(label)
            if text.strip().lower() not in _SCHEDULE_KEYWORDS and text not in out:
                out.append(text)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="the same yaml the training run uses")
    parser.add_argument("--cache-dir", default=None, help="override runtime.text_embed_cache_dir")
    parser.add_argument("--device-map", default="auto", help='accelerate device_map for Gemma')
    parser.add_argument("--dry-run", action="store_true", help="enumerate and report, do not load Gemma")
    args = parser.parse_args()

    from alaya.config.loader import load_config
    from alaya.data.dataloader import build_train_dataloader
    from alaya.data.text_embed_cache import cache_path, disk_put, enumerate_all_prompts
    from alaya.utils.distributed import init_distributed

    cfg = load_config(args.config)
    _assert_finite_prompt_set(cfg)

    cache_dir = args.cache_dir or cfg.runtime.text_embed_cache_dir
    if not cache_dir:
        raise SystemExit("no cache dir: set runtime.text_embed_cache_dir or pass --cache-dir")
    os.makedirs(cache_dir, exist_ok=True)

    # No torchrun env here, so this is a plain single-process state with no
    # process group -- init_distributed() short-circuits when world_size is 1.
    dist_state = init_distributed()
    loader = build_train_dataloader(cfg, dist_state)
    dataset = loader.dataset
    # The loader may hand back a concat wrapper; mirror how RolloutTrainer.setup()
    # reaches the subsets that actually carry .samples.
    subsets = [dataset] if hasattr(dataset, "samples") else list(getattr(dataset, "datasets", []))

    prompts: list[str] = []
    seen: set[str] = set()
    for subset in subsets:
        for text in enumerate_all_prompts(cfg, subset):
            if text not in seen:  # "" is a legal prompt: an empty negative prompt
                seen.add(text)
                prompts.append(text)

    # Interactive generation from the same config reads its scheduled prompts from this cache too.
    for text in _schedule_prompts(cfg):
        if text not in seen:
            seen.add(text)
            prompts.append(text)

    missing = [p for p in prompts if not os.path.exists(cache_path(cache_dir, p))]
    print(
        f"[Precache] {len(prompts)} distinct prompts, "
        f"{len(prompts) - len(missing)} already cached, {len(missing)} to encode",
        flush=True,
    )
    if args.dry_run or not missing:
        return 0

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
