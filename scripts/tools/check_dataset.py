"""Check every dataset declared under data.datasets before training on it.

    python scripts/tools/check_dataset.py --config configs/examples/finetune_video_caption_camera.yaml

Prints one summary per dataset, then every error and warning by file. Exits 1 when
any dataset has an error. The window layout (fps, window length, rollout rounds)
comes from the same config, so run it with the recipe you are going to train.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="the training yaml")
    parser.add_argument("--max-messages", type=int, default=50, help="per dataset, per kind; 0 = all")
    args = parser.parse_args()

    from alaya.config.loader import load_config
    from alaya.data.standard import parse_dataset_specs
    from alaya.data.standard_check import check_dataset, layout_from_config

    cfg = load_config(args.config)
    specs = parse_dataset_specs(cfg.data.datasets)
    if not specs:
        print(f"{args.config}: no data.datasets entries to check")
        return 1
    layout = layout_from_config(cfg)
    print(
        f"layout: {layout['window_frames']} frames per window at {layout['fps']:g} fps "
        f"({layout['prefix_frames']} history + {layout['chunk_frames']} target)"
    )

    failed = False
    for spec in specs:
        report = check_dataset(spec, **layout)
        mode = f", prompt_mode={spec.prompt_mode}" if spec.prompt_mode else ""
        status = "FAIL" if report.errors else "OK"
        unit = {"segment": "segments long enough for a window", "per_chunk": "round-aligned windows"}.get(
            spec.prompt_mode, "usable clips"
        )
        print(
            f"\n[{status}] {spec.name} ({spec.format}{mode}) at {spec.root}: {report.clips} clips, "
            f"{report.seconds / 60:.1f} min, {report.windows} {unit}, "
            f"{len(report.errors)} errors, {len(report.warnings)} warnings"
        )
        for kind, messages in (("error", report.errors), ("warning", report.warnings)):
            shown = messages if args.max_messages <= 0 else messages[: args.max_messages]
            for message in shown:
                print(f"  {kind}: {message}")
            if len(shown) < len(messages):
                print(f"  ... {len(messages) - len(shown)} more {kind}s (--max-messages 0 shows all)")
        failed = failed or bool(report.errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
