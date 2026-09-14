"""Score validation rollouts against ground truth, and compare two runs.

    # score one run (reads the sidecars + pred_clean.mp4 written by VALIDATE_ONLY=1)
    python scripts/tools/eval_rollouts.py score --config configs/overfit_c3vd_eval_base.yaml \
        --out outputs/overfit_c3vd_eval/base_scores.json
    # compare two scored runs
    python scripts/tools/eval_rollouts.py compare base_scores.json trained_scores.json

Ground truth is taken from the validation dataset itself: sidecars record sample_idx,
and the dataset is rebuilt with the same settings validate() uses, so the frames scored
are exactly the frames the rollout was conditioned on and compared with. The frame offset
between the prediction video and the ground truth is found by matching prediction frame 0
(the reconstructed conditioning frame) and reported, rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

EXPECTED_OFFSET = 24  # prediction frame 0 is ground-truth frame 24 (see alaya/data/c3vd.py)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-10 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Gaussian-window SSIM on luma (Wang et al. 2004 constants)."""
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mx, my = blur(x), blur(y)
    sx, sy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx * mx + my * my + c1) * (sx + sy + c2)
    return float(np.mean(num / den))


def _to_uint8_frames(pixels) -> np.ndarray:
    """Dataset pixel tensor [F,C,H,W] in [-1,1] or [0,1] -> [F,H,W,3] uint8 RGB."""
    arr = pixels.detach().float().cpu().numpy() if hasattr(pixels, "detach") else np.asarray(pixels, np.float32)
    if arr.ndim == 4 and arr.shape[1] == 3:
        arr = arr.transpose(0, 2, 3, 1)
    lo = float(arr.min())
    arr = (arr + 1.0) * 127.5 if lo < -0.01 else (arr * 255.0 if arr.max() <= 1.01 else arr)
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def score(config: str, mode: str | None, run_dir: str | None, out: str) -> None:
    import decord

    from alaya.config.loader import load_config
    from alaya.data.dataloader import build_validation_dataset

    cfg = load_config(config)
    mode_name = mode or next(iter(cfg.validation.modes))
    mode_cfg = cfg.validation.modes[mode_name]
    rounds, K, stride = int(mode_cfg.rollout_rounds), 4, int(cfg.sample.temporal_stride)
    needed = 1 + 3 * stride + rounds * K * stride  # 25-frame prefix + rounds * 32
    dataset = build_validation_dataset(cfg, mode_cfg, frames=needed)

    base = Path(run_dir) if run_dir else None
    if base is None:
        steps = sorted((Path(cfg.run.output_dir) / "validation").glob("step-*"))
        if not steps:
            raise SystemExit(f"no validation output under {cfg.run.output_dir}")
        base = steps[-1] / mode_name
    manifest_path = REPO_ROOT / cfg.paths.annotation_base_dir / "c3vd" / "eval_manifest.json"
    split = json.loads(manifest_path.read_text())["eval_clips"] if manifest_path.exists() else {}

    samples = []
    for side in sorted(base.glob("rank-*_sample-*.json")):
        meta = json.loads(side.read_text())
        pred_path = side.with_name(side.stem + "_pred_clean.mp4")
        if not pred_path.exists():
            raise SystemExit(f"missing {pred_path.name}; was save_debug_videos enabled?")
        # MultiSourceVideoDataset returns a tuple: (pixel_values, ref_img, caption, intrinsic,
        # cam_c2w, videoid, has_camera, source, caption_type, pose_orig_w, pose_orig_h,
        # frame_start, frame_end)
        item = dataset[int(meta["sample_idx"])]
        pixel_values, clip, frame_start = item[0], str(item[5]), int(item[11])
        gt = _to_uint8_frames(pixel_values)
        vr = decord.VideoReader(str(pred_path))
        pred = np.stack([vr[i].asnumpy() for i in range(len(vr))])
        if pred.shape[1:3] != gt.shape[1:3]:
            pred = np.stack([cv2.resize(f, (gt.shape[2], gt.shape[1]), interpolation=cv2.INTER_AREA) for f in pred])
        # locate the conditioning frame instead of trusting the layout arithmetic
        search = range(0, max(1, len(gt) - len(pred) + 1))
        offset = max(search, key=lambda o: psnr(pred[0], gt[o]))
        n = min(len(pred), len(gt) - offset)
        per_round = []
        for r in range(rounds):
            lo, hi = 1 + r * K * stride, min(1 + (r + 1) * K * stride, n)
            if lo >= hi:
                break
            idx = range(lo, hi)
            lat = next((m for m in meta.get("metrics", []) if int(m.get("round", -1)) == r), {})
            per_round.append({
                "round": r + 1,
                "psnr": float(np.mean([psnr(pred[i], gt[offset + i]) for i in idx])),
                "ssim": float(np.mean([ssim(pred[i], gt[offset + i]) for i in idx])),
                "latent_cos": lat.get("cos"),
                "latent_l2": lat.get("l2"),
            })
        samples.append({
            "sidecar": side.name, "sample_idx": int(meta["sample_idx"]), "clip": clip,
            "split": split.get(clip, "unknown"), "gt_frame_start": frame_start, "frame_offset": int(offset),
            "offset_as_expected": int(offset) == EXPECTED_OFFSET, "rounds": per_round,
        })
        # side-by-side strip for eyeballing: GT over prediction at five points
        picks = [0, 48, 81, 113, n - 1]  # start, round 2, mid-transition, round 4, end
        top = np.hstack([cv2.resize(gt[offset + i], (368, 208)) for i in picks])
        bot = np.hstack([cv2.resize(pred[i], (368, 208)) for i in picks])
        cv2.imwrite(str(side.with_name(side.stem + "_gt_vs_pred.png")), cv2.cvtColor(np.vstack([top, bot]), cv2.COLOR_RGB2BGR))
        print(f"  {side.stem}  clip={clip}  split={samples[-1]['split']}  offset={offset}  "
              + " ".join(f"r{x['round']}:{x['psnr']:.1f}dB/{x['ssim']:.3f}" for x in per_round), flush=True)

    Path(out).write_text(json.dumps({"config": config, "run_dir": str(base), "samples": samples}, indent=1))
    print(f"[score] {len(samples)} samples -> {out}")


def _summary(samples: list[dict]) -> dict:
    out = {}
    for split in ("train", "heldout"):
        rows = [s for s in samples if s["split"] == split]
        if not rows:
            continue
        rounds = max(len(s["rounds"]) for s in rows)
        out[split] = []
        for r in range(rounds):
            vals = [s["rounds"][r] for s in rows if len(s["rounds"]) > r]
            cos = [v["latent_cos"] for v in vals if v["latent_cos"] is not None]
            out[split].append({
                "round": r + 1,
                "psnr": float(np.mean([v["psnr"] for v in vals])),
                "ssim": float(np.mean([v["ssim"] for v in vals])),
                "latent_cos": float(np.mean(cos)) if cos else None,
                "n": len(vals),
            })
    return out


def compare(a_path: str, b_path: str) -> None:
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    sa, sb = _summary(a["samples"]), _summary(b["samples"])
    print(f"A = {a_path}\nB = {b_path}")
    for split in ("train", "heldout"):
        if split not in sa or split not in sb:
            continue
        print(f"\n[{split}]  round |   PSNR A -> B (delta)  |   SSIM A -> B   | latent cos A -> B")
        for ra, rb in zip(sa[split], sb[split]):
            cos = (f"{ra['latent_cos']:.3f} -> {rb['latent_cos']:.3f}"
                   if ra["latent_cos"] is not None and rb["latent_cos"] is not None else "n/a")
            print(f"          {ra['round']:>5} | {ra['psnr']:6.2f} -> {rb['psnr']:6.2f} ({rb['psnr'] - ra['psnr']:+5.2f}) "
                  f"| {ra['ssim']:.3f} -> {rb['ssim']:.3f} | {cos}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--config", required=True)
    s.add_argument("--mode", default=None)
    s.add_argument("--run-dir", default=None, help="validation/step-XXXXXX/<mode> (default: latest step)")
    s.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "score":
        score(args.config, args.mode, args.run_dir, args.out)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
