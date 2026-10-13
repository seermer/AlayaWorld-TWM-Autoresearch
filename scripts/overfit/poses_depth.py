"""ViGeo camera poses and depth for the 100 training clips of the overfit experiment.

    CUDA_VISIBLE_DEVICES=R python scripts/overfit/poses_depth.py --rank R --world W     (one process per GPU)

Clips run in chunks of 16 frames that carry ViGeo's cache; the focal length comes from the first
chunk, which ViGeo sees with full attention.
Writes data/overfit/canny_depth/vigeo/<scene>.npz: cam_c2w [N,4,4] (OpenCV camera-to-world, frame 0
the identity), intrinsics [3,3] in video pixels, depth [N,h,w] float16 at the model size.
"""
import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch

sys.path.insert(0, "third_party/ViGeo")
from vigeo import ViGeo  # noqa: E402
from vigeo.utils import recover_focal_from_xy  # noqa: E402

CHUNK = 16             # ViGeo's default chunk size
NUM_TOKENS = 1369      # ViGeo infer() default
CACHE_BUDGET = 786432  # cache tokens kept across chunks

ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--world", type=int, default=1)
args = ap.parse_args()
root = Path("data/overfit/canny_depth")
out = root / "vigeo"
out.mkdir(exist_ok=True)
scenes = json.loads(Path("configs/overfit/canny_depth_scenes.json").read_text())[:100]
device = torch.device("cuda")
model = ViGeo.from_pretrained("third_party/ViGeo/checkpoints/ViGeo1.1").to(device).eval()


def decode(path: Path):
    """All frames as uint8 RGB [N,h,w,3] at the model size, plus the video's (W, H)."""
    frames, wh = [], None
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            rgb = frame.to_ndarray(format="rgb24")
            if wh is None:
                wh = (rgb.shape[1], rgb.shape[0])
                h, w, _, _ = model._resolve_model_size(wh[1], wh[0], NUM_TOKENS)
            frames.append(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR))
    return np.stack(frames), wh


for i, scene in enumerate(scenes):
    name = scene.replace("/", "_")
    if i % args.world != args.rank or (out / f"{name}.npz").exists():
        continue
    frames, (W, H) = decode(root / "raw" / f"{name}.mp4")
    n, h, w = frames.shape[:3]
    poses, depths, focal, kv = [], [], None, None
    with torch.inference_mode():
        for s in range(0, n, CHUNK):
            x = torch.from_numpy(frames[s:s + CHUNK]).to(device).permute(0, 3, 1, 2).float() / 255.0
            o = model.infer(x, mode="chunk", chunk_size=CHUNK, num_tokens=NUM_TOKENS, total_budget=CACHE_BUDGET,
                            resize_output=False, kv_caches=kv)
            kv = o["kv_caches"]
            poses.append(o["pose_pred"].float().cpu())
            depths.append(o["depth_pred"].float().reshape(-1, h, w).cpu())
            if focal is None:
                p = o["points_pred"].float()
                xy = (p[..., :2] / p[..., 2:].clamp(min=1e-6)).permute(0, 3, 1, 2).unsqueeze(0)
                focal = float(recover_focal_from_xy(xy)[0].median())
            del o, x
    del kv
    c2w = np.tile(np.eye(4), (n, 1, 1))
    c2w[:, :3, :] = torch.cat(poses).double().numpy()
    c2w = np.linalg.inv(c2w[0]) @ c2w
    f = focal * (w * w + h * h) ** 0.5 / 2
    k = np.array([[f * W / w, 0, W / 2.0], [0, f * H / h, H / 2.0], [0, 0, 1]], dtype=np.float32)
    np.savez(out / f"{name}.npz", cam_c2w=c2w.astype(np.float32), intrinsics=k,
             depth=torch.cat(depths).numpy().astype(np.float16))
    print(f"rank {args.rank}: {name} frames={n}", flush=True)
    torch.cuda.empty_cache()
