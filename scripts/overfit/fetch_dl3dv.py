"""Cut one 314-frame clip (24 fps, 960x544) from each DL3DV scene of the overfit experiment.

    HF_TOKEN=... python scripts/overfit/fetch_dl3dv.py

Scenes: configs/overfit/canny_depth_scenes.json (the first 100 are trained on, the last 3 are held
out). Reads the scene's video.mp4 from 10 s in; the full file is not downloaded. The dataset
(DL3DV/DL3DV-ALL-video) is gated, so the token must have access to it.
Writes data/overfit/canny_depth/raw/<scene>.mp4.
"""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "DL3DV/DL3DV-ALL-video"
START_S, FRAMES = 10, 314

out = Path("data/overfit/canny_depth/raw")
out.mkdir(parents=True, exist_ok=True)
scenes = json.loads(Path("configs/overfit/canny_depth_scenes.json").read_text())
header = f"Authorization: Bearer {os.environ['HF_TOKEN']}\r\n"


def fetch(scene: str) -> None:
    dest = out / f"{scene.replace('/', '_')}.mp4"
    if dest.exists():
        return
    tmp = dest.with_suffix(".part.mp4")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{scene}/video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-headers", header, "-ss", str(START_S), "-i", url,
                    "-frames:v", str(FRAMES), "-vf", "fps=24,scale=960:544", "-an", "-c:v", "libx264",
                    "-crf", "17", "-pix_fmt", "yuv420p", str(tmp)], check=True, timeout=1200)
    n = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_frames", "-of", "csv=p=0", str(tmp)], capture_output=True, text=True).stdout.strip()
    if n != str(FRAMES):
        raise RuntimeError(f"{scene}: got {n!r} frames, need {FRAMES}")
    tmp.rename(dest)
    print(f"ok {scene}", flush=True)


with ThreadPoolExecutor(4) as pool:
    list(pool.map(fetch, scenes))
print(f"{len(scenes)} clips in {out}")
