"""The overfit experiment's training clips and evaluation cases, from the raw clips and ViGeo's output.

    python scripts/overfit/build_dataset.py        (after fetch_dl3dv.py and poses_depth.py)

Training, data/overfit/canny_depth/dataset (video_timed_prompts_camera): clip i of the first 100
scenes; even i is raw -> canny, odd i is canny -> depth. Label 1 lasts 57..145 frames, label 2
32..120 (seeded). The text changes on label 2's first frame; the picture cross-fades over label 2's
first 24 frames and then holds. A label's text is the scene description plus the sentence of its own
transform.

Evaluation, data/overfit/canny_depth/cases (WBench cases, camera held, three turns): the last 3
scenes, which are not trained on. Case 91tk is scene k in test t:
    0  no transform sentence
    1  turn 2 says canny, turn 3 says depth
    2  starts from the canny version of the first frame; turn 1 says canny, turns 2 and 3 say depth
"""
import json
import random
import subprocess
from pathlib import Path

import av
import cv2
import numpy as np

ROOT = Path("data/overfit/canny_depth")
CANNY, DEPTH = "Canny edge map.", "Depth map."
FADE, FPS, SEED = 24, 24, 0
W, H = 960, 544

captions = json.loads(Path("configs/overfit/canny_depth_captions.json").read_text())
scenes = json.loads(Path("configs/overfit/canny_depth_scenes.json").read_text())


def canny(frames):
    kernel = np.ones((3, 3), np.uint8)
    out = [cv2.dilate(cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), 100, 200), kernel) for f in frames]
    return np.repeat(np.stack(out)[..., None], 3, axis=-1)


def depth_map(depth):
    """Near is bright: inverse depth, scaled over the whole clip between its 1st and 99th percentile."""
    inv = 1.0 / np.clip(depth.astype(np.float32), 1e-3, None)
    lo, hi = np.percentile(inv, [1, 99])
    grey = (np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    out = [cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR) for g in grey]
    return np.repeat(np.stack(out)[..., None], 3, axis=-1)


def read(name, n=None):
    with av.open(str(ROOT / "raw" / f"{name}.mp4")) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)][:n])


data = ROOT / "dataset"
for sub in ("videos", "captions", "poses"):
    (data / sub).mkdir(parents=True, exist_ok=True)
rng = random.Random(SEED)
manifest = []
for i, scene in enumerate(scenes[:100]):
    name = scene.replace("/", "_")
    n1, n2 = rng.randint(57, 145), rng.randint(32, 120)
    n = n1 + n2
    raw = read(name, n)
    geo = np.load(ROOT / "vigeo" / f"{name}.npz")
    edges = canny(raw)
    first, second = (raw, edges) if i % 2 == 0 else (edges, depth_map(geo["depth"][:n]))
    frames = first.copy()
    frames[n1:] = second[n1:]
    for t in range(FADE):
        a = (t + 1) / FADE
        frames[n1 + t] = np.round((1 - a) * first[n1 + t] + a * second[n1 + t]).astype(np.uint8)

    clip = f"{i:03d}"
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "14",
                            "-pix_fmt", "yuv420p", str(data / "videos" / f"{clip}.mp4")], stdin=subprocess.PIPE)
    enc.communicate(frames.tobytes())
    if enc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on clip {clip}")
    scene_text = captions[f"{name}.mp4"]
    texts = (scene_text, f"{scene_text} {CANNY}") if i % 2 == 0 else (f"{scene_text} {CANNY}", f"{scene_text} {DEPTH}")
    (data / "captions" / f"{clip}.json").write_text(json.dumps({
        "caption": scene_text,
        "segments": [{"time_range_s": [0.0, n1 / FPS], "prompt": texts[0]},
                     {"time_range_s": [n1 / FPS, n / FPS], "prompt": texts[1]}]}, indent=1))
    np.savez(data / "poses" / f"{clip}.npz", cam_c2w=geo["cam_c2w"][:n], intrinsics=geo["intrinsics"])
    manifest.append({"clip": clip, "scene": scene, "kind": "raw_to_canny" if i % 2 == 0 else "canny_to_depth",
                     "label1_frames": n1, "label2_frames": n2})
    print(clip, manifest[-1]["kind"], n1, n2, flush=True)
(data / "manifest.json").write_text(json.dumps(manifest, indent=1))

cases = ROOT / "cases"
for sub in ("cases", "images"):
    (cases / sub).mkdir(parents=True, exist_ok=True)
TESTS = {0: {}, 1: {2: CANNY, 3: DEPTH}, 2: {1: CANNY, 2: DEPTH, 3: DEPTH}}
for k, scene in enumerate(scenes[100:], start=1):
    name = scene.replace("/", "_")
    frame = read(name, 1)[0]
    cv2.imwrite(str(cases / f"images/dl3dv_{k}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(cases / f"images/dl3dv_{k}_canny.png"), canny(frame[None])[0])
    for test, events in TESTS.items():
        case_id = str(9100 + 10 * test + k)
        image = f"images/dl3dv_{k}_canny.png" if test == 2 else f"images/dl3dv_{k}.png"
        (cases / "cases" / f"case_{case_id}.json").write_text(json.dumps({
            "id": case_id, "environment_prompt": captions[f"{name}.mp4"], "character_prompt": "",
            "perspective_prompt": "",
            "settings": {"perspective": "first_person", "subject": {"type": "unknown", "desc": ""},
                         "tracking_object": None, "initial_image": image, "subject_mask": None},
            "interactions": [{"type": "navigation", "action": "stop", "turn": t} for t in (1, 2, 3)]
                            + [{"type": "event_edit", "action": text, "turn": turn} for turn, text in events.items()],
            "metric_list": [], "source_scene": scene}, indent=1))
print(f"{len(manifest)} training clips, {3 * len(TESTS)} cases")
