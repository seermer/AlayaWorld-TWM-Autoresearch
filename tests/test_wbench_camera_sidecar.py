"""The WBench-mode render saves the camera path of the frames it wrote.

Layout of wbench_full.yaml (ViGeo prefix mode): r=8, history N=4 latents
(prefix = 1+(N-1)*r = 25 pixels), K=4, 2 output rounds -> 8 generated latents.
The causal VAE decodes 4+8 latents to 1+11*8 = 89 frames; the render drops 4*8 = 32,
so it writes 57 frames, and mp4 frame i is cam_c2w[25 + (8-1) + i] = cam_c2w[32 + i].
"""
import json
import math
from types import SimpleNamespace as NS

import numpy as np
import torch

from alaya.trainer.rollout_trainer import RolloutTrainer


def test_wbench_render_saves_written_camera_path(tmp_path):
    r, n_hist, k, rounds = 8, 4, 4, 2
    trainer = object.__new__(RolloutTrainer)
    trainer.dtype = torch.float32
    trainer.cfg = NS(
        sample=NS(temporal_stride=r, fps=24),
        layout=NS(sink_latent_frames=1, history_latent_frames=n_hist),
        validation=NS(video_history_latent_frames=n_hist),
        spatial_memory=NS(enabled=True, context_mode="vigeo_prefix_last_frame", depth_backend="vigeo"),
        run=NS(output_dir=str(tmp_path)),
    )
    written = {}
    trainer._decode_latent_to_video_frames = lambda lat: torch.zeros(1 + (lat.shape[2] - 1) * r, 2, 2, 3)
    trainer._write_video = lambda path, frames: written.update(path=path, n=int(frames.shape[0]))

    total = 1 + (n_hist - 1) * r + rounds * k * r  # needed_pixels = 89
    yaw = math.radians(30.0)
    rot = torch.tensor([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    cam = torch.eye(4).repeat(total, 1, 1)
    cam[:, :3, :3] = rot
    cam[:, 0, 3] = torch.arange(total, dtype=torch.float32) ** 2  # nonlinear ramp, so the offset matters
    metadata = {
        "cam_c2w": cam, "wbench_case_id": "7", "wbench_output_rounds": rounds,
        "wbench_chunks_per_turn": 3, "wbench_used_actions": ["W"], "wbench_target_base_start": 1 + n_hist,
    }
    trainer._save_wbench_output_video(
        mode_cfg=NS(wbench_output_dir=str(tmp_path), layout=NS(history_latent_frames=n_hist)),
        mode_dir=tmp_path, stem="s", metadata=metadata,
        pred_latents=[torch.zeros(1, 1, k, 1, 1) for _ in range(rounds)],
        latent_full=torch.zeros(1, 1, 1 + n_hist + rounds * k, 1, 1), N=n_hist,
    )

    assert written["n"] == 57
    npz = np.load(tmp_path / "case_7_combined_camera.npz")
    c2w = npz["cam_c2w"]
    assert c2w.shape == (57, 4, 4) and c2w.dtype == np.float32
    np.testing.assert_allclose(c2w[0], np.eye(4), atol=1e-6)
    start = (1 + (n_hist - 1) * r) + (r - 1)  # 32
    ramp = np.arange(total, dtype=np.float64) ** 2
    dx = ramp[start:start + 57] - ramp[start]
    expected_t = rot.numpy().T @ np.stack([dx, np.zeros(57), np.zeros(57)])
    np.testing.assert_allclose(c2w[:, :3, 3], expected_t.T, rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(c2w[:, :3, :3], np.broadcast_to(np.eye(3), (57, 3, 3)), atol=1e-6)
    assert start + 57 == total  # the written frames end exactly at the trajectory's end
    sidecar = json.loads((tmp_path / "case_7_combined.json").read_text())
    assert sidecar["camera_file"] == "case_7_combined_camera.npz"
    assert sidecar["num_frames"] == 57
