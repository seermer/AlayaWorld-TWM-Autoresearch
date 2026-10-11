"""The WBench-mode render saves the camera path of the frames it wrote.

Layout of wbench_full.yaml (ViGeo prefix mode): r=8, history N=4 latents
(prefix = 1+(N-1)*r = 25 pixels), K=4, 2 output rounds -> 8 generated latents.
The causal VAE decodes 4+8 latents to 1+11*8 = 89 frames; the render keeps the input frame
(decoded frame 24) and drops the last one, so it writes 64 frames, and mp4 frame i is
cam_c2w[24 + i].
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
        validation=NS(video_history_latent_frames=n_hist, vigeo_pred_decode_mode="chunk"),
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
        mode_cfg=NS(wbench_output_dir=str(tmp_path), wbench_keep_last_frame=False,
                    layout=NS(history_latent_frames=n_hist)),
        mode_dir=tmp_path, stem="s", metadata=metadata,
        pred_latents=[torch.zeros(1, 1, k, 1, 1) for _ in range(rounds)],
        latent_full=torch.zeros(1, 1, 1 + n_hist + rounds * k, 1, 1), N=n_hist,
    )

    assert written["n"] == 64
    npz = np.load(tmp_path / "case_7_combined_camera.npz")
    c2w = npz["cam_c2w"]
    assert c2w.shape == (64, 4, 4) and c2w.dtype == np.float32
    np.testing.assert_allclose(c2w[0], np.eye(4), atol=1e-6)
    start = (1 + (n_hist - 1) * r) - 1  # 24, the input frame
    ramp = np.arange(total, dtype=np.float64) ** 2
    dx = ramp[start:start + 64] - ramp[start]
    expected_t = rot.numpy().T @ np.stack([dx, np.zeros(64), np.zeros(64)])
    np.testing.assert_allclose(c2w[:, :3, 3], expected_t.T, rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(c2w[:, :3, :3], np.broadcast_to(np.eye(3), (64, 3, 3)), atol=1e-6)
    assert start + 64 == total - 1  # the last generated frame is not written
    sidecar = json.loads((tmp_path / "case_7_combined.json").read_text())
    assert sidecar["camera_file"] == "case_7_combined_camera.npz"
    assert sidecar["num_frames"] == 64


def test_wbench_render_decodes_chunk_by_chunk_like_the_documented_path(tmp_path):
    r, n_hist, k, rounds = 8, 4, 4, 2
    trainer = object.__new__(RolloutTrainer)
    trainer.dtype = torch.float32
    trainer.cfg = NS(
        sample=NS(temporal_stride=r, fps=24),
        layout=NS(sink_latent_frames=1, history_latent_frames=n_hist),
        validation=NS(video_history_latent_frames=n_hist, vigeo_pred_decode_mode="chunk"),
        spatial_memory=NS(enabled=True, context_mode="vigeo_prefix_last_frame", depth_backend="vigeo"),
        run=NS(output_dir=str(tmp_path)),
    )
    written, decoded = {}, {}

    def chunk_decode(*, nearby_latents, chunks):  # the input frame, then every generated frame
        decoded.update(nearby=len(nearby_latents), chunks=len(chunks))
        return torch.zeros(1 + len(chunks) * k * r, 2, 2, 3)

    trainer._decode_i2v_rollout_chunks_to_video_frames = chunk_decode
    trainer._write_video = lambda path, frames: written.update(n=int(frames.shape[0]))
    for keep_last, expected in ((False, 64), (True, 65)):
        trainer._save_wbench_output_video(
            mode_cfg=NS(wbench_output_dir=str(tmp_path), wbench_keep_last_frame=keep_last,
                        layout=NS(history_latent_frames=n_hist)),
            mode_dir=tmp_path, stem="s",
            metadata={"wbench_case_id": "7", "wbench_output_rounds": rounds, "wbench_chunks_per_turn": 3,
                      "wbench_used_actions": ["W"]},
            pred_latents=[torch.zeros(1, 1, k, 1, 1) for _ in range(rounds + 1)],
            nearby_condition_latents=[torch.zeros(1, 1, 2, 1, 1) for _ in range(rounds + 1)],
        )
        assert written["n"] == expected and decoded == {"nearby": rounds, "chunks": rounds}
