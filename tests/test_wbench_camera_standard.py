"""The WBench render's camera: holds on non-navigation turns, and follows WBench's reference
pose layout (pose 0 is the input frame, so the first turn has one step fewer)."""
import json
import math
from types import SimpleNamespace as NS

import numpy as np
from PIL import Image

from alaya.data.wbench import WBenchNaviDataset
from alaya.trainer.rollout_trainer import RolloutTrainer


def test_camera_holds_on_turns_without_a_navigation_action(tmp_path):
    (tmp_path / "cases").mkdir()
    (tmp_path / "images").mkdir()
    cases = {
        "1": [{"turn": 1, "type": "subject_action", "action": "Sit."},
              {"turn": 2, "type": "event_edit", "action": "It rains."}],
        "2": [{"turn": 1, "type": "navigation", "action": "left"},
              {"turn": 2, "type": "event_edit", "action": "It rains."},
              {"turn": 3, "type": "navigation", "action": "W"}],
    }
    for cid, its in cases.items():
        Image.new("RGB", (64, 32)).save(tmp_path / "images" / f"case_{cid}.jpg")
        (tmp_path / "cases" / f"case_{cid}.json").write_text(json.dumps({
            "id": cid, "interactions": its,
            "settings": {"initial_image": f"images/case_{cid}.jpg", "perspective": "first_person"},
        }))
    ds = WBenchNaviDataset(root=str(tmp_path), width=32, height=32, frames=1, include_non_navigation=True)
    actions = {ds[i]["metadata"]["wbench_case_id"]: ds[i]["metadata"]["wbench_turn_actions"] for i in range(len(ds))}
    assert actions == {"1": ["stop", "stop"], "2": ["left", "stop", "W"]}


def _yaw_pitch(pose):
    forward = pose[:3, 2]
    return math.degrees(math.atan2(forward[0], forward[2])), math.degrees(-math.asin(forward[1]))


def test_first_person_camera_follows_the_reference_layout():
    r, n_hist, k, cpt = 8, 4, 4, 3
    trainer = object.__new__(RolloutTrainer)
    trainer.cfg = NS(
        sample=NS(temporal_stride=r, fps=24),
        spatial_memory=NS(enabled=True, context_mode="vigeo_prefix_last_frame", depth_backend="vigeo"),
    )
    per_turn = cpt * k * r  # 96
    start = 1 + (n_hist - 1) * r  # 25: the first generated frame
    cam = trainer._build_wbench_camera_trajectory(
        actions=["right", "stop", "up"], perspective="first_person",
        mode_cfg=NS(wbench_move_per_sec=0.256, wbench_yaw_deg_per_sec=9.6, wbench_pitch_deg_per_sec=7.6,
                    layout=NS(history_latent_frames=n_hist)),
        K=k, chunks_per_turn=cpt, target_base_start=1 + n_hist, frames=start + 3 * per_turn,
    ).numpy()
    video = cam[start - 1:]  # video frame i is cam[start - 1 + i]; frame 0 is the input frame
    np.testing.assert_allclose(video[0], np.eye(4), atol=1e-6)
    assert _yaw_pitch(video[1])[0] > 0  # the first generated frame already turned one step
    yaw, pitch = _yaw_pitch(video[per_turn - 1])  # last frame of turn 0: 95 steps of 0.4 deg
    assert abs(yaw - 38.0) < 1e-3 and abs(pitch) < 1e-3
    np.testing.assert_allclose(video[2 * per_turn - 1], video[per_turn - 1], atol=1e-6)  # held through turn 1
    np.testing.assert_allclose(video[per_turn], video[per_turn - 1], atol=1e-6)
    assert not np.allclose(video[2 * per_turn], video[2 * per_turn - 1])  # turn 2 starts on its first frame
    np.testing.assert_allclose(video[3 * per_turn], video[3 * per_turn - 1], atol=1e-6)  # nothing after the last turn
