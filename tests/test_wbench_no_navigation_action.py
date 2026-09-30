"""dataset.no_navigation_action sets the camera of cases that never navigate."""
import json

import pytest
from PIL import Image

from alaya.data.wbench import WBenchNaviDataset


@pytest.fixture
def root(tmp_path):
    (tmp_path / "cases").mkdir()
    (tmp_path / "images").mkdir()
    Image.new("RGB", (64, 32)).save(tmp_path / "images" / "case_1.jpg")
    Image.new("RGB", (64, 32)).save(tmp_path / "images" / "case_2.jpg")
    cases = {
        "1": [{"turn": 1, "type": "subject_action", "action": "Sit."},
              {"turn": 2, "type": "event_edit", "action": "It rains."}],
        "2": [{"turn": 1, "type": "navigation", "action": "left"},
              {"turn": 2, "type": "event_edit", "action": "It rains."}],
    }
    for cid, its in cases.items():
        (tmp_path / "cases" / f"case_{cid}.json").write_text(json.dumps({
            "id": cid, "interactions": its,
            "settings": {"initial_image": f"images/case_{cid}.jpg", "perspective": "first_person"},
        }))
    return tmp_path


def _turn_actions(root, **kw):
    ds = WBenchNaviDataset(root=str(root), width=32, height=32, frames=1, include_non_navigation=True, **kw)
    return {ds[i]["metadata"]["wbench_case_id"]: ds[i]["metadata"]["wbench_turn_actions"] for i in range(len(ds))}


def test_default_keeps_moving_forward(root):
    assert _turn_actions(root) == {"1": ["W", "W"], "2": ["left", "left"]}


def test_stop_holds_only_cases_without_navigation(root):
    assert _turn_actions(root, no_navigation_action="stop") == {"1": ["stop", "stop"], "2": ["left", "left"]}
