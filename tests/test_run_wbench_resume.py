"""run_wbench.py reports requested cases that have no rendered video.

The trainer skips a bucket whose mode dir already exists, so a relaunch could
render nothing for a bucket and still exit 0; the driver now checks the output.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_wbench", Path(__file__).resolve().parents[1] / "scripts" / "tools" / "run_wbench.py")
run_wbench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_wbench)


def test_unrendered_lists_cases_without_video(tmp_path):
    (tmp_path / "case_1_combined.mp4").write_bytes(b"")
    (tmp_path / "case_e_5_combined.mp4").write_bytes(b"")
    assert run_wbench.unrendered(tmp_path, ["1", "7", "e_5", "23"]) == ["23", "7"]


def test_unrendered_missing_dir(tmp_path):
    assert run_wbench.unrendered(tmp_path / "nope", ["1"]) == ["1"]
