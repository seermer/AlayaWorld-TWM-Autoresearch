"""run_wbench.py case handling: which requested cases are unrendered, and --cases parsing.

The trainer skips a bucket whose mode dir already exists, so a relaunch could
render nothing for a bucket and still exit 0; the driver now checks the output.
A long comma list passed to --cases must not be probed as a file path.
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


def test_parse_cases_long_list_is_not_a_path():
    ids = [str(i) for i in range(1, 290)]
    assert run_wbench.parse_cases(",".join(ids)) == set(ids)


def test_parse_cases_from_file(tmp_path):
    f = tmp_path / "cases.txt"
    f.write_text("1,7\ne_5 23\n")
    assert run_wbench.parse_cases(str(f)) == {"1", "7", "e_5", "23"}
