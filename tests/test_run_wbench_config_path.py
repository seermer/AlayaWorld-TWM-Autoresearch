"""CONFIG_PATH handling for run_wbench.py.

run_wbench unconditionally called out.relative_to(REPO), which assumes the
generated bucket config lives inside this checkout. A harness that drives
WBench generation while writing its configs under its own run directory hit
ValueError and died before any generation started. train.sh accepts an
absolute CONFIG_PATH, so falling back to one is both sufficient and correct.
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "run_wbench", Path(__file__).resolve().parents[1] / "scripts" / "tools" / "run_wbench.py")
run_wbench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_wbench)


def test_config_inside_repo_stays_relative(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    out = repo / "configs" / "run.yaml"
    out.write_text("{}")
    assert run_wbench.config_path_for(out, repo) == "configs/run.yaml"


def test_config_outside_repo_becomes_absolute(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    elsewhere = tmp_path / "harness" / "runs" / "node0"
    elsewhere.mkdir(parents=True)
    out = elsewhere / "run.yaml"
    out.write_text("{}")
    got = run_wbench.config_path_for(out, repo)
    assert Path(got).is_absolute()
    assert Path(got) == out.resolve()
