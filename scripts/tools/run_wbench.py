"""Drive WBench generation over a set of cases, bucketed by turn count.

The validation loop rolls every sample of a launch out for the same number of
rounds (max(turns) * wbench_chunks_per_turn) because FSDP needs every rank to
issue the same forwards. On the full split that means a 2-turn case would be
rolled out as far as the longest 9-turn case — 2.5x wasted compute. So group the
cases by their turn count and launch one job per group; inside a group every case
needs exactly the same number of rounds.

    # all 289 cases
    python scripts/tools/run_wbench.py --config configs/wbench_full.yaml

    # a specific subset
    python scripts/tools/run_wbench.py --config configs/wbench_full.yaml --cases 1,7,23

The trainer skips a validation mode whose output dir already exists, and the bucket
modes keep their names across launches, so every launch writes under its own
timestamped step dir. Resuming is by rendered video (--resume), and a launch that
leaves any requested case unrendered exits non-zero.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def case_max_turn(case: dict) -> int:
    turns = [int(it.get("turn", 0) or 0) for it in (case.get("interactions") or [])]
    return max(turns) if turns else 1


def load_cases(cases_dir: Path, wanted: set[str] | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(cases_dir.glob("case_*.json")):
        cid = path.stem.replace("case_", "")
        if wanted is not None and cid not in wanted:
            continue
        out[cid] = json.loads(path.read_text(encoding="utf-8"))
    return out


def unrendered(video_dir: Path, case_ids) -> list[str]:
    return sorted(c for c in case_ids if not (video_dir / f"case_{c}_combined.mp4").exists())


def config_path_for(out: Path, repo: Path) -> str:
    """CONFIG_PATH for train.sh: repo-relative when possible, absolute otherwise.

    train.sh accepts an absolute path, so only relativise when the generated
    config actually lives inside the repo. A caller driving this from its own
    work tree -- a harness writing configs under its own run dir -- otherwise
    dies in relative_to() before generation ever starts.
    """
    try:
        return str(out.relative_to(repo))
    except ValueError:
        return str(out.resolve())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/wbench_full.yaml")
    ap.add_argument("--cases", default=None, help="comma-separated case ids, or a file with one id per line")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--resume", action="store_true", help="skip cases whose combined.mp4 already exists")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--master-port", default="29531")
    args = ap.parse_args()

    cfg_path = REPO / args.config
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mode_name, mode = next(
        (n, m) for n, m in cfg["validation"]["modes"].items() if m["dataset"]["source"] == "wbench_navi"
    )

    # paths in the shipped configs are relative to the repository root, which is
    # also the cwd train.sh launches from
    data_root = (REPO / mode["dataset"]["root"]).resolve()
    cases_dir = data_root / "cases"
    video_dir = (REPO / mode["wbench_output_dir"]).resolve()

    wanted: set[str] | None = None
    if args.cases:
        p = Path(args.cases)
        raw = p.read_text().split() if p.exists() else args.cases.replace(",", " ").split()
        wanted = {s.strip() for s in raw if s.strip()}

    cases = load_cases(cases_dir, wanted)
    if wanted:
        missing = wanted - set(cases)
        if missing:
            raise SystemExit(f"unknown case ids: {sorted(missing)}")
    if not cases:
        raise SystemExit(f"no cases matched under {cases_dir}")

    if args.resume:
        before = len(cases)
        todo = set(unrendered(video_dir, cases))
        cases = {c: d for c, d in cases.items() if c in todo}
        print(f"[run_wbench] resume: {before - len(cases)}/{before} cases already rendered")
        if not cases:
            print("[run_wbench] nothing to do")
            return 0

    buckets: dict[int, list[str]] = defaultdict(list)
    for cid, case in cases.items():
        buckets[case_max_turn(case)].append(cid)

    cpt = int(mode.get("wbench_chunks_per_turn", 3))
    n_gpus = len([g for g in args.gpus.split(",") if g.strip()])
    total_rounds = sum(t * cpt * -(-len(ids) // n_gpus) for t, ids in buckets.items())
    print(
        f"[run_wbench] {len(cases)} cases in {len(buckets)} turn-count buckets "
        f"{ {t: len(v) for t, v in sorted(buckets.items())} }; "
        f"{total_rounds} collective rollout rounds on {n_gpus} GPUs"
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env["VALIDATE_ONLY"] = "1"
    env["LOG_FILTER"] = "all"
    env["MASTER_PORT"] = args.master_port
    # FA2 is the launcher's default now, so this only matters if the caller opted into
    # FA3 -- setdefault so that stays their choice (it will abort loudly: ViGeo pulls in
    # xformers, which cannot coexist with a local flash-attn-3 build).
    env.setdefault("ALAYA_USE_FA3", "0")
    # 13B bf16 = 26GB does not fit on a 24GB card unsharded: build on CPU, let FSDP shard.
    env.setdefault("ALAYA_INIT_TRANSFORMER_ON_CPU", "1")
    # Gemma-3-12B is another 24GB; every prompt is served from the on-disk cache instead.
    env.setdefault("ALAYA_SKIP_TEXT_ENCODER", "1")
    # One rank per GPU each build a 13B model on the host; loading them all at once
    # needs ~250GB of RAM, so let the ranks take turns.
    env.setdefault("ALAYA_SERIAL_MODEL_LOAD", "1")
    # torchrun defaults OMP_NUM_THREADS=1, which makes building and dtype-casting a
    # 13B model on the host take ~10min per rank. The GPU work is not OpenMP-bound,
    # so give the host side some threads back.
    env.setdefault("OMP_NUM_THREADS", "8")

    cfg_dir = REPO / cfg["run"]["output_dir"] / "_bucket_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # One launch, one validation *mode* per bucket: validate() walks every mode in
    # the config, so the 13B model is loaded once instead of once per bucket.
    run_cfg = copy.deepcopy(cfg)
    run_cfg["validation"]["modes"] = {}
    # A fresh step dir per launch: a bucket dir left by an earlier (crashed or subset)
    # launch would otherwise make the trainer skip that whole bucket.
    run_cfg["validation"]["step_dir_suffix"] = (
        f"{cfg['validation'].get('step_dir_suffix') or ''}_{time.strftime('%Y%m%d-%H%M%S')}"
    )
    for turns in sorted(buckets):
        ids = sorted(buckets[turns], key=lambda x: int(x) if x.isdigit() else 0)
        bmode = copy.deepcopy(mode)
        bmode["dataset"]["case_ids"] = ids
        run_cfg["validation"]["modes"][f"wbench_t{turns:02d}"] = bmode
        rounds = turns * cpt
        print(
            f"[run_wbench]   bucket t{turns:02d}: {len(ids):3d} cases x {rounds:2d} rounds "
            f"({rounds * 4 * 8 / 24:5.1f}s of video each)"
        )
    out = cfg_dir / "run.yaml"
    out.write_text(yaml.safe_dump(run_cfg, sort_keys=True), encoding="utf-8")
    print(f"[run_wbench] config -> {out}")

    rc = 0
    if not args.dry_run:
        config_path = config_path_for(out, REPO)
        proc = subprocess.run(
            ["bash", "scripts/finetune/train.sh"],
            cwd=REPO,
            env={**env, "CONFIG_PATH": config_path},
        )
        rc = proc.returncode
        if rc != 0:
            print(f"[run_wbench] generation FAILED (rc={rc})", file=sys.stderr)
        missing = unrendered(video_dir, cases)
        if missing:
            print(f"[run_wbench] {len(missing)} requested case(s) not rendered: {missing[:10]}", file=sys.stderr)
            rc = rc or 1

    rendered = len(list(video_dir.glob("case_*_combined.mp4"))) if video_dir.exists() else 0
    print(f"\n[run_wbench] {rendered} videos in {video_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
