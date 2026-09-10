"""Pick a small WBench subset that spans the split's axes, for a smoke run.

Greedy stratified pick: repeatedly take the case that most improves coverage of
interaction-type mix, scene category, perspective, subject type, visual style and
turn count. Deterministic for a given (n, seed).

    python scripts/tools/select_wbench_sample.py --n 20 --data-root ../WBench/data
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import Counter


def facets(case: dict) -> list[tuple[str, str]]:
    s = case.get("settings", {}) or {}
    types = sorted({str(i.get("type")) for i in case.get("interactions") or []})
    turns = max([int(i.get("turn", 0) or 0) for i in case.get("interactions") or []] or [1])
    out = [
        ("mix", "+".join(types)),
        ("category", str((s.get("scene") or {}).get("category"))),
        ("environment", str((s.get("scene") or {}).get("environment"))),
        ("attribute", str((s.get("scene") or {}).get("attribute"))),
        ("perspective", str(s.get("perspective"))),
        ("subject", str((s.get("subject") or {}).get("type"))),
        ("style", str(s.get("style_new"))),
        ("nav_cate", str(case.get("nav_cate"))),
        ("turns", str(turns)),
    ]
    out += [("type", t) for t in types]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--data-root", default="../WBench/data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cases = {}
    for f in sorted(glob.glob(f"{args.data_root}/cases/case_*.json")):
        d = json.loads(open(f).read())
        cases[str(d["id"])] = d

    rng = random.Random(args.seed)
    order = sorted(cases)
    rng.shuffle(order)

    seen: Counter = Counter()
    picked: list[str] = []
    while len(picked) < args.n:
        best, best_gain = None, None
        for cid in order:
            if cid in picked:
                continue
            # a facet value not seen yet is worth most; the score decays as it repeats
            gain = sum(1.0 / (1 + seen[f]) for f in facets(cases[cid]))
            if best_gain is None or gain > best_gain:
                best, best_gain = cid, gain
        picked.append(best)
        seen.update(facets(cases[best]))

    picked.sort(key=lambda c: int(c) if c.isdigit() else 0)
    print(",".join(picked))

    cov: dict[str, Counter] = {}
    for cid in picked:
        for k, v in facets(cases[cid]):
            cov.setdefault(k, Counter())[v] += 1
    import sys
    for k in ("mix", "type", "category", "environment", "perspective", "subject", "style", "turns"):
        print(f"  {k:12s} {dict(cov.get(k, {}))}", file=sys.stderr)
    print(f"  total turns  {sum(int(t) * c for t, c in cov['turns'].items())}", file=sys.stderr)


if __name__ == "__main__":
    main()
