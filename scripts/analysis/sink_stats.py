"""Decision-rule statistics for the compute-sink sweeps.

Protocol: `results/sink-2026-11-01/protocol.md`. For each arm against `vurs`
and each c_max, the paired log ratio of the full-budget error over the
(case, seed) pairs, a Wilcoxon signed-rank test, and Holm correction over all
arm x c_max tests. Also the vurs vs space-filling gap per c_max (the sink
penalty), mean sink share and points placed, and the wiring check that
`vurs-cost` reproduces `vurs` at c_max = 1.

    python -m ND_scan_samplers.scripts.analysis.sink_stats --input ND_scan_samplers/results/sink-2026-11-01/6d
"""
import argparse
import gzip
import json
import pathlib

import numpy as np
from scipy.stats import wilcoxon

BASELINE = "vurs"
ALPHA = .05


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(pvalues)-rank)*pvalues[i]))
        adjusted[i] = running
    return adjusted


def final_rows(path):
    # Committed runs are gzipped: the 6D results.json exceeds GitHub's file limit.
    raw = path/"results.json"
    text = (raw.read_text(encoding="utf-8") if raw.exists() else
            gzip.decompress((path/"results.json.gz").read_bytes()).decode("utf-8"))
    rows = json.loads(text)["rows"]
    failed = [r for r in rows if r["status"] != "ok"]
    final = {(r["case"], r["seed"], r["arm"], r["c_max"]): r for r in rows
             if r["status"] == "ok" and "seconds" in r}
    return final, failed


def paired(final, arm, other, c_max, metric):
    keys = sorted({(case, seed) for case, seed, a, c in final if a == arm and c == c_max})
    pairs = [(final[(case, seed, arm, c_max)][metric], final[(case, seed, other, c_max)][metric])
             for case, seed in keys if (case, seed, other, c_max) in final]
    return np.log(np.array([a/b for a, b in pairs]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--metric", default="end_to_end_error")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    final, failed = final_rows(args.input)
    arms = sorted({a for _, _, a, _ in final})
    costs = sorted({c for *_, c in final})

    wiring = [k for k in final if k[2] == BASELINE and k[3] == 1. and (k[0], k[1], "vurs-cost", 1.) in final]
    mismatched = sum(final[k]["x"] != final[(k[0], k[1], "vurs-cost", 1.)]["x"] for k in wiring)
    print(f"{len(final)} complete trajectories, {len(failed)} failed rows; "
          f"vurs-cost == vurs at c=1 on {len(wiring)-mismatched}/{len(wiring)} pairs")

    tests = []
    for arm in arms:
        if arm == BASELINE:
            continue
        for c in costs:
            logs = paired(final, arm, BASELINE, c, args.metric)
            p = 1. if not np.any(logs) else float(wilcoxon(logs).pvalue)
            tests.append(dict(arm=arm, c_max=c, pairs=len(logs), median_ratio=float(np.exp(np.median(logs))),
                              geomean_ratio=float(np.exp(logs.mean())), p=p))
    for test, adjusted in zip(tests, holm(np.array([t["p"] for t in tests]))):
        test["p_holm"] = float(adjusted)
        test["verdict"] = ("better" if adjusted < ALPHA and test["median_ratio"] < 1 else
                           "worse" if adjusted < ALPHA and test["median_ratio"] > 1 else "tie")

    summary = {}
    for arm in arms:
        for c in costs:
            group = [r for k, r in final.items() if k[2] == arm and k[3] == c]
            summary[f"{arm}|{c:g}"] = dict(
                error=float(np.exp(np.mean(np.log([r[args.metric] for r in group])))),
                sink_share=float(np.mean([r["sink_share"] for r in group])),
                points=float(np.mean([r["n"] for r in group])))

    print(f"\n{args.metric}, ratio vs {BASELINE} (median over pairs), Holm over {len(tests)} tests")
    print(f"{'arm':14s}" + "".join(f"{'c=' + format(c, 'g'):>22s}" for c in costs))
    for arm in arms:
        if arm == BASELINE:
            continue
        cells = []
        for c in costs:
            t = next(t for t in tests if t["arm"] == arm and t["c_max"] == c)
            cells.append(f"{t['median_ratio']:.3f} {t['verdict']:>6s} p{t['p_holm']:.3f}")
        print(f"{arm:14s}" + "".join(f"{cell:>22s}" for cell in cells))
    print(f"\ngeometric-mean {args.metric} / sink share / points")
    for arm in arms:
        print(f"{arm:14s}" + "".join(
            f"{summary[f'{arm}|{c:g}']['error']:>10.4f} {summary[f'{arm}|{c:g}']['sink_share']:.2f} "
            f"{summary[f'{arm}|{c:g}']['points']:5.0f}" for c in costs))
    if args.output:
        args.output.write_text(json.dumps(dict(metric=args.metric, tests=tests, summary=summary,
                                               wiring_pairs=len(wiring), wiring_mismatched=mismatched,
                                               failed_rows=len(failed)), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
