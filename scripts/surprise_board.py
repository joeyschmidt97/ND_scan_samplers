"""Bayesian-surprise scoring and the registered VURS comparison on the MGKDB pools.

Protocol (registered before the full scoring, vault note
2026-10-07-vurs-surprise-scoring, "Confirmatory protocol"):

- reference `vurs`; comparators the ten-arm MGKDB board;
- primary endpoints per seed, each the mean over campaign checkpoints
  N = 23, 38, 61, 98: S1 label surprise (bits) and S2 growth-rate surprise
  (nats), both from `benchmarknd.surprise`, lower is better;
- paired per-seed differences vurs - comparator: mean, 95% bootstrap CI
  (10,000 resamples), Wilcoxon signed-rank p, Holm across 9 comparators x 2
  primary endpoints within each pool;
- margins 0.05 bits (S1) and 0.10 nats (S2); verdicts faster / same speed /
  not slower / slower / inconclusive; "at least as fast as X" needs faster,
  same speed or not slower on S1 and S2 in both pools.

Score every stored trajectory (placements unchanged, no rerun), then analyse:

    python -m scripts.surprise_board score --runs results/confirm-2026-10-05 results/board20-2026-10-07 \
        --output results/surprise-2026-10-07
    python -m scripts.surprise_board analyse --root results/surprise-2026-10-07
"""
import argparse
import glob
import json
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import wilcoxon

from benchmarknd.pool import initial_pool_design, load_pool
from benchmarknd.surprise import prequential_label_bits, surprise_scores

POOLS = ("hatch_pscans_global", "hatch_pscans3")
CHECKPOINTS = (9, 15, 23, 38, 61, 98, 159, 256)
CAMPAIGN = (23, 38, 61, 98)
REFERENCE = "vurs"
COMPARATORS = ("space-filling", "gpr-var", "gpr-grad", "gpr-u50-g50", "gpr-m05-blend",
               "vwrs", "vwrs-m", "vurs-m", "moe")
PRIMARY = {"S1": ("s_label_bits", .05), "S2": ("s_gamma_nats", .10)}
SECONDARY = {
    "S1 at N=256": ("s_label_bits", (256,)),
    "S2 at N=256": ("s_gamma_nats", (256,)),
    "S1 all checkpoints": ("s_label_bits", CHECKPOINTS),
    "S2 all checkpoints": ("s_gamma_nats", CHECKPOINTS),
    "overconfidence S1-H": ("overconfidence_bits", CAMPAIGN),
    "S2 median": ("s_gamma_median_nats", CAMPAIGN),
    "S2 single GP": ("s_gamma_global_nats", CAMPAIGN),
    "E1": ("ctr_macro_nrmse", CAMPAIGN),
    "E2 (higher is better)": ("label_accuracy", CAMPAIGN),
    "E1 at N=256": ("ctr_macro_nrmse", (256,)),
    "E2 at N=256 (higher is better)": ("label_accuracy", (256,)),
}


def trajectories(run_roots, pool):
    """{(arm, seed): (pool path, selected order, {n: board row})} across result sets."""
    found = {}
    for root in run_roots:
        for path in sorted(glob.glob(str(pathlib.Path(root)/pool/"seed*"/"results.json"))):
            payload = json.loads(pathlib.Path(path).read_text())
            board = {}
            for row in payload["rows"]:
                if row["status"] == "ok":
                    board.setdefault((row["arm"], row["seed"]), {})[row["n"]] = row
            for key, rows in board.items():
                order = rows[max(rows)].get("selected")
                if order is None:
                    continue
                if key in found:
                    raise ValueError(f"{key} appears in two result sets")
                found[key] = (payload["pool"]["path"], payload["pool"]["sha256"], order, rows)
    return found


def score_one(job):
    (arm, seed), (pool_path, sha, order, rows) = job
    oracle, meta = load_pool(pool_path)
    if meta["sha256"] != sha:
        raise ValueError("pool file changed since the run; refusing to score")
    start = len(initial_pool_design(oracle, seed))
    out = []
    for n in CHECKPOINTS:
        row = dict(arm=arm, seed=seed, **surprise_scores(oracle, np.asarray(order[:n])))
        row["overconfidence_bits"] = row["s_label_bits"]-row["label_entropy_bits"]
        board = rows.get(n, {})
        for key in ("ctr_macro_nrmse", "label_accuracy"):
            row[key] = board.get(key)
        out.append(row)
    bits = prequential_label_bits(oracle, order, start)
    out[-1]["prequential_label_bits"] = bits
    return out


def score(args):
    for pool in POOLS:
        target = args.output/pool
        target.mkdir(parents=True, exist_ok=True)
        jobs = sorted(trajectories(args.runs, pool).items())
        done = {(r["arm"], r["seed"]) for p in target.glob("*.json")
                for r in json.loads(p.read_text())}
        jobs = [job for job in jobs if job[0] not in done]
        print(f"{pool}: {len(jobs)} trajectories to score", flush=True)
        with ProcessPoolExecutor(args.workers) as executor:
            for rows in executor.map(score_one, jobs):
                arm, seed = rows[0]["arm"], rows[0]["seed"]
                (target/f"{arm}_seed{seed}.json").write_text(json.dumps(rows), encoding="utf-8")
                print(f"{pool} {arm} seed {seed}: S1 {rows[4]['s_label_bits']:.3f} "
                      f"S2 {rows[4]['s_gamma_nats']:.3f} at N={rows[4]['n']}", flush=True)


def load_scores(root, pool):
    rows = []
    for path in sorted((pathlib.Path(root)/pool).glob("*.json")):
        rows += json.loads(path.read_text())
    return rows


def endpoint(rows, arm, key, budgets):
    out = {}
    for seed in sorted({r["seed"] for r in rows}):
        values = [r[key] for r in rows if r["arm"] == arm and r["seed"] == seed and r["n"] in budgets]
        if len(values) == len(budgets) and all(v is not None for v in values):
            out[seed] = float(np.mean(values))
    return out


def paired(rows, other, key, budgets, rng):
    ref, comp = endpoint(rows, REFERENCE, key, budgets), endpoint(rows, other, key, budgets)
    seeds = sorted(set(ref) & set(comp))
    d = np.array([ref[s]-comp[s] for s in seeds])
    if len(d) < 2:
        return None
    boot = rng.choice(d, (10_000, len(d))).mean(axis=1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.
    return dict(n_seeds=len(d), mean=float(d.mean()), ci=[float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5))], p=p, reference_mean=float(np.mean([ref[s] for s in seeds])),
                comparator_mean=float(np.mean([comp[s] for s in seeds])))


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(pvalues)-rank)*pvalues[i]))
        adjusted[i] = running
    return adjusted


def verdict(test, margin):
    low, high = test["ci"]
    if test["p_holm"] < .05 and high < 0:
        return "faster"
    if test["p_holm"] < .05 and low > 0:
        return "slower"
    if -margin < low and high < margin:
        return "same speed"
    if high < margin:
        return "not slower"
    return "inconclusive"


def analyse(args):
    rng = np.random.default_rng(20261007)
    report = dict(protocol=__doc__, pools={})
    for pool in POOLS:
        rows = load_scores(args.root, pool)
        primary, keys = {}, []
        for name, (key, _) in PRIMARY.items():
            for other in COMPARATORS:
                test = paired(rows, other, key, CAMPAIGN, rng)
                if test is not None:
                    primary.setdefault(name, {})[other] = test
                    keys.append((name, other))
        adjusted = holm(np.array([primary[n][o]["p"] for n, o in keys]))
        for (name, other), p in zip(keys, adjusted):
            primary[name][other]["p_holm"] = float(p)
            primary[name][other]["verdict"] = verdict(primary[name][other], PRIMARY[name][1])
        secondary = {name: {other: paired(rows, other, key, budgets, rng) for other in COMPARATORS}
                     for name, (key, budgets) in SECONDARY.items()}
        arms = sorted({r["arm"] for r in rows})
        curves = {arm: {str(n): {k: [r[k] for r in rows if r["arm"] == arm and r["n"] == n]
                                 for k in ("s_label_bits", "s_gamma_nats", "label_entropy_bits")}
                        for n in CHECKPOINTS} for arm in arms}
        prequential = {arm: [r["prequential_label_bits"] for r in rows
                             if r["arm"] == arm and "prequential_label_bits" in r] for arm in arms}
        report["pools"][pool] = dict(primary=primary, secondary=secondary, curves=curves,
                                     prequential=prequential, seeds={arm: len(prequential[arm]) for arm in arms})
    claims = {}
    for other in COMPARATORS:
        verdicts = [report["pools"][p]["primary"][n][other]["verdict"] for p in POOLS for n in PRIMARY
                    if other in report["pools"][p]["primary"].get(n, {})]
        complete = len(verdicts) == len(POOLS)*len(PRIMARY)
        claims[other] = dict(
            verdicts=verdicts,
            at_least_as_fast=complete and all(v in ("faster", "same speed", "not slower") for v in verdicts),
            faster=complete and "slower" not in verdicts and all(
                any(report["pools"][p]["primary"][n][other]["verdict"] == "faster" for n in PRIMARY)
                for p in POOLS))
    report["claims"] = claims
    out = pathlib.Path(args.root)/"surprise_analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for pool in POOLS:
        print(f"\n{pool} (vurs - comparator, campaign mean; negative = vurs less surprised)")
        for name in PRIMARY:
            for other, t in report["pools"][pool]["primary"].get(name, {}).items():
                print(f"  {name} {other:14s} n={t['n_seeds']:2d} {t['mean']:+.3f} "
                      f"[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] p_holm={t['p_holm']:.3g} {t['verdict']}")
    print("\nclaims:", json.dumps({k: (v["at_least_as_fast"], v["faster"]) for k, v in claims.items()}))
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--runs", type=pathlib.Path, nargs="+", required=True)
    s.add_argument("--output", type=pathlib.Path, required=True)
    s.add_argument("--workers", type=int, default=8)
    a = sub.add_parser("analyse")
    a.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    score(args) if args.command == "score" else analyse(args)


if __name__ == "__main__":
    main()
