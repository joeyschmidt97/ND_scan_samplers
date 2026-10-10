"""Batched VURS/VWRS on the MGKDB pools against sequential vurs (registered 2026-10-10).

Protocol, registered before any batched run on the pools:

Question. How do batched VWRS/VURS placements (k points per acquisition,
placed before any of their values return) compare with sequential `vurs`
at the board's budgets? This is a measurement; it does not choose a default.

Arms.
- Batched: all 16 `strategies.BATCH_ARMS` = {vwrs, vurs} x k in {5, 10, 15,
  20} x {greedy (no suffix): re-rank after each pick with the fill distance
  updated; rank (suffix r): top k of one acquisition evaluation}.
- Sequential references: `vurs` (REFERENCE) and `vwrs` (batch size 1).
- The board's COMPARATORS (`surprise_board.COMPARATORS`). Space-filling is
  value-blind, so a batch size does not change its placements; it is one row.
Pools, seeds, budget, design: `surprise_board.POOLS` (hatch_pscans_global,
hatch_pscans3), seeds 0-19, budget 256, the shared Latin-hypercube start
snapped to the pool (`pool.initial_pool_design`, 9 points), checkpoints
`surprise_board.CHECKPOINTS` = 9, 15, 23, 38, 61, 98, 159, 256 (`pool.replay`
with 8 checkpoints), scorers unchanged (`pool.score_prefix`,
`surprise.surprise_scores` via `surprise_board.score_one`).

Reuse. Sequential arms are not rerun if their stored trajectories
(results/confirm-2026-10-05, results/board20-2026-10-07) and surprise scores
(results/surprise-2026-10-07) were produced under this protocol. Checked
before registration: vurs, space-filling and gpr-var, seeds 0-1, both pools,
rerun at HEAD reproduce the stored `selected` order exactly and the pool
sha256 matches. Batched trajectories go to results/pool-batch-2026-10-10.

Mid-batch checkpoints. Inside a batch the k points are paid in pick order
(greedy: the order chosen; rank: best first). A checkpoint N that falls
inside a batch is scored on the first N points in that order, exactly as a
sequential arm's prefix; this is the primary view. Secondary view: each
endpoint at the last completed batch at or below N, n_c = 9 + k*floor((N-9)/k)
(for k = 15 and 20 at N = 23 this is the 9-point start, the same for every
arm), compared with vurs scored on its own first n_c points (same number of
paid runs).

Primary endpoints (as the board): per seed, the mean over CAMPAIGN
checkpoints N = 23, 38, 61, 98 of
- S1 label surprise, bits (`s_label_bits`), margin 0.05 bits;
- S2 growth-rate surprise, nats (`s_gamma_nats`), margin 0.10 nats;
lower is better.

Comparison. For each batched arm, paired per-seed difference
d = arm - vurs (positive: the batched arm is more surprised, i.e. worse;
note the sign is the reverse of surprise_board's vurs - comparator), mean,
95% bootstrap CI (10,000 resamples of seeds), Wilcoxon signed-rank p, Holm
across 16 batched arms x 2 primary endpoints = 32 tests within each pool.

Verdict (`surprise_board.verdict` applied to d, so it describes the batched
arm relative to vurs): "faster" (Holm p < .05 and CI below 0), "slower"
(Holm p < .05 and CI above 0), "same speed" (CI inside +-margin), "not
slower" (CI upper end below +margin), else "inconclusive". A batched arm
"keeps pace with vurs" only if it is faster, same speed or not slower on S1
and S2 in both pools.

Secondary endpoints (paired vs vurs, bootstrap CI and uncorrected Wilcoxon,
no verdicts): campaign means of `ctr_macro_nrmse`,
`region_design_band_nrmse`, `region_peak_nrmse`, `label_accuracy` (higher
is better); `ctr_macro_nrmse` at N = 61 and at N = 98; S1 and S2 at N = 256;
S1, S2 and `ctr_macro_nrmse` in the last-completed-batch view.
Descriptive contrasts (uncorrected): greedy - rank at each (base, k); each
vwrs-b<k> - sequential vwrs. Comparator rows are reported with the same
d = arm - vurs statistics in their own Holm family (9 x 2) for context; the
board's registered verdicts for them stand.
Cost: placement wall time per run (`run_arm` only, scoring excluded) and the
number of acquisitions (GP fits for vurs-b<k>: about (256-9)/k against 247).
vurs and vwrs are re-timed under the same runner and load as the batched
arms (their orders are checked against the stored ones); the comparators'
times are the stored `seconds`, which include checkpoint scoring and were
taken under other load, so they are context only.

    python -m ND_scan_samplers.scripts.experiments.pool_batch --output results/pool-batch-2026-10-10
    python -m ND_scan_samplers.scripts.analysis.pool_batch_board score --root results/pool-batch-2026-10-10
    python -m ND_scan_samplers.scripts.analysis.pool_batch_board analyse --root results/pool-batch-2026-10-10
"""
import argparse
import json
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import wilcoxon

from ND_scan_samplers import ROOT
from ND_scan_samplers.scripts.analysis.surprise_board import (CAMPAIGN, CHECKPOINTS, COMPARATORS, POOLS, PRIMARY,
                                                              holm, score_one, trajectories, verdict)
from ND_scan_samplers.src.strategies import BATCH_ARMS
from ND_scan_samplers.tests.benchmarks.benchmarknd.pool import load_pool, score_prefix, transition_band
from ND_scan_samplers.tests.benchmarks.benchmarknd.surprise import surprise_scores

REFERENCE = "vurs"
SEQUENTIAL = ("vurs", "vwrs")
SEEDS = tuple(range(20))
BUDGET = 256
START = 9
STORED_RUNS = ("results/confirm-2026-10-05", "results/board20-2026-10-07")
STORED_SCORES = "results/surprise-2026-10-07"
SECONDARY = {
    "E1 ctr_macro_nrmse": ("ctr_macro_nrmse", CAMPAIGN),
    "band error": ("region_design_band_nrmse", CAMPAIGN),
    "peak error": ("region_peak_nrmse", CAMPAIGN),
    "E2 label_accuracy (higher is better)": ("label_accuracy", CAMPAIGN),
    "E1 at N=61": ("ctr_macro_nrmse", (61,)),
    "E1 at N=98": ("ctr_macro_nrmse", (98,)),
    "S1 at N=256": ("s_label_bits", (256,)),
    "S2 at N=256": ("s_gamma_nats", (256,)),
}


def completed(n, k):
    """Points paid at the last completed batch at or below checkpoint n."""
    return n if k == 1 or n <= START else START + k*((n-START)//k)


def batch_size(arm):
    return int(arm.split("-b")[1].rstrip("r")) if arm in BATCH_ARMS else 1


BOARD_KEYS = ("ctr_macro_nrmse", "label_accuracy", "region_design_band_nrmse", "region_peak_nrmse")
VIEW_KEYS = {"S1": "s_label_bits", "S2": "s_gamma_nats", "E1": "ctr_macro_nrmse"}
RANK_ARMS = tuple(a for a in BATCH_ARMS if a.endswith("r"))


def completed_points(arm):
    """Prefix sizes the last-completed-batch view needs beyond the checkpoints."""
    sizes = (5, 10, 15, 20) if arm in SEQUENTIAL else (batch_size(arm),)
    return sorted({completed(n, k) for k in sizes for n in CAMPAIGN} - set(CHECKPOINTS))


def extra_rows(oracle, order, arm, seed):
    band = transition_band(oracle)
    rows = []
    for n in completed_points(arm):
        paid = np.asarray(order[:n])
        row = dict(arm=arm, seed=seed, **surprise_scores(oracle, paid))
        scores = score_prefix(oracle, paid, band)
        row.update({key: scores[key] for key in BOARD_KEYS})
        rows.append(row)
    return rows


def score_job(job):
    """Board checkpoint scores (batched arms only) plus the completed-batch prefixes."""
    (arm, seed), (pool_path, sha, order, board) = job
    rows = []
    if arm in BATCH_ARMS:
        rows = score_one(job)
        for row in rows:
            row.update({key: board.get(row["n"], {}).get(key) for key in BOARD_KEYS})
    oracle, meta = load_pool(pool_path)
    if meta["sha256"] != sha:
        raise ValueError("pool file changed since the run; refusing to score")
    return arm, seed, rows + extra_rows(oracle, order, arm, seed)


def score(args):
    stored = [ROOT/r for r in STORED_RUNS]
    for pool in POOLS:
        target = args.root/"scores"/pool
        target.mkdir(parents=True, exist_ok=True)
        batched = trajectories([args.root], pool)
        sequential = {key: value for key, value in trajectories(stored, pool).items() if key[0] in SEQUENTIAL}
        jobs = sorted(batched.items()) + sorted(sequential.items())
        jobs = [job for job in jobs if not (target/f"{job[0][0]}_seed{job[0][1]}.json").exists()]
        print(f"{pool}: {len(jobs)} trajectories to score", flush=True)
        with ProcessPoolExecutor(args.workers) as executor:
            for arm, seed, rows in executor.map(score_job, jobs):
                (target/f"{arm}_seed{seed}.json").write_text(json.dumps(rows), encoding="utf-8")
        # The reused surprise scores must be what the current scorer gives.
        arm, seed = REFERENCE, 0
        recomputed = score_one(((arm, seed), sequential[arm, seed]))
        stored_rows = json.loads((ROOT/STORED_SCORES/pool/f"{arm}_seed{seed}.json").read_text())
        same = all(abs(a[k]-b[k]) < 1e-9 for a, b in zip(recomputed, stored_rows)
                   for k in ("s_label_bits", "s_gamma_nats"))
        (target.parent/f"{pool}_reuse_check.json").write_text(json.dumps(dict(
            arm=arm, seed=seed, surprise_matches_stored=same)), encoding="utf-8")
        print(f"{pool}: stored vurs seed 0 surprise reproduced: {same}", flush=True)


def load_table(root, pool):
    """{(arm, seed): {n: metrics}} from the stored board, stored scores and this study."""
    table = {}
    for (arm, seed), (_, _, _, board) in trajectories([ROOT/r for r in STORED_RUNS] + [root], pool).items():
        for n, row in board.items():
            table.setdefault((arm, seed), {})[n] = {key: row.get(key) for key in BOARD_KEYS}
    for path in sorted((ROOT/STORED_SCORES/pool).glob("*.json")):
        for row in json.loads(path.read_text()):
            table.setdefault((row["arm"], row["seed"]), {}).setdefault(row["n"], {}).update(
                s_label_bits=row["s_label_bits"], s_gamma_nats=row["s_gamma_nats"])
    for path in sorted((pathlib.Path(root)/"scores"/pool).glob("*.json")):
        for row in json.loads(path.read_text()):
            entry = table.setdefault((row["arm"], row["seed"]), {}).setdefault(row["n"], {})
            for key in ("s_label_bits", "s_gamma_nats") + BOARD_KEYS:
                if entry.get(key) is None:
                    entry[key] = row.get(key)
                elif row.get(key) is not None and abs(entry[key]-row[key]) > 1e-9:
                    raise ValueError(f"{pool} {row['arm']} seed {row['seed']} n={row['n']} {key} disagrees")
    return table


def endpoint(table, arm, key, budgets):
    out = {}
    for seed in SEEDS:
        rows = table.get((arm, seed), {})
        values = [rows.get(n, {}).get(key) for n in budgets]
        if values and all(v is not None for v in values):
            out[seed] = float(np.mean(values))
    return out


def paired(first, second, rng):
    """first - second, paired by seed."""
    seeds = sorted(set(first) & set(second))
    d = np.array([first[s]-second[s] for s in seeds])
    if len(d) < 2:
        return None
    boot = rng.choice(d, (10_000, len(d))).mean(axis=1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.
    return dict(n_seeds=len(d), mean=float(d.mean()), ci=[float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5))], p=p, arm_mean=float(np.mean([first[s] for s in seeds])),
                other_mean=float(np.mean([second[s] for s in seeds])))


def primary_family(table, arms, rng):
    tests, keys = {}, []
    for name, (key, _) in PRIMARY.items():
        reference = endpoint(table, REFERENCE, key, CAMPAIGN)
        for arm in arms:
            test = paired(endpoint(table, arm, key, CAMPAIGN), reference, rng)
            if test is not None:
                tests.setdefault(name, {})[arm] = test
                keys.append((name, arm))
    for (name, arm), p in zip(keys, holm(np.array([tests[n][a]["p"] for n, a in keys]))):
        tests[name][arm]["p_holm"] = float(p)
        tests[name][arm]["verdict"] = verdict(tests[name][arm], PRIMARY[name][1])
    return tests


def wall_times(root, pool):
    """Median placement seconds and acquisitions per arm; comparators from stored seconds."""
    out = {}
    for (arm, seed), (_, _, _, board) in trajectories([root], pool).items():
        last = board[max(board)]
        out.setdefault(arm, dict(seconds=[], acquisitions=[], source="placement only, this runner"))
        out[arm]["seconds"].append(last["placement_seconds"])
        out[arm]["acquisitions"].append(last["acquisitions"])
    timing = pathlib.Path(root)/pool/"sequential_timing.json"
    if timing.exists():
        payload = json.loads(timing.read_text())
        for row in payload["rows"]:
            entry = out.setdefault(row["arm"], dict(seconds=[], acquisitions=[], matches=[],
                                                    source="placement only, this runner"))
            entry["seconds"].append(row["placement_seconds"])
            entry["acquisitions"].append(row["acquisitions"])
            entry["matches"].append(row["order_matches_stored"])
    for root_name in STORED_RUNS:
        for path in sorted((ROOT/root_name/pool).glob("seed*/results.json")):
            for row in json.loads(path.read_text())["rows"]:
                if row.get("selected") is not None and row["arm"] in COMPARATORS and row["arm"] not in SEQUENTIAL:
                    out.setdefault(row["arm"], dict(seconds=[], acquisitions=[],
                                                    source="stored seconds, placement + scoring, other load"))
                    out[row["arm"]]["seconds"].append(row["seconds"])
    return {arm: dict(v, median_seconds=float(np.median(v["seconds"])),
                      median_acquisitions=float(np.median(v["acquisitions"])) if v["acquisitions"] else None,
                      runs=len(v["seconds"])) for arm, v in out.items()}


def analyse(args):
    rng = np.random.default_rng(20261010)
    report = dict(protocol=__doc__, pools={})
    others = tuple(a for a in COMPARATORS)
    every = BATCH_ARMS + others
    for pool in POOLS:
        table = load_table(args.root, pool)
        result = dict(primary=primary_family(table, BATCH_ARMS, rng),
                      comparators=primary_family(table, others, rng))
        result["secondary"] = {name: {arm: paired(endpoint(table, arm, key, budgets),
                                                  endpoint(table, REFERENCE, key, budgets), rng)
                                      for arm in every}
                               for name, (key, budgets) in SECONDARY.items()}
        view = {}
        for name, key in VIEW_KEYS.items():
            for arm in BATCH_ARMS:
                points = tuple(completed(n, batch_size(arm)) for n in CAMPAIGN)
                test = paired(endpoint(table, arm, key, points), endpoint(table, REFERENCE, key, points), rng)
                view.setdefault(name, {})[arm] = dict(test, points=points)
        result["completed_batch_view"] = view
        contrasts = {}
        for base in ("vwrs", "vurs"):
            for k in (5, 10, 15, 20):
                greedy, rank = f"{base}-b{k}", f"{base}-b{k}r"
                for name, (key, budgets) in dict(S1=("s_label_bits", CAMPAIGN), S2=("s_gamma_nats", CAMPAIGN),
                                                 **SECONDARY).items():
                    contrasts.setdefault(f"{greedy} - {rank}", {})[name] = paired(
                        endpoint(table, greedy, key, budgets), endpoint(table, rank, key, budgets), rng)
                    if base == "vwrs":
                        for arm in (greedy, rank):
                            contrasts.setdefault(f"{arm} - vwrs", {})[name] = paired(
                                endpoint(table, arm, key, budgets), endpoint(table, "vwrs", key, budgets), rng)
        result["contrasts"] = contrasts
        result["means"] = {arm: {name: float(np.mean(list(endpoint(table, arm, key, budgets).values())))
                                 for name, (key, budgets) in dict(S1=("s_label_bits", CAMPAIGN),
                                                                  S2=("s_gamma_nats", CAMPAIGN),
                                                                  **SECONDARY).items()}
                           for arm in (REFERENCE,) + every}
        result["seeds"] = {arm: len(endpoint(table, arm, "s_label_bits", CAMPAIGN))
                           for arm in (REFERENCE,) + every}
        result["wall_time"] = wall_times(args.root, pool)
        report["pools"][pool] = result
    report["keeps_pace"] = {arm: all(report["pools"][p]["primary"][n][arm]["verdict"] in
                                     ("faster", "same speed", "not slower") for p in POOLS for n in PRIMARY)
                            for arm in BATCH_ARMS}
    out = pathlib.Path(args.root)/"analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(tables(report))
    print(f"saved {out}")


def tables(report):
    lines = []
    for pool, result in report["pools"].items():
        lines += [f"\n### {pool}  (d = arm - vurs; positive = arm worse; verdict describes the arm)\n",
                  "| arm | S1 bits | d S1 [95% CI] | verdict | S2 nats | d S2 [95% CI] | verdict "
                  "| E1 N=61 | E1 N=98 | median s | acq |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        means, wall = result["means"], result["wall_time"]
        for arm in (REFERENCE, "vwrs") + BATCH_ARMS + tuple(a for a in COMPARATORS if a != "vwrs"):
            cells = [arm]
            for name in PRIMARY:
                family = result["primary"] if arm in BATCH_ARMS else result["comparators"]
                cells.append(f"{means[arm][name]:.3f}")
                t = family.get(name, {}).get(arm)
                cells += (["--", "reference"] if t is None else
                          [f"{t['mean']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]",
                           t["verdict"] + ("" if arm in BATCH_ARMS else "*")])
            cells += [f"{means[arm]['E1 at N=61']:.3f}", f"{means[arm]['E1 at N=98']:.3f}"]
            w = wall.get(arm)
            cells += ["--", "--"] if w is None else [
                f"{w['median_seconds']:.1f}" + ("" if w["source"].startswith("placement") else "+"),
                "--" if w["median_acquisitions"] is None else f"{w['median_acquisitions']:.0f}"]
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("\n* comparator verdicts: own Holm family (9 x 2), context only; the board's verdicts stand. "
                 "+ stored seconds incl. scoring, other load.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--root", type=pathlib.Path, required=True)
    s.add_argument("--workers", type=int, default=12)
    a = sub.add_parser("analyse")
    a.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    score(args) if args.command == "score" else analyse(args)


if __name__ == "__main__":
    main()
