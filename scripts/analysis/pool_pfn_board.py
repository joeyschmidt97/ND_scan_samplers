"""TabPFN label arms on the MGKDB pools against VURS (registered 2026-10-13).

Protocol, registered before any TabPFN-arm run on the pools:

Question. Does replacing VURS's nearest-paid (kNN) boundary term with
TabPFN's posterior entropy make a campaign learn the mode map faster, without
slowing how fast it learns the growth rate? Context: the FMC matched-budget
learning curves (Fusion_Microinstability_Classifier `1437fb8`) showed TabPFN
beats tuned XGBoost on held-out label surprise at every n from 50 to 1,000
with random training sets. This study asks whether that label model also
places runs better. It is a measurement; it does not choose a default.

Arms.
- vurs-boundary-pfn: the vurs-boundary profile (coverage .2, variation .2,
  GP uncertainty .2, boundary .4) with the boundary term replaced by TabPFN's
  normalized entropy over the unpaid candidates (`label_model`).
- pfn-entropy: pure TabPFN entropy sampling (the classic active-learning arm;
  no coverage, variation or GP term; space-fills until two modes are paid).
- References, reused from the stored board (they reproduce exactly in this
  environment, checked 2026-10-13): vurs (REFERENCE), vurs-boundary (the
  kNN-boundary twin), space-filling (value-blind context).
TabPFN: keyless v2 weights, CPU, 4 ensemble members inside the acquisition
loop. Pools, seeds, budget, design, checkpoints: as `pool_batch_board`
(hatch_pscans_global and hatch_pscans3, seeds 0-19, budget 256, shared
9-point start, checkpoints 9 ... 256).

Primary endpoints (as the board): per seed, the mean over CAMPAIGN
checkpoints N = 23, 38, 61, 98 of
- S1 label surprise, bits (`s_label_bits`, Dirichlet-kNN scorer), margin 0.05;
- S2 growth-rate surprise, nats (`s_gamma_nats`), margin 0.10;
lower is better. The scorers are the board's: neither is TabPFN, so no arm
grades itself.

Comparisons, d = arm - other paired by seed (negative: the arm is less
surprised, i.e. better), mean, 95% bootstrap CI (10,000 seed resamples),
Wilcoxon signed-rank p, Holm within each pool over 3 comparisons x 2
endpoints = 6 tests:
  A  vurs-boundary-pfn - vurs-boundary   (the registered question)
  B  vurs-boundary-pfn - vurs
  C  pfn-entropy - vurs
Verdicts: `surprise_board.verdict` ("faster", "slower", "same speed",
"not slower", "inconclusive").

Claim. "The TabPFN boundary term helps" only if comparison A is "faster" on
S1 in both pools and its S2 verdict is "faster", "same speed" or "not
slower" in both pools. Anything else is reported as found.

Secondary (paired, bootstrap CI and uncorrected Wilcoxon, no verdicts): S1
on held transition-band runs (`pool.transition_band`); S1 under a TabPFN
scorer (8 members, shrunk toward uniform with the board's Jeffreys
pseudo-counts) on all held runs and on band runs, read with the caveat that
it favours the TabPFN arms; `label_accuracy`, `ctr_macro_nrmse`; S1 and S2 at
N = 256; seeds won out of 20 for every comparison. Descriptive: realized
label surprise of each run the TabPFN arms paid for, -log2 p_before(label),
and placement wall time.

    python -m ND_scan_samplers.scripts.experiments.pool_pfn --output results/pool-pfn-2026-10-13
    python -m ND_scan_samplers.scripts.analysis.pool_pfn_board score --root results/pool-pfn-2026-10-13
    python -m ND_scan_samplers.scripts.analysis.pool_pfn_board analyse --root results/pool-pfn-2026-10-13
"""
import argparse
import json
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ND_scan_samplers import ROOT
from ND_scan_samplers.scripts.analysis.pool_batch_board import (BOARD_KEYS, SEEDS, STORED_RUNS, STORED_SCORES,
                                                                endpoint, paired)
from ND_scan_samplers.scripts.analysis.surprise_board import (CAMPAIGN, CHECKPOINTS, POOLS, PRIMARY, holm,
                                                              score_one, trajectories, verdict)
from ND_scan_samplers.src.label_model import pfn_posterior
from ND_scan_samplers.src.strategies import PFN_ARMS
from ND_scan_samplers.tests.benchmarks.benchmarknd.pool import load_pool, transition_band
from ND_scan_samplers.tests.benchmarks.benchmarknd.surprise import ALPHA, LN2, macro, surprise_scores

REFERENCE = "vurs"
REFERENCES = ("vurs", "vurs-boundary", "space-filling")
COMPARISONS = {"A": ("vurs-boundary-pfn", "vurs-boundary"),
               "B": ("vurs-boundary-pfn", "vurs"),
               "C": ("pfn-entropy", "vurs")}
CONTEXT = {"space-filling vs vurs": ("space-filling", "vurs")}
EXTRA_KEYS = ("s_label_bits_band", "s_label_bits_pfn", "s_label_bits_pfn_band")
SECONDARY = {
    "S1 band": ("s_label_bits_band", CAMPAIGN),
    "S1 TabPFN scorer": ("s_label_bits_pfn", CAMPAIGN),
    "S1 TabPFN scorer, band": ("s_label_bits_pfn_band", CAMPAIGN),
    "label accuracy": ("label_accuracy", CAMPAIGN),
    "ctr_macro_nrmse": ("ctr_macro_nrmse", CAMPAIGN),
    "S1 at N=256": ("s_label_bits", (256,)),
    "S2 at N=256": ("s_gamma_nats", (256,)),
}
SCORER_SEED = 0


def extra_scores(oracle, paid, band):
    """Band-restricted kNN label surprise and the TabPFN-scored label surprise."""
    n_modes = len(oracle.label_names)
    knn = surprise_scores(oracle, paid, return_points=True)
    held = knn["held_index"]
    labels = oracle.labels[held]
    in_band = band[held]
    post = pfn_posterior(oracle.pool[paid], oracle.labels[paid], oracle.pool[held], n_modes,
                         seed=SCORER_SEED, n_estimators=None)
    post = (len(paid)*post+ALPHA)/(len(paid)+n_modes*ALPHA)
    pfn_bits = -np.log(post[np.arange(len(held)), labels])/LN2
    return dict(s_label_bits_band=macro(knn["label_bits"][in_band], labels[in_band], n_modes)[0],
                s_label_bits_pfn=macro(pfn_bits, labels, n_modes)[0],
                s_label_bits_pfn_band=macro(pfn_bits[in_band], labels[in_band], n_modes)[0])


def score_job(job):
    """S1/S2 board scores for the new arms; extra scores for every arm."""
    (arm, seed), (pool_path, sha, order, board) = job
    pool_path = str(ROOT/pool_path)             # stored runs record repo-relative paths
    job = ((arm, seed), (pool_path, sha, order, board))
    oracle, meta = load_pool(pool_path)
    if meta["sha256"] != sha:
        raise ValueError("pool file changed since the run; refusing to score")
    band = transition_band(oracle)
    rows = score_one(job) if arm in PFN_ARMS else [dict(arm=arm, seed=seed, n=n) for n in CHECKPOINTS]
    for row in rows:
        row.update(extra_scores(oracle, np.asarray(order[:row["n"]]), band))
        row.update({key: board.get(row["n"], {}).get(key) for key in BOARD_KEYS})
    metadata = board[max(board)].get("metadata", {})
    if "realized_label_bits" in metadata:
        rows[-1]["realized_label_bits"] = metadata["realized_label_bits"]
        rows[-1]["placement_seconds"] = board[max(board)].get("placement_seconds")
    return arm, seed, rows


def _init(threads):
    os.environ["ND_PFN_THREADS"] = str(threads)


def score(args):
    stored = [ROOT/r for r in STORED_RUNS]
    for pool in POOLS:
        target = args.root/"scores"/pool
        target.mkdir(parents=True, exist_ok=True)
        new = {k: v for k, v in trajectories([args.root], pool).items() if k[0] in PFN_ARMS}
        refs = {k: v for k, v in trajectories(stored, pool).items() if k[0] in REFERENCES}
        jobs = [job for job in sorted(new.items()) + sorted(refs.items())
                if not (target/f"{job[0][0]}_seed{job[0][1]}.json").exists()]
        print(f"{pool}: {len(jobs)} trajectories to score", flush=True)
        with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.threads,)) as executor:
            for arm, seed, rows in executor.map(score_job, jobs):
                (target/f"{arm}_seed{seed}.json").write_text(json.dumps(rows), encoding="utf-8")
                print(f"{pool} {arm} seed {seed}: S1-band {rows[4]['s_label_bits_band']:.3f} "
                      f"S1-pfn {rows[4]['s_label_bits_pfn']:.3f} at N={rows[4]['n']}", flush=True)


def load_table(root, pool):
    """{(arm, seed): {n: metrics}} from the stored board, stored scores and this study."""
    table = {}
    for (arm, seed), (_, _, _, board) in trajectories([ROOT/r for r in STORED_RUNS] + [root], pool).items():
        if arm not in REFERENCES and arm not in PFN_ARMS:
            continue
        for n, row in board.items():
            table.setdefault((arm, seed), {})[n] = {key: row.get(key) for key in BOARD_KEYS}
    for path in sorted((ROOT/STORED_SCORES/pool).glob("*.json")):
        for row in json.loads(path.read_text()):
            if (row["arm"], row["seed"]) in table:
                table[row["arm"], row["seed"]].setdefault(row["n"], {}).update(
                    s_label_bits=row["s_label_bits"], s_gamma_nats=row["s_gamma_nats"])
    realized = {}
    for path in sorted((pathlib.Path(root)/"scores"/pool).glob("*.json")):
        for row in json.loads(path.read_text()):
            entry = table.setdefault((row["arm"], row["seed"]), {}).setdefault(row["n"], {})
            for key in ("s_label_bits", "s_gamma_nats") + EXTRA_KEYS:
                if row.get(key) is not None:
                    entry[key] = row[key]
            if "realized_label_bits" in row:
                realized[row["arm"], row["seed"]] = (row["realized_label_bits"], row.get("placement_seconds"))
    return table, realized


def wins(first, second, key=None):
    """Seeds where the first arm is better: lower surprise or error, higher accuracy."""
    seeds = sorted(set(first) & set(second))
    better = (lambda a, b: a > b) if key == "label_accuracy" else (lambda a, b: a < b)
    return f"{sum(better(first[s], second[s]) for s in seeds)}/{len(seeds)}"


def analyse(args):
    rng = np.random.default_rng(20261013)
    report = dict(protocol=__doc__, pools={})
    for pool in POOLS:
        table, realized = load_table(args.root, pool)
        primary, keys = {}, []
        for name, (key, _) in PRIMARY.items():
            for label, (arm, other) in COMPARISONS.items():
                first, second = endpoint(table, arm, key, CAMPAIGN), endpoint(table, other, key, CAMPAIGN)
                test = paired(first, second, rng)
                if test is not None:
                    test["wins"] = wins(first, second)
                    primary.setdefault(name, {})[label] = test
                    keys.append((name, label))
        for (name, label), p in zip(keys, holm(np.array([primary[n][c]["p"] for n, c in keys]))):
            primary[name][label]["p_holm"] = float(p)
            primary[name][label]["verdict"] = verdict(primary[name][label], PRIMARY[name][1])
        secondary = {}
        for name, (key, budgets) in SECONDARY.items():
            for label, (arm, other) in {**COMPARISONS, **CONTEXT}.items():
                first, second = endpoint(table, arm, key, budgets), endpoint(table, other, key, budgets)
                test = paired(first, second, rng)
                if test is not None:
                    test["wins"] = wins(first, second, key)
                    secondary.setdefault(name, {})[label] = test
        curves = {arm: {n: {key: float(np.median([table[arm, s][n][key] for s in SEEDS
                                                  if table.get((arm, s), {}).get(n, {}).get(key) is not None]))
                            for key in ("s_label_bits", "s_gamma_nats", "s_label_bits_band", "s_label_bits_pfn")
                            if any(table.get((arm, s), {}).get(n, {}).get(key) is not None for s in SEEDS)}
                        for n in CHECKPOINTS}
                  for arm in REFERENCES + tuple(PFN_ARMS)}
        trace = {arm: dict(mean_realized_bits=float(np.mean([np.mean(v[0]) for k, v in realized.items()
                                                              if k[0] == arm])),
                           median_placement_seconds=float(np.median([v[1] for k, v in realized.items()
                                                                     if k[0] == arm and v[1] is not None])))
                 for arm in PFN_ARMS if any(k[0] == arm for k in realized)}
        report["pools"][pool] = dict(primary=primary, secondary=secondary, curves=curves, realized=trace)
    ok = []
    for pool, result in report["pools"].items():
        a = result["primary"]
        ok.append(a.get("S1", {}).get("A", {}).get("verdict") == "faster"
                  and a.get("S2", {}).get("A", {}).get("verdict") in ("faster", "same speed", "not slower"))
    report["claim_tabpfn_boundary_helps"] = bool(ok and all(ok))
    (args.root/"pfn_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.root/"pfn_summary.md").write_text(summary(report), encoding="utf-8")
    plot(report, args.root/"pfn_curves.png")
    print(summary(report))


def summary(report):
    lines = ["# TabPFN label arms vs VURS on the MGKDB pools", ""]
    for pool, result in report["pools"].items():
        lines += [f"## {pool}", "", "| endpoint | comparison | arm | other | d (CI) | wins | Holm p | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, tests in result["primary"].items():
            for label, t in tests.items():
                arm, other = COMPARISONS[label]
                lines.append(f"| {name} | {label} {arm} - {other} | {t['arm_mean']:.3f} | {t['other_mean']:.3f} | "
                             f"{t['mean']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] | {t['wins']} | "
                             f"{t['p_holm']:.3g} | {t['verdict']} |")
        lines += ["", "| secondary | comparison | d (CI) | wins | p |", "|---|---|---|---|---|"]
        for name, tests in result["secondary"].items():
            for label, t in tests.items():
                lines.append(f"| {name} | {label} | {t['mean']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] | "
                             f"{t['wins']} | {t['p']:.3g} |")
        for arm, t in result["realized"].items():
            lines.append(f"\n{arm}: mean realized label surprise {t['mean_realized_bits']:.3f} bits per paid run, "
                         f"median placement {t['median_placement_seconds']:.0f} s")
        lines.append("")
    lines.append(f"Claim (TabPFN boundary term helps): {report['claim_tabpfn_boundary_helps']}")
    return "\n".join(lines)+"\n"


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = (("s_label_bits", "S1 label surprise (bits)"), ("s_label_bits_band", "S1 on transition band (bits)"),
            ("s_gamma_nats", "S2 growth-rate surprise (nats)"))
    pools = list(report["pools"])
    fig, axes = plt.subplots(len(pools), len(keys), figsize=(4.2*len(keys), 3.4*len(pools)), squeeze=False)
    colors = {"vurs": "#555555", "vurs-boundary": "#e08a1e", "space-filling": "#aaaaaa",
              "vurs-boundary-pfn": "#2a62d9", "pfn-entropy": "#18a3a3"}
    for i, pool in enumerate(pools):
        for j, (key, title) in enumerate(keys):
            ax = axes[i, j]
            for arm, curve in report["pools"][pool]["curves"].items():
                ns = [n for n in CHECKPOINTS if key in curve.get(n, {})]
                if ns:
                    ax.plot(ns, [curve[n][key] for n in ns], marker="o", ms=3, lw=1.8,
                            color=colors.get(arm), label=arm)
            ax.set_xscale("log")
            ax.set_title(f"{pool}: {title}", fontsize=9)
            ax.set_xlabel("paid runs N")
            ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--root", type=pathlib.Path, required=True)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--threads", type=int, default=4)
    a = sub.add_parser("analyse")
    a.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    score(args) if args.command == "score" else analyse(args)


if __name__ == "__main__":
    main()
