"""Pool replay of the batched VWRS/VURS arms on the MGKDB pools (protocol in
`ND_scan_samplers.scripts.analysis.pool_batch_board`).

Each (pool, arm, seed) is the board's `pool.replay` -- budget 256, the shared
9-point start, checkpoints 9 ... 256 scored with `pool.score_prefix` -- with
the placement (`run_arm`) timed apart from the scoring, so the wall time of a
batch of k (about 247/k acquisitions, one GP fit each for vurs) can be set
against the sequential arm. Output per seed, in the board's shape:

    <output>/<pool>/seed<k>/results.json

Sequential vurs and vwrs are not rescored (the stored board trajectories are
reused); with --time-sequential they are rerun under the same runner and load
for their wall time only, and their order is checked against the stored one:

    <output>/<pool>/sequential_timing.json

    python -m ND_scan_samplers.scripts.experiments.pool_batch --output results/pool-batch-2026-10-10 --time-sequential
"""
import argparse
import json
import pathlib
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ND_scan_samplers import ROOT
from ND_scan_samplers.scripts.analysis.pool_batch_board import BUDGET, SEEDS, SEQUENTIAL, STORED_RUNS
from ND_scan_samplers.scripts.analysis.surprise_board import CHECKPOINTS, POOLS
from ND_scan_samplers.src.strategies import BATCH_ARMS, run_arm
from ND_scan_samplers.tests.benchmarks.benchmarknd.core import Observations
from ND_scan_samplers.tests.benchmarks.benchmarknd.pool import (checkpoints, initial_pool_design, load_pool,
                                                                score_prefix, transition_band)

_CACHE = {}


def pool_path(pool):
    return ROOT/"data"/"pools"/f"{pool}.csv"


def oracle_for(pool):
    if pool not in _CACHE:
        oracle, meta = load_pool(pool_path(pool))
        band = transition_band(oracle)
        meta["band_size"] = int(band.sum())
        _CACHE[pool] = (oracle, meta, band)
    return _CACHE[pool]


def place(oracle, arm, seed):
    start = time.perf_counter()
    obs = Observations(oracle, BUDGET, oracle.dim, seed, shared=initial_pool_design(oracle, seed))
    _, metadata = run_arm(arm, obs, seed)
    return oracle.indices(obs.x), metadata, time.perf_counter()-start


def batch_job(job):
    """`pool.replay` with placement and scoring timed separately."""
    pool, arm, seed = job
    oracle, _, band = oracle_for(pool)
    try:
        order, metadata, placement = place(oracle, arm, seed)
        start = time.perf_counter()
        rows = []
        for n in checkpoints(len(initial_pool_design(oracle, seed)), BUDGET, len(CHECKPOINTS)):
            row = dict(arm=arm, seed=seed, budget=BUDGET, status="ok")
            row.update(score_prefix(oracle, order[:n], band))
            rows.append(row)
        scoring = time.perf_counter()-start
        sizes = metadata.get("batch_sizes", [])
        rows[-1].update(seconds=placement+scoring, placement_seconds=placement, scoring_seconds=scoring,
                        acquisitions=len(sizes), selected=order.tolist(),
                        metadata={k: v for k, v in metadata.items() if k in
                                  ("acquisition_weights", "variation", "anisotropic", "candidates",
                                   "batch", "batch_mode", "batch_sizes", "fit_warnings")})
    except Exception as error:                                  # noqa: BLE001
        rows = [dict(arm=arm, seed=seed, budget=BUDGET, status="failed",
                     reason=f"{type(error).__name__}: {error}")]
    return pool, arm, seed, rows


def stored_order(pool, arm, seed):
    for root in STORED_RUNS:
        path = ROOT/root/pool/f"seed{seed}"/"results.json"
        if path.exists():
            for row in json.loads(path.read_text())["rows"]:
                if row["arm"] == arm and row.get("selected") is not None:
                    return row["selected"]
    return None


def timing_job(job):
    pool, arm, seed = job
    oracle, _, _ = oracle_for(pool)
    order, _, placement = place(oracle, arm, seed)
    stored = stored_order(pool, arm, seed)
    return pool, arm, seed, dict(arm=arm, seed=seed, placement_seconds=placement,
                                 acquisitions=BUDGET-len(initial_pool_design(oracle, seed)),
                                 order_matches_stored=stored is not None and order.tolist() == stored)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--pools", nargs="+", default=list(POOLS))
    parser.add_argument("--arms", nargs="+", choices=BATCH_ARMS, default=list(BATCH_ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--time-sequential", action="store_true",
                        help="also rerun vurs and vwrs for their placement wall time")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=ROOT, text=True).strip())
    config = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
    config.update(budget=BUDGET, checkpoints=len(CHECKPOINTS))

    payloads, timing = {}, {}
    for pool in args.pools:
        _, meta, _ = oracle_for(pool)
        for seed in args.seeds:
            path = args.output/pool/f"seed{seed}"/"results.json"
            if path.exists():
                parser.error(f"{path} already exists; choose a new --output")
            payloads[pool, seed] = dict(pool=meta, commit=commit, src_dirty=dirty,
                                        config=dict(config, seeds=[seed], pools=[pool]), rows={})
        timing[pool] = dict(commit=commit, src_dirty=dirty, workers=args.workers, rows=[])

    jobs = [(pool, arm, seed) for pool in args.pools for seed in args.seeds for arm in args.arms]
    sequential = ([(pool, arm, seed) for pool in args.pools for seed in args.seeds for arm in SEQUENTIAL]
                  if args.time_sequential else [])
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as executor:
        # The long sequential reruns go first so the batched arms are timed under
        # the same load rather than on an idle machine at the end.
        futures = [executor.submit(timing_job, job) for job in sequential]
        futures += [executor.submit(batch_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            pool, arm, seed, result = future.result()
            if arm in SEQUENTIAL:
                timing[pool]["rows"].append(result)
                path = args.output/pool/"sequential_timing.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(timing[pool], indent=2), encoding="utf-8")
                status = f"placement {result['placement_seconds']:.1f}s match={result['order_matches_stored']}"
            else:
                payload = payloads[pool, seed]
                payload["rows"][arm] = result
                ordered = [row for a in args.arms if a in payload["rows"] for row in payload["rows"][a]]
                path = args.output/pool/f"seed{seed}"/"results.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(dict(payload, rows=ordered), indent=2, allow_nan=False),
                                encoding="utf-8")
                last = result[-1]
                status = (f"placement {last['placement_seconds']:.1f}s N={last['n']} "
                          f"E1 {last['ctr_macro_nrmse']:.3f}" if last["status"] == "ok"
                          else f"FAILED {last['reason']}")
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(futures)} {pool} seed={seed} {arm}: {status}",
                  flush=True)
    print(f"done in {time.perf_counter()-started:.0f}s")


if __name__ == "__main__":
    main()
