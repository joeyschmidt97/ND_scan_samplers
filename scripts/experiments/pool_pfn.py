"""Pool replay of the TabPFN label arms on the MGKDB pools (protocol in
`ND_scan_samplers.scripts.analysis.pool_pfn_board`).

Each (pool, arm, seed) is the board's `pool.replay` -- budget 256, the shared
9-point start, checkpoints 9 ... 256 scored with `pool.score_prefix` -- for the
arms in `strategies.PFN_ARMS`. The references (vurs, vurs-boundary,
space-filling) are not rerun: their stored board trajectories reproduce
exactly under this environment (checked 2026-10-13, seed 0 both pools, and
vurs-boundary seed 1 after the PFN change to `strategies`). Output per seed,
in the board's shape:

    <output>/<pool>/seed<k>/results.json

Each worker runs TabPFN on --threads torch threads (ND_PFN_THREADS), so
--workers x --threads should not exceed the core count.

    python -m ND_scan_samplers.scripts.experiments.pool_pfn --output results/pool-pfn-2026-10-13
"""
import argparse
import json
import os
import pathlib
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from ND_scan_samplers import ROOT
from ND_scan_samplers.scripts.analysis.pool_batch_board import BUDGET, SEEDS
from ND_scan_samplers.scripts.analysis.surprise_board import CHECKPOINTS, POOLS
from ND_scan_samplers.scripts.experiments.pool_batch import oracle_for, place
from ND_scan_samplers.src.strategies import PFN_ARMS
from ND_scan_samplers.tests.benchmarks.benchmarknd.pool import checkpoints, initial_pool_design, score_prefix

KEPT_METADATA = ("acquisition_weights", "variation", "candidates", "fit_warnings", "boundary_model",
                 "acquisition", "realized_label_bits")


def _init(threads):
    os.environ["ND_PFN_THREADS"] = str(threads)


def pfn_job(job):
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
        rows[-1].update(seconds=placement+scoring, placement_seconds=placement, scoring_seconds=scoring,
                        selected=order.tolist(),
                        metadata={k: v for k, v in metadata.items() if k in KEPT_METADATA})
    except Exception as error:                                  # noqa: BLE001
        rows = [dict(arm=arm, seed=seed, budget=BUDGET, status="failed",
                     reason=f"{type(error).__name__}: {error}")]
    return pool, arm, seed, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--pools", nargs="+", default=list(POOLS))
    parser.add_argument("--arms", nargs="+", choices=list(PFN_ARMS), default=list(PFN_ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=ROOT, text=True).strip())
    config = {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()}
    config.update(budget=BUDGET, checkpoints=len(CHECKPOINTS))

    payloads, jobs = {}, []
    for pool in args.pools:
        _, meta, _ = oracle_for(pool)
        for seed in args.seeds:
            path = args.output/pool/f"seed{seed}"/"results.json"
            done = {}
            if path.exists():                       # resume: keep finished arms
                for row in json.loads(path.read_text())["rows"]:
                    if row["status"] == "ok":
                        done.setdefault(row["arm"], []).append(row)
            payloads[pool, seed] = dict(pool=meta, commit=commit, src_dirty=dirty,
                                        config=dict(config, seeds=[seed], pools=[pool]), rows=done)
            jobs += [(pool, arm, seed) for arm in args.arms if arm not in done]

    started = time.perf_counter()
    print(f"{len(jobs)} runs, {args.workers} workers x {args.threads} threads", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.threads,)) as executor:
        futures = [executor.submit(pfn_job, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            pool, arm, seed, result = future.result()
            payload = payloads[pool, seed]
            payload["rows"][arm] = result
            ordered = [row for a in PFN_ARMS if a in payload["rows"] for row in payload["rows"][a]]
            path = args.output/pool/f"seed{seed}"/"results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(payload, rows=ordered), indent=2, allow_nan=False),
                            encoding="utf-8")
            last = result[-1]
            status = (f"placement {last['placement_seconds']:.0f}s N={last['n']} "
                      f"label_acc {last['label_accuracy']:.3f}" if last["status"] == "ok"
                      else f"FAILED {last['reason']}")
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(futures)} {pool} seed={seed} {arm}: {status}",
                  flush=True)
    print(f"done in {time.perf_counter()-started:.0f}s")


if __name__ == "__main__":
    main()
