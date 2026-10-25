"""Trajectories of the core field on the VURS confirmation geometries.

Protocol: `wiki/queries/2026-10-24-vurs-confirmation-protocol` (Vulcan wiki);
cases in `ND_scan_samplers.scripts.data.confirmation_cases`. One (block, arm)
is one trajectory to N = --budget (512 primary; 256 in the 2-D/5-D dev pilots); checkpoints are prefixes of the paid order, so
nothing is rerun per budget. Per trajectory:

    <output>/<namespace>/block<k>/<arm>.json

holding the paid order, values, labels, wall time and, on failure, the error.
A finished file is never recomputed (checkpoint recovery); a failure is
written, not dropped, and is retried only with --retry-failed.

Sequential arms share one Latin-hypercube start of 2d+1 points drawn from the
block's acquisition seed. sg_lib pays only its own prescribed nodes, with no
LHS start; it has no acquisition randomness, so one run per geometry is all
there is. sg_lib runs at tol = 1e-6 per direction, the value of sg_lib's own
example drivers (the 2-D development pilot of 2026-10-25 used the historical
tol = 0 adapter). --dim picks the case dimension; 2-D is guidance, 5-D/6-D the
primary test.

    python -m ND_scan_samplers.scripts.experiments.vurs_confirm --namespace dev --dim 5 --blocks 0-7,20-27,40-47,60-67 --output outputs/vurs-confirm-5d
"""
import argparse
import json
import os
import pathlib
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.stats import qmc

from ND_scan_samplers.scripts.data.confirmation_cases import (DEV_ENTROPY, ConfirmationCase,
                                                              confirmation_entropy, role_seed)
from ND_scan_samplers.src import strategies
from ND_scan_samplers.tests.benchmarks.benchmarknd.core import Observations

BUDGET = 512
ARMS = ("vurs", "space-filling", "gpr-var-m12", "gpr-var-m32", "gpr-var-m52", "moe", "sglib")
GP_NU = {"gpr-var-m12": .5, "gpr-var-m32": 1.5, "gpr-var-m52": 2.5}
SGLIB_TOL = 1e-6


def parse_blocks(text):
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo)+1))
    return out


def shared_start(entropy, block, dim=2):
    return qmc.LatinHypercube(dim, seed=role_seed(entropy, block, "acquisition", dim)).random(2*dim+1)


def trajectory(entropy, block, arm, dim=2, budget=BUDGET):
    case = ConfirmationCase(entropy, block, dim)
    start = np.empty((0, case.dim)) if arm == "sglib" else shared_start(entropy, block, case.dim)
    obs = Observations(case, budget, case.dim, shared=start)
    seed = role_seed(entropy, block, "acquisition", dim)
    out = dict(block=block, arm=arm, case=case.describe(), budget=budget, start=len(start))
    if arm == "sglib":
        out["sglib_tol"] = SGLIB_TOL
    began = time.perf_counter()
    try:
        if arm in GP_NU:
            _, info = strategies.gpr(obs, seed, nu=GP_NU[arm])
        else:
            _, info = strategies.run_arm(arm, obs, seed, sglib_tol=SGLIB_TOL)
        out.update(status="complete", info={k: v for k, v in info.items()
                                             if k in ("batch_sizes", "kernel", "matern_nu", "fit_warnings")})
    except Exception as error:
        out.update(status="failed", error=f"{type(error).__name__}: {error}",
                   traceback=traceback.format_exc())
    out.update(seconds=time.perf_counter()-began, n=len(obs.x),
               x=obs.x.tolist(), y=obs.y.tolist(),
               labels=case.region(obs.x).tolist() if len(obs.x) else [])
    return out


def _work(entropy, block, arm, path, dim, budget):
    result = trajectory(entropy, block, arm, dim, budget)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return block, arm, result["status"], result["n"], result["seconds"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", choices=("dev", "confirm"), required=True)
    parser.add_argument("--manifest-commit", help="frozen manifest commit; required for --namespace confirm")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--blocks", required=True, help="e.g. 0-7,20-27")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.namespace == "confirm":
        if not args.manifest_commit:
            parser.error("--namespace confirm needs --manifest-commit")
        entropy = confirmation_entropy(args.manifest_commit)
    else:
        entropy = DEV_ENTROPY
    root = pathlib.Path(args.output, args.namespace)
    jobs = []
    for block in parse_blocks(args.blocks):
        for arm in args.arms.split(","):
            if arm not in ARMS:
                parser.error(f"unknown arm {arm}")
            path = root/f"block{block}"/f"{arm}.json"
            if path.exists():
                done = json.loads(path.read_text())["status"] == "complete"
                if done or not args.retry_failed:
                    continue
            jobs.append((block, arm, path))
    # slow sg_lib jobs first so they do not trail the pool
    jobs.sort(key=lambda job: job[1] != "sglib")
    print(f"{len(jobs)} trajectories to run, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_work, entropy, *job, args.dim, args.budget) for job in jobs]
        for future in as_completed(futures):
            block, arm, status, n, seconds = future.result()
            print(f"block{block:3d} {arm:14s} {status:8s} n={n:3d} {seconds:7.1f}s", flush=True)


if __name__ == "__main__":
    # one BLAS thread per worker; spawned workers inherit this before numpy loads
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    main()
