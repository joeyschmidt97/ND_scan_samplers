"""Refit one common GP on every arm's final design: design quality without its kernel.

Each GP-based arm is normally scored by its own surrogate, so a comparison of
native errors mixes two things: where the arm put its points and which kernel
its predictor uses. VURS acquires and predicts with Matern-1/2; the
uncertainty-only GP arms with Matern-3/2 or Matern-1/2. Refitting the same GP
(identical kernel family, bounds, restarts and seed as
`benchmarknd.strategies.fit_gp`) on each frozen final design isolates the
design. Both kernels are refitted, so a design that only works with the kernel
it was acquired with shows up as kernel-dependent.

The designs come from a committed results file; nothing is re-acquired.

    python -m scripts.refit_common_gp --source results/6d-ionut-spine-2026-09-20 \
        --output results/6d-common-gp-refit-2026-10-04
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import pathlib
import subprocess
import types

import numpy as np
from scipy.stats import qmc

from ND_scan_samplers.tests.benchmarks.benchmarknd.ionut import IonutSurface
from ND_scan_samplers.src.strategies import fit_gp

ARMS = ("vurs", "gpr-var", "gpr-m05-var", "vwrs", "space-filling", "gpr-u50-g50")
KERNELS = (1.5, .5)
TEST_SEED, TEST_LOG2 = 91479, 14


def refit(job):
    case, arm, seed, x, y = job
    surface = IonutSurface(case)
    test = qmc.Sobol(6, scramble=True, seed=TEST_SEED).random_base2(TEST_LOG2)
    truth = surface(test)
    scale = float(np.ptp(truth))
    obs = types.SimpleNamespace(x=np.asarray(x), y=np.asarray(y), dim=6)
    out = dict(case=case, arm=arm, seed=seed, n=len(x))
    for nu in KERNELS:
        gp, warnings = fit_gp(obs, seed, nu=nu)
        out[f"error_m{nu}"] = float(np.sqrt(np.mean((gp.predict(test)-truth)**2))/scale)
        out[f"fit_warnings_m{nu}"] = warnings
    return out


def summarize(rows, arms):
    """Per kernel: mean over seeds per case, then ratio to the best arm per case."""
    summary = {}
    for nu in KERNELS:
        key = f"error_m{nu}"
        cases = sorted({r["case"] for r in rows})
        table = {c: {a: float(np.mean([r[key] for r in rows if r["case"] == c and r["arm"] == a]))
                     for a in arms} for c in cases}
        ratios = np.array([[table[c][a]/min(table[c].values()) for a in arms] for c in cases])
        summary[f"matern_{nu}"] = dict(
            per_case=table,
            ratio_mean={a: float(ratios[:, i].mean()) for i, a in enumerate(arms)},
            ratio_worst={a: float(ratios[:, i].max()) for i, a in enumerate(arms)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    source = json.loads((args.source/"results.json").read_text())["rows"]
    jobs = [(r["case"], r["arm"], r["seed"], r["x"], r["y"]) for r in source
            if r["status"] == "ok" and r["n"] == args.budget and r["arm"] in args.arms and r.get("x")]
    if not jobs:
        raise SystemExit("no final designs found for the requested arms and budget")
    with ProcessPoolExecutor(args.workers) as pool:
        rows = sorted(pool.map(refit, jobs), key=lambda r: (r["case"], r["arm"], r["seed"]))

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    payload = dict(source=str(args.source), budget=args.budget, arms=args.arms, kernels=KERNELS,
                   test=dict(sobol_seed=TEST_SEED, points=2**TEST_LOG2), code_commit=commit,
                   rows=rows, summary=summarize(rows, args.arms))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"results.json").write_text(json.dumps(payload, indent=1))
    for nu in KERNELS:
        s = payload["summary"][f"matern_{nu}"]
        print(f"Matern-{nu}: " + "  ".join(f"{a} {s['ratio_mean'][a]:.2f}/{s['ratio_worst'][a]:.2f}" for a in args.arms))


if __name__ == "__main__":
    main()
