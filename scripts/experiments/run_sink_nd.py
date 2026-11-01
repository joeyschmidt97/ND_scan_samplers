"""6D compute-sink sweep on Ionut's microinstability proxies.

The 2D sweep (`run_sink_2d`) puts the spread and the cost on distance to a
geometric fold. Here both come from the proxies' own branch competition,

    s(x) = exp(-|G_lead - G_second| / (T * ptp(G)))      (`resolution.noise`)

the transition-competition field that sets the spread of `NoisyIonutSurface`,
so a contested point is one where two microinstability branches are tied. Cost,
budget, arms and the cost-aware arm are those of `run_sink_2d`; the start is the
shared 2d+1 Latin hypercube of the ND benchmarks instead of the 2D corners.

Grading follows the confirmation scorer (C2 reconstruction, `vurs_confirm_score`):
a fixed Matern-3/2 GP with a white-noise term, fitted to each arm's paid prefix,
RMSE over a fixed Sobol grading cloud divided by the case scale. As in 2D it is
fitted twice from one design -- `placement` on the true values at the paid
points, `end_to_end` on the noisy values the sampler saw -- and also reported on
the contested part of the cloud (s > SINK_AT). Label surprise in bits uses the
Dirichlet-kNN posterior of `benchmarknd.surprise`.

    python -m ND_scan_samplers.scripts.experiments.run_sink_nd --output results/sink-6d-<date>
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import pathlib
import platform
import subprocess
import time

import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

from ND_scan_samplers import ROOT
from ND_scan_samplers.scripts.analysis.vurs_confirm_score import fit_reconstruction
from ND_scan_samplers.scripts.experiments.run_sink_2d import (ARMS, DEFAULT_COSTS, SINK_AT,
                                                              ChargedObservations, compute_checkpoints,
                                                              run_strategy)
from ND_scan_samplers.src.resolution.noise import TransitionNoise, competition
from ND_scan_samplers.tests.benchmarks.benchmarknd.ionut import CASES
from ND_scan_samplers.tests.benchmarks.benchmarknd.noisy import NoisyIonutSurface
from ND_scan_samplers.tests.benchmarks.benchmarknd.surprise import label_posterior

DEFAULT_CASES = ("ionut-itg-tem-argmax-gamma", "ionut-itg-tem-softmax-gamma",
                 "ionut-itg-kbm-argmax-gamma", "ionut-itg-kbm-softmax-gamma")
DEFAULT_PEAK = .25               # TransitionNoise's declared peak
GRADING_POINTS = 4096
GRADING_SEED = 91479             # the ND evaluation-set scramble


class ChargedIonutSurface(NoisyIonutSurface):
    """Noisy proxy that also exposes the competition field the cost is charged on."""

    def __init__(self, case, peak=DEFAULT_PEAK, seed=0):
        super().__init__(case, TransitionNoise(peak=peak), seed)

    def competition(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        return competition(self.truth.branches(x), self.noise.temperature, self.scale())


def grading(surface):
    x = qmc.Sobol(surface.dim, scramble=True, seed=GRADING_SEED).random_base2(
        GRADING_POINTS.bit_length()-1)
    y = surface.truth(x)
    region = surface.truth.region(x)
    return dict(x=x, y=y, scale=float(np.ptp(y)), region=region,
                n_modes=int(surface.truth.branches(x[:1]).shape[1]),
                contested=surface.competition(x) > SINK_AT)


def score_design(surface, design, noisy, test, seed):
    out = {}
    truth = test["y"]
    for label, values in (("placement", surface.truth(design)), ("end_to_end", noisy)):
        yhat = fit_reconstruction(design, values, 1.5, seed).predict(test["x"])
        error = (yhat-truth)/test["scale"]
        out[f"{label}_error"] = float(np.sqrt(np.mean(error**2)))
        out[f"{label}_band_error"] = (float(np.sqrt(np.mean(error[test["contested"]]**2)))
                                      if test["contested"].any() else None)
    out["corruption_penalty"] = out["end_to_end_error"]-out["placement_error"]
    labels = surface.truth.region(design)
    posterior = label_posterior(design, labels, test["x"], test["n_modes"])
    out["label_bits"] = float(np.mean(-np.log2(posterior[np.arange(len(truth)), test["region"]])))
    return out


def execute(task):
    case, seed, arm, c_max, peak, budget, checkpoint_count = task
    trial = dict(case=case, seed=seed, arm=arm, c_max=c_max, noise_peak=peak, budget=budget)
    started = time.perf_counter()
    try:
        surface = ChargedIonutSurface(case, peak, seed)
        test = grading(surface)
        obs = ChargedObservations(surface, budget, surface.dim, seed, c_max=c_max)
        start = len(obs.costs)
        with threadpool_limits(limits=1):
            run_strategy(arm, obs, seed)
            design, noisy = obs.x, obs.y
            costs = np.array(obs.costs)
            spent = np.cumsum(costs)
            sunk = surface.competition(design) > SINK_AT
            rows = []
            for compute in compute_checkpoints(spent[start-1], budget, checkpoint_count)[1:]:
                n = int(np.searchsorted(spent, compute+1e-9, side="right"))
                rows.append(dict(**trial, compute=compute, n=n, status="ok",
                                 sink_share=float(costs[:n][sunk[:n]].sum()/spent[n-1]),
                                 contested_share=float(sunk[:n].mean()),
                                 **score_design(surface, design[:n], noisy[:n], test, seed)))
        rows[-1]["seconds"] = time.perf_counter()-started
        rows[-1]["x"] = design.tolist()
        rows[-1]["y_observed"] = noisy.tolist()
        rows[-1]["costs"] = costs.tolist()
        return rows
    except Exception as exc:
        return [dict(**trial, compute=0., n=0, status="failed", reason=f"{type(exc).__name__}: {exc}")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(DEFAULT_CASES))
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--costs", type=float, nargs="+", default=list(DEFAULT_COSTS),
                        help="c_max values: compute charged at a tied branch pair, per unit-cost run")
    parser.add_argument("--noise-peak", type=float, default=DEFAULT_PEAK)
    parser.add_argument("--budget", type=float, default=256.)
    parser.add_argument("--checkpoints", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="keep completed trajectories from a previous run")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    result_path = args.output/"results.json"
    identity = lambda r: (r["case"], r["seed"], r["arm"], r["c_max"])
    previous, done = [], set()
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))["rows"]
        done = {identity(r) for r in previous if r["status"] == "ok" and "seconds" in r}
        previous = [r for r in previous if identity(r) in done]
    elif result_path.exists():
        parser.error("output exists; pass --resume or choose a new --output")

    tasks = [(case, seed, arm, c_max, args.noise_peak, args.budget, args.checkpoints)
             for c_max, case, seed, arm in itertools.product(args.costs, args.cases, args.seeds, args.arms)
             if (case, seed, arm, c_max) not in done]
    # Unit cost places the most points and runs longest: start it first so
    # the slowest trajectories do not trail the pool.
    tasks.sort(key=lambda t: (t[3], t[1], t[0]))

    sources = sorted((ROOT/"src"/"resolution").glob("*.py")) + [
        ROOT/"tests/benchmarks/benchmarknd/noisy.py", ROOT/"src/strategies.py", ROOT/"src/ask_tell.py",
        ROOT/"scripts/experiments/run_sink_2d.py", ROOT/"scripts/analysis/vurs_confirm_score.py",
        pathlib.Path(__file__)]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not-a-checkout"
    payload = dict(config=vars(args) | {"output": str(args.output)},
                   protocol="budget in compute units, cost 1+(c_max-1)*branch competition; scores "
                            "at equal compute; common Matern-3/2 GP fitted to true (placement) or "
                            "noisy (end_to_end) values at the design; graded on the noiseless proxy",
                   sink_at=SINK_AT, grading_points=GRADING_POINTS, commit=commit,
                   source_hash=hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest(),
                   python=platform.python_version(), rows=list(previous))
    print(f"{len(tasks)} trajectories on {args.workers} workers ({len(done)} already complete)",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for finished, future in enumerate(as_completed(futures), 1):
            try:
                rows = future.result()
            except Exception as exc:
                print(f"  worker lost: {type(exc).__name__}: {exc}", flush=True)
                continue
            payload["rows"].extend(rows)
            result_path.write_text(json.dumps(payload, indent=1, allow_nan=False), encoding="utf-8")
            last = rows[-1]
            note = (f"n={last['n']} sink {last['sink_share']:.2f} placement {last['placement_error']:.4f} "
                    f"end_to_end {last['end_to_end_error']:.4f} {last['seconds']:.0f}s"
                    if last["status"] == "ok" else last["reason"])
            print(f"[{finished}/{len(tasks)}] {last['case']} s{last['seed']} {last['arm']} "
                  f"c={last['c_max']:g} {note}", flush=True)
    failures = [r for r in payload["rows"] if r["status"] != "ok"]
    print(f"saved {result_path} ({len(payload['rows'])} rows, {len(failures)} failures)")


if __name__ == "__main__":
    main()
