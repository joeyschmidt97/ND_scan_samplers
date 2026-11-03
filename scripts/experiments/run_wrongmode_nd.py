"""6D wrong-mode sweep: what if a contested run locks onto the wrong branch?

The compute-sink sweep (`run_sink_nd`) gives a contested run a wide Gaussian
spread around the right answer. A real GENE run near a mode transition can fail
worse than that: the subdominant branch wins the race inside the simulated
time, the trace settles on it, and the run reports that branch's growth rate
as a clean, converged number. Averaging cannot catch it and the reported spread
does not warn of it. A sampler that is drawn to contested points will collect
more of these, so this is where attraction to contested regions could hurt.

Here each evaluation, with probability

    p_wrong(x) = flip_peak * s(x)          (s the branch-competition field)

returns the subdominant branch's growth rate instead of the observed value,
with only the floor spread and the subdominant branch as its reported label.
flip_peak = 0.5 makes a perfectly tied pair a coin flip. The flip draws come
from their own generator, so flip_peak = 0 reproduces `run_sink_nd` seed for
seed. Cost, budget, arms, grading and checkpoints are those of `run_sink_nd`.

What each arm sees of a wrong run: its value and floor spread (`vurs-prod`
trusts it), and its wrong label (`vurs-avoid` reads it). `vurs-flag` still flags
from the competition field, standing in for a campaign whose convergence check
does not depend on which branch the trace settled on.

    python -m ND_scan_samplers.scripts.experiments.run_wrongmode_nd --output ND_scan_samplers/results/wrongmode-<date>
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
from threadpoolctl import threadpool_limits

from ND_scan_samplers import ROOT
from ND_scan_samplers.scripts.experiments.run_sink_2d import (ARMS, SINK_AT, ChargedObservations,
                                                              compute_checkpoints, run_strategy)
from ND_scan_samplers.scripts.experiments.run_sink_nd import (DEFAULT_CASES, DEFAULT_PEAK,
                                                              ChargedIonutSurface, grading, score_design)
from ND_scan_samplers.src.resolution.noise import ABSOLUTE_FLOOR
from ND_scan_samplers.tests.benchmarks.benchmarknd.ionut import CASES

DEFAULT_COSTS = (1., 4.)
DEFAULT_FLIPS = (.25, .5)


class WrongModeIonutSurface(ChargedIonutSurface):
    """Charged proxy whose contested runs sometimes settle on the subdominant branch."""

    def __init__(self, case, peak=DEFAULT_PEAK, seed=0, flip_peak=0.):
        super().__init__(case, peak, seed)
        self.flip_peak = float(flip_peak)
        self._flip_rng = np.random.default_rng(20261102+seed)
        self.labels = {}           # reported branch label per paid point
        self.wrong = {}            # whether that run settled on the wrong branch

    def observe(self, x, replicates=1):
        x = np.atleast_2d(np.asarray(x, float))
        drawn, spread = super().observe(x, replicates)
        branches = self.truth.branches(x)
        rows = np.arange(len(x))
        # argmax, as truth.region labels: tied branches (both stable) keep the
        # first index, which argsort would not.
        leading = np.argmax(branches, axis=1)
        masked = branches.copy()
        masked[rows, leading] = -np.inf
        runner_up = np.argmax(masked, axis=1)
        second = branches[rows, runner_up]
        flip = self._flip_rng.random(len(x)) < self.flip_peak*self.competition(x)
        floor = self.noise.floor*np.abs(second) + ABSOLUTE_FLOOR
        wrong_value = second + floor*self._flip_rng.standard_normal(len(x))
        drawn = np.where(flip, wrong_value, drawn)
        spread = np.where(flip, floor, spread)
        for point, label, flipped in zip(x, np.where(flip, runner_up, leading), flip):
            key = ChargedObservations.key(point)
            self.labels[key], self.wrong[key] = int(label), bool(flipped)
        return drawn, spread


class ReportedLabels:
    """The branch each paid run reported, wrong or not, for label-reading arms."""

    def __init__(self, surface):
        self.surface = surface

    def region(self, x):
        return np.array([self.surface.labels[ChargedObservations.key(p)] for p in np.atleast_2d(x)])


def execute(task):
    case, seed, arm, c_max, flip_peak, peak, budget, checkpoint_count = task
    trial = dict(case=case, seed=seed, arm=arm, c_max=c_max, flip_peak=flip_peak,
                 noise_peak=peak, budget=budget)
    started = time.perf_counter()
    try:
        surface = WrongModeIonutSurface(case, peak, seed, flip_peak)
        test = grading(surface)
        obs = ChargedObservations(surface, budget, surface.dim, seed, c_max=c_max)
        obs._evaluate = ReportedLabels(surface)
        start = len(obs.costs)
        with threadpool_limits(limits=1):
            run_strategy(arm, obs, seed)
            design, noisy = obs.x, obs.y
            costs = np.array(obs.costs)
            spent = np.cumsum(costs)
            sunk = surface.competition(design) > SINK_AT
            wrong = np.array([surface.wrong[obs.key(p)] for p in design])
            rows = []
            for compute in compute_checkpoints(spent[start-1], budget, checkpoint_count)[1:]:
                n = int(np.searchsorted(spent, compute+1e-9, side="right"))
                rows.append(dict(**trial, compute=compute, n=n, status="ok",
                                 sink_share=float(costs[:n][sunk[:n]].sum()/spent[n-1]),
                                 contested_share=float(sunk[:n].mean()),
                                 wrong_share=float(wrong[:n].mean()),
                                 **score_design(surface, design[:n], noisy[:n], test, seed)))
        rows[-1]["seconds"] = time.perf_counter()-started
        rows[-1]["x"] = design.tolist()
        rows[-1]["y_observed"] = noisy.tolist()
        rows[-1]["costs"] = costs.tolist()
        rows[-1]["wrong"] = wrong.tolist()
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
    parser.add_argument("--flip-peaks", type=float, nargs="+", default=list(DEFAULT_FLIPS),
                        help="wrong-branch probability at a tied branch pair")
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
    identity = lambda r: (r["case"], r["seed"], r["arm"], r["c_max"], r["flip_peak"])
    previous, done = [], set()
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))["rows"]
        done = {identity(r) for r in previous if r["status"] == "ok" and "seconds" in r}
        previous = [r for r in previous if identity(r) in done]
    elif result_path.exists():
        parser.error("output exists; pass --resume or choose a new --output")

    tasks = [(case, seed, arm, c_max, flip, args.noise_peak, args.budget, args.checkpoints)
             for flip, c_max, case, seed, arm in itertools.product(args.flip_peaks, args.costs, args.cases,
                                                                   args.seeds, args.arms)
             if (case, seed, arm, c_max, flip) not in done]
    # Unit cost places the most points and runs longest: start it first so
    # the slowest trajectories do not trail the pool.
    tasks.sort(key=lambda t: (t[3], t[1], t[0]))

    sources = sorted((ROOT/"src"/"resolution").glob("*.py")) + [
        ROOT/"tests/benchmarks/benchmarknd/noisy.py", ROOT/"src/strategies.py", ROOT/"src/ask_tell.py",
        ROOT/"scripts/experiments/run_sink_2d.py", ROOT/"scripts/experiments/run_sink_nd.py",
        ROOT/"scripts/analysis/vurs_confirm_score.py", pathlib.Path(__file__)]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not-a-checkout"
    payload = dict(config=vars(args) | {"output": str(args.output)},
                   protocol="run_sink_nd with wrong-branch returns: probability flip_peak*competition, "
                            "subdominant growth rate at floor spread and its label; graded on the "
                            "noiseless dominant-branch proxy",
                   sink_at=SINK_AT, commit=commit,
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
            note = (f"n={last['n']} wrong {last['wrong_share']:.3f} sink {last['sink_share']:.2f} "
                    f"end_to_end {last['end_to_end_error']:.4f} {last['seconds']:.0f}s"
                    if last["status"] == "ok" else last["reason"])
            print(f"[{finished}/{len(tasks)}] {last['case']} s{last['seed']} {last['arm']} "
                  f"c={last['c_max']:g} p={last['flip_peak']:g} {note}", flush=True)
    failures = [r for r in payload["rows"] if r["status"] != "ok"]
    print(f"saved {result_path} ({len(payload['rows'])} rows, {len(failures)} failures)")


if __name__ == "__main__":
    main()
