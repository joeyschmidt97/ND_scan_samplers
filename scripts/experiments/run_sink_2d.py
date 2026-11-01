"""2D compute-sink sweep: does VURS waste compute on contested points?

The 2D noise sweep (`run_noise_2d`) charges every evaluation one unit. A GENE
run near a mode transition does not cost one unit: two branches are tied, the
trace never settles, and the run burns to its time limit and still comes back
with a wide spread. A sampler drawn to contested regions therefore pays more
per point there, and the unit-cost benchmark cannot see it.

Here each evaluation costs

    cost(x) = 1 + (c_max - 1) * s(x),   s(x) = exp(-|d_fold(x)| / FOLD_SCALE)

with s the same competition field that sets the spread, and the budget is in
compute units. c_max is swept rather than chosen: c_max = 1 is the unit-cost
control and must reproduce `run_noise_2d` at the same noise peak, seed for
seed. The sampler never sees the cost before paying it.

Arms beyond the benchmark registry drive the production ask/tell sampler
(`src.ask_tell.ResolutionSampler`), which is what a GENE campaign deploys and
which, unlike the benchmark arms, gives noisy points a heteroscedastic GP:

    vurs-prod   tell() with the measured spread: GP alpha (sigma/std y)^2,
                denoised variation
    vurs-flag   tell() with converged=False where s > UNCONVERGED_AT and no
                spread: GP alpha = unconverged_noise, the flag-only path

One arm is cost-aware: `vurs-cost` is plain VURS with each candidate's merit
divided by its predicted cost, the realized cost of the nearest paid run. A
campaign knows a run's wall time once it has paid for it and nothing before,
so this transfers cost the way `candidate_noise` transfers spread. At c_max = 1
every predicted cost is one and the arm must reproduce `vurs` exactly.

Scores are taken at equal compute spent, not equal point count.

    python -m ND_scan_samplers.scripts.experiments.run_sink_2d --output results/sink-2d-<date>
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
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from ND_scan_samplers import ROOT
from ND_scan_samplers.scripts.experiments.run_noise_2d import (CORNERS, Noisy2DSurface, score_design,
                                                               truth_curvature)
from ND_scan_samplers.src.ask_tell import ResolutionSampler
from ND_scan_samplers.src.strategies import (Acquisition, candidates_for, mask_duplicates,
                                             resolution_weights, run_arm)
from ND_scan_samplers.tests.benchmarks.benchmark2d.core import evaluation_set
from ND_scan_samplers.tests.benchmarks.benchmarknd.noisy import NoisyObservations

DEFAULT_COSTS = (1., 2., 4., 8.)
DEFAULT_PEAK = .20               # the completed block of the unit-cost sweep
SINK_AT = .5                     # competition above which compute counts as sunk
UNCONVERGED_AT = .5              # competition above which vurs-flag reports converged=False
ARMS = ("space-filling", "vwrs", "vurs", "vurs-n", "vurs-a", "vurs-avoid",
        "vurs-prod", "vurs-flag", "vurs-cost")
ASK_TELL_ARMS = ("vurs-prod", "vurs-flag")


class ChargedObservations(NoisyObservations):
    """Noisy observations whose budget is compute, charged by competition.

    The last evaluation may overrun the budget by less than one evaluation's
    cost: a campaign does not cancel a run already submitted. Every arm gets
    the same rule, and scores are read at compute checkpoints at or below the
    budget, so the overrun never enters a score.
    """

    def __init__(self, surface, budget, dim, seed=0, shared=None, c_max=1.):
        self.c_max = float(c_max)
        self.costs = []            # compute charged per evaluation, in call order
        # observed_labels() reads branch labels through `_evaluate.region`;
        # the vurs-avoid profile needs them, the oracle knows them.
        self._evaluate = surface.truth
        super().__init__(surface, budget, dim, seed, shared)

    def cost(self, x):
        return 1.+(self.c_max-1.)*self.surface.competition(np.atleast_2d(x))

    @property
    def remaining(self):
        return max(0., self.budget-self.spent)

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        if x.shape[1] != self.dim or not np.isfinite(x).all() or (x < 0).any() or (x > 1).any():
            raise ValueError("finite unit-box coordinates of the right dimension required")
        if self.costs and not self.remaining:
            raise RuntimeError("compute budget already spent")
        drawn, single = self.surface.observe(x)
        for point, value, spread, charge in zip(x, drawn, np.atleast_1d(single), self.cost(x)):
            key = self.key(point)
            if key in self._sum:
                raise RuntimeError("the sink sweep admits no replicates")
            self._order.append(np.asarray(point, float))
            self._sum[key], self._count[key] = float(value), 1
            self._single[key] = float(spread)
            self.costs.append(float(charge))
            self.spent += float(charge)
        self.requests += len(x)
        return np.array([self._sum[self.key(p)] for p in x])


def cost_aware_vurs(obs, seed):
    """Plain VURS with merit divided by the nearest paid run's realized cost.

    Mirrors the sequential `resolution_sampling` loop (same candidate stream,
    acquisition and duplicate mask) so that unit costs reproduce `vurs`.
    """
    rng = np.random.default_rng(seed)
    weights = resolution_weights(uncertainty=True)
    while obs.remaining:
        candidates = candidates_for(obs, rng)
        acquisition = Acquisition(obs.x, obs.y, candidates, seed, weights, uncertainty=True)
        predicted = np.asarray(obs.costs)[cKDTree(obs.x).query(candidates)[1]]
        merit = mask_duplicates(acquisition.merit()/predicted, acquisition.spacing, obs)
        obs(candidates[int(np.argmax(merit))])


def run_strategy(arm, obs, seed):
    if arm in ASK_TELL_ARMS:
        run_ask_tell(arm, obs, seed)
    elif arm == "vurs-cost":
        cost_aware_vurs(obs, seed)
    else:
        run_arm(arm, obs, seed)


def run_ask_tell(arm, obs, seed):
    """Drive the production sampler one point at a time against the oracle."""
    names = [f"x{j}" for j in range(obs.dim)]
    sampler = ResolutionSampler({n: (0., 1.) for n in names}, uncertainty=True,
                                batch=1, seed=seed)
    competition = obs.surface.competition

    def report(points, values):
        if arm == "vurs-prod":
            return dict(noise=[obs._single[obs.key(p)] for p in points])
        return dict(converged=list(competition(points) <= UNCONVERGED_AT))

    sampler.observe(obs.x, obs.y, **report(obs.x, obs.y))
    while obs.remaining:
        asked = sampler.ask(1)
        if not asked:
            raise RuntimeError("ask() returned nothing before the budget was spent")
        point = np.array([[asked[0][n] for n in names]])
        value = obs(point)
        sampler.tell(list(value), **report(point, value))
    return sampler


def compute_checkpoints(start, budget, count=8):
    return [float(c) for c in np.unique(np.round(np.geomspace(start, budget, count), 6))]


def execute(task):
    case, seed, arm, c_max, peak, budget, checkpoint_count = task
    trial = dict(case=case, seed=seed, arm=arm, c_max=c_max, noise_peak=peak, budget=budget)
    started = time.perf_counter()
    try:
        surface = Noisy2DSurface(case, seed, peak=peak)
        test = evaluation_set(surface.truth)
        test = (*test, truth_curvature(surface.truth, test[0]))
        obs = ChargedObservations(surface, budget, 2, seed, shared=CORNERS, c_max=c_max)
        with threadpool_limits(limits=1):
            run_strategy(arm, obs, seed)
        design, noisy = obs.x, obs.y
        costs = np.array(obs.costs)
        spent = np.cumsum(costs)
        competition = surface.competition(design)
        sunk = competition > SINK_AT
        rows = []
        for compute in compute_checkpoints(spent[len(CORNERS)-1], budget, checkpoint_count):
            n = int(np.searchsorted(spent, compute+1e-9, side="right"))
            rows.append(dict(**trial, compute=compute, n=n, status="ok",
                             sink_share=float(costs[:n][sunk[:n]].sum()/spent[n-1]),
                             contested_share=float(sunk[:n].mean()),
                             **score_design(surface, design[:n], noisy[:n], test)))
        rows[-1]["seconds"] = time.perf_counter()-started
        rows[-1]["x"] = design.tolist()
        rows[-1]["y_observed"] = noisy.tolist()
        rows[-1]["costs"] = costs.tolist()
        return rows
    except Exception as exc:
        return [dict(**trial, compute=0., n=0, status="failed", reason=f"{type(exc).__name__}: {exc}")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+",
                        default=["two-plane-four-peaks", "three-plane-three-peaks",
                                 "two-plane-asymmetric"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--costs", type=float, nargs="+", default=list(DEFAULT_COSTS),
                        help="c_max values: compute charged at a tied fold, per unit-cost run")
    parser.add_argument("--noise-peak", type=float, default=DEFAULT_PEAK)
    parser.add_argument("--budget", type=float, default=256.)
    parser.add_argument("--checkpoints", type=int, default=8)
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
    # Highest cost first: those trajectories place the fewest points and finish
    # soonest, and they are the regime the sweep exists to test.
    tasks.sort(key=lambda t: (-t[3], t[1], t[0]))

    sources = sorted((ROOT/"src"/"resolution").glob("*.py")) + [
        ROOT/"tests/benchmarks/benchmarknd/noisy.py", ROOT/"src/strategies.py", ROOT/"src/ask_tell.py",
        ROOT/"scripts/experiments/run_noise_2d.py", pathlib.Path(__file__)]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not-a-checkout"
    payload = dict(config=vars(args) | {"output": str(args.output)},
                   protocol="budget in compute units, cost 1+(c_max-1)*competition; scores "
                            "at equal compute; placement from true values at the design, "
                            "end_to_end from the noisy ones; both on the noiseless surface",
                   sink_at=SINK_AT, unconverged_at=UNCONVERGED_AT, commit=commit,
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
                    f"end_to_end {last['end_to_end_error']:.4f}"
                    if last["status"] == "ok" else last["reason"])
            print(f"[{finished}/{len(tasks)}] {last['case']} s{last['seed']} {last['arm']} "
                  f"c={last['c_max']:g} {note}", flush=True)
    failures = [r for r in payload["rows"] if r["status"] != "ok"]
    print(f"saved {result_path} ({len(payload['rows'])} rows, {len(failures)} failures)")


if __name__ == "__main__":
    main()
