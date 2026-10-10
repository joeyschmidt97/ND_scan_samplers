import numpy as np
import pytest

from ND_scan_samplers import ROOT
from ND_scan_samplers.src.ask_tell import ResolutionSampler, CONVERGED, UNCONVERGED
from ND_scan_samplers.src.strategies import run_arm
from ND_scan_samplers.tests.benchmarks.benchmarknd.core import Observations
from ND_scan_samplers.tests.benchmarks.benchmarknd.pool import initial_pool_design, load_pool

UNIT_AXES = {"a": (0., 1.), "b": (0., 1.), "c": (0., 1.), "d": (0., 1.)}


@pytest.fixture(scope="module")
def pool():
    oracle, _ = load_pool(ROOT/"data/pools/hatch_pscans_global.csv")
    return oracle


def drive(sampler, oracle):
    """Run a sampler to its budget against a pool oracle; return the told order."""
    while not sampler.finished:
        batch = sampler.ask()
        if not batch:
            break
        sampler.tell(oracle(np.array([[p[n] for n in sampler.names] for p in batch])))
    return oracle.indices(sampler.x)


@pytest.mark.parametrize("arm,uncertainty,batch,mode", [
    ("vurs", True, 1, "greedy"), ("vwrs", False, 1, "greedy"),
    ("vurs-b5", True, 5, "greedy"), ("vurs-b5r", True, 5, "rank")])
def test_ask_tell_reproduces_the_benchmark_arm(pool, arm, uncertainty, batch, mode):
    seed, budget = 1, 34
    obs = Observations(pool, budget, pool.dim, seed, shared=initial_pool_design(pool, seed))
    run_arm(arm, obs, seed)
    sampler = ResolutionSampler(UNIT_AXES, uncertainty=uncertainty, batch=batch, batch_mode=mode,
                                candidates=pool.pool, budget=budget, seed=seed)
    assert drive(sampler, pool).tolist() == pool.indices(obs.x).tolist()


def test_failed_runs_never_steer_the_scan(pool):
    sampler = ResolutionSampler(UNIT_AXES, batch=4, candidates=pool.pool, seed=0)
    initial = sampler.ask()
    values = list(pool(np.array([[p[n] for n in sampler.names] for p in initial])))
    values[2] = None                                   # did not execute
    sampler.tell(values)
    assert len(sampler.failed_points) == 1 and len(sampler.x) == len(initial)-1
    lost = np.array([[sampler.failed_points[0][n] for n in sampler.names]])
    for _ in range(3):
        batch = sampler.ask()
        asked = np.array([[p[n] for n in sampler.names] for p in batch])
        assert not np.isclose(asked, lost).all(axis=1).any()
        sampler.tell(pool(asked))
    assert lost.tolist()[0] not in sampler.x.tolist()


def test_a_failed_initial_design_is_topped_up_before_fitting(pool):
    sampler = ResolutionSampler(UNIT_AXES, batch=4, candidates=pool.pool, seed=0)
    initial = sampler.ask()
    sampler.tell([None]*5 + list(pool(np.array([[p[n] for n in sampler.names] for p in initial[5:]]))))
    top_up = sampler.ask()
    assert len(top_up) == sampler.dim+2-len(sampler.x)
    sampler.tell(pool(np.array([[p[n] for n in sampler.names] for p in top_up])))
    assert len(sampler.ask()) == 4


def test_unconverged_values_are_kept_but_trusted_less(pool):
    told = {}
    for flag in (True, False):
        sampler = ResolutionSampler(UNIT_AXES, batch=4, candidates=pool.pool, seed=0)
        initial = sampler.ask()
        x = np.array([[p[n] for n in sampler.names] for p in initial])
        converged = [True]*len(x)
        converged[0] = flag
        sampler.tell(pool(x), converged=converged)
        told[flag] = (sampler, x)
    sampler, x = told[False]
    assert sampler.status[0] == UNCONVERGED and sampler.status[1] == CONVERGED
    exact = told[True][0].surrogate()(x[:1])[0]
    loose = sampler.surrogate()(x[:1])[0]
    assert np.isclose(exact, pool(x[:1])[0], rtol=1e-4) and not np.isclose(loose, exact, rtol=1e-6)
    assert len(sampler.ask()) == 4


def test_pending_points_count_as_placed(pool):
    sampler = ResolutionSampler(UNIT_AXES, batch=6, candidates=pool.pool, seed=0)
    initial = sampler.ask()
    sampler.tell(pool(np.array([[p[n] for n in sampler.names] for p in initial])))
    first, second = sampler.ask(), sampler.ask()
    rows = {tuple(np.round([p[n] for n in sampler.names], 12)) for p in first+second}
    assert len(rows) == 12 and len(sampler.pending) == 12


def test_state_round_trip_continues_identically(pool):
    sampler = ResolutionSampler(UNIT_AXES, batch=5, candidates=pool.pool, seed=3, budget=40)
    for _ in range(2):
        batch = sampler.ask()
        sampler.tell(pool(np.array([[p[n] for n in sampler.names] for p in batch])))
    sampler.ask()                                      # leave a batch pending
    clone = ResolutionSampler.from_state(sampler.state())
    assert clone.ask() == sampler.ask()


def test_physical_axes_and_log_axes_round_trip():
    axes = {"omt": (1., 4.), "ky": (.05, 2.)}
    sampler = ResolutionSampler(axes, batch=3, seed=0, log_axes=("ky",))
    initial = sampler.ask()
    assert len(initial) == 5
    for p in initial:
        assert 1. <= p["omt"] <= 4. and .05 <= p["ky"] <= 2.
    truth = lambda p: np.sin(p["omt"])+np.log(p["ky"])
    sampler.tell([truth(p) for p in initial])
    for p in sampler.ask():
        assert 1. <= p["omt"] <= 4. and .05 <= p["ky"] <= 2.
    assert np.allclose(sampler._to_unit(sampler._to_physical(sampler.x)), sampler.x)
