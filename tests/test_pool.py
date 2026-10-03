"""Pool-mode contracts: arms choose only among pool points and pay each once."""
import csv

import numpy as np
import pytest

from benchmarknd.core import Observations
from benchmarknd.pool import (DEFAULT_POOL_ARMS, POOL_REFUSED, PoolOracle, initial_pool_design,
                              load_pool, replay, score_prefix, transition_band)


def grid_oracle(side=7):
    """A 3-D grid with a two-branch fold: the smallest pool every arm can run on."""
    axis = np.linspace(0, 1, side)
    x = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    labels = (x[:, 0]+.5*x[:, 1] > .7).astype(int)
    y = np.where(labels == 1, 1.5*x[:, 0]+x[:, 2], .2*x[:, 1]) + .3*np.sin(3*x[:, 2])
    return PoolOracle(x, y, labels, ("low", "high"))


def test_oracle_answers_at_pool_points_and_refuses_others():
    oracle = grid_oracle()
    assert np.allclose(oracle(oracle.pool[:5]), oracle.y[:5])
    assert list(oracle.region(oracle.pool[:5])) == list(oracle.labels[:5])
    with pytest.raises(KeyError):
        oracle(np.full((1, 3), .123))


def test_initial_design_is_distinct_pool_points():
    oracle = grid_oracle()
    start = initial_pool_design(oracle, seed=1)
    assert len(start) == 2*oracle.dim+1
    assert len(set(oracle.indices(start))) == len(start)


@pytest.mark.parametrize("arm", DEFAULT_POOL_ARMS)
def test_every_default_arm_pays_only_distinct_pool_points(arm):
    oracle = grid_oracle()
    rows = replay(oracle, arm, seed=0, budget=30, n_checkpoints=3)
    selected = rows[-1]["selected"]
    assert len(selected) == len(set(selected)) == 30
    assert rows[-1]["n"] == 30
    assert 0 <= rows[-1]["label_accuracy"] <= 1


@pytest.mark.parametrize("arm", POOL_REFUSED[:2])
def test_prescribed_node_arms_are_refused(arm):
    with pytest.raises(ValueError):
        replay(grid_oracle(), arm, seed=0, budget=30)


def test_scores_use_only_unpaid_points():
    oracle = grid_oracle()
    band = transition_band(oracle)
    assert band.any() and not band.all()
    everything_but_one = np.arange(len(oracle.pool)-1)
    out = score_prefix(oracle, everything_but_one, band)
    assert out["n"] == len(oracle.pool)-1
    assert out["label_accuracy"] in (0., 1.)


def test_load_pool_drops_sentinels_and_averages_repeats(tmp_path):
    path = tmp_path/"pool.csv"
    rows = [dict(factor_T=a, factor_n=b, x0=c, ky=k, gamma=g, mode_ID=m)
            for a, b, c, k, g, m in [
                (1., 1., .93, .1, .5, "KBM"), (1., 1., .93, .1, .7, "KBM"),
                (1.1, 1., .93, .1, -1, "MTM"), (1.1, .9, .95, .2, .2, "MTM"),
                (.9, 1.1, .97, .4, .1, ""), (.9, .9, .99, 1., .3, "ETG")]]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + ["omega"])
        writer.writeheader()
        writer.writerows(dict(r, omega=0.) for r in rows)
    oracle, meta = load_pool(path)
    assert meta["dropped"] == {"gamma_sentinel": 1, "unlabelled": 1}
    assert meta["repeats_averaged"] == 1
    assert meta["pool_size"] == 3
    assert np.isclose(oracle.y.max(), .6)
    assert (oracle.pool >= 0).all() and (oracle.pool <= 1).all()


def test_continuous_oracles_are_unchanged():
    """A non-pool oracle still gets Sobol candidates; Observations still works."""
    from benchmarknd.strategies import candidates_for, candidate_count
    obs = Observations(lambda x: x.sum(axis=1), 40, 3, seed=0)
    assert len(candidates_for(obs, np.random.default_rng(0))) == candidate_count(3)
