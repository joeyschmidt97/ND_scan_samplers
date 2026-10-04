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


def test_per_mode_scores_use_each_modes_own_scale():
    oracle = grid_oracle()
    band = transition_band(oracle)
    out = score_prefix(oracle, np.arange(0, len(oracle.pool), 3), band)
    assert set(out["mode_nrmse"]) == {"low", "high"}
    live = [v for v in out["mode_nrmse"].values() if v is not None]
    assert np.isclose(out["macro_nrmse"], np.mean(live))
    assert out["worst_mode_nrmse"] == max(live)


def test_classify_then_regress_separates_a_small_branch_from_a_large_one():
    """A flat small branch beside a steep large one: per-mode fits recover it."""
    axis = np.linspace(0, 1, 7)
    x = np.array(np.meshgrid(axis, axis, axis, indexing="ij")).reshape(3, -1).T
    labels = (x[:, 0] > .5).astype(int)
    y = np.where(labels == 1, 10+5*x[:, 1], .01*x[:, 2])
    oracle = PoolOracle(x, y, labels, ("small", "large"))
    out = score_prefix(oracle, np.arange(0, len(x), 2), transition_band(oracle))
    # With true labels the small branch is recovered exactly; the global
    # interpolant smears the large branch into it.
    assert out["ctr_oracle_mode_nrmse"]["small"] < 1e-6 < out["mode_nrmse"]["small"]
    assert out["ctr_fallback_points"] == 0
    # With predicted labels a misclassified point beside the fold takes the
    # other branch's value, so the predicted-label score is finite but can be
    # worse than the global one: classification error is not hidden.
    assert np.isfinite(out["ctr_macro_nrmse"])


def test_rescore_reproduces_pooled_scores_and_adds_per_mode(tmp_path):
    import json
    from benchmarknd.pool import rescore
    oracle = grid_oracle(side=6)
    names = oracle.label_names
    path = tmp_path/"pool.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["factor_T", "factor_n", "x0", "ky", "gamma", "omega", "mode_ID"])
        for p, y, k in zip(oracle.pool, oracle.y, oracle.labels):
            writer.writerow([1+p[0], 1+p[1], .9+.1*p[2], .1+p[2], y, 0., names[k]])
    loaded, meta = load_pool(path)
    rows = replay(loaded, "vwrs", seed=0, budget=25, n_checkpoints=3)
    for row in rows:
        for key in ("mode_nrmse", "mode_nmae", "macro_nrmse", "worst_mode_nrmse"):
            row.pop(key)
    results = tmp_path/"results.json"
    results.write_text(json.dumps(dict(pool=meta, rows=rows)))
    again = json.loads(rescore(results).read_text())["rows"]
    assert len(again) == len(rows)
    for old, new in zip(rows, again):
        assert np.isclose(old["nrmse"], new["nrmse"])
        assert new["macro_nrmse"] is not None


def test_truth_free_scores_need_only_paid_runs():
    """Same answer whatever the unpaid truth is: no reference values are read."""
    from benchmarknd.pool import truth_free_scores
    oracle = grid_oracle()
    paid = np.arange(0, len(oracle.pool), 4)
    a = truth_free_scores(oracle.pool[paid], oracle.y[paid], oracle.labels[paid], oracle.pool)
    scrambled = oracle.y.copy()
    scrambled[np.setdiff1d(np.arange(len(scrambled)), paid)] = 99.
    b = truth_free_scores(oracle.pool[paid], scrambled[paid], oracle.labels[paid], oracle.pool)
    assert a == b
    assert set(a) == {"tf_fill_p95", "tf_observed_vwfd_p95", "tf_paid_boundary",
                      "tf_cv_nrmse", "tf_loo_label"}


def test_region_scores_cover_the_three_jobs():
    from benchmarknd.pool import exploration_regions
    oracle = grid_oracle()
    regions = exploration_regions(oracle)
    assert regions["design_band"].any() and regions["peak"].any() and regions["quiet"].any()
    assert not (regions["quiet"] & regions["peak"]).any()
    rows = replay(oracle, "space-filling", seed=0, budget=40, n_checkpoints=2)
    last = rows[-1]
    assert last["modes_found"] == 2
    assert all(1 <= v <= 40 for v in last["first_hit"].values())
    assert 0 <= last["transition_budget_share"] <= 1
    assert 0 <= last["transition_detection"] <= 1
    assert 0 < last["peak_best_ratio"] <= 1


def test_continuous_oracles_are_unchanged():
    """A non-pool oracle still gets Sobol candidates; Observations still works."""
    from benchmarknd.strategies import candidates_for, candidate_count
    obs = Observations(lambda x: x.sum(axis=1), 40, 3, seed=0)
    assert len(candidates_for(obs, np.random.default_rng(0))) == candidate_count(3)
