import numpy as np
import pytest

from ND_scan_samplers.src.label_model import normalized_entropy, pfn_posterior
from ND_scan_samplers.src.strategies import ARMS, PFN_ARMS, realized_bits


def test_normalized_entropy_bounds():
    p = np.array([[1., 0., 0.], [.5, .5, 0.], [1/3, 1/3, 1/3]])
    h = normalized_entropy(p)
    assert h[0] == pytest.approx(0.)
    assert h[2] == pytest.approx(1.)
    assert 0. < h[1] < 1.


def test_single_paid_mode_is_a_point_mass():
    post = pfn_posterior(np.zeros((3, 2)), [2, 2, 2], np.ones((4, 2)), 4)
    assert np.allclose(post[:, 2], 1.) and np.allclose(post.sum(axis=1), 1.)


def test_realized_bits_floors_unseen_labels():
    assert realized_bits(np.array([.5, .5]), 0) == pytest.approx(1.)
    assert realized_bits(np.array([.5, .5]), 3) == pytest.approx(-np.log2(1e-4))


def test_pfn_arms_stay_out_of_the_harness_registry():
    assert not set(PFN_ARMS) & set(ARMS)


def test_pfn_arm_runs_on_a_pool():
    pytest.importorskip("tabpfn")
    from ND_scan_samplers.src.strategies import run_arm
    from ND_scan_samplers.tests.benchmarks.benchmarknd.core import Observations
    from ND_scan_samplers.tests.benchmarks.benchmarknd.pool import PoolOracle
    rng = np.random.default_rng(0)
    x = rng.random((60, 2))
    labels = (x[:, 0] > .5).astype(int)
    oracle = PoolOracle(x, x.sum(axis=1), labels, ("a", "b"))
    for arm in PFN_ARMS:
        obs = Observations(oracle, 14, 2, 0, shared=x[[0, 1, 2, 3, 59]])
        _, metadata = run_arm(arm, obs, 0)
        assert len(obs.x) == 14 and len(np.unique(oracle.indices(obs.x))) == 14
        assert metadata["boundary_model"] == "pfn"
