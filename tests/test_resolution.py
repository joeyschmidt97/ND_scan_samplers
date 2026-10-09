"""The dimension-free variation estimator and fit-free resolution spine.

Every behavioural test runs at 2, 5 and 8 dimensions with the same assertion.
That is the property under test: the spine must not change meaning with the
input dimension, which is what the Delaunay estimator could not promise.
"""
import numpy as np
import pytest

from ND_scan_samplers.benchmarknd.core import Observations, SurfaceND, evaluation_set, score, truth_variation
from ND_scan_samplers.src.strategies import resolution_sampling, run_arm
from ND_scan_samplers.src.resolution import fit_free_scores, knn_variation, metric_fill, region_shapes, spine_targets
from scipy.spatial import cKDTree

from ND_scan_samplers.src.resolution.variation import FOLD_RATIO, fold_floor, shrinkage, stencil_size

DIMS = (2, 5, 8)


def design(dim, n=400, seed=0):
    return np.random.default_rng(seed).random((n, dim))


@pytest.mark.parametrize("dim", DIMS)
def test_plane_slope_is_recovered_and_its_curvature_is_zero(dim):
    """A steep plane is reconstructed exactly, so it must not attract budget."""
    x = design(dim)
    y = 3.*x[:, 0]
    out = knn_variation(x, y, design(dim, 200, seed=1))
    assert np.allclose(out.gradient, 3., atol=.05)
    assert out.curvature.max() < 1e-3


@pytest.mark.parametrize("dim", DIMS)
def test_curvature_separates_a_narrow_peak_from_a_steep_plane(dim):
    """The ordering slope alone gets wrong: modest peak beats steep plane."""
    x = design(dim)
    query = design(dim, 200, seed=1)
    plane = knn_variation(x, 3.*x[:, 0], query)
    peak = knn_variation(x, np.exp(-.5*np.sum(((x-.5)/.1)**2, axis=1)), query)
    assert plane.gradient.mean() > peak.gradient.mean()
    assert plane.curvature.mean() < peak.curvature.mean()


@pytest.mark.parametrize("dim", DIMS)
def test_a_jump_raises_the_residual_and_branch_restriction_removes_it(dim):
    x = design(dim, 600)
    labels = (x[:, 0] > .5).astype(int)
    y = x[:, 1] + 2.*labels                       # unit-scale jump across x0 = 0.5
    near = x[np.abs(x[:, 0]-.5) < .08][:50]
    spanning = knn_variation(x, y, near)
    restricted = knn_variation(x, y, near, labels=labels)
    assert spanning.residual.mean() > 5*restricted.residual.mean()


@pytest.mark.parametrize("dim", DIMS)
def test_indicator_modes_have_slope_units_and_rank_differently(dim):
    x = design(dim)
    query = design(dim, 100, seed=1)
    out = knn_variation(x, np.exp(-.5*np.sum(((x-.5)/.1)**2, axis=1)), query)
    spacing = np.full(len(query), .1)
    curvature = out.indicator(spacing, "curvature")
    gradient = out.indicator(spacing, "gradient")
    blend = out.indicator(spacing, "blend")
    assert np.allclose(blend, gradient+curvature)
    with pytest.raises(ValueError):
        out.indicator(spacing, "nonsense")


def test_stencil_must_leave_a_residual():
    assert stencil_size(5) >= 7
    with pytest.raises(ValueError):
        stencil_size(5, k=6)


@pytest.mark.parametrize("dim", DIMS)
def test_too_few_observations_is_refused_not_guessed(dim):
    with pytest.raises(ValueError):
        knn_variation(design(dim, dim+1), np.zeros(dim+1), design(dim, 3))


@pytest.mark.parametrize("dim", DIMS)
def test_refining_the_design_lowers_every_spine_quantile(dim):
    probes = design(dim, 500, seed=2)
    variation = np.ones(len(probes))
    coarse = fit_free_scores(design(dim, 60, seed=3), probes, variation, 1.)
    fine = fit_free_scores(design(dim, 600, seed=3), probes, variation, 1.)
    for key in ("fill_p95", "fill_max", "vwfd_p95", "vwfd_max", "vwfd_rms"):
        assert fine[key] < coarse[key], key
    for tau in spine_targets():
        key = f"vwfd_coverage_{int(round(tau*100)):02d}"
        assert 0. <= coarse[key] <= 1. and fine[key] >= coarse[key]


@pytest.mark.parametrize("dim", DIMS)
def test_spine_reports_the_worst_axis_and_refuses_bad_input(dim):
    probes = design(dim, 200, seed=2)
    points = design(dim, 100, seed=3)
    points[:, 1] *= .01                            # collapse one axis
    out = fit_free_scores(points, probes, np.ones(len(probes)), 1.)
    assert out["axis_spread_worst_index"] == 1
    assert out["axis_spread_min"] < out["axis_spread_mean"]
    with pytest.raises(ValueError):
        fit_free_scores(points, probes, np.ones(len(probes)), 0.)
    with pytest.raises(ValueError):
        fit_free_scores(points, probes, np.ones(len(probes)+1), 1.)


def test_truth_variation_is_exact_on_a_known_surface():
    surface = SurfaceND("5d-m1-p1-anis", 0)
    x = design(5, 64, seed=4)
    gradient, curvature = truth_variation(surface, x)
    assert gradient.shape == (64,) and np.isfinite(gradient).all()
    assert curvature.shape == (64,) and np.isfinite(curvature).all()
    assert (gradient > 0).all()


@pytest.mark.parametrize("case", ("5d-m2-rotated", "8d-m2-disjoint"))
def test_resolution_sampling_spends_the_exact_budget_and_declares_its_weights(case):
    surface = SurfaceND(case, 0)
    for arm, uncertainty in (("vwrs", 0.), ("vurs", 1/3)):
        obs = Observations(surface, 2*surface.dim+8, surface.dim, 0)
        predict, metadata = run_arm(arm, obs, 0)
        assert len(obs.x) == obs.budget
        assert metadata["acquisition_weights"]["uncertainty"] == pytest.approx(uncertainty)
        assert sum(metadata["acquisition_weights"].values()) == pytest.approx(1.)
        assert (predict is None) == (arm == "vwrs")


def top_direction(shapes):
    return np.linalg.eigh(shapes)[1][..., -1]


@pytest.mark.parametrize("dim", DIMS)
def test_a_fold_gets_a_rank_one_shape_along_its_normal(dim):
    """At a max of two planes the fold normal is the mean-gradient difference."""
    rng = np.random.default_rng(6)
    first, second = rng.normal(size=dim), rng.normal(size=dim)
    x = design(dim, 800)
    values = np.stack([(x - .5) @ first, (x - .5) @ second], axis=1)   # fold through the centre
    labels, y = values.argmax(axis=1), values.max(axis=1)
    normal = (first - second)/np.linalg.norm(first - second)
    probe = design(dim, 20000, seed=1)
    gap = (probe - .5) @ normal
    near = probe[np.abs(gap) < .02][:40]
    shapes = region_shapes(x, y, labels, near)
    assert np.allclose(np.trace(shapes, axis1=1, axis2=2), dim)
    alignment = np.abs(top_direction(shapes) @ normal)
    assert np.median(alignment) > .95
    tangent = np.linalg.svd(normal[None, :])[2][-1]          # any unit vector orthogonal to the normal
    across = np.einsum("i,mij,j->m", normal, shapes, normal)
    along = np.einsum("i,mij,j->m", tangent, shapes, tangent)
    # Fixed anisotropy: across/along = FOLD_RATIO at every dimension.
    a = fold_floor(dim)
    assert np.median(along) == pytest.approx(a, rel=.05)
    assert np.median(across)/np.median(along) == pytest.approx(FOLD_RATIO, rel=.05)
    sharp = region_shapes(x, y, labels, near, fold_ratio=np.inf)
    sharp_along = np.einsum("i,mij,j->m", tangent, sharp, tangent)
    assert np.median(sharp_along) < .1          # only the small-sample shrinkage (d+1)/n remains


def test_fold_floor_keeps_the_2d_value_and_grows_with_dimension():
    assert fold_floor(2) == pytest.approx(.25)          # unchanged from 8050a3f in 2D
    assert fold_floor(6) == pytest.approx(.5)
    assert fold_floor(8) > fold_floor(6) > fold_floor(2)


@pytest.mark.parametrize("dim", DIMS)
def test_centering_ignores_a_steep_slope_and_finds_the_peak_axis(dim):
    """Raw gradients point along the slope; the bending is along axis 0."""
    x = design(dim, 800)
    width = np.full(dim, .6); width[0] = .08
    y = 3.*x[:, 1] + np.exp(-.5*np.sum(((x - .5)/width)**2, axis=1))
    near = .5 + design(dim, 40, seed=1)*.1 - .05
    shapes = region_shapes(x, y, np.zeros(len(x), int), near)
    assert np.median(np.abs(top_direction(shapes)[:, 0])) > .9


@pytest.mark.parametrize("dim", DIMS)
def test_without_labels_the_metric_is_exactly_plain_vwrs(dim):
    x = design(dim, 200)
    y = np.exp(-.5*np.sum(((x - .5)/.2)**2, axis=1))
    query = design(dim, 30, seed=1)
    spacing = cKDTree(x).query(query)[0]
    curvature = knn_variation(x, y, query).curvature
    assert np.allclose(metric_fill(x, y, query), spacing**2*curvature, rtol=1e-9)


@pytest.mark.parametrize("dim", DIMS)
def test_small_regions_fall_back_to_the_isotropic_shape(dim):
    assert shrinkage(1, dim) == 1. and shrinkage(10*(dim + 1), dim) == pytest.approx(.1)
    x = design(dim, 400)
    y = x[:, 0]**2
    labels = np.zeros(len(x), int)
    labels[:dim + 2] = 1                                      # too few to fit a shape
    query = x[:dim + 2] + 1e-6
    shapes = region_shapes(x, y, labels, query, k=dim + 2)
    lonely = [i for i, row in enumerate(cKDTree(x).query(query, k=dim + 2)[1])
              if (labels[row] == 1).all()]
    for i in lonely:
        assert np.allclose(shapes[i], np.eye(dim))


@pytest.mark.parametrize("case", ("5d-m2-rotated", "8d-m2-disjoint"))
def test_anisotropic_arms_spend_the_exact_budget_and_declare_the_metric(case):
    surface = SurfaceND(case, 0)
    for arm in ("vwrs-m", "vurs-m"):
        obs = Observations(surface, 2*surface.dim+8, surface.dim, 0)
        predict, metadata = run_arm(arm, obs, 0)
        assert len(obs.x) == obs.budget
        assert metadata["anisotropic"] is True
        assert (predict is None) == (arm == "vwrs-m")


def test_anisotropic_arms_refuse_noise_flags_and_unlabelled_oracles():
    surface = SurfaceND("5d-m2-rotated", 0)
    with pytest.raises(ValueError):
        resolution_sampling(Observations(surface, 20, 5, 0), 0, anisotropic=True, replicate=True)
    unlabelled = Observations(lambda x: surface(x), 20, 5, 0)
    with pytest.raises(ValueError):
        run_arm("vwrs-m", unlabelled, 0)


def test_holistic_score_is_not_silently_inherited_across_dimension():
    # A folded case: the fold-band term of H only exists where a fold does.
    surface = SurfaceND("5d-m2-rotated", 0)
    test = evaluation_set(surface, 4096)
    obs = Observations(surface, 32, 5, 0)
    while obs.remaining:
        obs(np.random.default_rng(obs.remaining).random((1, 5)))

    uncalibrated = score(surface, obs, test)
    assert uncalibrated["holistic_uncalibrated"] is True
    assert uncalibrated["holistic_error"] is None

    targets = dict(nmae=.05, band_nmae=.10, p95_error=.15, vwfd_p95=.25)
    calibrated = score(surface, obs, test, targets=targets)
    assert calibrated["holistic_uncalibrated"] is False
    assert calibrated["holistic_driver"] in targets
    assert calibrated["holistic_error"] == pytest.approx(
        max(calibrated[name]/limit for name, limit in targets.items()))
    with pytest.raises(ValueError):
        score(surface, obs, test, targets=dict(does_not_exist=.1))


def test_a_fold_free_control_refuses_a_fold_band_tolerance():
    """The one-mode controls have no fold, so H cannot carry a fold term."""
    surface = SurfaceND("5d-m1-p1-anis", 0)
    test = evaluation_set(surface, 4096)
    obs = Observations(surface, 32, 5, 0)
    while obs.remaining:
        obs(np.random.default_rng(obs.remaining).random((1, 5)))
    assert score(surface, obs, test)["band_nmae"] is None
    with pytest.raises(ValueError):
        score(surface, obs, test, targets=dict(nmae=.05, band_nmae=.10))


def test_the_rbf_evaluator_is_labelled_secondary_in_every_row():
    surface = SurfaceND("5d-m1-p1-anis", 0)
    test = evaluation_set(surface, 4096)
    obs = Observations(surface, 24, 5, 0)
    while obs.remaining:
        obs(np.random.default_rng(100+obs.remaining).random((1, 5)))
    row = score(surface, obs, test)
    assert row["reconstruction_role"] == "secondary"
    assert row["reconstruction"] == "thin-plate-spline-rbf"


def test_declared_tolerances_exist_for_five_and_refuse_eight():
    from ND_scan_samplers.benchmarknd.core import tolerances_for
    spine, ported = tolerances_for(5)
    assert set(spine) == {"vwfd_p95", "nonlinear_p95", "fill_p95"}
    assert set(ported) == {"nmae", "band_nmae", "p95_error", "vwfd_p95"}
    assert tolerances_for(5, band_available=False)[1].keys() == {"nmae", "p95_error", "vwfd_p95"}
    # 8D has not been calibrated; inheriting the 5D limits would be the bug.
    with pytest.raises(ValueError, match="no calibrated tolerances"):
        tolerances_for(8)


def test_spine_holistic_is_primary_and_ported_rides_along_labelled():
    from ND_scan_samplers.benchmarknd.core import tolerances_for
    surface = SurfaceND("5d-m2-rotated", 0)
    test = evaluation_set(surface, 4096)
    obs = Observations(surface, 40, 5, 0)
    while obs.remaining:
        obs(np.random.default_rng(obs.remaining).random((1, 5)))
    spine, ported = tolerances_for(5)
    row = score(surface, obs, test, targets=spine, secondary_targets=ported)
    assert row["holistic_uncalibrated"] is False
    assert row["holistic_driver"] in spine
    assert row["ported_holistic_uncalibrated"] is False
    assert row["ported_holistic_driver"] in ported
    # The primary score must not contain a reconstruction-based term.
    assert not {"nmae", "band_nmae", "p95_error"} & set(row["holistic_targets"])
