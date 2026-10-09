"""Ionut's native 6D microinstability proxies as a high-dimensional surface.

The 3D work used declared conditional slices of these formulas. This runs them
in their own dimension, with no coordinate held fixed, so the sampler faces the
full parameter space the proxy was written over:

    RLTi, RLTe, RLn, nu, beta   (scripts/ionut_proxies._RANGES)
    ky scale                    (the sixth axis, 0.30*(0.5 + x5))

Available branch pairs are ITG-TEM and ITG-KBM. There is no ETG or MTM branch
in the upstream formulas, so no ETG case can be run here; `branch_ITG`,
`branch_TEM` and `branch_KBM` are the whole set.

These are phenomenological proxies, not gyrokinetic solves. The transition mask
is derived from the stored branch growth rates rather than from the hard/soft
selection weights, so argmax and softmax variants of one pair share a physical
mask and stay comparable.
"""
import numpy as np
from scipy.stats import qmc

from ND_scan_samplers.scripts.generate_ionut_data import CASES as NATIVE_CASES, values

# The blend case carries no branch decomposition, so it has no transition mask
# and no per-branch error; excluded from the matched field rather than scored
# with a silently different metric set.
CASES = tuple(case for case in NATIVE_CASES if case != "ionut-stellarator-itg-kbm")
TRANSITION_FRACTION = .05       # branch-gap width counted as "at the transition"

# Isolated-bump variants: the same proxy plus two anisotropic Gaussian bumps on
# the reported output, placed where one branch clearly dominates, is growing,
# and |omega| is large. They mimic a mode that is neither weak nor competing.
# Bumps are added to y only, never to the branch growth rates, so the region
# labels and the live-competition mask are those of the unbumped proxy.
BUMPED = "-bumped"
BUMPED_CASES = tuple(case + BUMPED for case in CASES)
BUMP_NARROW, BUMP_WIDE = .08, .20   # sigma on the two narrow axes, and the rest
BUMP_AMPLITUDE = .35                # fraction of the unbumped output range
_BUMPS = {}


def _admissible_center(probe, eligible, region, want, sigma, live, omegas, bumps, rng):
    """First admissible center for one bump shape, preferring region `want`."""
    for restrict in (True, False):
        pool = np.flatnonzero(eligible & ((region == want) if restrict else True))
        for index in rng.permutation(pool):
            center = probe[index]
            if any(np.linalg.norm(center-b["center"]) < .5 for b in bumps):
                continue
            if len(live) and np.min(np.sum(((live-center)/sigma)**2, axis=1)) < 9.:
                continue
            ball = np.sum(((probe-center)/sigma)**2, axis=1) < 4.
            sign = np.sign(omegas[0][index])
            if any((np.sign(w[ball]) != sign).any() for w in omegas):
                continue
            return int(index)
    return None


def isolated_bumps(case):
    """Declared bump geometry for one branch pair, shared by all four outputs.

    Centers come from a fixed Sobol cloud, so the gamma/omega and argmax/softmax
    cases of a pair carry identical bumps, as one GENE run reports both
    outputs. A center is admissible when it sits 0.15 inside the box, its
    dominant branch is growing (above 30% of the branch range), |omega| is above
    its median, every live-competition point is at least three sigma away in
    the bump's own anisotropic frame (so the bump is below 1.1% of its height
    in the transition band), and omega keeps one sign, under both argmax and
    softmax selection, everywhere within two sigma (so a frequency bump never
    pushes omega through zero). One center per dominant region where possible,
    each narrow on a different pair of axes: the first pair, in a fixed order,
    for which an admissible center exists. The order matters on ITG-KBM, where
    a bump wide along beta reaches the KBM onset from almost anywhere.
    """
    kind = case.split("-")[1] + "-" + case.split("-")[2]
    if kind in _BUMPS:
        return _BUMPS[kind]
    probe = qmc.Sobol(6, scramble=True, seed=4243).random_base2(14)
    data = values(f"ionut-{kind}-argmax-gamma", probe)
    g = data["G"]
    omegas = [values(f"ionut-{kind}-{mode}-omega", probe)["y"] for mode in ("argmax", "softmax")]
    scale = IonutSurface(f"ionut-{kind}-argmax-gamma").branch_scale()
    # The ITG-KBM live band is ~1% of the domain, so the clearance test uses a
    # denser cloud than the center search; with 2^14 points it missed enough of
    # the band to leave a 5% bump inside it.
    dense = qmc.Sobol(6, scramble=True, seed=4244).random_base2(17)
    dense_g = values(f"ionut-{kind}-argmax-gamma", dense)["G"]
    gap = np.abs(dense_g[:, 0]-dense_g[:, 1])/scale
    live = dense[(gap < TRANSITION_FRACTION) & (dense_g.max(axis=1) > TRANSITION_FRACTION*scale)]
    eligible = ((probe > .15).all(axis=1) & (probe < .85).all(axis=1)
                & (g.max(axis=1) > .3*scale)
                & (np.abs(omegas[0]) > np.median(np.abs(omegas[0]))))
    region = np.argmax(g, axis=1)
    rng = np.random.default_rng(20261002)
    orders = [(0, 3), (1, 4), (2, 5), (0, 4), (1, 3), (2, 4), (0, 5), (1, 5), (3, 4),
              (0, 1), (2, 3), (4, 5), (0, 2), (1, 2), (3, 5)]
    bumps = []
    for want in (0, 1):
        found = None
        for narrow in orders:
            if any(set(narrow) == set(b["narrow"]) for b in bumps):
                continue
            sigma = np.full(6, BUMP_WIDE)
            sigma[list(narrow)] = BUMP_NARROW
            found = _admissible_center(probe, eligible, region, want, sigma, live, omegas, bumps, rng)
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"{kind}: no admissible isolated-bump center")
        bumps.append(dict(center=probe[found].copy(), sigma=sigma, narrow=tuple(narrow),
                          region=int(region[found]), omega_sign=float(np.sign(omegas[0][found]))))
    _BUMPS[kind] = bumps
    return bumps


class IonutSurface:
    """Adapter presenting a native 6D proxy through the SurfaceND interface.

    `distance` and `peak_distance` carry the same meaning their synthetic
    counterparts do -- distance to the feature the mask selects -- so the
    existing band and peak masks keep working without a second scoring path.
    """
    dim = 6
    band_threshold = TRANSITION_FRACTION

    def __init__(self, case, seed=0):
        base = case[:-len(BUMPED)] if case.endswith(BUMPED) else case
        if base not in CASES:
            raise ValueError(f"unknown or unscoreable Ionut case: {case}")
        if seed != 0:
            raise ValueError("the proxy formulas carry no surface seed; use seed 0")
        self.case = case
        self.base = base
        self.bumped = case.endswith(BUMPED)
        self.seed = 0

    def __call__(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        y = values(self.base, x)["y"]
        return y + self.bump_values(x) if self.bumped else y

    def bump_amplitudes(self):
        """Signed amplitude per bump: positive on gamma, along the local sign on omega.

        A frequency bump pushes |omega| up and never through zero, so it does
        not flip the drift direction and with it the apparent mode identity.
        """
        if not hasattr(self, "_amplitudes"):
            probe = qmc.Sobol(self.dim, scramble=True, seed=4242).random_base2(14)
            size = BUMP_AMPLITUDE*float(np.ptp(values(self.base, probe)["y"]))
            omega = self.base.endswith("omega")
            self._amplitudes = [size*(b["omega_sign"] if omega else 1.)
                                for b in isolated_bumps(self.base)]
        return self._amplitudes

    def bump_values(self, x):
        out = np.zeros(len(x))
        for bump, amplitude in zip(isolated_bumps(self.base), self.bump_amplitudes()):
            out += amplitude*np.exp(-.5*np.sum(((x-bump["center"])/bump["sigma"])**2, axis=1))
        return out

    def branches(self, x):
        return values(self.base, np.atleast_2d(np.asarray(x, float)))["G"]

    def region(self, x):
        """Dominant branch by growth rate, independent of the selection rule."""
        return np.argmax(self.branches(x), axis=1)

    def branch_scale(self):
        """Branch growth-rate range on one fixed probe cloud, computed once.

        The mask used to normalize by the range of whatever batch it was called
        on, so the band depended on the query set and a single point had range
        zero. A fixed cloud makes the band a property of the surface alone.
        """
        if not hasattr(self, "_branch_scale"):
            probe = qmc.Sobol(self.dim, scramble=True, seed=4242).random_base2(14)
            self._branch_scale = float(np.ptp(self.branches(probe)))
        return self._branch_scale

    def distance(self, x):
        """Normalized branch-growth gap for *live* competition; inf elsewhere.

        Small means the two branches are nearly tied. A near-tie only counts as
        a transition while at least one branch is growing: where both sit near
        zero -- ITG stable and KBM below onset -- nothing is competing. With a
        gap test alone that dead zone was 93% of the ITG-KBM band (21% for
        ITG-TEM) and dominated every ITG-KBM transition score.
        """
        g = self.branches(x)
        scale = self.branch_scale()
        if scale <= 0:
            return np.full(len(g), np.inf)
        gap = np.abs(g[:, 0]-g[:, 1])/scale
        live = g.max(axis=1) > self.band_threshold*scale
        return np.where(live, gap, np.inf)

    def peak_distance(self, x):
        """Rank distance into the high-response tail, in the same units the
        synthetic peak mask uses: below 2 selects the top decile.

        For a bumped case it is the distance to the nearest isolated bump in
        that bump's own sigma units, so the peak mask (below 2) selects the
        isolated bumps -- the feature the variant exists to test.
        """
        if self.bumped:
            x = np.atleast_2d(np.asarray(x, float))
            return np.min([np.sqrt(np.sum(((x-b["center"])/b["sigma"])**2, axis=1))
                           for b in isolated_bumps(self.base)], axis=0)
        y = self(x)
        cut = float(np.quantile(y, .9))
        top = float(np.max(y))
        if top <= cut:
            return np.full(len(y), np.inf)
        return 2.*np.clip((cut-y)/(top-cut)+1., 0., None)

    @property
    def normals(self):
        """Two branches, so per-branch errors are reported for both."""
        return np.zeros((2, self.dim))
