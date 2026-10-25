"""Fresh analytic geometries (2-D guidance, 5-D/6-D primary) for the VURS confirmation study.

Protocol: `wiki/queries/2026-10-24-vurs-confirmation-protocol` (Vulcan wiki).
Each case is K smooth branch growth rates g_k on the unit square; the oracle
reports only the winner, gamma = max_k g_k, and its label argmax_k g_k (the
dominant mode). Hidden subdominant branch values never leave the case.

Four equally weighted strata, one generator each:

  smooth   two broad branches, tilt plus a wide bump; gently curved switch
  curved   two or three branches with strong quadratic terms, so the switch
           bends; broad peaks (sigma 0.15-0.30)
  narrow   two smooth branches carrying one or two narrow anisotropic peaks
           (sigma 0.03-0.08) placed anywhere in their own region, off center
  islands  a broad base branch plus one or two island branches that win only
           inside small disconnected bumps (1-5% of the square each)

Above 2-D the same four strata are drawn by `_*_nd` generators (`dim` > 2).
Their peaks are anisotropic in a random orthonormal frame: broad bumps have
every axis width in the 2-D broad range; a narrow peak is narrow (sigma
0.03-0.08, the 2-D narrow range) along one or two directions and broad (0.15-
0.35) along the rest, so it occupies a resolvable volume at N = 256 while its
fit-free width l still reports the narrow direction. Tilts are scaled by
sqrt(2/d) so their range over the box matches 2-D. Island bumps take the drawn
volume share through the d-ball volume formula. The 2-D generators are
unchanged, so 2-D development blocks reproduce exactly.

Orientation and anisotropy are drawn per case. Draws failing the validity
checks on a fixed reference cloud are redrawn from the next child seed; the
attempt count is recorded, never hidden.

Seeds. Every case is a (namespace, block) pair. Development blocks use
DEV_ENTROPY and may be inspected freely. Confirmation blocks use an entropy
derived from the frozen manifest's commit hash, so no confirmation geometry
can exist before the manifest does. Within a block, separate SeedSequence
children carry the geometry, acquisition, grader and predictor roles.
"""
from dataclasses import dataclass, field
import hashlib

import numpy as np
from scipy.special import gammaln
from scipy.stats import qmc

STRATA = ("smooth", "curved", "narrow", "islands")
BLOCKS_PER_STRATUM = 20
DEV_ENTROPY = 20261024
ROLES = dict(geometry=0, acquisition=1, grader=2, predictor=3)
REFERENCE_SEED = 5150          # reference cloud: validity checks and scale only
MAX_ATTEMPTS = 200
MIN_MODE_MASS = dict(smooth=.15, curved=.10, narrow=.15, islands=.005)
ISLAND_MASS = (.01, .05)


def confirmation_entropy(manifest_commit):
    """Confirmation entropy from the frozen manifest commit; unknowable before it."""
    return int(hashlib.sha256(manifest_commit.encode()).hexdigest()[:16], 16)


def stratum_of(block):
    return STRATA[block // BLOCKS_PER_STRATUM % len(STRATA)]


def role_seed(entropy, block, role, dim=2):
    """Integer seed for one role of one block; roles and dimensions never share a stream.

    2-D keeps its original three-word key so existing 2-D blocks reproduce.
    """
    key = [entropy, block, ROLES[role]] + ([] if dim == 2 else [dim])
    sequence = np.random.SeedSequence(key)
    return int(sequence.generate_state(1, np.uint32)[0])


def reference_cloud(n=16384, dim=2):
    return qmc.Sobol(dim, scramble=True, seed=REFERENCE_SEED).random_base2(int(np.log2(n)))


def grader_cloud(entropy, block, replicate=0, n=65536, dim=2):
    """Independent scrambled-Sobol grading cloud; replicate 1 is the integration check."""
    seed = np.random.SeedSequence(role_seed(entropy, block, "grader", dim)).spawn(2)[replicate]
    return qmc.Sobol(dim, scramble=True, seed=np.random.default_rng(seed)).random_base2(int(np.log2(n)))


def _rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _metric(rng, major):
    """Inverse-square-width matrix of a rotated anisotropic Gaussian."""
    rho = np.exp(rng.uniform(0., np.log(4.)))
    widths = np.array([major, max(major/rho, .02)])
    r = _rotation(rng.uniform(0, np.pi))
    return r @ np.diag(1/widths**2) @ r.T


@dataclass
class Branch:
    """g(x) = c + b.(x-.5) + (x-m)^T Q (x-m) + sum_j A_j exp(-.5 (x-p_j)^T M_j (x-p_j))."""
    c: float
    b: np.ndarray
    m: np.ndarray = None
    q: np.ndarray = None
    bumps: list = field(default_factory=list)       # (amplitude, center, metric)

    def __post_init__(self):
        d = len(self.b)
        self.m = np.full(d, .5) if self.m is None else self.m
        self.q = np.zeros((d, d)) if self.q is None else self.q

    def __call__(self, x):
        d = x-self.m
        y = self.c + (x-.5) @ self.b + np.einsum("ni,ij,nj->n", d, self.q, d)
        for amplitude, center, metric in self.bumps:
            e = x-center
            y = y + amplitude*np.exp(-.5*np.einsum("ni,ij,nj->n", e, metric, e))
        return y


def _tilt(rng, size):
    return size*_rotation(rng.uniform(0, 2*np.pi)) @ np.array([1., 0.])


def _smooth(rng):
    # one wide bump shared by both branches cancels in g_0 - g_1, so the switch
    # is the tilt difference bent only by a mild quadratic
    bump = (rng.uniform(.3, .7), rng.uniform(.15, .85, 2), _metric(rng, rng.uniform(.25, .45)))
    branches = []
    for _ in range(2):
        r = _rotation(rng.uniform(0, np.pi))
        q = r @ np.diag(rng.uniform(-.6, .6, 2)) @ r.T
        branches.append(Branch(rng.uniform(-.1, .1), _tilt(rng, rng.uniform(.3, .8)),
                               m=rng.uniform(.2, .8, 2), q=q, bumps=[bump]))
    return branches


def _curved(rng):
    branches = []
    for _ in range(rng.integers(2, 4)):
        r = _rotation(rng.uniform(0, np.pi))
        q = r @ np.diag(rng.uniform(-2.5, 2.5, 2)) @ r.T
        bump = (rng.uniform(.3, .7), rng.uniform(.15, .85, 2), _metric(rng, rng.uniform(.15, .30)))
        branches.append(Branch(rng.uniform(-.15, .15), _tilt(rng, rng.uniform(.2, .6)),
                               m=rng.uniform(.2, .8, 2), q=q, bumps=[bump]))
    return branches


def _narrow(rng):
    branches = [Branch(rng.uniform(-.05, .05), _tilt(rng, rng.uniform(.3, .8))) for _ in range(2)]
    for k, branch in enumerate(branches):
        for _ in range(rng.integers(1, 3)):
            for _ in range(500):
                center = rng.uniform(.05, .95, 2)
                if _winner(branches, center[None])[0] == k:
                    break
            else:
                return None
            branch.bumps.append((rng.uniform(.5, 1.), center, _metric(rng, rng.uniform(.03, .08))))
    return branches


def _islands(rng):
    base = Branch(0., _tilt(rng, rng.uniform(.3, .8)),
                  bumps=[(rng.uniform(.3, .6), rng.uniform(.2, .8, 2), _metric(rng, rng.uniform(.25, .45)))])
    branches, centers = [base], []
    for _ in range(rng.integers(1, 3)):
        island = Branch(base.c, base.b.copy(), bumps=list(base.bumps))
        island.c -= rng.uniform(.15, .3)                 # loses everywhere but its bumps
        for _ in range(rng.integers(1, 3)):
            for _ in range(500):
                center = rng.uniform(.12, .88, 2)
                if all(np.linalg.norm(center-c) > .25 for c in centers):
                    break
            else:
                return None
            centers.append(center)
            # bump > depth on an ellipse of area 2 pi ln(A/depth)/sqrt(det M);
            # pick the metric scale that gives the drawn island area
            area = rng.uniform(*ISLAND_MASS)
            amplitude, depth = rng.uniform(.5, .9), base.c-island.c
            shape = _metric(rng, 1.)
            scale = 2*np.pi*np.log(amplitude/depth)/(area*np.sqrt(np.linalg.det(shape)))
            island.bumps.append((amplitude, center, shape*scale))
        branches.append(island)
    return branches


GENERATORS = dict(smooth=_smooth, curved=_curved, narrow=_narrow, islands=_islands)


# --- d > 2 -------------------------------------------------------------------

def _frame(rng, d):
    """Random orthonormal frame (Haar, sign-fixed QR)."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q*np.sign(np.diag(r))


def _metric_nd(rng, widths):
    r = _frame(rng, len(widths))
    return r @ np.diag(1/np.asarray(widths)**2) @ r.T


def _broad_widths(rng, d, major):
    """Major width on one axis, the rest shrunk by log-uniform ratios in [1, 4]."""
    rho = np.exp(rng.uniform(0., np.log(4.), d))
    rho[0] = 1.
    return np.maximum(major/rho, .05)


def _narrow_widths(rng, d):
    """Narrow (0.03-0.08) along one or two directions, broad (0.15-0.35) along the rest."""
    widths = rng.uniform(.15, .35, d)
    narrow = rng.integers(1, 3)
    widths[:narrow] = rng.uniform(.03, .08, narrow)
    return widths


def _tilt_nd(rng, d, size):
    u = rng.standard_normal(d)
    return size*np.sqrt(2/d)*u/np.linalg.norm(u)


def _quadratic_nd(rng, d, bound):
    r = _frame(rng, d)
    return r @ np.diag(rng.uniform(-bound, bound, d)) @ r.T


def _smooth_nd(rng, d):
    bump = (rng.uniform(.3, .7), rng.uniform(.15, .85, d),
            _metric_nd(rng, _broad_widths(rng, d, rng.uniform(.25, .45))))
    return [Branch(rng.uniform(-.1, .1), _tilt_nd(rng, d, rng.uniform(.3, .8)), m=rng.uniform(.2, .8, d),
                   q=_quadratic_nd(rng, d, .6), bumps=[bump]) for _ in range(2)]


def _curved_nd(rng, d):
    branches = []
    for _ in range(rng.integers(2, 4)):
        bump = (rng.uniform(.3, .7), rng.uniform(.15, .85, d),
                _metric_nd(rng, _broad_widths(rng, d, rng.uniform(.15, .30))))
        branches.append(Branch(rng.uniform(-.15, .15), _tilt_nd(rng, d, rng.uniform(.2, .6)),
                               m=rng.uniform(.2, .8, d), q=_quadratic_nd(rng, d, 2.5), bumps=[bump]))
    return branches


def _narrow_nd(rng, d):
    branches = [Branch(rng.uniform(-.05, .05), _tilt_nd(rng, d, rng.uniform(.3, .8))) for _ in range(2)]
    for k, branch in enumerate(branches):
        for _ in range(rng.integers(1, 3)):
            for _ in range(500):
                center = rng.uniform(.05, .95, d)
                if _winner(branches, center[None])[0] == k:
                    break
            else:
                return None
            branch.bumps.append((rng.uniform(.5, 1.), center, _metric_nd(rng, _narrow_widths(rng, d))))
    return branches


def _islands_nd(rng, d):
    base = Branch(0., _tilt_nd(rng, d, rng.uniform(.3, .8)),
                  bumps=[(rng.uniform(.3, .6), rng.uniform(.2, .8, d),
                          _metric_nd(rng, _broad_widths(rng, d, rng.uniform(.25, .45))))])
    branches, centers = [base], []
    log_ball = d/2*np.log(np.pi)-gammaln(d/2+1)          # log volume of the unit d-ball
    for _ in range(rng.integers(1, 3)):
        island = Branch(base.c, base.b.copy(), bumps=list(base.bumps))
        island.c -= rng.uniform(.15, .3)                 # loses everywhere but its bumps
        for _ in range(rng.integers(1, 3)):
            for _ in range(500):
                center = rng.uniform(.15, .85, d)
                if all(np.linalg.norm(center-c) > .4 for c in centers):
                    break
            else:
                return None
            centers.append(center)
            # bump > depth inside e^T M e < r^2 with r^2 = 2 ln(A/depth), volume
            # V_d r^d / sqrt(det M); scaling a fixed shape sets the drawn share
            area = rng.uniform(*ISLAND_MASS)
            amplitude, depth = rng.uniform(.5, .9), base.c-island.c
            shape = _metric_nd(rng, _broad_widths(rng, d, 1.))
            r2 = 2*np.log(amplitude/depth)
            log_scale = 2/d*(log_ball+d/2*np.log(r2)-np.log(area)-.5*np.linalg.slogdet(shape)[1])
            island.bumps.append((amplitude, center, shape*np.exp(log_scale)))
        branches.append(island)
    return branches


GENERATORS_ND = dict(smooth=_smooth_nd, curved=_curved_nd, narrow=_narrow_nd, islands=_islands_nd)


def _winner(branches, x):
    return np.argmax(np.column_stack([g(x) for g in branches]), axis=1)


def _valid(stratum, branches, cloud):
    labels = _winner(branches, cloud)
    mass = np.bincount(labels, minlength=len(branches))/len(cloud)
    if mass.min() < MIN_MODE_MASS[stratum]:
        return False
    if stratum == "islands":
        # every island mode must stay small; the base must hold the rest
        return mass[1:].max() <= 2*ISLAND_MASS[1] and mass[0] > .7
    return True


class ConfirmationCase:
    """Winner-only oracle for one block: callable gamma, `region` labels.

    `scale` is the case's response range on the reference cloud, fixed before
    any acquisition and shared by every arm.
    """
    def __init__(self, entropy, block, dim=2):
        self.entropy, self.block = int(entropy), int(block)
        self.stratum = stratum_of(block)
        self.dim = int(dim)
        cloud = reference_cloud(dim=self.dim)
        sequence = np.random.SeedSequence(role_seed(entropy, block, "geometry", self.dim))
        for attempt, child in enumerate(sequence.spawn(MAX_ATTEMPTS)):
            rng = np.random.default_rng(child)
            branches = (GENERATORS[self.stratum](rng) if self.dim == 2
                        else GENERATORS_ND[self.stratum](rng, self.dim))
            if branches is not None and _valid(self.stratum, branches, cloud):
                break
        else:
            raise RuntimeError(f"block {block}: no valid {self.stratum} geometry in {MAX_ATTEMPTS} draws")
        self.branches, self.attempts = branches, attempt+1
        self.n_modes = len(branches)
        self.scale = float(np.ptp(self(cloud)))
        self.reference_mass = np.bincount(self.region(cloud), minlength=self.n_modes)/len(cloud)

    def _stack(self, x):
        return np.column_stack([g(np.atleast_2d(np.asarray(x, float))) for g in self.branches])

    def region(self, x):
        return self._stack(x).argmax(axis=1)

    def __call__(self, x):
        return self._stack(x).max(axis=1)

    def peak_widths(self, step=1e-3, cap=1.):
        """Fit-free peak width per mode: l_m = sqrt(h_m / lambda_m) at the mode's maximum.

        lambda_m is the largest eigenvalue of -Hessian of gamma at the mode's
        highest point, h_m the height of that point above the mode's median.
        For a Gaussian bump of amplitude A and width sigma on a flat floor,
        l = sigma exactly. A maximum on the box edge, or one without negative
        curvature, has no interior peak and gets `cap` (broad).
        """
        from scipy.optimize import minimize
        cloud = reference_cloud(dim=self.dim)
        truth, region = self(cloud), self.region(cloud)
        out = []
        for m in range(self.n_modes):
            inside = region == m
            if not inside.any():
                out.append(None)
                continue
            start = cloud[inside][int(np.argmax(truth[inside]))]
            fit = minimize(lambda p: -self.branches[m](p[None])[0], start, method="L-BFGS-B",
                           bounds=[(0., 1.)]*self.dim)
            peak = fit.x if self.region(fit.x)[0] == m else start
            height = float(self(peak)[0]-np.median(truth[inside]))
            if np.any(np.minimum(peak, 1-peak) < 2*step) or height <= 0:
                out.append(cap)
                continue
            hessian = np.empty((self.dim, self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    e_i, e_j = np.eye(self.dim)[i]*step, np.eye(self.dim)[j]*step
                    hessian[i, j] = (self(peak+e_i+e_j)-self(peak+e_i-e_j)
                                     - self(peak-e_i+e_j)+self(peak-e_i-e_j))[0]/(4*step**2)
            curvature = float(np.linalg.eigvalsh(-hessian).max())
            out.append(cap if curvature <= 0 else float(min(cap, np.sqrt(height/curvature))))
        return out

    def peak_width(self):
        """Case width: geometric mean of the mode widths, weighted by true mode mass as C2 is."""
        widths, mass = self.peak_widths(), self.reference_mass
        live = [(w, a) for w, a in zip(widths, mass) if w is not None]
        return float(np.exp(sum(a*np.log(w) for w, a in live)/sum(a for _, a in live)))

    def describe(self):
        return dict(block=self.block, dim=self.dim, stratum=self.stratum, n_modes=self.n_modes, attempts=self.attempts,
                    scale=self.scale, reference_mass=self.reference_mass.round(4).tolist())
