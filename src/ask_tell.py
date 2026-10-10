"""VWRS/VURS as an ask/tell sampler, for campaigns whose evaluations take hours.

The benchmark arms (`strategies.resolution_sampling`) own their oracle loop:
they call it and wait. A GENE campaign cannot wait inside a loop -- a batch
sits in the NERSC queue for hours and comes back partly failed -- so here the
loop is inverted. `ask()` proposes a batch, the caller runs it, `tell()` hands
the results back, and the sampler keeps no state the caller cannot save with
`state()` and restore with `from_state()`.

The acquisition is the same `strategies.Acquisition` the benchmarks score, and
a batch is chosen by `Acquisition.select`, so a batch of one reproduces the
sequential arm and a batch of k reproduces arm vurs-b<k> (`batch_mode`
"greedy") or vurs-b<k>r ("rank").

Outcomes a run can have, and what each does here:

    converged       value used with the GP's usual jitter
    not converged   value used, but flagged as uncertain: the GP gets
                    `unconverged_noise` as that point's variance (in units of
                    the response variance) and the variation estimate treats
                    the matching spread as noise, not structure
    failed          the simulation did not execute: no value exists. The point
                    is left out of the fit and of every later candidate set,
                    and is listed in `failed` for the caller to raise as an
                    alert. It never steers the scan.

Axes are physical; the sampler works in the unit box, mapping each axis
linearly (or logarithmically, for `log_axes`) onto [0, 1].
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from ND_scan_samplers.src.strategies import (Acquisition, BATCH_MODES, candidate_count,
                                             duplicate_radius, fit_gp_xy, resolution_weights,
                                             sobol_candidates)

CONVERGED, UNCONVERGED, FAILED = "converged", "unconverged", "failed"


def key(u):
    return tuple(np.round(np.asarray(u, float), 12))


class ResolutionSampler:
    """Batched VWRS/VURS behind ask()/tell().

    axes        {name: (low, high)} in physical units, in a fixed order
    uncertainty True for VURS, False for VWRS
    batch       points per ask(); the first ask() returns the 2d+1 initial design
    batch_mode  "greedy" or "rank" (see `strategies.Acquisition.select`)
    candidates  optional finite set of allowed points (physical units, one row
                or dict per point) -- e.g. the pre-built equilibrium library.
                Omitted: a fresh Sobol cloud over the box at every ask().
    budget      optional cap on points placed (told plus failed plus pending)
    initial     optional initial design (physical units) replacing the
                Latin hypercube, e.g. to share one start across arms
    """

    def __init__(self, axes, *, uncertainty=True, weights=None, nu=.5, batch=8,
                 batch_mode="greedy", candidates=None, budget=None, seed=0, log_axes=(),
                 unconverged_noise=.1, initial=None):
        if batch_mode not in BATCH_MODES:
            raise ValueError(f"batch_mode must be one of {BATCH_MODES}")
        if int(batch) < 1:
            raise ValueError("batch must be at least one")
        self.axes = {name: (float(lo), float(hi)) for name, (lo, hi) in axes.items()}
        for name, (lo, hi) in self.axes.items():
            if not hi > lo:
                raise ValueError(f"axis {name}: high must exceed low")
            if name in log_axes and lo <= 0:
                raise ValueError(f"log axis {name} needs a positive range")
        self.names = list(self.axes)
        self.log_axes = tuple(log_axes)
        self.dim = len(self.names)
        self.uncertainty = bool(uncertainty)
        self.weights = resolution_weights(uncertainty, False, weights)
        self.nu, self.batch, self.batch_mode = nu, int(batch), batch_mode
        self.budget = None if budget is None else int(budget)
        self.seed = int(seed)
        self.unconverged_noise = float(unconverged_noise)
        self.pool = None if candidates is None else self._to_unit(candidates)
        if self.pool is not None and len({key(p) for p in self.pool}) != len(self.pool):
            raise ValueError("candidate points must be distinct")
        self.initial = None if initial is None else self._to_unit(initial)
        self.x = np.empty((0, self.dim))
        self.y = np.empty(0)
        self.status = []          # CONVERGED or UNCONVERGED, one per told point
        self.failed = []          # unit coordinates of runs that did not execute
        self.pending = np.empty((0, self.dim))
        self.fit_warnings = 0

    # -- units ---------------------------------------------------------

    def _to_unit(self, points):
        rows = [[p[n] for n in self.names] if isinstance(p, dict) else list(p) for p in points]
        raw = np.array(rows, float).reshape(-1, self.dim)
        unit = np.empty_like(raw)
        for j, name in enumerate(self.names):
            lo, hi = self.axes[name]
            if name in self.log_axes:
                unit[:, j] = (np.log(raw[:, j])-np.log(lo))/(np.log(hi)-np.log(lo))
            else:
                unit[:, j] = (raw[:, j]-lo)/(hi-lo)
        if (unit < -1e-9).any() or (unit > 1+1e-9).any():
            raise ValueError("points must lie inside the axis bounds")
        return np.clip(unit, 0., 1.)

    def _to_physical(self, unit):
        out = []
        for u in np.atleast_2d(unit):
            point = {}
            for j, name in enumerate(self.names):
                lo, hi = self.axes[name]
                point[name] = float(np.exp(np.log(lo)+u[j]*(np.log(hi)-np.log(lo)))
                                    if name in self.log_axes else lo+u[j]*(hi-lo))
            out.append(point)
        return out

    # -- protocol ------------------------------------------------------

    @property
    def placed(self):
        return len(self.x) + len(self.failed) + len(self.pending)

    @property
    def finished(self):
        if self.budget is not None and self.placed >= self.budget:
            return True
        return self.pool is not None and not len(self._open_pool())

    def _open_pool(self):
        taken = {key(p) for p in np.vstack([self.x, self.pending, np.reshape(self.failed, (-1, self.dim))])}
        return np.array([p for p in self.pool if key(p) not in taken]).reshape(-1, self.dim)

    def _initial_design(self):
        if self.initial is not None:
            target = self.initial
        else:
            target = qmc.LatinHypercube(self.dim, seed=20260912+self.seed).random(2*self.dim+1)
        if self.pool is None or self.initial is not None:
            return target
        # Snap to distinct library points, as the pool benchmark does.
        tree, taken = cKDTree(self.pool), []
        for point in target:
            for index in np.atleast_1d(tree.query(point, k=min(len(self.pool), 32))[1]):
                if int(index) not in taken:
                    taken.append(int(index))
                    break
        return self.pool[taken]

    def ask(self, n=None):
        """Propose the next batch as [{axis: value}], or [] when finished.

        Points already proposed and not yet told count as placed: the next
        batch spreads around them rather than repeating them.
        """
        if self.finished:
            return []
        room = np.inf if self.budget is None else self.budget-self.placed
        n = int(min(self.batch if n is None else n, room))
        if not (len(self.x) or len(self.pending) or len(self.failed)):
            chosen = self._initial_design()[:int(min(room, 2*self.dim+1))]
        elif len(self.x) < self.dim+2:
            # Failures left too few values for the variation fit: top up by
            # fill distance alone until the fit is possible.
            chosen = self._fill(int(min(room, self.dim+2-len(self.x)-len(self.pending))))
        else:
            chosen = self._propose(n)
        self.pending = np.vstack([self.pending, chosen])
        return self._to_physical(chosen) if len(chosen) else []

    def _propose(self, n):
        rng = np.random.default_rng([self.seed, len(self.x), len(self.failed)])
        if self.pool is None:
            candidates = sobol_candidates(rng, self.dim, candidate_count(self.dim))
        else:
            candidates = self._open_pool()
        unconverged = np.array([s == UNCONVERGED for s in self.status])
        spread = None
        if unconverged.any():
            spread = np.where(unconverged, np.sqrt(self.unconverged_noise)*float(np.std(self.y)), 0.)
        alpha = np.where(unconverged, self.unconverged_noise, 1e-8) if unconverged.any() else None
        acquisition = Acquisition(self.x, self.y, candidates, self.seed, self.weights,
                                  uncertainty=self.uncertainty, nu=self.nu, noise=spread,
                                  gp_alpha=alpha)
        self.fit_warnings += acquisition.fit_warnings
        merit = acquisition.merit()
        radius = None
        if self.pool is None:
            radius = duplicate_radius(self.dim, len(self.x))
            merit[acquisition.spacing < radius] = -np.inf
        if len(self.pending):
            # Proposed but not yet told: treat as placed, exactly as a point
            # chosen earlier in this batch would be.
            spacing = np.minimum(acquisition.spacing, cKDTree(self.pending).query(candidates)[0])
            merit = acquisition.merit(spacing)
            if radius is not None:
                merit[spacing < radius] = -np.inf
            acquisition.spacing = spacing
        picks = acquisition.select(n, self.batch_mode, radius, merit)
        return candidates[picks]

    def _fill(self, n):
        if n <= 0:
            return np.empty((0, self.dim))
        rng = np.random.default_rng([self.seed, len(self.x), len(self.failed)])
        candidates = (sobol_candidates(rng, self.dim, candidate_count(self.dim))
                      if self.pool is None else self._open_pool())
        placed = np.vstack([self.x, self.pending, np.reshape(self.failed, (-1, self.dim))])
        chosen = []
        for _ in range(min(n, len(candidates))):
            spacing = cKDTree(np.vstack([placed, *chosen]) if chosen else placed).query(candidates)[0]
            chosen.append(candidates[int(np.argmax(spacing))][None, :])
        return np.vstack(chosen) if chosen else np.empty((0, self.dim))

    def tell(self, values, points=None, converged=None):
        """Report the oldest pending points' outcomes, in ask() order.

        values     one per point; None or NaN marks a run that failed to execute
        points     where each run actually happened, if not the asked point
                   (e.g. the library equilibrium it was snapped to)
        converged  one bool per point; False keeps the value but marks it
                   uncertain. Default: all converged.
        """
        values = list(values)
        n = len(values)
        if n > len(self.pending):
            raise ValueError(f"{n} values for {len(self.pending)} pending points")
        asked, self.pending = self.pending[:n], self.pending[n:]
        where = asked if points is None else self._to_unit(points)
        if len(where) != n:
            raise ValueError("one point per value required")
        flags = [True]*n if converged is None else [bool(c) for c in converged]
        if len(flags) != n:
            raise ValueError("one convergence flag per value required")
        known = {key(p) for p in self.x}
        for u, value, ok in zip(where, values, flags):
            if value is None or not np.isfinite(value):
                self.failed.append(u)
                continue
            if key(u) in known:
                raise ValueError(f"point {self._to_physical(u)[0]} was already told")
            known.add(key(u))
            self.x = np.vstack([self.x, u])
            self.y = np.append(self.y, float(value))
            self.status.append(CONVERGED if ok else UNCONVERGED)

    def surrogate(self):
        """GP mean over physical points (dict or row per point); VURS's own model."""
        if len(self.x) < 2:
            raise RuntimeError("nothing to fit yet")
        unconverged = np.array([s == UNCONVERGED for s in self.status])
        alpha = np.where(unconverged, self.unconverged_noise, 1e-8)
        gp, _ = fit_gp_xy(self.x, self.y, self.seed, self.nu, alpha)
        return lambda points: gp.predict(self._to_unit([points] if isinstance(points, dict) else points))

    @property
    def failed_points(self):
        """Physical coordinates of runs that failed to execute: dashboard alerts."""
        return self._to_physical(np.reshape(self.failed, (-1, self.dim))) if self.failed else []

    # -- persistence ---------------------------------------------------

    def state(self):
        return dict(axes={k: list(v) for k, v in self.axes.items()}, log_axes=list(self.log_axes),
                    uncertainty=self.uncertainty, weights=dict(self.weights), nu=self.nu,
                    batch=self.batch, batch_mode=self.batch_mode, budget=self.budget,
                    seed=self.seed, unconverged_noise=self.unconverged_noise,
                    pool=None if self.pool is None else self.pool.tolist(),
                    initial=None if self.initial is None else self.initial.tolist(),
                    x=self.x.tolist(), y=self.y.tolist(), status=list(self.status),
                    failed=np.reshape(self.failed, (-1, self.dim)).tolist(),
                    pending=self.pending.tolist(), fit_warnings=self.fit_warnings)

    @classmethod
    def from_state(cls, state):
        weights = state["weights"] if state["uncertainty"] else None
        sampler = cls(state["axes"], uncertainty=state["uncertainty"], weights=weights,
                      nu=state["nu"], batch=state["batch"], batch_mode=state["batch_mode"],
                      budget=state["budget"], seed=state["seed"], log_axes=state["log_axes"],
                      unconverged_noise=state["unconverged_noise"])
        unit = lambda rows: np.array(rows, float).reshape(-1, sampler.dim)
        sampler.pool = None if state["pool"] is None else unit(state["pool"])
        sampler.initial = None if state["initial"] is None else unit(state["initial"])
        sampler.x, sampler.y = unit(state["x"]), np.array(state["y"], float)
        sampler.status = list(state["status"])
        sampler.failed = list(unit(state["failed"]))
        sampler.pending = unit(state["pending"])
        sampler.fit_warnings = state["fit_warnings"]
        return sampler
