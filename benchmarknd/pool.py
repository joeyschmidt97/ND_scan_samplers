"""Pool-mode replay: run the placement policies against a fixed set of real runs.

A pool is a finite table of completed simulations. The oracle answers only at
pool points and refuses anything else, so every arm chooses among real runs
exactly as it would choose among Sobol candidates (see
`strategies.candidates_for`). Truth is known at every pool point, so scoring
uses the points an arm did not pay for, never an analytic surface.

    python -m benchmarknd.pool --pool data/pools/hatch_pscans_global.csv \
        --output results/pool-hatch-global-2026-10-03

What this measures and what it does not: a ranking here is conditional on the
pool's support. Methods with prescribed node positions (sparse grids) cannot
be evaluated natively on a pool and are refused rather than snapped.
"""
import argparse
import csv
import hashlib
import json
import pathlib
import subprocess
import time
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from resolution import knn_variation

from .core import Observations, reconstruct
from .strategies import ARMS, NOISE_AWARE_ARMS, run_arm

# Coordinates of a pool row, in the order they become unit-box axes. ky is
# sampled on a log axis because GENE scans space it that way and the response
# changes by mode over decades, not over equal steps.
AXES = ("factor_T", "factor_n", "x0", "ky")
LOG_AXES = ("ky",)

# Arms that cannot run on a pool: prescribed-node methods need their own node
# positions, and the noise-aware arms need a reported spread a pool lacks.
POOL_REFUSED = ("sglib", "sgpp") + NOISE_AWARE_ARMS
POOL_ARMS = tuple(arm for arm in ARMS if arm not in POOL_REFUSED)
DEFAULT_POOL_ARMS = ("space-filling", "gpr-var", "gpr-grad", "gpr-u50-g50",
                     "gpr-m05-blend", "vwrs", "vurs", "vwrs-m", "vurs-m", "moe")


class PoolOracle:
    """Exact lookup of a response and a branch label at pool points only."""

    def __init__(self, x, y, labels, label_names=()):
        x = np.asarray(x, float)
        if x.ndim != 2 or len(x) != len(y) or len(x) != len(labels):
            raise ValueError("pool coordinates, responses and labels must align")
        if (x < 0).any() or (x > 1).any():
            raise ValueError("pool coordinates must lie in the unit box")
        self.pool = x
        self.y = np.asarray(y, float)
        self.labels = np.asarray(labels, int)
        self.label_names = tuple(label_names)
        self._index = {Observations.key(p): i for i, p in enumerate(x)}
        if len(self._index) != len(x):
            raise ValueError("pool coordinates must be unique; aggregate repeats first")

    @property
    def dim(self):
        return self.pool.shape[1]

    def indices(self, x):
        out = []
        for point in np.atleast_2d(x):
            index = self._index.get(Observations.key(point))
            if index is None:
                raise KeyError("requested point is not in the pool")
            out.append(index)
        return np.array(out, int)

    def __call__(self, x):
        return self.y[self.indices(x)]

    def region(self, x):
        """Branch label at paid points, as `strategies.observed_labels` expects."""
        return self.labels[self.indices(x)]


def load_pool(path, target="gamma", drop_unlabelled=True):
    """Read an exported pool CSV into a unit-box oracle plus its provenance.

    Rows with the gamma == -1 fill value carry no response and are dropped.
    Repeated coordinates (two scan directories with the same factors) are
    averaged, with the label taken by majority; the count is reported.
    """
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    kept, dropped = [], Counter()
    for row in rows:
        if float(row["gamma"]) == -1.:
            dropped["gamma_sentinel"] += 1
            continue
        if drop_unlabelled and not row["mode_ID"]:
            dropped["unlabelled"] += 1
            continue
        kept.append(row)
    if not kept:
        raise ValueError("no usable rows in the pool")

    raw = np.array([[float(r[a]) for a in AXES] for r in kept])
    for i, axis in enumerate(AXES):
        if axis in LOG_AXES:
            raw[:, i] = np.log(raw[:, i])
    names = sorted({r["mode_ID"] or "Unknown" for r in kept})
    label = np.array([names.index(r["mode_ID"] or "Unknown") for r in kept])
    value = np.array([float(r[target]) for r in kept])

    # Aggregate exact repeats before normalizing, keyed on the raw coordinates.
    groups = {}
    for i, key in enumerate(map(tuple, np.round(raw, 10))):
        groups.setdefault(key, []).append(i)
    coords = np.array([raw[g[0]] for g in groups.values()])
    y = np.array([value[g].mean() for g in groups.values()])
    labels = np.array([Counter(label[g]).most_common(1)[0][0] for g in groups.values()])
    repeats = sum(len(g)-1 for g in groups.values())

    low, high = coords.min(axis=0), coords.max(axis=0)
    if (high <= low).any():
        raise ValueError("a pool axis is constant; drop it before replay")
    unit = (coords-low)/(high-low)
    meta = dict(path=str(path), target=target, axes=list(AXES), log_axes=list(LOG_AXES),
                raw_low=low.tolist(), raw_high=high.tolist(), rows_read=len(rows),
                dropped=dict(dropped), repeats_averaged=repeats, pool_size=len(unit),
                label_names=names, label_counts={names[k]: int(v) for k, v in
                                                 sorted(Counter(labels).items())},
                sha256=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest())
    return PoolOracle(unit, y, labels, names), meta


def initial_pool_design(oracle, seed=0):
    """The shared Latin-hypercube start, snapped to distinct pool points."""
    target = Observations.initial_design(oracle.dim, seed)
    tree, taken = cKDTree(oracle.pool), []
    for point in target:
        for index in tree.query(point, k=min(len(oracle.pool), 32))[1]:
            if index not in taken:
                taken.append(int(index))
                break
    return oracle.pool[taken]


def transition_band(oracle, k=None):
    """Pool points whose nearest pool neighbours include another branch label."""
    k = k or 2*oracle.dim+1
    _, neighbours = cKDTree(oracle.pool).query(oracle.pool, k=min(k+1, len(oracle.pool)))
    return np.array([len(set(oracle.labels[row])) > 1 for row in neighbours])


def score_prefix(oracle, paid_index, band):
    """Scores of one paid prefix, measured on every pool point it left unpaid."""
    paid = np.zeros(len(oracle.pool), bool)
    paid[paid_index] = True
    held = ~paid
    scale = float(np.ptp(oracle.y))
    yhat = reconstruct(oracle.pool[paid], oracle.y[paid], oracle.pool[held])
    absolute = np.abs(yhat-oracle.y[held])/scale
    nearest = cKDTree(oracle.pool[paid]).query(oracle.pool[held])[1]
    correct = oracle.labels[paid][nearest] == oracle.labels[held]
    band_held = band[held]
    out = dict(n=int(paid.sum()), nrmse=float(np.sqrt(np.mean(absolute**2))),
               nmae=float(np.mean(absolute)), p95_error=float(np.quantile(absolute, .95)),
               label_accuracy=float(correct.mean()),
               band_coverage=float(paid[band].mean()) if band.any() else None,
               band_nmae=float(np.mean(absolute[band_held])) if band_held.any() else None,
               band_label_accuracy=float(correct[band_held].mean()) if band_held.any() else None)
    counts = Counter(oracle.labels[paid])
    out["paid_per_label"] = {oracle.label_names[k]: int(counts.get(k, 0))
                             for k in range(len(oracle.label_names))}
    out.update(per_mode_scores(oracle, held, yhat))

    # Classify-then-regress: the label each held point is predicted to carry
    # chooses which mode's own interpolant reconstructs it. The oracle-label
    # variant uses the true held labels instead; it is a diagnostic that
    # separates regression error from classification error and is not
    # available to a real campaign.
    predicted = oracle.labels[paid][nearest]
    ctr = classify_then_regress(oracle, paid, held, predicted, yhat)
    ctr_oracle = classify_then_regress(oracle, paid, held, oracle.labels[held], yhat)
    ctr_absolute = np.abs(ctr["yhat"]-oracle.y[held])/scale
    out.update(ctr_nrmse=float(np.sqrt(np.mean(ctr_absolute**2))),
               ctr_fallback_points=ctr["fallback_points"])
    out.update({"ctr_" + k: v for k, v in per_mode_scores(oracle, held, ctr["yhat"]).items()})
    out.update({"ctr_oracle_" + k: v for k, v in
                per_mode_scores(oracle, held, ctr_oracle["yhat"]).items()
                if k in ("mode_nrmse", "macro_nrmse", "worst_mode_nrmse")})
    out.update(truth_free_scores(oracle.pool[paid], oracle.y[paid], oracle.labels[paid],
                                 oracle.pool))
    return out


def truth_free_scores(x, y, labels, candidates, folds=10, seed=0):
    """Scores computable without ground truth, from paid runs alone.

    These are what a live campaign (NSTX GENE, no reference manifold) can
    report. Every input is something the campaign owns: the paid inputs and
    responses, the labels classified at paid runs, and the candidate set or
    domain it could have sampled. Prefixed `tf_`.

    Validated against truth on the Hatch DIII-D pool (seeds 0-3, ten arms):
    - `tf_observed_vwfd_p95` (fill distance times the variation estimated
      from paid runs) tracks per-mode truth error, Spearman +0.84 at N=61 and
      N=256. It is the same quantity VWRS/VURS minimize, so it flatters them;
      the GP and space-filling arms still order correctly under it.
    - `tf_fill_p95` (pure coverage) tracks label accuracy, Spearman -0.84.
    - `tf_paid_boundary` (share of paid runs with a differently labelled paid
      neighbour) tracks label accuracy, +0.92 at N=256.
    - `tf_cv_nrmse` and `tf_loo_label` are reported as warnings, not merits:
      under concentrated designs they reward clustering (space-filling has
      the lowest CV error and a poor truth error; gradient GPs have the
      highest neighbour-label agreement and the worst truth error).
    """
    x, y, labels = np.asarray(x, float), np.asarray(y, float), np.asarray(labels)
    scale = max(float(np.ptp(y)), 1e-12)
    spacing = cKDTree(x).query(candidates)[0]
    gradient = knn_variation(x, y, candidates).gradient
    order = np.random.default_rng(seed).permutation(len(x)) % folds
    residual = np.empty(len(x))
    for fold in range(folds):
        test = order == fold
        residual[test] = reconstruct(x[~test], y[~test], x[test])-y[test]
    k = min(2*x.shape[1]+2, len(x))
    neighbours = cKDTree(x).query(x, k=k)[1]
    return dict(tf_fill_p95=float(np.quantile(spacing, .95)),
                tf_observed_vwfd_p95=float(np.quantile(spacing*gradient/scale, .95)),
                tf_paid_boundary=float(np.mean([len(set(labels[row])) > 1 for row in neighbours])),
                tf_cv_nrmse=float(np.sqrt(np.mean((residual/scale)**2))),
                tf_loo_label=float(np.mean(labels[neighbours[:, 1]] == labels)))


def classify_then_regress(oracle, paid, held, held_labels, fallback):
    """Reconstruct each held point from the paid points of its assigned mode.

    One thin-plate RBF per mode, fitted only on paid points carrying that
    label, so a small-growth branch is not smeared by a neighbouring branch
    with growth rates ten times larger. A mode whose paid points cannot carry
    the interpolant's linear tail -- fewer than dim + 2, or all on one
    hyperplane, e.g. every paid MTM point at a single ky -- falls back to the
    global reconstruction for its held points; the count is reported.
    """
    yhat = np.array(fallback, float)
    fallback_points = 0
    paid_x, paid_y, paid_labels = oracle.pool[paid], oracle.y[paid], oracle.labels[paid]
    held_x = oracle.pool[held]
    for k in np.unique(held_labels):
        target = held_labels == k
        source = paid_labels == k
        tail = np.column_stack([np.ones(source.sum()), paid_x[source]])
        if source.sum() < oracle.dim+2 or np.linalg.matrix_rank(tail) < oracle.dim+1:
            fallback_points += int(target.sum())
            continue
        yhat[target] = reconstruct(paid_x[source], paid_y[source], held_x[target])
    return dict(yhat=yhat, fallback_points=fallback_points)


def per_mode_scores(oracle, held, yhat):
    """Reconstruction error inside each true mode, on that mode's own scale.

    The pooled `nrmse` divides by the range of the whole pool, so the branch
    with the largest growth rates (high-ky ETG) dominates it and an arm that
    spends its budget there wins by construction. Here each mode's error is
    normalized by that mode's own response range over the pool, and the
    macro average weights every mode equally regardless of size or scale.
    """
    truth, labels = oracle.y[held], oracle.labels[held]
    per_nrmse, per_nmae = {}, {}
    for k, name in enumerate(oracle.label_names):
        mask = labels == k
        span = float(np.ptp(oracle.y[oracle.labels == k]))
        if not mask.any() or span <= 0:
            per_nrmse[name] = per_nmae[name] = None
            continue
        error = np.abs(yhat[mask]-truth[mask])/span
        per_nrmse[name] = float(np.sqrt(np.mean(error**2)))
        per_nmae[name] = float(np.mean(error))
    live = [v for v in per_nrmse.values() if v is not None]
    return dict(mode_nrmse=per_nrmse, mode_nmae=per_nmae,
                macro_nrmse=float(np.mean(live)) if live else None,
                worst_mode_nrmse=float(np.max(live)) if live else None)


def rescore(results_path, output_path=None):
    """Recompute every checkpoint score from the stored selection orders.

    Placement does not depend on the scorer, so a scoring change needs no
    rerun: each finished trajectory stores its full paid order, and every
    checkpoint is a prefix of it.
    """
    results_path = pathlib.Path(results_path)
    payload = json.loads(results_path.read_text())
    oracle, meta = load_pool(payload["pool"]["path"], payload["pool"]["target"])
    if meta["sha256"] != payload["pool"]["sha256"]:
        raise ValueError("pool file changed since the run; refusing to rescore")
    band = transition_band(oracle)
    ends = {(r["arm"], r["seed"]): r["selected"] for r in payload["rows"]
            if r["status"] == "ok" and "selected" in r}
    rows = []
    for row in payload["rows"]:
        if row["status"] != "ok" or (row["arm"], row["seed"]) not in ends:
            continue
        order = ends[(row["arm"], row["seed"])]
        fresh = dict(row)
        fresh.update(score_prefix(oracle, np.asarray(order[:row["n"]]), band))
        rows.append(fresh)
    out = dict(payload, rows=rows, rescored_from=str(results_path),
               rescore_note="per-mode scores added; placements unchanged")
    output_path = pathlib.Path(output_path or results_path.with_name("rescored.json"))
    output_path.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    return output_path


def checkpoints(start, budget, count):
    grid = np.unique(np.round(np.geomspace(start, budget, count)).astype(int))
    return [int(n) for n in grid if n >= start]


def replay(oracle, arm, seed, budget, n_checkpoints=8, band=None):
    if arm in POOL_REFUSED:
        raise ValueError(f"{arm} cannot run on a pool")
    if budget >= len(oracle.pool):
        raise ValueError("budget must be smaller than the pool")
    band = transition_band(oracle) if band is None else band
    start = time.perf_counter()
    obs = Observations(oracle, budget, oracle.dim, seed, shared=initial_pool_design(oracle, seed))
    _, metadata = run_arm(arm, obs, seed)
    order = oracle.indices(obs.x)
    rows = []
    for n in checkpoints(len(initial_pool_design(oracle, seed)), budget, n_checkpoints):
        row = dict(arm=arm, seed=seed, budget=budget, status="ok")
        row.update(score_prefix(oracle, order[:n], band))
        rows.append(row)
    rows[-1].update(seconds=time.perf_counter()-start, selected=order.tolist(),
                    metadata={k: v for k, v in metadata.items() if k in
                              ("acquisition_weights", "variation", "anisotropic", "candidates")})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore", type=pathlib.Path,
                        help="recompute scores of an existing results.json and exit")
    parser.add_argument("--pool", type=pathlib.Path)
    parser.add_argument("--target", default="gamma", choices=("gamma", "omega"))
    parser.add_argument("--arms", nargs="+", choices=POOL_ARMS, default=list(DEFAULT_POOL_ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--checkpoints", type=int, default=8)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.rescore:
        print(f"saved {rescore(args.rescore, args.output)}")
        return
    if args.pool is None or args.output is None:
        parser.error("--pool and --output are required unless --rescore is given")

    oracle, meta = load_pool(args.pool, args.target)
    band = transition_band(oracle)
    meta["band_size"] = int(band.sum())
    root = pathlib.Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "not-a-checkout"
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/"results.json"
    if path.exists():
        parser.error("output already exists; choose a new --output")
    payload = dict(pool=meta, commit=commit,
                   config={k: (str(v) if isinstance(v, pathlib.Path) else v)
                           for k, v in vars(args).items()}, rows=[])
    print(f"pool {meta['pool_size']} points, band {meta['band_size']}, labels "
          f"{meta['label_counts']}, dropped {meta['dropped']}", flush=True)
    for seed in args.seeds:
        for arm in args.arms:
            try:
                rows = replay(oracle, arm, seed, args.budget, args.checkpoints, band)
            except Exception as error:                          # noqa: BLE001
                rows = [dict(arm=arm, seed=seed, budget=args.budget, status="failed",
                             reason=f"{type(error).__name__}: {error}")]
            payload["rows"].extend(rows)
            path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            last = rows[-1]
            if last["status"] == "ok":
                print(f"[{time.strftime('%H:%M:%S')}] seed={seed} {arm:14s} N={last['n']} "
                      f"nrmse {last['nrmse']:.4f} band_cov {last['band_coverage']:.3f} "
                      f"label_acc {last['label_accuracy']:.3f} in {last['seconds']:.0f}s", flush=True)
            else:
                print(f"seed={seed} {arm} FAILED: {last['reason']}", flush=True)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
