"""Common scorers for the VURS confirmation trajectories (C1 and C2).

Protocol: `wiki/queries/2026-10-24-vurs-confirmation-protocol` (Vulcan wiki).
Every arm is graded on the same block grading cloud with the same models fitted
only to its paid prefix; no arm's own surrogate enters a score.

C1, broad mode map, at N = 32, 64, 98:
    volume-weighted held-out label surprise, bits/point: the mean of
    -log2 p(true mode | x) over the uniform grading cloud. Primary posterior is
    Dirichlet-kNN (Jeffreys 1/2, k = min(2d+1, N)); the GP classifier of
    `benchmarknd.surprise` is the second common scorer. Diagnostics: per-mode
    recall of the kNN argmax and fill-distance P95.
C2, high-growth regions, at the full budget (N = 512 primary, 256 secondary;
    the 256-budget dev pilots used 256 and 159):
    a fixed GP reconstruction (Matern-3/2 primary; 1/2 and 5/2 robustness) is
    fitted to the prefix; RMSE on each true mode's top growth-rate decile is
    divided by the case scale and the mode errors are averaged with the modes'
    true volume shares as weights. Global RMSE and peak simple regret are
    reported beside it.
The primary C1 and C2 scores are repeated on the second grading scramble as an
integration-resolution check.

    python -m ND_scan_samplers.scripts.analysis.vurs_confirm_score --input outputs/vurs-confirm/dev
"""
import argparse
import json
import os
import pathlib
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial import cKDTree
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from ND_scan_samplers.scripts.data.confirmation_cases import (DEV_ENTROPY, ConfirmationCase,
                                                              confirmation_entropy, grader_cloud, role_seed)
from ND_scan_samplers.tests.benchmarks.benchmarknd.surprise import gp_label_posterior, label_posterior

C1_CHECKPOINTS = (32, 64, 98)
C2_CHECKPOINTS = {256: (256, 159), 512: (512, 256)}     # budget -> (primary, secondary)
KERNELS = (1.5, .5, 2.5)            # primary first
TOP = .9                            # top-decile threshold within each mode


def label_bits(px, pl, query, truth, n_modes, classifier):
    post = (label_posterior(px, pl, query, n_modes) if classifier == "knn"
            else gp_label_posterior(px, pl, query, n_modes))
    return -np.log2(post[np.arange(len(truth)), truth]), post.argmax(axis=1)


def fit_reconstruction(x, y, nu, seed):
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (1e-3, 1e3))*Matern([.3]*x.shape[1], (1e-2, 1e2), nu=nu)
        + WhiteKernel(1e-8, (1e-10, 1e-3)),
        normalize_y=True, n_restarts_optimizer=2, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(x, y)
    return gp


def c2_scores(yhat, truth, region, n_modes, scale):
    errors, masses = [], []
    for m in range(n_modes):
        inside = region == m
        masses.append(inside.mean())
        if not inside.any():
            errors.append(None)
            continue
        top = inside & (truth >= np.quantile(truth[inside], TOP))
        errors.append(float(np.sqrt(np.mean((yhat[top]-truth[top])**2))/scale))
    weighted = sum(w*e for w, e in zip(masses, errors) if e is not None)/sum(
        w for w, e in zip(masses, errors) if e is not None)
    return dict(top_decile=float(weighted), top_decile_modes=errors,
                global_rmse=float(np.sqrt(np.mean((yhat-truth)**2))/scale),
                simple_regret=float((truth.max()-truth[int(np.argmax(yhat))])/scale))


def score_trajectory(entropy, path):
    run = json.loads(path.read_text())
    out = dict(block=run["block"], arm=run["arm"], status=run["status"], n_paid=run["n"],
               seconds=run["seconds"], stratum=run["case"]["stratum"], dim=run["case"].get("dim", 2), budget=run["budget"])
    if run["status"] != "complete":
        return out
    dim = run["case"].get("dim", 2)
    case = ConfirmationCase(entropy, run["block"], dim)
    x, y, labels = np.array(run["x"]), np.array(run["y"]), np.array(run["labels"])
    seed = role_seed(entropy, run["block"], "predictor", dim)
    for replicate in (0, 1):
        grid = grader_cloud(entropy, run["block"], replicate, dim=dim)
        truth, region = case(grid), case.region(grid)
        tag = "" if replicate == 0 else "_g2"
        for n in C1_CHECKPOINTS:
            bits, guess = label_bits(x[:n], labels[:n], grid, region, case.n_modes, "knn")
            out[f"c1_knn_{n}{tag}"] = float(bits.mean())
            if replicate:
                continue
            out[f"recall_{n}"] = [float((guess[region == m] == m).mean()) if (region == m).any() else None
                                  for m in range(case.n_modes)]
            out[f"fill_p95_{n}"] = float(np.quantile(cKDTree(x[:n]).query(grid)[0], .95))
            out[f"c1_gpc_{n}"] = float(label_bits(x[:n], labels[:n], grid, region,
                                                  case.n_modes, "gpc")[0].mean())
        for n in C2_CHECKPOINTS[run["budget"]]:
            for nu in (KERNELS[:1] if replicate else KERNELS):
                yhat = fit_reconstruction(x[:n], y[:n], nu, seed).predict(grid)
                for key, value in c2_scores(yhat, truth, region, case.n_modes, case.scale).items():
                    out[f"c2_m{int(2*nu)}2_{key}_{n}{tag}"] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="<output>/<namespace> of vurs_confirm")
    parser.add_argument("--manifest-commit", help="frozen manifest commit; required for confirm runs")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)-1))
    args = parser.parse_args()
    root = pathlib.Path(args.input)
    entropy = confirmation_entropy(args.manifest_commit) if args.manifest_commit else DEV_ENTROPY
    if root.name == "confirm" and not args.manifest_commit:
        parser.error("confirm runs need --manifest-commit")
    paths = [p for p in sorted(root.glob("block*/*.json")) if not p.name.endswith(".score.json")]
    jobs = [p for p in paths if not p.with_suffix(".score.json").exists()]
    print(f"{len(jobs)} of {len(paths)} trajectories to score", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(score_trajectory, entropy, p): p for p in jobs}
        for future in as_completed(futures):
            path = futures[future]
            path.with_suffix(".score.json").write_text(json.dumps(future.result()))
            print(f"scored {path.parent.name}/{path.stem}", flush=True)


if __name__ == "__main__":
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    main()
