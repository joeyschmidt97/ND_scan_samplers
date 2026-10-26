"""Paired stratified bootstrap-t, practical-margin tests, width slope and power for the VURS confirmation.

Protocol: `wiki/queries/2026-10-24-vurs-confirmation-protocol` (Vulcan wiki),
with the 2026-10-25 amendment. One analytic block is one independent paired
observation, reduced to one value per contrast:

    C1: v = S_VURS - S_comp, kNN label surprise averaged over N = 32, 64, 98 (bits)
    C2: v = log(E_VURS / E_comp), Matern-3/2 top-decile error at N = 256

The estimate is the equal-stratum-weight mean of v. For C2 its exponential,
R = exp(mean v), is the geometric-mean error ratio: every case counts equally,
whatever its error scale. (The pre-amendment ratio of mean errors let the
narrow-peak stratum's large errors decide R and failed its null control.)

Practical superiority of VURS over one comparator, on one endpoint, needs both
  * the pointwise 95% bootstrap-t upper bound below the margin
    (C1: -0.05 bits; C2: R = 0.90), and
  * the one-sided practical-margin test, H0: mean v >= margin, rejected after
    Holm adjustment over the 12-contrast core family at 0.05.
Both come from one studentized (bootstrap-t) distribution: whole blocks are
resampled within strata, T* = (est* - est)/se*, with se the stratified
standard error; p = (1 + #{T* <= (est - margin)/se}) / (B + 1). B = 10,000,
frozen seed.

C2-w (registered secondary): per comparator, the OLS slope of the block C2 log
ratio on log peak width l (`ConfirmationCase.peak_width`). The log ratio falls
as VURS gains, so a positive slope means VURS's advantage grows as peaks
narrow. Stratified percentile bootstrap.

Subcommands:
    board     decision table and width slopes (dev tables are labelled DEV)
    controls  type-I error with the population exactly at the margin
    power     power at the planning alternatives (0.10 bits; R = 0.80) for
              --per-stratum blocks per stratum (20/30/40 by default),
              resampled from dev-block variance

--comparators restricts the family (the 2026-10-25 (3) amendment keeps
space-filling, gpr-var-m32, moe and sglib: Holm over 8).

    python -m ND_scan_samplers.scripts.analysis.vurs_confirm_stats power --input outputs/vurs-confirm/dev
"""
import argparse
import json
import pathlib

import numpy as np

from ND_scan_samplers.scripts.data.confirmation_cases import (DEV_ENTROPY, STRATA, ConfirmationCase,
                                                              confirmation_entropy)

REFERENCE = "vurs"
COMPARATORS = ("space-filling", "gpr-var-m12", "gpr-var-m32", "gpr-var-m52", "moe", "sglib")
C1_KEYS = ("c1_knn_32", "c1_knn_64", "c1_knn_98")
C2_KEY = "c2_m32_top_decile_{budget}"     # primary C2 at the full budget
MARGIN = dict(c1=-.05, c2=float(np.log(.90)))
ALTERNATIVE = dict(c1=-.10, c2=float(np.log(.80)))
FAMILY_ALPHA = .05
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261028
ERROR_FLOOR = 1e-8


def load(root):
    """block -> arm -> score dict, complete or not."""
    table = {}
    for path in pathlib.Path(root).glob("block*/*.score.json"):
        row = json.loads(path.read_text())
        table.setdefault(row["block"], {})[row["arm"]] = row
    return table


def block_values(table, comparators=COMPARATORS, budget=256):
    """Per-block contrast values; blocks missing any arm are reported, not dropped."""
    arms = (REFERENCE,)+tuple(comparators)
    out, incomplete = {}, []
    for block, rows in sorted(table.items()):
        if any(rows.get(a, {}).get("status") != "complete" for a in arms):
            incomplete.append(block)
            continue
        c1 = {a: float(np.mean([rows[a][k] for k in C1_KEYS])) for a in arms}
        c2 = {a: max(rows[a][C2_KEY.format(budget=budget)], ERROR_FLOOR) for a in arms}
        out[block] = dict(stratum=next(iter(rows.values()))["stratum"],
                          c1={c: c1[REFERENCE]-c1[c] for c in comparators},
                          c2={c: float(np.log(c2[REFERENCE]/c2[c])) for c in comparators})
    return out, incomplete


def _groups(strata):
    return [np.flatnonzero(strata == s) for s in STRATA if (strata == s).any()]


def _estimate(values, groups):
    """Equal-stratum-weight mean and its stratified standard error, one per row of `values`."""
    means = np.stack([values[..., g].mean(axis=-1) for g in groups])
    variances = np.stack([values[..., g].var(axis=-1, ddof=1)/len(g) for g in groups])
    return means.mean(axis=0), np.sqrt(variances.sum(axis=0))/len(groups)


def contrast(strata, values, endpoint, rng, size=BOOTSTRAP):
    """Estimate, bootstrap-t interval and practical-margin p-value for one contrast."""
    strata, values = np.asarray(strata), np.asarray(values, float)
    groups = _groups(strata)
    estimate, se = _estimate(values, groups)
    rows = np.concatenate([g[rng.integers(0, len(g), (size, len(g)))] for g in groups], axis=1)
    offsets = np.cumsum([0]+[len(g) for g in groups])
    boot_groups = [np.arange(offsets[i], offsets[i+1]) for i in range(len(groups))]
    boot_est, boot_se = _estimate(values[rows], boot_groups)
    t = (boot_est-estimate)/np.maximum(boot_se, 1e-12)
    q_low, q_high = np.quantile(t, [.025, .975])
    low, high = estimate-q_high*se, estimate-q_low*se
    margin = MARGIN[endpoint]
    p = (1+np.sum(t <= (estimate-margin)/max(se, 1e-12)))/(size+1)
    out = dict(estimate=float(estimate), se=float(se), low=float(low), high=float(high),
               p_margin=float(p), ci_below_margin=bool(high < margin))
    if endpoint == "c2":
        out.update(ratio=float(np.exp(estimate)), ratio_low=float(np.exp(low)), ratio_high=float(np.exp(high)))
    return out


def holm(pvalues, alpha=FAMILY_ALPHA):
    """Holm step-down rejections, in input order."""
    order = np.argsort(pvalues)
    reject = np.zeros(len(pvalues), bool)
    for rank, i in enumerate(order):
        if pvalues[i] > alpha/(len(pvalues)-rank):
            break
        reject[i] = True
    return reject


def decide(values, rng, comparators=COMPARATORS):
    """The core family: C1 and C2 of VURS against each comparator (Holm over 2 x comparators)."""
    blocks = sorted(values)
    strata = [values[b]["stratum"] for b in blocks]
    rows = [dict(endpoint=e, comparator=c,
                 **contrast(strata, [values[b][e][c] for b in blocks], e, rng))
            for e in ("c1", "c2") for c in comparators]
    for r, rejected in zip(rows, holm(np.array([r["p_margin"] for r in rows]))):
        r["holm_reject"] = bool(rejected)
        r["superior"] = bool(rejected and r["ci_below_margin"])
    return rows


def width_slopes(values, widths, rng, size=BOOTSTRAP, comparators=COMPARATORS):
    """C2-w: OLS slope of the C2 log ratio on log peak width, per comparator."""
    blocks = sorted(values)
    strata = np.array([values[b]["stratum"] for b in blocks])
    x = np.log([widths[b] for b in blocks])
    groups = _groups(strata)
    rows = np.concatenate([g[rng.integers(0, len(g), (size, len(g)))] for g in groups], axis=1)
    out = []
    for c in comparators:
        y = np.array([values[b]["c2"][c] for b in blocks])
        slope = float(np.polyfit(x, y, 1)[0])
        xs, ys = x[rows], y[rows]
        xc, yc = xs-xs.mean(axis=1, keepdims=True), ys-ys.mean(axis=1, keepdims=True)
        boot = np.sum(xc*yc, axis=1)/np.maximum(np.sum(xc**2, axis=1), 1e-12)
        low, high = np.quantile(boot, [.025, .975])
        out.append(dict(comparator=c, slope=slope, low=float(low), high=float(high),
                        p_nonpositive=float((1+np.sum(boot <= 0))/(size+1))))
    return out


# -- synthetic checks of the procedure ------------------------------------------

def shifted_population(strata, values, target, inflate=1.):
    """Dev block values moved so their *population* mean sits exactly at `target`.

    Each block keeps its value's offset from its stratum mean (times `inflate`);
    the equal-stratum-weight mean is moved to the target. Studies are then
    resampled from this population, so a study's estimate scatters around the
    target as a real study's would.
    """
    values = np.asarray(values, float).copy()
    groups = _groups(strata)
    for g in groups:
        mean = values[g].mean()
        values[g] = mean+inflate*(values[g]-mean)
    return values-_estimate(values, groups)[0]+target


def rejection_rate(strata, values, endpoint, target, per_stratum, threshold, sims, size, seed, inflate=1.):
    """Share of synthetic studies with the CI below the margin and p at or below `threshold`."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    population = shifted_population(strata, values, target, inflate)
    groups = [np.flatnonzero(strata == s) for s in STRATA]
    hits = 0
    for _ in range(sims):
        pick = np.concatenate([g[rng.integers(0, len(g), per_stratum)] for g in groups])
        result = contrast(strata[pick], population[pick], endpoint, rng, size)
        hits += result["ci_below_margin"] and result["p_margin"] <= threshold
    return hits/sims


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("board", "controls", "power"))
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest-commit", help="frozen manifest commit; required for confirm runs")
    parser.add_argument("--sims", type=int, default=400)
    parser.add_argument("--size", type=int, default=2000, help="bootstrap draws inside each simulation")
    parser.add_argument("--output", help="write the JSON result here")
    parser.add_argument("--comparators", default=",".join(COMPARATORS))
    parser.add_argument("--per-stratum", default="20,30,40", help="power only: blocks per stratum to simulate")
    args = parser.parse_args()
    comparators = tuple(args.comparators.split(","))
    if not set(comparators) <= set(COMPARATORS):
        parser.error(f"unknown comparator in {comparators}")
    family = 2*len(comparators)
    dev = pathlib.Path(args.input).name == "dev"
    if not dev and not args.manifest_commit:
        parser.error("confirm runs need --manifest-commit")
    entropy = DEV_ENTROPY if dev else confirmation_entropy(args.manifest_commit)
    table = load(args.input)
    dims = {row.get("dim", 2) for rows in table.values() for row in rows.values()}
    if len(dims) != 1:
        parser.error(f"one dimension per run directory, found {sorted(dims)}")
    dim = dims.pop()
    budgets = {row.get("budget", 256) for rows in table.values() for row in rows.values()}
    if len(budgets) != 1:
        parser.error(f"one budget per run directory, found {sorted(budgets)}")
    budget = budgets.pop()
    values, incomplete = block_values(table, comparators, budget=budget)
    blocks = sorted(values)
    strata = [values[b]["stratum"] for b in blocks]
    result = dict(namespace="dev" if dev else "confirm", dim=dim, budget=budget, blocks=len(values),
                  incomplete=incomplete, comparators=list(comparators), family=family)
    if args.command == "board":
        cases = {b: ConfirmationCase(entropy, b, dim) for b in blocks}
        widths = {b: cases[b].peak_width() for b in blocks}
        result["rows"] = decide(values, np.random.default_rng(BOOTSTRAP_SEED), comparators)
        result["width_slopes"] = width_slopes(values, widths, np.random.default_rng(BOOTSTRAP_SEED+1),
                                              comparators=comparators)
        result["blocks_detail"] = [dict(block=b, stratum=values[b]["stratum"], peak_width=widths[b],
                                        mode_widths=cases[b].peak_widths(), c1=values[b]["c1"],
                                        c2_log_ratio=values[b]["c2"]) for b in blocks]
        for r in result["rows"]+result["width_slopes"]:
            print(("DEV " if dev else "")+json.dumps(r))
    else:
        settings = ([(20, MARGIN, "null at margin", 1.)] if args.command == "controls" else
                    [(int(n), ALTERNATIVE, "planning alternative", f)
                     for n in args.per_stratum.split(",") for f in (1., 1.25)])
        rows = []
        for per_stratum, targets, label, inflate in settings:
            for endpoint in ("c1", "c2"):
                for k, comparator in enumerate(comparators):
                    seed = BOOTSTRAP_SEED+1000*per_stratum+10*k+(endpoint == "c2")
                    common = (strata, [values[b][endpoint][comparator] for b in blocks], endpoint,
                              targets[endpoint], per_stratum)
                    row = dict(per_stratum=per_stratum, endpoint=endpoint, comparator=comparator,
                               scenario=label, inflate=inflate,
                               # one alternative among the nulls: Holm's first step, alpha/family
                               reject_bonferroni=rejection_rate(*common, FAMILY_ALPHA/family, args.sims,
                                                                args.size, seed, inflate),
                               # all alternatives true: Holm's last step, the CI bound binds
                               reject_unadjusted=rejection_rate(*common, FAMILY_ALPHA, args.sims,
                                                                args.size, seed, inflate))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        result["rows"] = rows
    if args.output:
        pathlib.Path(args.output).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
