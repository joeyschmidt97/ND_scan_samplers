"""Apply the registered decision rule to the confirmatory VURS-profile runs.

Protocol (registered before the runs, vault note
2026-10-04-vurs-exploration-profiles, "Confirmatory comparison -- protocol"):

- primary endpoints per seed, each the mean over campaign checkpoints
  N = 23, 38, 61, 98: E1 per-mode classify-then-regress error (lower is
  better) and E2 mode-label accuracy (higher is better);
- paired per-seed differences from plain `vurs`: mean, 95% bootstrap CI
  (10,000 resamples), Wilcoxon signed-rank p, Holm across 6 candidates x 2
  primary endpoints within each pool;
- a profile replaces `vurs` only if, in both pools, the E1 difference's upper
  CI is below +5% of vurs's mean E1 (non-inferior) and E2 is superior (Holm
  p < 0.05, CI excluding zero).

    python scripts/confirm_profiles.py --root results/confirm-2026-10-05
"""
import argparse
import glob
import json
import pathlib

import numpy as np
from scipy.stats import wilcoxon

CAMPAIGN = (23, 38, 61, 98)
REFERENCE = "vurs"
CANDIDATES = ("vurs-default", "vurs-modeid", "vurs-coverage", "vurs-boundary",
              "vurs-avoid", "vurs-nu25")
BASELINES = ("space-filling", "gpr-var", "vwrs")
MARGIN = .05
PRIMARY = {"E1": ("ctr_macro_nrmse", "lower"), "E2": ("label_accuracy", "higher")}
SECONDARY = {
    "worst-mode sensitivity": ("mode_sensitivity_worst", CAMPAIGN),
    "transition budget share": ("transition_budget_share", CAMPAIGN),
    "transition detection": ("transition_detection", CAMPAIGN),
    "fill P95": ("tf_fill_p95", CAMPAIGN),
    "observed VWFD P95": ("tf_observed_vwfd_p95", CAMPAIGN),
    "E1 at N=256": ("ctr_macro_nrmse", (256,)),
    "E2 at N=256": ("label_accuracy", (256,)),
}


def load(pool_dir):
    rows = []
    for path in sorted(glob.glob(str(pool_dir/"seed*"/"results.json"))):
        rows += json.loads(pathlib.Path(path).read_text())["rows"]
    return rows


def endpoint(rows, arm, key, budgets):
    """{seed: mean of key over the budget checkpoints} for completed trajectories."""
    out = {}
    for seed in sorted({r["seed"] for r in rows}):
        values = [r[key] for r in rows if r["arm"] == arm and r["seed"] == seed
                  and r["status"] == "ok" and r["n"] in budgets]
        if len(values) == len(budgets) and all(v is not None for v in values):
            out[seed] = float(np.mean(values))
    return out


def paired(candidate, reference, rng):
    seeds = sorted(set(candidate) & set(reference))
    d = np.array([candidate[s]-reference[s] for s in seeds])
    boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.
    return dict(n=len(d), mean=float(d.mean()), lo=float(np.quantile(boot, .025)),
                hi=float(np.quantile(boot, .975)), p=p)


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(pvalues)-rank)*pvalues[index]))
        adjusted[index] = running
    return adjusted


def analyse(pool_dir, rng):
    rows = load(pool_dir)
    failures = [r for r in rows if r["status"] != "ok"]
    arms = (REFERENCE,) + CANDIDATES + BASELINES
    means = {}
    for arm in arms:
        means[arm] = {name: endpoint(rows, arm, key, CAMPAIGN)
                      for name, (key, _) in PRIMARY.items()}
        means[arm].update({name: endpoint(rows, arm, key, budgets)
                           for name, (key, budgets) in SECONDARY.items()})
    tests = {}
    for arm in CANDIDATES + BASELINES:
        tests[arm] = {name: paired(means[arm][name], means[REFERENCE][name], rng)
                      for name in list(PRIMARY) + list(SECONDARY)}
    flat = [(arm, name) for arm in CANDIDATES for name in PRIMARY]
    adjusted = holm(np.array([tests[a][n]["p"] for a, n in flat]))
    for (arm, name), p in zip(flat, adjusted):
        tests[arm][name]["p_holm"] = float(p)
    ref_e1 = float(np.mean(list(means[REFERENCE]["E1"].values())))
    verdict = {}
    for arm in CANDIDATES:
        e1, e2 = tests[arm]["E1"], tests[arm]["E2"]
        verdict[arm] = dict(noninferior_E1=e1["hi"] < MARGIN*ref_e1,
                            superior_E2=e2["p_holm"] < .05 and e2["lo"] > 0)
    return dict(pool=pool_dir.name, n_rows=len(rows), failures=len(failures),
                seeds=sorted({r["seed"] for r in rows}), reference_E1=ref_e1,
                means={a: {k: (float(np.mean(list(v.values()))) if v else None,
                               float(np.std(list(v.values()))) if v else None, len(v))
                           for k, v in m.items()} for a, m in means.items()},
                tests=tests, verdict=verdict)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261005)
    pools = sorted(p for p in args.root.iterdir() if p.is_dir())
    report = {p.name: analyse(p, rng) for p in pools}
    qualifying = [arm for arm in CANDIDATES
                  if all(report[p]["verdict"][arm]["noninferior_E1"]
                         and report[p]["verdict"][arm]["superior_E2"] for p in report)]
    if qualifying:
        gain = {arm: np.mean([report[p]["tests"][arm]["E2"]["mean"] for p in report])
                for arm in qualifying}
        decision = max(gain, key=gain.get)
    else:
        decision = REFERENCE
    out = dict(protocol=__doc__, decision=decision, qualifying=qualifying, pools=report)
    (args.root/"confirmatory_analysis.json").write_text(json.dumps(out, indent=2))
    for name, r in report.items():
        print(f"=== {name}: {r['n_rows']} rows, {r['failures']} failed, seeds {len(r['seeds'])}, "
              f"vurs E1 {r['reference_E1']:.3f}")
        for arm in CANDIDATES + BASELINES:
            t = r["tests"][arm]
            v = r["verdict"].get(arm, {})
            print(f"  {arm:14s} dE1 {t['E1']['mean']:+.3f} [{t['E1']['lo']:+.3f},{t['E1']['hi']:+.3f}]"
                  f" p={t['E1'].get('p_holm', t['E1']['p']):.3g} | dE2 {t['E2']['mean']:+.4f}"
                  f" [{t['E2']['lo']:+.4f},{t['E2']['hi']:+.4f}] p={t['E2'].get('p_holm', t['E2']['p']):.3g}"
                  + (f" | noninf {v['noninferior_E1']} sup {v['superior_E2']}" if v else ""))
    print(f"DECISION: {decision}  (qualifying: {qualifying})")


if __name__ == "__main__":
    main()
