"""Batched VURS/VWRS on the MGKDB pools against sequential vurs (registered 2026-10-10).

Protocol, registered before any batched run on the pools:

Question. How do batched VWRS/VURS placements (k points per acquisition,
placed before any of their values return) compare with sequential `vurs`
at the board's budgets? This is a measurement; it does not choose a default.

Arms.
- Batched: all 16 `strategies.BATCH_ARMS` = {vwrs, vurs} x k in {5, 10, 15,
  20} x {greedy (no suffix): re-rank after each pick with the fill distance
  updated; rank (suffix r): top k of one acquisition evaluation}.
- Sequential references: `vurs` (REFERENCE) and `vwrs` (batch size 1).
- The board's COMPARATORS (`surprise_board.COMPARATORS`). Space-filling is
  value-blind, so a batch size does not change its placements; it is one row.
Pools, seeds, budget, design: `surprise_board.POOLS` (hatch_pscans_global,
hatch_pscans3), seeds 0-19, budget 256, the shared Latin-hypercube start
snapped to the pool (`pool.initial_pool_design`, 9 points), checkpoints
`surprise_board.CHECKPOINTS` = 9, 15, 23, 38, 61, 98, 159, 256 (`pool.replay`
with 8 checkpoints), scorers unchanged (`pool.score_prefix`,
`surprise.surprise_scores` via `surprise_board.score_one`).

Reuse. Sequential arms are not rerun if their stored trajectories
(results/confirm-2026-10-05, results/board20-2026-10-07) and surprise scores
(results/surprise-2026-10-07) were produced under this protocol. Checked
before registration: vurs, space-filling and gpr-var, seeds 0-1, both pools,
rerun at HEAD reproduce the stored `selected` order exactly and the pool
sha256 matches. Batched trajectories go to results/pool-batch-2026-10-10.

Mid-batch checkpoints. Inside a batch the k points are paid in pick order
(greedy: the order chosen; rank: best first). A checkpoint N that falls
inside a batch is scored on the first N points in that order, exactly as a
sequential arm's prefix; this is the primary view. Secondary view: each
endpoint at the last completed batch at or below N, n_c = 9 + k*floor((N-9)/k)
(for k = 15 and 20 at N = 23 this is the 9-point start, the same for every
arm), compared with vurs scored on its own first n_c points (same number of
paid runs).

Primary endpoints (as the board): per seed, the mean over CAMPAIGN
checkpoints N = 23, 38, 61, 98 of
- S1 label surprise, bits (`s_label_bits`), margin 0.05 bits;
- S2 growth-rate surprise, nats (`s_gamma_nats`), margin 0.10 nats;
lower is better.

Comparison. For each batched arm, paired per-seed difference
d = arm - vurs (positive: the batched arm is more surprised, i.e. worse;
note the sign is the reverse of surprise_board's vurs - comparator), mean,
95% bootstrap CI (10,000 resamples of seeds), Wilcoxon signed-rank p, Holm
across 16 batched arms x 2 primary endpoints = 32 tests within each pool.

Verdict (`surprise_board.verdict` applied to d, so it describes the batched
arm relative to vurs): "faster" (Holm p < .05 and CI below 0), "slower"
(Holm p < .05 and CI above 0), "same speed" (CI inside +-margin), "not
slower" (CI upper end below +margin), else "inconclusive". A batched arm
"keeps pace with vurs" only if it is faster, same speed or not slower on S1
and S2 in both pools.

Secondary endpoints (paired vs vurs, bootstrap CI and uncorrected Wilcoxon,
no verdicts): campaign means of `ctr_macro_nrmse`,
`region_design_band_nrmse`, `region_peak_nrmse`, `label_accuracy` (higher
is better); `ctr_macro_nrmse` at N = 61 and at N = 98; S1 and S2 at N = 256;
S1, S2 and `ctr_macro_nrmse` in the last-completed-batch view.
Descriptive contrasts (uncorrected): greedy - rank at each (base, k); each
vwrs-b<k> - sequential vwrs. Comparator rows are reported with the same
d = arm - vurs statistics in their own Holm family (9 x 2) for context; the
board's registered verdicts for them stand.
Cost: placement wall time per run (`run_arm` only, scoring excluded) and the
number of acquisitions (GP fits for vurs-b<k>: about (256-9)/k against 247).
vurs and vwrs are re-timed under the same runner and load as the batched
arms (their orders are checked against the stored ones); the comparators'
times are the stored `seconds`, which include checkpoint scoring and were
taken under other load, so they are context only.

    python -m ND_scan_samplers.scripts.experiments.pool_batch --output results/pool-batch-2026-10-10
    python -m ND_scan_samplers.scripts.analysis.pool_batch_board score --root results/pool-batch-2026-10-10
    python -m ND_scan_samplers.scripts.analysis.pool_batch_board analyse --root results/pool-batch-2026-10-10
"""
from ND_scan_samplers.scripts.analysis.surprise_board import CAMPAIGN, CHECKPOINTS, COMPARATORS, POOLS, PRIMARY
from ND_scan_samplers.src.strategies import BATCH_ARMS

REFERENCE = "vurs"
SEQUENTIAL = ("vurs", "vwrs")
SEEDS = tuple(range(20))
BUDGET = 256
START = 9
STORED_RUNS = ("results/confirm-2026-10-05", "results/board20-2026-10-07")
STORED_SCORES = "results/surprise-2026-10-07"
SECONDARY = {
    "E1 ctr_macro_nrmse": ("ctr_macro_nrmse", CAMPAIGN),
    "band error": ("region_design_band_nrmse", CAMPAIGN),
    "peak error": ("region_peak_nrmse", CAMPAIGN),
    "E2 label_accuracy (higher is better)": ("label_accuracy", CAMPAIGN),
    "E1 at N=61": ("ctr_macro_nrmse", (61,)),
    "E1 at N=98": ("ctr_macro_nrmse", (98,)),
    "S1 at N=256": ("s_label_bits", (256,)),
    "S2 at N=256": ("s_gamma_nats", (256,)),
}


def completed(n, k):
    """Points paid at the last completed batch at or below checkpoint n."""
    return n if k == 1 or n <= START else START + k*((n-START)//k)


def batch_size(arm):
    return int(arm.split("-b")[1].rstrip("r")) if arm in BATCH_ARMS else 1


def main():
    raise NotImplementedError("analysis is implemented after the runs; see the protocol above")


if __name__ == "__main__":
    main()
