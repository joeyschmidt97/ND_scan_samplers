# Batched VURS/VWRS on the MGKDB pools (2026-10-10)

Registered protocol: module docstring of `scripts/analysis/pool_batch_board.py`
(committed in 5d22d6b before any batched run). 16 batched arms
(`strategies.BATCH_ARMS`: vwrs/vurs x k = 5, 10, 15, 20 x greedy / rank `r`)
against sequential `vurs`, 20 seeds, both Hatch pools, budget 256, board
checkpoints and scorers. Sequential arms and comparators are the stored board
trajectories (results/confirm-2026-10-05, results/board20-2026-10-07) and
surprise scores (results/surprise-2026-10-07), reused after checks:
vurs, space-filling, gpr-var seeds 0-1 rerun at HEAD reproduce their stored
order; all 80 vurs/vwrs timing reruns reproduce their stored order
(`<pool>/sequential_timing.json`); the stored vurs seed-0 surprise scores are
reproduced by the current scorer (`scores/<pool>_reuse_check.json`). The
analysis also reproduces the board's vurs - comparator differences exactly.

## Commands

    # runs (3dc8357, 12 workers, 1372 s)
    PYTHONPATH=C:\Users\joesc\git python -m ND_scan_samplers.scripts.experiments.pool_batch \
        --output results/pool-batch-2026-10-10 --time-sequential
    # surprise scores of the batched trajectories + last-completed-batch prefixes
    PYTHONPATH=C:\Users\joesc\git python -m ND_scan_samplers.scripts.analysis.pool_batch_board score \
        --root results/pool-batch-2026-10-10
    # registered analysis -> analysis.json, tables below
    PYTHONPATH=C:\Users\joesc\git python -m ND_scan_samplers.scripts.analysis.pool_batch_board analyse \
        --root results/pool-batch-2026-10-10

## Files

- `<pool>/seed<k>/results.json`: board-shaped rows per batched arm; the last row
  holds `selected` (pick order), `placement_seconds`, `scoring_seconds`,
  `acquisitions`, `metadata.batch_sizes`.
- `<pool>/sequential_timing.json`: vurs/vwrs placement time under the same load.
- `scores/<pool>/<arm>_seed<k>.json`: S1/S2 per checkpoint (batched arms) and at
  the last-completed-batch prefixes (batched arms, vurs, vwrs).
- `analysis.json`: primary family (16 x 2, Holm per pool), comparator family
  (9 x 2, context only), secondaries, last-completed-batch view, greedy - rank
  and vwrs-b<k> - vwrs contrasts, means, wall times.
- `run.log`: runner console output.

## Primary results

d = arm - vurs, campaign mean over N = 23, 38, 61, 98, paired over 20 seeds;
positive = the arm is more surprised (worse). Verdicts describe the arm.
E1 = `ctr_macro_nrmse`. "median s" = placement wall time per run (scoring
excluded), 12 concurrent workers; "acq" = acquisitions per run.

### hatch_pscans_global  (d = arm - vurs; positive = arm worse; verdict describes the arm)

| arm | S1 bits | d S1 [95% CI] | verdict | S2 nats | d S2 [95% CI] | verdict | E1 N=61 | E1 N=98 | median s | acq |
|---|---|---|---|---|---|---|---|---|---|---|
| vurs | 1.464 | -- | reference | 1.358 | -- | reference | 0.636 | 0.610 | 155.5 | 247 |
| vwrs | 1.577 | +0.113 [+0.076, +0.151] | slower* | 1.599 | +0.241 [-0.180, +0.650] | inconclusive* | 0.741 | 0.849 | 6.1 | 247 |
| vwrs-b5 | 1.496 | +0.033 [-0.001, +0.069] | inconclusive | 1.010 | -0.348 [-0.712, +0.019] | not slower | 0.691 | 0.684 | 1.6 | 50 |
| vwrs-b5r | 1.590 | +0.126 [+0.091, +0.162] | slower | 1.104 | -0.254 [-0.764, +0.334] | inconclusive | 0.831 | 0.683 | 1.6 | 50 |
| vwrs-b10 | 1.493 | +0.029 [-0.005, +0.067] | inconclusive | 1.437 | +0.079 [-0.277, +0.464] | inconclusive | 0.678 | 0.639 | 0.8 | 25 |
| vwrs-b10r | 1.598 | +0.134 [+0.089, +0.178] | slower | 1.082 | -0.276 [-0.681, +0.142] | inconclusive | 0.850 | 0.741 | 0.8 | 25 |
| vwrs-b15 | 1.478 | +0.014 [-0.023, +0.051] | inconclusive | 1.286 | -0.072 [-0.415, +0.269] | inconclusive | 0.685 | 0.673 | 0.5 | 17 |
| vwrs-b15r | 1.638 | +0.174 [+0.125, +0.225] | slower | 1.177 | -0.181 [-0.636, +0.289] | inconclusive | 0.907 | 0.697 | 0.5 | 17 |
| vwrs-b20 | 1.474 | +0.010 [-0.024, +0.046] | same speed | 1.284 | -0.074 [-0.425, +0.263] | inconclusive | 0.748 | 0.702 | 0.4 | 13 |
| vwrs-b20r | 1.656 | +0.192 [+0.140, +0.248] | slower | 1.181 | -0.177 [-0.624, +0.268] | inconclusive | 0.895 | 0.792 | 0.4 | 13 |
| vurs-b5 | 1.464 | -0.000 [-0.033, +0.034] | same speed | 1.353 | -0.005 [-0.490, +0.500] | inconclusive | 0.647 | 0.682 | 44.4 | 50 |
| vurs-b5r | 1.532 | +0.068 [+0.028, +0.111] | inconclusive | 1.293 | -0.065 [-0.485, +0.324] | inconclusive | 0.602 | 0.581 | 45.1 | 50 |
| vurs-b10 | 1.462 | -0.001 [-0.030, +0.027] | same speed | 1.613 | +0.254 [-0.278, +0.917] | inconclusive | 0.623 | 0.654 | 24.6 | 25 |
| vurs-b10r | 1.580 | +0.116 [+0.073, +0.163] | slower | 0.918 | -0.440 [-0.820, -0.060] | not slower | 0.758 | 0.615 | 23.2 | 25 |
| vurs-b15 | 1.465 | +0.001 [-0.032, +0.034] | same speed | 1.593 | +0.234 [-0.323, +0.904] | inconclusive | 0.675 | 0.621 | 16.8 | 17 |
| vurs-b15r | 1.604 | +0.140 [+0.098, +0.184] | slower | 0.955 | -0.403 [-0.693, -0.131] | not slower | 0.837 | 0.628 | 17.5 | 17 |
| vurs-b20 | 1.472 | +0.008 [-0.025, +0.041] | same speed | 1.502 | +0.144 [-0.411, +0.810] | inconclusive | 0.713 | 0.645 | 13.7 | 13 |
| vurs-b20r | 1.636 | +0.172 [+0.122, +0.225] | slower | 1.030 | -0.329 [-0.636, -0.041] | not slower | 0.987 | 0.650 | 14.3 | 13 |
| space-filling | 1.463 | -0.001 [-0.029, +0.028] | same speed* | 1.110 | -0.248 [-0.730, +0.267] | inconclusive* | 0.742 | 0.725 | 3.8+ | -- |
| gpr-var | 1.492 | +0.028 [-0.011, +0.065] | inconclusive* | 1.261 | -0.097 [-0.549, +0.333] | inconclusive* | 0.571 | 0.561 | 218.2+ | -- |
| gpr-grad | 1.784 | +0.320 [+0.270, +0.378] | slower* | 1.529 | +0.171 [-0.399, +0.833] | inconclusive* | 1.718 | 2.143 | 108.6+ | -- |
| gpr-u50-g50 | 1.692 | +0.228 [+0.173, +0.293] | slower* | 1.345 | -0.014 [-0.531, +0.534] | inconclusive* | 1.125 | 1.290 | 116.2+ | -- |
| gpr-m05-blend | 1.796 | +0.332 [+0.249, +0.418] | slower* | 1.834 | +0.475 [-0.218, +1.251] | inconclusive* | 1.287 | 1.869 | 108.9+ | -- |
| vwrs-m | 1.576 | +0.112 [+0.074, +0.154] | slower* | 1.602 | +0.243 [-0.117, +0.598] | inconclusive* | 0.811 | 0.797 | 16.7+ | -- |
| vurs-m | 1.470 | +0.006 [-0.016, +0.033] | same speed* | 1.395 | +0.037 [-0.023, +0.099] | same speed* | 0.607 | 0.561 | 116.6+ | -- |
| moe | 1.469 | +0.005 [-0.027, +0.038] | same speed* | 0.765 | -0.593 [-0.921, -0.294] | faster* | 0.859 | 0.871 | 121.3+ | -- |

### hatch_pscans3  (d = arm - vurs; positive = arm worse; verdict describes the arm)

| arm | S1 bits | d S1 [95% CI] | verdict | S2 nats | d S2 [95% CI] | verdict | E1 N=61 | E1 N=98 | median s | acq |
|---|---|---|---|---|---|---|---|---|---|---|
| vurs | 1.252 | -- | reference | 2.057 | -- | reference | 0.177 | 0.158 | 218.4 | 247 |
| vwrs | 1.258 | +0.006 [-0.032, +0.046] | same speed* | 2.336 | +0.279 [-0.300, +0.884] | inconclusive* | 0.186 | 0.159 | 5.3 | 247 |
| vwrs-b5 | 1.262 | +0.011 [-0.028, +0.047] | same speed | 2.020 | -0.038 [-0.628, +0.442] | inconclusive | 0.185 | 0.162 | 1.2 | 50 |
| vwrs-b5r | 1.276 | +0.024 [-0.022, +0.071] | inconclusive | 1.791 | -0.266 [-0.841, +0.185] | inconclusive | 0.208 | 0.168 | 1.1 | 50 |
| vwrs-b10 | 1.265 | +0.013 [-0.025, +0.047] | same speed | 1.984 | -0.073 [-0.482, +0.385] | inconclusive | 0.179 | 0.156 | 0.6 | 25 |
| vwrs-b10r | 1.354 | +0.103 [+0.062, +0.141] | slower | 1.792 | -0.265 [-0.842, +0.191] | inconclusive | 0.210 | 0.172 | 0.6 | 25 |
| vwrs-b15 | 1.289 | +0.037 [-0.000, +0.069] | inconclusive | 1.882 | -0.175 [-0.694, +0.214] | inconclusive | 0.188 | 0.164 | 0.4 | 17 |
| vwrs-b15r | 1.412 | +0.160 [+0.117, +0.205] | slower | 1.838 | -0.219 [-0.730, +0.216] | inconclusive | 0.221 | 0.175 | 0.4 | 17 |
| vwrs-b20 | 1.303 | +0.051 [+0.014, +0.087] | inconclusive | 2.020 | -0.038 [-0.571, +0.340] | inconclusive | 0.181 | 0.160 | 0.3 | 13 |
| vwrs-b20r | 1.454 | +0.203 [+0.150, +0.259] | slower | 1.838 | -0.219 [-0.722, +0.202] | inconclusive | 0.235 | 0.175 | 0.3 | 13 |
| vurs-b5 | 1.270 | +0.019 [-0.026, +0.063] | inconclusive | 2.101 | +0.044 [-0.639, +0.648] | inconclusive | 0.182 | 0.158 | 45.9 | 50 |
| vurs-b5r | 1.278 | +0.026 [-0.008, +0.060] | inconclusive | 2.057 | -0.000 [-0.633, +0.590] | inconclusive | 0.190 | 0.161 | 44.0 | 50 |
| vurs-b10 | 1.288 | +0.036 [-0.004, +0.077] | inconclusive | 2.038 | -0.020 [-0.437, +0.279] | inconclusive | 0.185 | 0.160 | 23.0 | 25 |
| vurs-b10r | 1.331 | +0.079 [+0.034, +0.122] | inconclusive | 2.044 | -0.013 [-0.569, +0.432] | inconclusive | 0.205 | 0.166 | 21.9 | 25 |
| vurs-b15 | 1.283 | +0.032 [-0.010, +0.075] | inconclusive | 2.245 | +0.188 [-0.364, +0.647] | inconclusive | 0.181 | 0.160 | 16.7 | 17 |
| vurs-b15r | 1.402 | +0.150 [+0.103, +0.199] | slower | 2.109 | +0.052 [-0.319, +0.420] | inconclusive | 0.214 | 0.171 | 17.9 | 17 |
| vurs-b20 | 1.281 | +0.030 [-0.012, +0.069] | inconclusive | 2.273 | +0.216 [-0.355, +0.706] | inconclusive | 0.184 | 0.159 | 13.4 | 13 |
| vurs-b20r | 1.416 | +0.165 [+0.115, +0.213] | slower | 2.163 | +0.105 [-0.281, +0.485] | inconclusive | 0.213 | 0.176 | 13.9 | 13 |
| space-filling | 1.304 | +0.052 [+0.011, +0.096] | inconclusive* | 2.278 | +0.221 [-0.113, +0.538] | inconclusive* | 0.180 | 0.159 | 2.1+ | -- |
| gpr-var | 1.313 | +0.062 [+0.016, +0.105] | inconclusive* | 2.687 | +0.630 [+0.124, +1.375] | inconclusive* | 0.193 | 0.163 | 117.6+ | -- |
| gpr-grad | 1.503 | +0.251 [+0.204, +0.298] | slower* | 2.033 | -0.024 [-0.652, +0.542] | inconclusive* | 0.388 | 0.321 | 116.8+ | -- |
| gpr-u50-g50 | 1.368 | +0.117 [+0.075, +0.168] | slower* | 2.029 | -0.028 [-0.325, +0.273] | inconclusive* | 0.276 | 0.217 | 114.4+ | -- |
| gpr-m05-blend | 1.639 | +0.388 [+0.297, +0.487] | slower* | 2.502 | +0.445 [-0.369, +1.339] | inconclusive* | 0.412 | 0.393 | 97.2+ | -- |
| vwrs-m | 1.256 | +0.005 [-0.031, +0.041] | same speed* | 2.169 | +0.112 [-0.449, +0.725] | inconclusive* | 0.183 | 0.161 | 9.6+ | -- |
| vurs-m | 1.256 | +0.004 [-0.011, +0.018] | same speed* | 2.113 | +0.055 [-0.130, +0.259] | inconclusive* | 0.183 | 0.156 | 94.4+ | -- |
| moe | 1.268 | +0.017 [-0.016, +0.052] | inconclusive* | 1.582 | -0.475 [-0.886, -0.161] | not slower* | 0.183 | 0.158 | 97.3+ | -- |

* comparator verdicts: own Holm family (9 x 2), context only; the board's verdicts stand. + stored seconds incl. scoring, other load.

No batched arm "keeps pace with vurs" under the registered rule: S2 is
inconclusive for almost every arm (bootstrap CIs about +-0.3 to +-0.7 nats
against a 0.10 margin), so the rule cannot be met by any arm at 20 seeds.

## Notes

- Mid-batch checkpoints are scored on the pick-order prefix (primary). The
  last-completed-batch view (`completed_batch_view`, vurs scored at the same
  number of paid runs) is secondary.
- On hatch_pscans3 the campaign-mean E1 of the rank arms is dominated by one
  seed at N = 23 (E1 up to 111 for vwrs-b10r, 42 for vurs-b15r/b20r): the
  clustered first batch leaves a mode whose per-mode interpolant extrapolates.
  E1 at N = 61 and 98 is unaffected.
- For k >= 15 the first 14 picks after the 9-point start coincide across
  batch sizes within a mode (greedy or rank), so N = 23 is identical for
  b15 and b20 of the same mode.
- Space-filling is value-blind; batching does not change it.
- Comparator wall times (+) are the stored `seconds` (placement + checkpoint
  scoring, other load) and are context only.
