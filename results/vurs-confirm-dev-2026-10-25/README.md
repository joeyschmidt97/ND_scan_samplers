# VURS confirmation: development pilot (2026-10-24/25)

**Development evidence only.** These are development blocks (`DEV_ENTROPY`),
inspected freely while designing the study. They are not confirmation and may
not be cited as a result. The confirmation geometries do not exist yet: their
seeds come from the frozen manifest's commit hash.

Protocol: Vulcan wiki `wiki/queries/2026-10-24-vurs-confirmation-protocol`,
with the 2026-10-25 amendment (C2 as a geometric-mean ratio, bootstrap-t
tests, registered C2-w width slope).

## What ran

- 32 development blocks, 8 per stratum (blocks 0-7, 20-27, 40-47 and 60-67), each run by 7 arms to N = 256.
- That makes 224 trajectories, with no failures.
- Arms:
    - vurs
    - space-filling
    - gpr-var-m12
    - gpr-var-m32
    - gpr-var-m52
    - moe
    - sglib (historical `tol=0` adapter, no LHS start, deterministic per geometry)
- Mean wall time per trajectory with 15 workers in parallel, in seconds:

  | Arm | Mean wall time (s) |
  |---|---|
  | vurs | 85 |
  | space-filling | 1 |
  | gpr-var-m12 | 84 |
  | gpr-var-m32 | 93 |
  | gpr-var-m52 | 110 |
  | moe | 130 |
  | sglib | 741 |

## Commands

Run from the directory holding `ND_scan_samplers/`, with `PYTHONPATH` set to it.

```bash
python -m ND_scan_samplers.scripts.experiments.vurs_confirm --namespace dev --blocks 0-7,20-27,40-47,60-67 --output ND_scan_samplers/outputs/vurs-confirm --workers 15
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_score --input ND_scan_samplers/outputs/vurs-confirm/dev --workers 15
# dev/ here is a copy of outputs/vurs-confirm/dev (trajectories + scores)
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_stats board --input ND_scan_samplers/results/vurs-confirm-dev-2026-10-25/dev --output ND_scan_samplers/results/vurs-confirm-dev-2026-10-25/board.json
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_stats controls --input ND_scan_samplers/results/vurs-confirm-dev-2026-10-25/dev --sims 1000 --output ND_scan_samplers/results/vurs-confirm-dev-2026-10-25/controls.json
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_stats power --input ND_scan_samplers/results/vurs-confirm-dev-2026-10-25/dev --output ND_scan_samplers/results/vurs-confirm-dev-2026-10-25/power.json
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_plots --results ND_scan_samplers/results/vurs-confirm-dev-2026-10-25
```

## Findings (development, descriptive)

**The procedure is now valid on its null controls.** With the population exactly at the margin (1,000 simulations, 20 blocks per stratum), the false-superiority rate is:

| Endpoint | At Holm's first step (nominal 0.0042) | Unadjusted, with the CI condition (nominal 0.025) |
|---|---|---|
| C1 | 0.003-0.009 | 0.018-0.033 |
| C2 | 0.002-0.006 | 0.016-0.027 |

The superseded version (a percentile bootstrap on the ratio of mean errors) reached 0.066 at the first step and failed.

**C1, the mode map, at N = 32/64/98:**

- VURS is 0.004-0.013 bits worse than space-filling, the three GP-variance arms and MoE.
- It is 0.024 bits better than sglib.
- No contrast approaches the -0.05 margin, so superiority on C1 is implausible.
- Noninferiority (within +0.05 bits) holds against every arm.

**C2, high-growth regions, at N = 256, as the geometric-mean error ratio VURS/comparator:**

| Comparator | Ratio | 95% interval |
|---|---|---|
| space-filling | 0.61 | 0.52-0.72 |
| gpr-var-m12 | 0.59 | 0.50-0.70 |
| gpr-var-m32 | 0.59 | 0.45-0.75 |
| gpr-var-m52 | 0.60 | 0.48-0.72 |
| sglib | 0.30 | 0.19-0.46 |
| **moe** | **1.15** | **0.96-1.32** |

VURS does not beat MoE on C2.

**C2-w: VURS's peak-error advantage grows as peaks narrow.** The slope of the log error ratio on the log of the peak width l, with 95% interval:

| Comparator | Slope | 95% interval |
|---|---|---|
| space-filling | 0.38 | 0.25-0.50 |
| gpr-var-m12 | 0.44 | 0.32-0.58 |
| gpr-var-m32 | 0.51 | 0.37-0.67 |
| gpr-var-m52 | 0.49 | 0.37-0.63 |
| sglib | 0.71 | 0.50-0.93 |
| moe | 0.12 | 0.00-0.23 |

- On the narrowest peaks (l about 0.02) VURS's error is about one fifth of the GP arms'.
- On broad cases (l about 1) VURS is level with or worse than them.
- Against MoE the trend is weak. VURS is roughly level at l about 0.02 and worse on broad cases.

**Power at the planning alternative (R = 0.80, Holm's first step, dev spread)** reaches 80% only against MoE and space-filling at 40 blocks per stratum:

| Comparator | 20 blocks/stratum | 30 blocks/stratum | 40 blocks/stratum |
|---|---|---|---|
| moe | 0.50 | 0.74 | 0.86 |
| space-filling | 0.39 | 0.67 | 0.81 |
| gpr-var-m52 | 0.28 | 0.55 | 0.73 |
| gpr-var-m12 | 0.37 | 0.55 | 0.72 |
| gpr-var-m32 | 0.14 | 0.26 | 0.36 |
| sglib | 0.05 | 0.08 | 0.08 |

- sglib's low power comes from huge block-to-block spread, not from a small effect: its dev ratio is 0.30.
- C1 power is 1.00 at every size, because the block spread is about 0.01 bits.
- Caveat: the variance comes from only 8 development blocks per stratum. The x1.25 inflated-spread rows are in `power.json`.

## Figures

- `geometries.png`: four development cases per stratum, with peak width l.
- `board.png`: the 12-contrast decision table with bootstrap-t intervals.
- `width.png`: C2-w, the block error ratio against l for each comparator.
- `power.png`: C2 power against blocks per stratum.

## Files

- `dev/block<k>/<arm>.json`: paid order, values, labels, wall time and status.
- `dev/block<k>/<arm>.score.json`: C1 and C2 scores, diagnostics, and the second-grader check.
- `board.json`: decision rows, width slopes and per-block detail (stratum, l, mode widths, contrast values).
- `controls.json`: type-I rates at the margin.
- `power.json`: power rows.
