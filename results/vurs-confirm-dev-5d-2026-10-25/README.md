# VURS confirmation: 5-D development pilot (2026-10-25)

**Development evidence only.** Development blocks (`DEV_ENTROPY`, 5-D seed key),
inspected freely while designing the study. Not confirmation; not citable.
The confirmation geometries do not exist yet: their seeds come from the frozen
manifest's commit hash.

Why 5-D: the 2-D pilot (`../vurs-confirm-dev-2026-10-25`) is kept as guidance;
the target problem is 5-6-D, so 5-D is the primary confirmation dimension and
6-D a separately frozen extension. Protocol: Vulcan wiki
`wiki/queries/2026-10-24-vurs-confirmation-protocol`, amendments of 2026-10-25.

## What changed from the 2-D pilot

- Geometries: `_*_nd` generators in `scripts/data/confirmation_cases.py`, same four strata.
    - Peaks are anisotropic in a random orthonormal frame.
    - A narrow peak is narrow (sigma 0.03-0.08) along one or two directions and broad (0.15-0.35) along the rest. A peak narrow in all five directions would occupy ~1e-9 of the box and be unresolvable by any arm at N = 256.
    - Tilts are scaled by sqrt(2/d) to keep the 2-D range over the box.
    - Island shares (1-5%) use the d-ball volume.
- `sglib` runs at `tol=1e-6` per direction, the value in sg_lib's own example drivers (`adapt_code_full.py`). The 2-D pilot used the historical `tol=0` adapter.
- Everything else (arms, N = 256, C1/C2 scorers, bootstrap-t, Holm, margins) is unchanged.

## What ran

32 development blocks, 8 per stratum (0-7, 20-27, 40-47, 60-67), 7 arms to N = 256: 224 trajectories, 0 failures (28 of them first as the timing pilot, `pilot.log`). Mean wall time per trajectory with 15 workers in parallel, in seconds:

| Arm | Mean wall time (s) |
|---|---|
| vurs | 193 |
| space-filling | 1 |
| gpr-var-m12 | 180 |
| gpr-var-m32 | 194 |
| gpr-var-m52 | 206 |
| moe | 309 |
| sglib (tol 1e-6) | 185 |

160 blocks (1,120 trajectories) project to about 4 h on this machine.

## Commands

Run from the directory holding `ND_scan_samplers/`, with `PYTHONPATH` set to it.

```bash
python -m ND_scan_samplers.scripts.experiments.vurs_confirm --namespace dev --dim 5 --blocks 0-7,20-27,40-47,60-67 --output ND_scan_samplers/outputs/vurs-confirm-5d --workers 15
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_score --input ND_scan_samplers/outputs/vurs-confirm-5d/dev --workers 15
# dev/ here is a copy of outputs/vurs-confirm-5d/dev (trajectories + scores)
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_stats board --input ND_scan_samplers/results/vurs-confirm-dev-5d-2026-10-25/dev --output ND_scan_samplers/results/vurs-confirm-dev-5d-2026-10-25/board.json
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_stats controls --input ND_scan_samplers/results/vurs-confirm-dev-5d-2026-10-25/dev --sims 1000 --output ND_scan_samplers/results/vurs-confirm-dev-5d-2026-10-25/controls.json
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_stats power --input ND_scan_samplers/results/vurs-confirm-dev-5d-2026-10-25/dev --output ND_scan_samplers/results/vurs-confirm-dev-5d-2026-10-25/power.json
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_plots --results ND_scan_samplers/results/vurs-confirm-dev-5d-2026-10-25
```

## Findings (development, descriptive)

**Null controls pass.** With the population at the margin (1,000 simulations, 20 blocks per stratum), the false-superiority rate at Holm's first step (nominal 0.0042) is 0.001-0.005 for C1 and 0.000-0.008 for C2. Unadjusted rates are 0.015-0.028.

**C1, mode map:** VURS is 0.015-0.028 bits worse than space-filling, the GP-variance arms and MoE, and 0.027 bits better than sglib (interval crosses 0). Superiority is implausible; noninferiority (+0.05 bits) holds against every arm.

**C2, high-growth regions, N = 256, geometric-mean error ratio VURS/comparator:**

| Comparator | Ratio | 95% interval | Practical superiority (dev) |
|---|---|---|---|
| space-filling | 0.83 | 0.75-0.93 | no (CI crosses 0.90) |
| gpr-var-m12 | 0.73 | 0.66-0.82 | yes |
| gpr-var-m32 | 0.78 | 0.70-0.89 | CI clears, Holm does not |
| gpr-var-m52 | 0.76 | 0.69-0.86 | CI clears, Holm does not |
| sglib | 0.57 | 0.49-0.65 | yes |
| **moe** | **0.98** | **0.91-1.06** | no |

By stratum (dev geometric means; 8 blocks each):

| Stratum | median l | vs space-filling | vs GP-var (m12/m32/m52) | vs MoE | vs sglib |
|---|---|---|---|---|---|
| smooth | 0.73 | 0.71 | 0.57/0.64/0.61 | 0.88 | 0.54 |
| curved | 1.00 | 1.23 | 1.11/1.18/1.20 | 1.13 | 0.41 |
| narrow | 0.05 | 0.82 | 0.79/0.78/0.76 | 1.02 | 0.71 |
| islands | 0.11 | 0.68 | 0.58/0.63/0.61 | 0.90 | 0.65 |

- The advantage over the GP-variance arms is about 25% overall, smaller than in 2-D (about 40%).
- The 2-D narrow-peak effect (ratio about 0.24) largely disappears in 5-D (about 0.78).
- VURS is worse on curved switches (1.1-1.2).
- It still does not beat MoE.

**C2-w width slopes:** 0.08-0.09 against space-filling and GP-variance (intervals touch 0), about 0 against MoE, -0.17 against sglib. In 5-D the advantage no longer tracks peak width the way it did in 2-D (slopes 0.4-0.5).

**Power at the planning alternative (R = 0.80, Holm's first step, dev spread)** is much higher than in 2-D, because block-to-block spread is smaller:

| Comparator | 20 blocks/stratum | 30 blocks/stratum | 40 blocks/stratum |
|---|---|---|---|
| space-filling | 0.85 | 0.98 | 1.00 |
| gpr-var-m12 | 0.78 | 0.97 | 0.99 |
| gpr-var-m32 | 0.70 | 0.91 | 0.99 |
| gpr-var-m52 | 0.74 | 0.92 | 0.99 |
| moe | 0.99 | 1.00 | 1.00 |
| sglib | 0.56 | 0.85 | 0.94 |

With spread inflated x1.25, 40 blocks per stratum still gives 0.76-1.00. C1 power is 1.00 at every size.

## Figures

- `geometries.png`: four development cases per stratum as 2-D slices: an interior peak along its two most curved directions, otherwise the box centre across the mode switch.
- `board.png`, `width.png`, `power.png`: as in the 2-D pilot.

## Files

As in the 2-D pilot: `dev/block<k>/<arm>.json` and `.score.json`, `board.json`, `controls.json`, `power.json`, logs.
