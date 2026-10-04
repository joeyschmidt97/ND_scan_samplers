# 6D vwrs-m rerun with the fixed fold ratio, provenance

- **Run date:** 2026-10-03
- **Source branch:** `codex/benchmark-2d`
- **Source commit:** `c4efd2c` (`Hold the fold shape's anisotropy ratio fixed across dimension`)
- **Result status:** complete; 48 of 48 trajectories, 480 checkpoint rows, 0 failures
- **Command:** as `results/6d-anisotropic-2026-10-03`, with `--arms vwrs-m --output results/6d-fold-ratio-2026-10-03`

## What this tests

Whether holding the fold shape at 7:1 across:along (identity share 0.5 at d = 6, against 0.25 and 19:1 in stage 1) restores the transition band in 6D, and whether the isolated-bump targeting survives. space-filling and vwrs are deterministic and compared from stage 1 (`bd12609`).

## Reading at N=512

| Group | Band lift: vwrs / vwrs-m 19:1 / vwrs-m 7:1 | Bump lift: vwrs / 19:1 / 7:1 | VWFD P95: vwrs / 7:1 |
|---|---|---|---|
| ITG-TEM bumped | 1.69 / 1.04 / 1.55 | 3.25 / 21.6 / 5.50 | 0.2936 / 0.2933 |
| ITG-KBM bumped | 2.54 / 2.04 / 2.29 | 1.25 / 1.88 / 1.25 | 0.4642 / 0.4675 |
| ITG-TEM | 1.63 / 1.19 / 1.49 | — | 0.3119 / 0.3133 |
| ITG-KBM | 2.56 / 1.75 / 2.25 | — | 0.4658 / 0.4670 |

The fixed ratio restores most of the band and brings local-linear global error back to the vwrs level (ITG-TEM bumped 0.0697 vs vwrs 0.0689; 19:1 was 0.0730), but the bump targeting falls from 21.6x to 5.5x and the bump-error gain from 8% to 3%. Per case, vwrs-m at 7:1 is within about 3% of vwrs on VWFD P95 and local-linear error. The metric trades band against bumps along a continuum; at this budget no setting improves on isotropic VWRS.
