# 6D anisotropic VWRS benchmark, stage 1, provenance

- **Run date:** 2026-10-03
- **Source branch:** `codex/benchmark-2d`
- **Source commit:** `40251b9` (code as of `8050a3f`, fold floor 0.25)
- **Platform:** local Windows Python environment, 12 workers
- **Result status:** complete; 144 of 144 trajectories, 1,440 checkpoint rows, 0 failures

## Command

```bash
.venv/Scripts/python.exe -m benchmarknd --cases <8 Ionut cases> <same 8>-bumped --arms space-filling vwrs vwrs-m --seeds 0 1 2 --budget-6d 512 --workers 12 --output results/6d-anisotropic-2026-10-03
```

Checkpoints at N = 13, 20, 29, 44, 67, 100, 150, 226, 340, 512.

## What this tests

Whether the region-split metric (`vwrs-m`) beats isotropic curvature VWRS (`vwrs`, its exact S = I control above 2D) in six dimensions, where weak and inert axes should let anisotropy pay, and whether it finds isolated bumps without starving the transition. Stage 2 (`vurs`, `vurs-m`, GP 50/50 blend) is not in this run.

## Scoring caution

The `error`/`band_error`/`peak_error` fields use the thin-plate RBF evaluator, declared secondary since 2026-09-20 because its error rises as a design concentrates. The fit-free spine (`vwfd_*`, `nonlinear_*`, `fill_*`) is primary. A local-linear (kNN tricube) reconstruction was computed post hoc as a diagnostic; it is not stored here.

## Reading at N=512 (pooled over seeds)

| Group | VWFD P95 space-filling | vwrs | vwrs-m |
|---|---:|---:|---:|
| ITG-TEM | 0.347 | 0.312 | 0.316 |
| ITG-KBM | 0.525 | 0.466 | 0.470 |
| ITG-TEM bumped | 0.323 | 0.294 | 0.296 |
| ITG-KBM bumped | 0.528 | 0.464 | 0.470 |

vwrs-m ties vwrs on the spine. It moves points: 21.6x the space-filling share onto the ITG-TEM bumps (vwrs 3.3x) but only 1.04x in the ITG-TEM live band (vwrs 1.69x). The fold floor weakens with dimension: across:along = ((1-a)d + a)/a is 7:1 in 2D but 19:1 in 6D at a = 0.25. Local-linear global error: space-filling lowest, vwrs 5-11% higher, vwrs-m up to 27% higher on ITG-TEM; bump error 8% lower for vwrs-m on ITG-TEM, slightly higher on ITG-KBM (bump masks are ~0.6% of the volume, so noisy).
