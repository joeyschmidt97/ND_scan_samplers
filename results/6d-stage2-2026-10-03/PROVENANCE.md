# 6D anisotropic benchmark, stage 2 (GP-based arms), provenance

- **Run dates:** 2026-10-03 21:40 to 2026-10-04 05:32
- **Source branch:** `codex/benchmark-2d`
- **Source commit:** `fbb98e0` (code `c4efd2c`, fold ratio 7:1)
- **Platform:** local Windows Python environment, 12 workers
- **Result status:** complete; 144 of 144 trajectories, 1,440 checkpoint rows, 0 failures

## Command

```bash
.venv/Scripts/python.exe -m benchmarknd --cases <8 Ionut cases> <same 8>-bumped --arms vurs vurs-m gpr-u50-g50 --seeds 0 1 2 --budget-6d 512 --workers 12 --output results/6d-stage2-2026-10-03
```

## What this tests

Whether GP uncertainty changes the stage-1 verdict: does `vurs-m` (region-split metric plus Matern-1/2 GP uncertainty) beat `vurs` (isotropic curvature plus the same GP), and how both compare with the established Matern-3/2 GP 50/50 uncertainty/gradient blend. Stage-1 arms are in `results/6d-anisotropic-2026-10-03` and `results/6d-fold-ratio-2026-10-03`.

## Reading at N=512 (pooled RMS over seeds)

Native GP prediction error (normalized RMS on 65,536 reference points; independent of the secondary RBF evaluator):

| Group | vurs | vurs-m | gpr-u50-g50 |
|---|---:|---:|---:|
| ITG-TEM | 0.0519 | 0.0508 | 0.1017 |
| ITG-KBM | 0.0116 | 0.0117 | 0.0237 |
| ITG-TEM bumped | 0.0445 | 0.0443 | 0.0914 |
| ITG-KBM bumped | 0.0133 | 0.0130 | 0.0305 |

VURS halves the GP blend's native error on every group; vurs-m matches vurs within 2%. On the fit-free spine vurs and vurs-m tie the VWRS arms (VWFD P95 within about 2%), while the GP blend is worse than space-filling (ITG-KBM 0.582 vs 0.525) because it concentrates 10-11x the space-filling share in the ITG-KBM live band and leaves holes elsewhere (fill P95 0.48 vs 0.36). Caveat: vurs uses a Matern-1/2 GP and the blend a Matern-3/2 GP, so the VURS-versus-blend gap mixes acquisition rule with kernel.
