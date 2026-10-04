# Fold-floor rerun of the anisotropic arms (2D) provenance

- **Run date:** 2026-10-03
- **Source branch:** `codex/benchmark-2d`
- **Source commit:** `8050a3f` (`Floor the isotropic share of the fold metric shape`)
- **Platform:** local Windows Python environment
- **Result status:** complete; 6,072 of 6,072 expected trajectory rows succeeded

## Command

```powershell
.\.venv\Scripts\python.exe -m benchmark2d --cases two-plane-isolated two-plane-four-peaks three-plane-three-peaks two-plane-asymmetric --arms vwrs-m vurs-m --seeds 0 1 2 --budgets 32 64 128 256 --native-test-size 1024 --output results\2d-fold-floor-2026-10-03
```

## What this tests

Whether keeping at least 25% of the identity in the rank-one fold shape (`FOLD_FLOOR = 0.25`) removes the fold-band starvation found in `results/2d-anisotropic-2026-10-02` (`82036f9`). Only `vwrs-m`/`vurs-m` changed; every other arm is deterministic and is compared from that run, same cases, seeds and reference sets. Expected rows: `2 arms × 4 cases × 3 seeds × 253 = 6,072`.

## Reading at N=256 (pooled RMS over cases and seeds)

| Arm | Global | Fold band | Fold-adjacent peaks | Isolated peaks |
|---|---:|---:|---:|---:|
| vwrs-k (isotropic control, 82036f9) | 0.0128 | 0.0205 | 0.0203 | 0.0209 |
| vurs-k (isotropic control, 82036f9) | 0.0125 | 0.0165 | 0.0279 | 0.0291 |
| vwrs-m, no floor (82036f9) | 0.0181 | 0.0332 | 0.0198 | 0.0173 |
| vwrs-m, floor 0.25 | 0.0135 | 0.0234 | 0.0179 | 0.0193 |
| vurs-m, no floor (82036f9) | 0.0134 | 0.0228 | 0.0264 | 0.0288 |
| vurs-m, floor 0.25 | 0.0130 | 0.0190 | 0.0278 | 0.0293 |

The floor recovers most of the fold band (vwrs-m band share on `two-plane-four-peaks` 0.185 to 0.255, against 0.266 for vwrs-k) and cuts vwrs-m global error 25%. The isotropic curvature arms still lead the fold band; the anisotropic peak gains (8-12%) are inside three-seed spread. In two dimensions anisotropy can trade at most one direction against one; the higher-dimensional cases, with weak and inert axes, are the real test.
