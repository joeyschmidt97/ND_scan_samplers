# Anisotropic VWRS/VURS 2D benchmark provenance

- **Run date:** 2026-10-02
- **Source branch:** `codex/benchmark-2d`
- **Source commit:** `9c6f4df` (`Add the 2D isolated-bump case and anisotropic arms`)
- **Platform:** local Windows Python environment
- **Result status:** complete; 30,360 of 30,360 expected trajectory rows succeeded

## Command

```powershell
.\.venv\Scripts\python.exe -m benchmark2d --cases two-plane-isolated two-plane-four-peaks three-plane-three-peaks two-plane-asymmetric --arms grid triangles gpr-var gpr-blend vwrs vurs vwrs-k vurs-k vwrs-m vurs-m --seeds 0 1 2 --budgets 32 64 128 256 --native-test-size 1024 --output results\2d-anisotropic-2026-10-02
```

Every integer paid-point count from 4 through 256 is scored; `--budgets` sets the report and secondary-scorer caps.

## Scope

Ten arms, four folded surfaces, three geometry seeds, 16,384 independent reference points per test. `two-plane-isolated` is new in `9c6f4df`: one fold-adjacent peak pair plus one isolated anisotropic bump per region, 0.30 from the fold, elongated parallel to the fold in region 0 and perpendicular in region 1.

- `vwrs`/`vurs` — the original 2D arms, Delaunay-gradient variation floored at 0.1.
- `vwrs-k`/`vurs-k` — isotropic `h**2 * curvature`, the dimension-free indicator benchmarknd uses.
- `vwrs-m`/`vurs-m` — region-split local resolution metric; branch labels from the oracle at paid points only.

The -k and -m pairs differ only in the metric shape, so their gap isolates the anisotropy.

## Validation

- Expected rows: `10 arms × 4 cases × 3 seeds × 253 point counts = 30,360`; successful 30,360, failed 0.
- New score fields `peak_fold_*` and `peak_isolated_*` come from `Surface.peak_masks`; all pre-existing fields are computed as before.

## Reading at N=256 (pooled RMS of normalized RMS over cases and seeds)

| Arm | Global | Fold band | Fold-adjacent peaks | Isolated peaks |
|---|---:|---:|---:|---:|
| triangles | 0.0147 | 0.0188 | 0.0355 | 0.0399 |
| gpr-blend | 0.0186 | 0.0206 | 0.0374 | 0.0434 |
| vwrs | 0.0160 | 0.0202 | 0.0375 | 0.0308 |
| vwrs-k | 0.0128 | 0.0205 | 0.0203 | 0.0209 |
| vurs-k | 0.0125 | 0.0165 | 0.0279 | 0.0291 |
| vwrs-m | 0.0181 | 0.0332 | 0.0198 | 0.0173 |
| vurs-m | 0.0134 | 0.0228 | 0.0264 | 0.0288 |

Curvature beats the Delaunay gradient indicator. The anisotropic metric improves isolated peaks but raises fold-band error: on `two-plane-isolated`, `vwrs-m` puts 8.5% of its points in the fold band against 13.4% for uniform sampling. The rank-one fold shape scores gaps along the fold near zero, while the Delaunay error depends on the containing simplex rather than the nearest point. Three seeds only; peak differences under about 30% are within seed spread.
