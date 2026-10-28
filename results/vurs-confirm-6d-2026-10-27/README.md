# VURS confirmation: 6-D extension, reduced manifest (2026-10-27)

**Confirmation evidence, separate extension** (reported apart from the 5-D primary).
Protocol: Vulcan wiki `wiki/queries/2026-10-24-vurs-confirmation-protocol`,
section "6-D confirmation extension — complete".

## What ran

- Manifest: `manifests/vurs-confirm-6d-reduced.json` (`9116f41`); entropy from the
  parent manifest `9215f5a`. Code `db17d56`.
- 96 blocks, 24 per stratum, 5 arms (vurs, space-filling, gpr-var-m32, moe, sglib), N = 512.
- 480 trajectories, 0 failures, 0 incomplete blocks.

## Commands

```
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_score \
    --input ND_scan_samplers/outputs/vurs-confirm-6d-n512/confirm \
    --manifest-commit 9215f5a5b52f0714e9b5f22a5d15c1060c1fc46f --workers 14
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_stats board \
    --input ND_scan_samplers/results/vurs-confirm-6d-2026-10-27/confirm \
    --manifest-commit 9215f5a5b52f0714e9b5f22a5d15c1060c1fc46f \
    --comparators space-filling,gpr-var-m32,moe,sglib \
    --output ND_scan_samplers/results/vurs-confirm-6d-2026-10-27/board.json
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_plots \
    --results ND_scan_samplers/results/vurs-confirm-6d-2026-10-27
```

## Result

C2 (N = 512), geometric-mean ratio VURS/comparator:

| Comparator | Ratio | 95% interval | Verdict |
|---|---|---|---|
| sglib | 0.547 | 0.511-0.583 | practical superiority |
| gpr-var-m32 | 0.846 | 0.793-0.899 | not superior (margin p 0.023 fails Holm) |
| space-filling | 0.930 | 0.885-0.981 | not superior |
| moe | 1.016 | 0.978-1.060 | inconclusive |

C1 (bits): +0.018 space-filling, +0.012 gpr-var-m32, +0.012 moe, -0.034 sglib;
every upper bound below +0.05 (interval condition only).

C2-w slopes: +0.062 space-filling, +0.063 gpr-var-m32 (pass Holm), +0.018 moe
(fails), -0.148 sglib.
