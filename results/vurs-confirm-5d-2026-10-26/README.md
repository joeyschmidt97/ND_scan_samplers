# VURS confirmation: 5-D, reduced manifest (2026-10-26)

**Confirmation evidence.** Protocol: Vulcan wiki
`wiki/queries/2026-10-24-vurs-confirmation-protocol`, amendments of 2026-10-25,
including amendment (3). Results section there: "5-D confirmation — complete".

## What ran

- Manifest: `manifests/vurs-confirm-5d-reduced.json` (`9116f41`); entropy from the
  parent manifest `5732ecf`, so case geometries match the full design. Code `db17d56`.
- 96 blocks, 24 per stratum (0-83, 100-103, 120-123, 140-143), 5 arms
  (vurs, space-filling, gpr-var-m32, moe, sglib), N = 512.
- 480 trajectories, 0 failures, 0 incomplete blocks.
- `run1-crashed.log` (BrokenProcessPool, cause unidentified), `run2-stopped.log`
  and `run3.log` are the three launches; the runner resumes from saved files.

## Statistics

```
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_stats board \
    --input ND_scan_samplers/results/vurs-confirm-5d-2026-10-26/confirm \
    --manifest-commit 5732ecfe243f662cf406cb56eb3457d8d959c71a \
    --comparators space-filling,gpr-var-m32,moe,sglib \
    --output ND_scan_samplers/results/vurs-confirm-5d-2026-10-26/board.json
python -m ND_scan_samplers.scripts.analysis.vurs_confirm_plots \
    --results ND_scan_samplers/results/vurs-confirm-5d-2026-10-26
```

Holm over 8 core contrasts; C2-w Holm over 4.

## Result

C2 (Matern-3/2 top-decile error, N = 512), geometric-mean ratio VURS/comparator:

| Comparator | Ratio | 95% interval | Verdict |
|---|---|---|---|
| space-filling | 0.796 | 0.754-0.844 | practical superiority |
| gpr-var-m32 | 0.695 | 0.655-0.744 | practical superiority |
| sglib | 0.421 | 0.388-0.461 | practical superiority |
| moe | 0.987 | 0.943-1.034 | inconclusive |

C1 (kNN label surprise, VURS - comparator, bits): +0.022 space-filling,
+0.013 gpr-var-m32, +0.012 moe, -0.025 sglib. No superiority; every upper bound
is below the +0.05 noninferiority margin (interval condition only; no
multiplicity-controlled noninferiority test is implemented).

C2-w slopes: +0.130 space-filling, +0.129 gpr-var-m32, +0.080 moe (all pass
Holm), -0.163 sglib.

Not yet run: C2 at N = 256, Matern-1/2 and 5/2 refits, the GP-classifier C1 scorer.
