# Compute-sink sweep, 2D and 6D (2026-11-01)

Protocol: `protocol.md` (registered `d832cae`, before the full runs).
Code: `b8265cc` (sweeps), `6ef1471` (first-checkpoint fix, see below).

## What ran

- 2D: 3 folded cases x 20 seeds x 9 arms x c_max {1, 2, 4, 8} = 2160 trajectories, 0 failures.
- 6D: 4 Ionut gamma cases x 20 seeds x 9 arms x c_max {1, 2, 4, 8} = 2880 trajectories, 0 failures.
- Budget 256 compute units; cost per run `1 + (c_max - 1) * s(x)`.

Wiring checks: `vurs-cost` reproduces `vurs` exactly at c_max = 1 on 60/60 (2D)
and 80/80 (6D) pairs. 2D c_max = 1 reproduces `noise-2d-2026-09-21` (pilot,
5 arms).

**Checkpoint fix.** The first 2D run lost 783 of 2160 trajectories (87 per arm,
c_max > 1 only): rounded compute checkpoints put the first prefix below the
corners' cost, and a 3-corner Delaunay reconstruction failed. Fixed in
`6ef1471` and the failed trajectories rerun with `--resume`. Sampling was never
affected; completed trajectories are kept as run.

## Verdicts (primary: end-to-end error at compute 256, Holm over 32 tests)

| Rule | 2D | 6D | Verdict |
|---|---|---|---|
| Sink penalty exists | vurs vs space-filling ratio 1/1.215 at c=1, 1/1.292 at c=8: gap grows | tie at c=1, space-filling 1.085x worse at c=8: gap grows | **No.** VURS's lead over space-filling widens as contested runs get dearer |
| Avoidance helps | `vurs-a` worse at c=1,2,8; `vurs-avoid` tie; `vurs-cost` worse at c=2,4,8 (1.26x at c=8) | `vurs-cost` better at c=4 (0.951); others tie | **No.** No arm wins in both dimensions; `vurs-cost` reverses between them |
| Production is safe | `vurs-prod`, `vurs-flag` tie at every c_max | tie at every c_max | **Yes** |

Secondary (not part of the rule): in 2D, `vurs-flag` has higher fold-band
error at c >= 2 (1.08-1.13x) and `vurs-prod` at c = 8 (1.08x); `vurs-n` and
`vurs-prod` place worse than `vurs` at c = 1 (1.11x, 1.10x).

## Reading

Contested points cost more but carry the structure. In 2D, VURS spends 37% of
its compute on them at c = 8 (space-filling 31%) and still has 26% lower
error than space-filling. `vurs-cost` cuts that share to 13% and loses most of
the advantage (0.0435 vs 0.0342). Avoiding high-cost points is not a saving
on these surfaces.

Caveats: phenomenological proxies; contested runs return a value with Gaussian
spread, not a failed or wrong-mode answer; c_max swept, not measured from GENE.

## Files

- `2d/`, `6d/`: `results.json.gz` (gzipped: the 6D JSON exceeds GitHub's 100 MB
  file limit) and `stats-<metric>.{txt,json}`.
- `run-2d.log`, `run-2d-resume.log`, `run-6d.log`.

```
python -m ND_scan_samplers.scripts.analysis.sink_stats --input ND_scan_samplers/results/sink-2026-11-01/2d --metric end_to_end_error
```
