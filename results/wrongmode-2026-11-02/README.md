# Wrong-mode sweep, 6D (2026-11-02)

Protocol: `protocol.md` (registered `72d0cd5`, before the full run).
Code: `9adece7` (sweep), `c2e1df6` (stats over flip-peak cells).

## What ran

4 Ionut gamma cases x 20 seeds x 9 arms x flip_peak {0.25, 0.5} x c_max {1, 4}
= 2880 trajectories, 0 failures. Control (flip_peak = 0) is
`sink-2026-11-01/6d`, reproduced exactly by this code on 16/16 pilot
trajectories. `vurs-cost` reproduces `vurs` at c_max = 1 on 160/160 pairs.

## Verdicts (primary: end-to-end error at compute 256, Holm over 32 tests)

| Rule | Result | Verdict |
|---|---|---|
| Wrong-mode penalty exists | space-filling vs vurs: 0.999, 1.009, 1.024, 1.040 (all tie) | **No** |
| Avoidance helps | `vurs-a`, `vurs-avoid`, `vurs-cost` tie in every cell | **No** |
| Production is safe | `vurs-prod`, `vurs-flag` tie in every cell | **Yes** |

The sink-sweep verdict stands with wrong-branch returns included.

## Reading

Wrong-branch returns hurt every arm about equally. Geometric-mean end-to-end
error, flip_peak 0 / 0.5:

| | c = 1 | c = 4 |
|---|---|---|
| vurs | 0.0151 / 0.0188 (+25%) | 0.0173 / 0.0204 (+18%) |
| space-filling | 0.0151 / 0.0189 (+25%) | 0.0181 / 0.0208 (+15%) |

VURS does not collect more wrong runs than space-filling: wrong share 9.3% vs
10.1% at flip_peak 0.5 (4.6-4.7% vs 5.0-5.1% at 0.25). In 6D its contested
sampling is local to the transition, not piled on the tie. `vurs-cost` cuts the
wrong share to 8.4% at c = 4 without an error gain. The single-seed pilot that
suggested a VURS-specific penalty did not hold up.

Secondary: on placement error (true values at the same designs) space-filling
is Holm-significantly worse than `vurs` in every cell (1.04-1.07x), so VURS's
designs are better; the wrong values it is handed erase that edge end to end.
Space-filling has lower label surprise than every VURS arm
(0.88-0.90x, as in the sink sweep); `vurs-cost` also lower at c = 4 (0.96-0.97x).
`vwrs` is worse than `vurs` everywhere.

Caveats: 6D proxies only (the 2D folds have no subdominant branch); wrong runs
report the subdominant growth rate cleanly, not a failed run; flip rates swept,
not measured from GENE.

## Files

- `results.json.gz`, `stats-<metric>.{txt,json}` for end-to-end, placement,
  contested-region error and label bits; `run.log`.

```
python -m ND_scan_samplers.scripts.analysis.sink_stats --input ND_scan_samplers/results/wrongmode-2026-11-02 --metric end_to_end_error
```
