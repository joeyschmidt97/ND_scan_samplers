# Wrong-mode sweep (6D) — registered protocol

Registered before the full run. Script: `scripts/experiments/run_wrongmode_nd.py`.
Follows `results/sink-2026-11-01`, whose verdict (no sink penalty, avoidance
does not help, production safe) assumed a contested run returns the right
answer with a Gaussian spread.

## Question

A contested GENE run can instead settle on the subdominant branch and report
its growth rate as a clean, converged number. VURS samples contested regions
on purpose. Does that make it collect enough wrong-branch values to lose to
space-filling, and does steering away from contested points (by label, spread,
heteroscedastic GP or cost) then pay?

## Design

- Everything as in the 6D sink sweep: Ionut `itg-tem` and `itg-kbm`, argmax and
  softmax, gamma (4 cases); spread peak 25%; cost `1 + (c_max - 1) * s(x)`;
  budget 256 compute units; shared 2d+1 Latin-hypercube start; 9 arms; seeds
  0-19, paired; C2 reconstruction on the 4096-point Sobol grading cloud.
- New: each evaluation returns, with probability `flip_peak * s(x)`, the
  subdominant branch's growth rate at the floor spread (1%) and the subdominant
  label. Flip draws use their own generator.
- Cells: `flip_peak` in {0.25, 0.5} x `c_max` in {1, 4}. 0.5 makes a tied pair a
  coin flip. `flip_peak = 0` is the sink sweep itself (`sink-2026-11-01/6d`),
  used as the control and not rerun.
- What arms see of a wrong run: its value and floor spread (`vurs-prod` trusts
  it), its label (`vurs-avoid`). `vurs-flag` flags from s(x) as before.

## Wiring check (passed before registration)

`flip_peak = 0` reproduces `sink-2026-11-01/6d` designs and end-to-end errors
exactly: 16/16 trajectories (`ionut-itg-kbm-argmax-gamma` and
`ionut-itg-tem-softmax-gamma`, seed 0, `vurs-a`, `vurs-avoid`, `vurs-flag`,
`vurs-cost`, c_max 1 and 4) plus `space-filling`, `vurs`, `vurs-prod` at c_max 4.

## Scores (at compute = 256)

- Primary: end-to-end error (reconstruction from the values the sampler saw,
  wrong ones included), graded on the noiseless dominant-branch proxy.
- Secondary: placement error, contested-region error (s > 0.5), label surprise
  in bits, wrong share (fraction of paid points that settled on the wrong
  branch), sink share, points placed.

## Decision rule

For each arm A != vurs and each of the 4 cells: paired log ratio
`log(err_A / err_vurs)` over 80 (case, seed) pairs; Wilcoxon signed-rank,
Holm-corrected over 8 arms x 4 cells = 32 tests, family alpha 0.05.

- **Wrong-mode penalty exists** if `space-filling` beats `vurs` (median ratio
  < 1, Holm-significant) in any cell.
- **Avoidance helps** if `vurs-a`, `vurs-avoid` or `vurs-cost` beats `vurs`
  (median ratio < 1, Holm-significant) in any cell.
- **Production is safe** if `vurs-prod` and `vurs-flag` are not
  Holm-significantly worse than `vurs` in any cell.
- 6D only. If no rule fires against VURS, the sink-sweep verdict stands with
  wrong-mode returns included. If the penalty or avoidance rule fires, the
  sink-sweep verdict is restricted to right-branch returns, reported as a
  behaviour curve (error and wrong share vs flip_peak, with the flip_peak = 0
  control), and a 2D counterpart is needed before any recommendation changes.

## Not tested here

2D (the folded surfaces have no subdominant branch to return); runs that fail
outright; real GENE wrong-mode rates; replication or run extension.
