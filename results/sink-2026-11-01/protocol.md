# Compute-sink sweep (2D and 6D) — registered protocol (2026-11-01)

Registered before the full runs. Scripts: `scripts/experiments/run_sink_2d.py`,
`scripts/experiments/run_sink_nd.py`.

## Question

Does VURS's attraction to contested (high-spread) points cost it when those
points also cost more compute? Does steering away from them -- by label,
by spread, by the production sampler's heteroscedastic GP, or by predicted
cost -- do better at equal compute?

## Design

- Cost per evaluation: `1 + (c_max - 1) * s(x)`, s the competition field that
  also sets the spread. Budget 256 compute units. `c_max` in {1, 2, 4, 8}; no
  value is privileged. The sampler sees a run's cost only after paying for it.
- **2D:** the three folded cases of the unit-cost noise sweep, s from fold
  distance (scale 0.06), noise peak 20%, floor 1%. Shared corner start.
- **6D:** Ionut proxies `itg-tem` and `itg-kbm`, argmax and softmax, gamma
  (4 cases); s from the branch growth-rate gap (`resolution.noise.competition`),
  noise peak 25%. Shared 2d+1 Latin-hypercube start.
- Seeds 0-19, paired across arms (same start, same oracle draws per seed).
- Arms (9): `space-filling`, `vwrs`, `vurs`, `vurs-n`, `vurs-a` (negative
  aleatoric), `vurs-avoid` (boundary -0.3), `vurs-prod` (ask/tell, measured
  spread), `vurs-flag` (ask/tell, converged=False where s > 0.5), `vurs-cost`
  (VURS merit / realized cost of the nearest paid run).

## Wiring checks (must pass before any result is read)

- 2D, c_max = 1: benchmark arms reproduce `results/noise-2d-2026-09-21` at the
  same case, seed and 20% peak, to floating-point identity. **Passed in pilot**
  (5 arms, two-plane-four-peaks seed 0).
- c_max = 1: `vurs-cost` reproduces `vurs` design and scores exactly. **Passed in
  2D pilot.** Re-checked on every (case, seed) in the full runs.

## Scores (at compute = 256)

- Primary: end-to-end error (reconstruction from the noisy values the sampler
  saw). 2D: the preregistered Delaunay evaluator. 6D: the confirmation C2
  reconstruction (Matern-3/2 GP with white noise, fixed for every arm) on a
  4096-point Sobol grading cloud.
- Secondary: placement error, contested-region error (2D fold band; 6D s > 0.5
  on the grading cloud), sink share (fraction of compute spent where s > 0.5),
  points placed; 6D also kNN label surprise in bits.

## Decision rule (applied separately in 2D and 6D)

For each arm A != vurs and each c_max: paired log ratio
`log(err_A / err_vurs)` over the (case, seed) pairs (60 in 2D, 80 in 6D);
Wilcoxon signed-rank, Holm-corrected over 8 arms x 4 c_max = 32 tests at family
alpha 0.05.

- **Sink penalty exists** if the vurs-minus-space-filling gap in primary error
  shrinks monotonically with c_max and the c_max = 8 vurs vs space-filling test
  differs in sign or significance from c_max = 1.
- **Avoidance helps** if `vurs-a`, `vurs-avoid` or `vurs-cost` beats `vurs`
  (median ratio < 1, Holm-significant) at some c_max > 1 without being
  Holm-significantly worse at c_max = 1.
- **Production is safe** if `vurs-prod` and `vurs-flag` are not
  Holm-significantly worse than `vurs` at any c_max.
- A recommendation needs the same verdict in 2D and 6D. A verdict in one
  dimension only is reported as a behaviour curve (error and sink share vs
  c_max), not as a recommendation.

## Not tested here

Real GENE cost ratios (c_max swept instead), replication or run extension,
held-out data (MGKDB has no spread or cost fields).
