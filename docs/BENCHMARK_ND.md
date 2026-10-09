# 5-D and 8-D benchmark

Folded manifolds in five and eight dimensions where each mode is deliberately
non-responsive in several axes, mimicking a mode that ignores whole coordinate
ranges. Four cases per dimension: one mode with one peak, two modes with an
axis-aligned fold, two modes with a dense fold, and two modes with disjoint
active subspaces and an extra peak. Strength patterns are declared per mode
(`S` strong, `M` moderate, `W` weak, `.` inert); 5-D modes ignore two to three
axes and 8-D modes five. An inert axis changes the value by exactly zero.

Three things forced changes to the 2-D setup:

| 2-D choice | Why it fails above two dimensions | Replacement |
|---|---|---|
| Delaunay reconstruction | at d=8, N=512 it costs 346 s and leaves 85% of the test set outside the convex hull | one fixed thin-plate-spline RBF for every arm |
| `triangles`, MoE's triangle and bilinear-grid experts | Delaunay again; a bilinear grid needs `side**d` coefficients | `triangles` dropped; MoE now gates k-NN linear, quadratic-ANOVA ridge and GP |
| `grid`, 2^d charged corners, 0.06 duplicate radius, 1,024 candidates | 256 corners at d=8; the radius excludes nothing when neighbours sit 0.4 apart | Sobol space-filling baseline, shared `2d+1` Latin-hypercube start, radius `0.1*N^(-1/d)`, `512*d` candidates |

Scoring happens at ten log-spaced checkpoints: at d=8 the scorer costs about
32 minutes per trajectory when every integer N is scored and 19 seconds at ten,
and the trajectory is identical either way.

```bash
python -m ND_scan_samplers.tests.benchmarks.benchmarknd --cases 5d-m2-rotated --seeds 0 --output outputs/nd-run/5d-m2-rotated-s0
python -m ND_scan_samplers.tests.benchmarks.benchmarknd.collect --run outputs/nd-run --dims 5 8 --output results
python -m ND_scan_samplers.tests.benchmarks.benchmarknd.checks          # geometry and scorer validation
python src/nersc/nd_progress.py --logs $TEMP/ndlogs --results outputs/nd-run --watch
python -m ND_scan_samplers.tests.benchmarks.benchmarknd.tables      # strength tables into results/<dim>d/figures
```

`collect` refuses to write a dimension until every case, seed and arm finished,
so a partial run cannot be read as a complete one.

### 5-D result

**Adaptive placement loses to plain space-filling, and the gradient-weighted
acquisitions that won in 2-D are the worst performers here.** Combined
normalized RMS at N = 512, four cases and three seeds, common RBF scorer:

| Arm | Combined | m1-p1-anis | m2-aligned | m2-disjoint | m2-rotated |
|---|---:|---:|---:|---:|---:|
| Space-filling | 0.0543 | 0.0497 | 0.0544 | 0.0468 | 0.0622 |
| GP uncertainty | 0.0560 | 0.0513 | 0.0562 | 0.0483 | 0.0661 |
| Mixture of experts | 0.0600 | 0.0583 | 0.0595 | 0.0515 | 0.0699 |
| GP unc/grad 70/30 | 0.0621 | 0.0672 | 0.0584 | 0.0468 | 0.0711 |
| GP unc/grad 50/50 | 0.0729 | 0.0872 | 0.0673 | 0.0468 | 0.0809 |
| GP unc/grad 30/70 | 0.1415 | 0.2333 | 0.0818 | 0.0476 | 0.1037 |
| GP gradient | 0.1672 | 0.2887 | 0.1002 | 0.0567 | 0.1051 |

The gradient arms get **worse as the budget grows**. That is not a conditioning
failure: their RBF interpolation residual stays at 1e-13, the same as every
other arm. It is clustering. On `5d-m1-p1-anis` at N = 512 the minimum
neighbour distance is 0.035 for GP gradient and 0.305 for space-filling, so the
gradient rule spends its budget resolving one peak while the rest of the box
goes unsampled, and the global RMS pays for it. The ordering of the blend
ratios is monotone in the gradient share, which is consistent with that reading.

Two caveats belong with this result. The common RBF reconstructor is isotropic,
so a design that concentrates on the active axes gains nothing from doing so
(measured headroom over a space-filling design: 1.0x). An ARD-GP reconstructor
on the identical designs is 3-15x more accurate and recovers the planted length
scales, so the absolute error levels here are a property of the scorer as much
as of the samplers. Rankings under a second reconstructor are not yet measured.

![5-D combined performance](../results/5d/figures/performance.png)
