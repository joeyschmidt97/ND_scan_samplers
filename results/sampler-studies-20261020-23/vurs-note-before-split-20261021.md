---
created: 2026-10-20 21:30
up: "[[moc-vwrs-vurs]]"
tags: [query, numerics, active-learning, ai-generated/Vulcan]
summary: "When is a VURS campaign done (fixed budget; density and surprise are monitors, not stops), and how to replay other samplers post hoc on the NSTX VURS runs despite home-pool bias."
source: []
aliases: [VURS stopping rule, NSTX post-hoc sampler replay]
edit_notes:
  - "2026-10-20 22:19 — recorded two-stage mode mapping and response characterization; registered scoring principles and specified a proposed sparse-grid/Matérn comparison, with execution gates and deferred work"
  - "2026-10-20 22:31 — user authorized overnight proxy tests; recorded reduced development protocol, MGKDB predictor-learning-curve proposal and proxy validity limits"
  - "2026-10-20 22:41 — launched separate grouped MGKDB gamma/omega model search; fixed first-field settings, protected pscans3 grading and saved experiment artifacts"
  - "2026-10-21 09:30 — recorded completed proxy and MGKDB results; distinguished independent target selection from separate fitting and proposed validation/ablation follow-ups"
---

# VURS stopping and post-hoc replay on the NSTX campaign (2026-10-20)

> Two questions from the NSTX campaign. First, what makes a VURS run "complete": point density, surprise, or something else? Second, once VURS has chosen the NSTX runs, can the other samplers be replayed on those same runs to compare them? The second question really asks whether uncertainty in growth rate (or in convergence) is enough to drive sampling that covers the space.

**Structure.** Part 1 holds the stopping answer and the replay design, and does not depend on any run. Part 2 holds evidence and is empty until NSTX VURS runs exist. A reader may stop after Part 1.

**External review.** Kosmos was asked the same questions; its report is graded against the registered rules below in [[2026-10-20-kosmos-vurs-stopping-grade#Adopt (candidate changes, before NSTX runs start)]]. The candidates that matter before the NSTX runs are slice size by power calculation and seed count by power calculation. Its composite endpoint and off-policy weighting are rejected.

# Part 1 — Concept

## What stops a VURS campaign

- **A fixed paid-run budget.** The loop runs until budget minus paid runs reaches zero. Resolution tolerances, density scores and surprise are scored afterwards and never stop acquisition ([[2026-09-18-vwrs-vurs-resolution-sampling#Common evaluation protocol]]).
- **Density against variability** (spacing $h_N(x)$ times local variation $\widehat L_N(x)$, stop at the elbow) is a refinement indicator, not a certificate. A narrow peak that no point has landed in yet looks flat, so the curve can level off before that region is resolved ([[2026-09-15-uq-audit-and-data-proxy#When is the sample size sufficient?]]).
- **Surprise** on runs not yet paid for works for mode labels but not for growth rates. Growth-rate surprise is heavy-tailed: a few runs at mode switches dominate it, and 20 seeds separate no arms ([[2026-10-07-vurs-surprise-scoring]]).
- **Truth-free stopping indicators** were proposed on 09-18 and never implemented.

### Budget guide (MGKDB replay evidence, per discharge)

| Runs | What the benchmarks support |
|---|---|
| ≲60 | no method separates from seed scatter |
| ~98 | sensitivity recovery converged for every VURS profile (0.93–0.95) |
| ~100 | VURS advantage starts; below this it buys robustness, not resolution |
| 150+ | needed to resolve mode transitions and $\gamma$ peaks |

Sources: [[vwrs-vurs-committee-overview]], [[sampler-leaderboard]], [[2026-10-07-vurs-surprise-scoring#Check B — the survey regions: an advantage at N = 256, not at campaign budgets]].

### Rule for the NSTX campaign

- **Stop by budget.** Draft chapter (Oct 28 cut): about 60–100 total runs per discharge is a planning budget, not an established NSTX sufficiency threshold. The MGKDB budget guide does not certify NSTX sensitivity rankings or transition resolution. Reserving 25% for grading also reduces the acquisition budget.
- **Plot two monitors, never stop on them:** label surprise on held-out runs, and the change between successive reconstructions at fixed reference points. Flat curves can indicate stalled learning at substantial error; they do not alone support "sufficient".
- Turning a monitor into a stopping rule needs a registered rule with ≥20 paired seeds on held-out data first.

## Post-hoc replay on the NSTX VURS runs

The MGKDB replays already ran every arm on a fixed pool of completed runs, with each arm's starting design snapped to pool points ([[2026-10-03-mgkdb-pool-sampler-replay]]). The same pool mode can replay on the NSTX runs at no compute cost. The new problem is that VURS chose this pool.

### Why a VURS-built pool favors VURS

1. **Support bias.** Every arm can only pick points VURS already picked. Pool density follows VURS's preferences, so space-filling restricted to that pool looks less space-filling, and any arm whose ideal picks lie where VURS never went gets snapped toward VURS's choices.
2. **Grading bias.** Truth exists only at pool points, and those concentrate where VURS saw structure. Errors are averaged with VURS's weighting, not over the target input measure.
3. **Self-replay.** VURS from the same start reproduces its own path. Different starting designs per seed remove the trivial case but not the home advantage.

**So the evidence is lopsided.** VURS winning on its own pool is weak support. A competitor beating VURS on VURS's own pool is strong evidence against VURS. Read the replay as a falsification test, not as confirmation.

### Design that limits the bias

- **Reserve a space-filling slice of the campaign** (about 25% of each discharge's budget, scrambled Sobol over the 5D box, run alongside VURS). It is the unbiased grading set: never in the replay pool, scored by every arm. At an 80-run budget: 20 Sobol + 60 VURS.
- **Replay budget at most half the pool**, so every arm has to choose.
- **20 paired seeds**, each with its own Latin-hypercube start snapped to distinct pool points; the same starts for every arm.
- **Report snap distance** for the shared starts and any actual off-pool acquisition proposals, in normalized coordinates (median, P95 and maximum). Pool-native candidate selection has no acquisition snap distance: report not applicable. Reference-to-pool distances diagnose unsupported regions. Neither quantity measures the full statistical bias.
- **Prescribed-node arms (`sglib`, SG++) are refused**, as in the MGKDB replays: snapping their nodes would misrepresent the method.
- **Primary endpoint: label surprise on the Sobol slice.** Growth-rate surprise was inconclusive on MGKDB and stays secondary. Register the margin before looking.

### Testing whether uncertainty alone is enough

The question "does uncertainty in $\gamma$ drive enough coverage?" maps onto arms that already exist:

- **GP uncertainty** (posterior variance of $\gamma$ only). On MGKDB it ties VURS at campaign budgets and flips sign between pools (25% better on development, 20% worse on held-out): unreliable, not shown better ([[vwrs-vurs-committee-overview]]).
- **Gradient-seeking GPs**: label surprise stalls and the model becomes overconfident; slowest to learn the mode map.
- **VURS weight ablation**: the profile weights (coverage, variation, uncertainty) can isolate uncertainty-only against coverage-only against full VURS on the same pool ([[2026-10-04-vurs-exploration-profiles]]).
- **Convergence uncertainty** is untested. Candidate arm: treat unconverged or failed runs as their own label (they are recorded, never dropped — [[nstx-survey-sampling-procedure#Procedure (one discharge)]] step 5) and acquire on the entropy of predicted failure. New arm, so it needs its own registered rule before it runs.

## Registered rules

### Status and objective (2026-10-20)

The user approved recording the two-stage objective and scoring design. The scoring principles below are fixed before new results; this is **not an executed comparison or a fully frozen experiment manifest**. Exact case versions, sampler configurations, evaluation seeds and the statistical family must be recorded before a confirmatory run. No new winner or default follows from this design. Existing inspected results are development evidence only.

**Stage 1 — survey the mode map.** Sample the full declared parameter box to predict the dominant mode and approximate boundaries of its regions, including disconnected islands. Dominant means fastest growing at a location, not largest in volume. Finite sampling cannot certify that every island has been found. Rare modes are real regions, not discarded outliers; their contribution to the primary score is proportional to their volume under the declared input measure.

**Stage 2 — characterize growth within modes.** Refine growth-rate peaks, steep flanks and mode-switch regions, while retaining broad exploration. Peaks and sensitivities are different targets: a smooth maximum has zero gradient. Summarize local sensitivity directions across each mode's region to identify influential parameter combinations. The same sampler may continue or another may take over; a staged policy must be tested against keeping each sampler unchanged.

Scale and objective are independent: both scores can be evaluated over the full box or a prospectively declared zoom region. Do not choose a favorable zoom using grading outcomes. The primary measure is uniform in normalized declared coordinates, including a log coordinate for a logarithmic physical axis. This is a search-space measure, not a claim about experimental parameter probabilities.

### Stage 1 scoring — broad mapping

**Primary: volume-weighted held-out label surprise**, in bits per point. For an independent fixed Sobol reference set $Q$ and true dominant label $m(x)$, use $S_N=|Q|^{-1}\sum_{x\in Q}-\log_2 p_N(m(x)\mid x)$. Every design is scored by the same Dirichlet-kNN classifier fitted only to its acquired observations, with $k=\min(2d+1,N)$ and Jeffreys pseudocount $1/2$ for each label in a taxonomy frozen before results. The posterior denominator uses the actual neighbor count. Modes absent from training retain nonzero probability. Unlike the earlier macro-mode primary, this average weights regions by their volume ([[2026-10-07-vurs-surprise-scoring#The model: prior and posterior]]).

**Secondary:** label error; per-mode recall and per-mode surprise; boundary-band label error; P95 and maximum reference-to-design distance. Fix the boundary mask from evaluator-only truth, not an arm's predictions. On Ionut branch proxies use a branch-gap mask: the largest minus second-largest growth rate is at most 0.05 of a fixed truth response scale. Freeze that scale on separate reference data. An absent rare mode in a finite grading set is unassessed, not recovered. Report a common GP-classifier sensitivity check without switching the primary scorer after viewing rankings.

The sparse grid needs no native label posterior: its paid node labels can train the same classifier as every other design. Coverage is a diagnostic, not a substitute for prediction quality. A flat surprise curve at high error is not convergence.

### Stage 2 scoring — reconstruction, peaks and parameter levers

Keep three quantities separate; do not merge them into an arbitrary composite ranking.

1. **Response reconstruction:** normalized growth-rate RMSE on independent points, globally and inside each true dominant-mode region. Use one positive response scale per case, frozen from independent development/reference data, for every arm. Report per-mode errors and the volume-weighted aggregate; macro-mode averaging is secondary. Do not normalize each arm by its own observed range.
2. **Peak recovery:** RMSE over the true top growth-rate decile within each mode, and normalized simple regret (truth-reference maximum minus highest acquired growth rate in that mode). Fix decile masks independently of acquisitions. Report undiscovered modes explicitly rather than dropping them from averages. A finite reference maximum is an approximation; a sample exceeding it exposes reference inadequacy, not negative physical regret. On analytic cases refine the evaluator's peak reference independently. Peak recovery does not establish sensitivity recovery.
3. **Sensitivity recovery:** compare local finite-scale response changes and their aggregate direction matrix with independently evaluated truth. In normalized coordinates use centered differences with half-step $h=0.01$; report $h=0.005$ and $0.02$ as robustness checks. Keep only interior stencils that remain in the same true dominant-mode region; report excluded mass and assess boundary switching separately. Apply identical stencils to predictions and truth. This defines sensitivity at a declared scale, not an exact derivative of a rough GP or discontinuity.

Let $g_m(x)$ be that vector of normalized growth-rate differences, and $R_m$ the region where mode $m$ dominates. Define $C_m=\mathbb E[g_mg_m^\mathsf T\mid x\in R_m]$ under uniform conditional volume weighting. Report $\mathrm{tr}(C_m)$ (total sensitivity), normalized diagonal entries (axis contributions), and leading eigenspaces (parameter combinations). Score the relative Frobenius error of $C_m$ against truth as the primary sensitivity endpoint; report leading-direction angle and parameter-rank agreement secondarily. Near-zero truth matrices require absolute error and an unidentifiable direction flag, not division by zero. Near-degenerate eigenvalues require a subspace comparison rather than a supposedly unique eigenvector.

Aggregate per-mode sensitivity errors with **true region-volume weights fixed by the evaluator**, while displaying every per-mode result. Report macro-mode and growth-rate-weighted aggregates separately. These are conditional within-region summaries, not classical independent-input Sobol indices: conditioning on a mode can couple the coordinates. Boundary-crossing effects remain in Stage 1's boundary score. A winner-only oracle exposes each mode only where it dominates; hidden branch responses must not be supplied to training merely because synthetic truth has them.

### Fair UQ proxy test — sparse grids and Matérn GPs

**Yes, the documented truth-known UQ/Ionut proxy suite is the appropriate test bed.** The existing notes describe conditional 3D ITG–TEM and ITG–KBM cases, native 6D branch proxies, and synthetic peak/interface controls ([[2026-09-18-vwrs-vurs-resolution-sampling#NERSC and Ionut-proxy extension]]). Interpret “UQ-instability” as this suite pending identification of a different intended project. `C:\Users\joesc\git\ND_scan_samplers` was verified present; the update log identifies it as the repository of record ([[vwrs-vurs-update-log#2026-10-09]]). The October 3 note still names `UQ_discontinuties-2d`; this is flagged historical path drift, not silently repaired. No code repository was modified or new benchmark executed for this registration.

**Candidate field:** Ionut's real `sglib` adapter; genuine SG++ only where its backend is available; GP posterior-variance, gradient-seeking and fixed 50/50 blend acquisitions crossed with Matérn $\nu=1/2,3/2,5/2$; plain VURS and space-filling references. Freeze fitting, noise, optimization, candidate counts and all other acquisition settings across each kernel comparison. Matérn-1/2 is Ionut's recorded suggestion, not an established winner ([[2026-09-16-ionut-sampling-recommendations#Recommendations and clarification points]]). Finite differences make the gradient-arm definition explicit for every kernel.

**Sparse-grid configuration gate:** audit tolerances, refinement criterion, node family, anisotropy and level caps before admitting a configuration as “Ionut's method.” The earlier audit found zero tolerances made spectral score components tie; its budget adapter also differed from native termination ([[2026-09-15-uq-audit-and-data-proxy#Sparse grids in 2D]]). Preserve that historical configuration as a named baseline if needed; do not silently replace it or generalize its rank to all sparse grids. Any corrected configuration must be fixed on development cases before held-out evaluation.

**Two scoreboards prevent placement/kernel confounding.** First, score labels with the common classifier and refit every acquired design with each common Matérn predictor (1/2, 3/2, 5/2), holding fit settings fixed. Show the design-by-reconstruction-kernel matrix for growth, peaks and sensitivity. Second, report each method's native predictor separately: this evaluates the whole sampling-plus-reconstruction pipeline. Common fits are diagnostics too; a win dependent on one scorer kernel is conditional, not universal. Do not use one global RBF as unquestioned truth: the existing high-dimensional benchmark documented failures on concentrated peak designs ([[2026-09-18-vwrs-vurs-resolution-sampling#6D design versus kernel — VURS against GP sampling]]).

**Native nodes and honest costs:** query the analytic oracle wherever a sparse grid requires; never snap its nodes to a VURS pool. Count all unique paid evaluations, initialization and any failed attempts consistently; reuse cached identical points. At a common exact paid count a partial native refinement batch may be graded through common refits, but its native interpolant is unavailable until the batch completes. Show native errors against actual completed-batch costs without pretending a batch overshoot fits the budget. Shared LHS starts apply to compatible sequential samplers, not forced sparse-grid nodes. Pair sparse grids and other methods by the same truth realization and grading scramble. Twenty copies of a deterministic grid on one identical function are not twenty independent acquisition trials.

**Confirmatory execution gate:** at least 20 paired seeds/realizations, independently held-out truth locations and unseen case variations, and a timestamped manifest before outcomes are scored. Existing benchmark cases and results are development material. Use frozen peak locations/widths, anisotropy and island-size variations for held-out synthetic cases; any proxy transformation must remain inside its valid domain and be disclosed as a synthetic stress test. Freeze one reference cloud of 65,536 points per paired realization, inaccessible to acquisition. If only fixed Ionut functions are used, distinguish reference-integration uncertainty from variation across genuinely different functions.

Proposed cheap first study: the two conditional 3D growth-rate families with budgets 32, 64, 98, 159 and 256, subject to native-start feasibility checked without scoring. Stage 1 endpoint is mean label surprise over 32/64/98; Stage 2 endpoints are sensitivity-matrix error and peak-decile RMSE at 256, with all intermediate curves reported. Full 6D and switched-policy experiments are deferred unless the existing infrastructure makes them feasible before Oct 28. Case manifest, configuration gates and precise treatment of degenerate sensitivity cases must be closed before launch; the list is not authorization to implement missing machinery.

For label surprise retain the 0.05-bit practical margin from [[2026-10-07-vurs-surprise-scoring#Confirmatory protocol (registered 2026-10-07, before the full scoring)]]. For nonzero Stage 2 errors propose a 10% relative improvement margin. Average case scores within each paired realization before inference; do not treat reference points as independent experimental replicates. Report paired bootstrap 95% intervals (10,000 resamples), with multiplicity correction across all prospectively declared primary contrasts and endpoints. A practical superiority claim requires the interval beyond the relevant margin plus Holm-adjusted significance; equivalence requires its interval entirely inside the declared equivalence band. If neither condition holds, report noninferiority only when supported, otherwise inconclusive. Freeze the contrast family and zero-error rule in the manifest. No method becomes better or default from a pilot, an uncorrected secondary endpoint, or an inspected-case reranking.

### NSTX replay and staged-policy limits

Reserve the approximately 25% scrambled-Sobol grading slice **prospectively**, excluding its outcomes from the original campaign as well as all replays. An after-the-fact subset of VURS points is not an unbiased replacement. At 80 total runs, 20 grading plus 60 pool points permits a maximum replay budget of 30. This cannot test a claim about 98-point acquisition. Keep 20 shared snapped-LHS replay seeds, the half-pool cap and the refusal of prescribed-node sparse grids. Apply Stage 1's volume-weighted score; growth-rate and sensitivity results remain separate and may be unidentifiable at these budgets. Home-pool support bias survives independent grading, so a competitor win is stronger evidence against VURS than a VURS win is evidence for it.

### Sobol grading slice — deferred to a post-draft test (user, 2026-10-20)

**Manifest change, made before any NSTX VURS run.** The 25% slice above is not reserved in the draft campaign. The slice and the NSTX post-hoc replay become one test, run after the draft.

- **What a scrambled Sobol slice is.** A Sobol sequence is a low-discrepancy sequence: with $2^m$ points, every dyadic sub-box of the 5D box holds its fair share, including in every 1D and 2D projection. Owen scrambling randomizes the digits while keeping that uniformity, so error estimates over the slice are unbiased, and repeated scrambles give an error bar. Use 16 or 32 points.
- **Running it later is still valid.** The slice is unbiased because its points do not depend on any sampler's data, not because of when it runs. Fixing the Sobol points and scramble seed now, and running them after the VURS campaign, keeps it independent. The condition is that no slice outcome enters training, acquisition or tuning before grading. After grading, slice runs join the database.
- **Proposed size when run:** 16 points per discharge and radius, not a budget fraction; confirm by the power calculation in [[2026-10-20-kosmos-vurs-stopping-grade#Adopt (candidate changes, before NSTX runs start)]].
- **What the draft can claim without it.** VURS sampler performance rests on the truth-known proxies and the MGKDB pool replays. Those pools are grid scans not built by VURS, so they carry no home-pool bias. NSTX map and surrogate accuracy can be reported only as cross-validation on the campaign's own points, labeled as self-consistency, not independent accuracy.
- **Free partial check now.** The 129038 and 129015 corner runs were chosen independently of VURS and can grade the VURS-trained map without bias at the box corners. They say nothing about the interior.

**Committee point — must be answered in the dissertation.** Expected question: *"How do you know the NSTX mode map is accurate, when every point was chosen by your own sampler?"* Prepared answer:
1. Sampler validity is shown on independent data: proxies with known truth, plus MGKDB pools VURS did not build.
2. On NSTX, accuracy is stated as cross-validation self-consistency plus a corner-run check, and named as a limitation.
3. The independent Sobol test is designed and fixed in advance (points and seed frozen), and is the first follow-up.

Track in [[dissertation-outline#Open questions]] and [[phd-roadmap#Future-work parking lot (do not start before January)]].

#### Frozen slice v1 (2026-10-20)

**Frozen before any NSTX VURS run.** 192 points: 16 per discharge and radius role (129015, 129038, 132543, 132588 × $n_e$ top, $T_e$ top, steep). Record of truth: `NSTXU_lithium_study/data/sobol_grading_slice/sobol_slice_v1.csv`, commit `b016adc`, sha256 `43987f6a1c6b26e491d344c9e5042a4ac592d9bb78e22af5d15d7e6fc470dbf8`. Generator and check: [[cmd-sobol-slice]].

- **Construction:** Owen-scrambled Sobol (`scipy.stats.qmc.Sobol`, $d=5$, scipy 1.18.0), seed `default_rng(SeedSequence(20261020, spawn_key=(shot, role_index)))`. Axes and bounds as [[nstx-survey-sampling-procedure#Sampled axes and bounds]], `kymin` on a log scale.
- **Keyed by radius role, not $
ho_	ext{tor}$,** so a later Chapman refit (132xxx moved to profiles_* on 2026-10-16) does not change the points.
- **Checks:** every unit-axis range spans 0.002–0.999 with mean 0.50; no two points in a design snap to the same equilibrium; `--check` reproduces the CSV exactly.
- **Run values:** the profile scalings below are snapped to ped_grid's two-decimal case names; the CSV also keeps the raw values and unit-box coordinates. `kymin` is exact.
- **Cost when run:** 16 distinct equilibria per design, so 48 cheaseBS cases per discharge and 192 in total, each with one `kymin`. That does not match `vurs_loop`'s jobs, which put several $k_y$ on one equilibrium; running the slice needs a one-$k_y$-per-job path or packing several slice points per job.
- **Rule:** no slice outcome enters training, acquisition or tuning before grading. After grading, slice runs join the database.

> [!example]- 129015 — ne_top (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 1.11 | 0.92 | 1.09 | 1.00 | 0.1767 |
> | 1 | 0.92 | 1.11 | 0.88 | 0.85 | 0.6228 |
> | 2 | 0.86 | 0.81 | 1.20 | 1.20 | 1.5059 |
> | 3 | 1.07 | 1.00 | 0.99 | 0.95 | 0.0675 |
> | 4 | 1.04 | 0.86 | 0.84 | 0.91 | 0.1167 |
> | 5 | 0.83 | 1.07 | 1.03 | 1.14 | 0.8221 |
> | 6 | 0.99 | 0.96 | 0.95 | 0.89 | 0.3210 |
> | 7 | 1.18 | 1.17 | 1.14 | 1.06 | 0.2883 |
> | 8 | 1.17 | 0.84 | 0.92 | 1.11 | 0.6972 |
> | 9 | 0.96 | 1.03 | 1.10 | 0.94 | 0.1544 |
> | 10 | 0.81 | 0.94 | 0.82 | 1.09 | 0.0603 |
> | 11 | 1.00 | 1.13 | 1.01 | 0.86 | 1.7219 |
> | 12 | 1.08 | 0.98 | 1.17 | 0.80 | 1.1838 |
> | 13 | 0.90 | 1.20 | 0.95 | 1.05 | 0.0828 |
> | 14 | 0.94 | 0.88 | 1.07 | 0.99 | 0.2002 |
> | 15 | 1.15 | 1.10 | 0.86 | 1.15 | 0.4525 |

> [!example]- 129015 — Te_top (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 0.90 | 1.07 | 0.99 | 1.04 | 0.0790 |
> | 1 | 1.18 | 0.93 | 1.04 | 0.96 | 1.2525 |
> | 2 | 1.00 | 1.13 | 0.88 | 1.17 | 0.3213 |
> | 3 | 0.88 | 0.87 | 1.14 | 0.83 | 0.1282 |
> | 4 | 0.82 | 1.15 | 1.07 | 0.86 | 0.2796 |
> | 5 | 1.10 | 0.84 | 0.91 | 1.15 | 0.7007 |
> | 6 | 1.12 | 1.05 | 1.16 | 0.93 | 1.4500 |
> | 7 | 1.00 | 0.95 | 0.81 | 1.07 | 0.0915 |
> | 8 | 0.96 | 1.12 | 1.18 | 1.12 | 0.9654 |
> | 9 | 1.14 | 0.89 | 0.84 | 0.88 | 0.0605 |
> | 10 | 1.06 | 1.08 | 1.09 | 1.10 | 0.1669 |
> | 11 | 0.83 | 0.92 | 0.94 | 0.91 | 0.4214 |
> | 12 | 0.86 | 1.01 | 0.86 | 0.98 | 0.5450 |
> | 13 | 1.04 | 0.99 | 1.11 | 1.02 | 0.2159 |
> | 14 | 1.17 | 1.19 | 0.97 | 0.81 | 0.1172 |
> | 15 | 0.94 | 0.81 | 1.01 | 1.19 | 1.8710 |

> [!example]- 129015 — steep (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 1.18 | 1.15 | 1.08 | 1.08 | 0.1498 |
> | 1 | 0.86 | 0.99 | 0.99 | 0.91 | 1.6600 |
> | 2 | 0.91 | 1.07 | 1.20 | 1.12 | 0.3352 |
> | 3 | 1.05 | 0.81 | 0.88 | 0.89 | 0.1174 |
> | 4 | 1.06 | 1.01 | 0.93 | 0.82 | 0.0507 |
> | 5 | 0.98 | 0.87 | 1.04 | 1.19 | 0.7749 |
> | 6 | 0.85 | 1.19 | 0.84 | 0.98 | 0.9074 |
> | 7 | 1.11 | 0.95 | 1.13 | 1.01 | 0.2742 |
> | 8 | 1.13 | 1.09 | 0.81 | 1.16 | 1.2864 |
> | 9 | 0.81 | 0.83 | 1.11 | 0.85 | 0.1946 |
> | 10 | 0.97 | 1.12 | 0.91 | 1.04 | 0.0906 |
> | 11 | 1.09 | 0.95 | 1.02 | 0.96 | 0.4372 |
> | 12 | 1.00 | 1.15 | 1.17 | 0.93 | 0.5984 |
> | 13 | 0.93 | 0.92 | 0.86 | 1.05 | 0.0662 |
> | 14 | 0.89 | 1.03 | 1.05 | 0.87 | 0.2125 |
> | 15 | 1.17 | 0.89 | 0.97 | 1.15 | 1.1794 |

> [!example]- 129038 — ne_top (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 0.83 | 1.14 | 1.10 | 1.15 | 0.7748 |
> | 1 | 1.06 | 0.86 | 0.95 | 0.97 | 0.0526 |
> | 2 | 1.15 | 1.08 | 1.17 | 1.05 | 0.2080 |
> | 3 | 0.94 | 0.92 | 0.82 | 0.83 | 1.2498 |
> | 4 | 0.99 | 1.03 | 1.00 | 0.88 | 1.9152 |
> | 5 | 1.11 | 0.97 | 1.05 | 1.09 | 0.1281 |
> | 6 | 1.02 | 1.19 | 0.87 | 0.91 | 0.0794 |
> | 7 | 0.89 | 0.81 | 1.12 | 1.11 | 0.4842 |
> | 8 | 0.86 | 1.07 | 0.90 | 1.05 | 0.0722 |
> | 9 | 1.03 | 0.93 | 1.14 | 0.87 | 0.5374 |
> | 10 | 1.14 | 1.11 | 0.97 | 1.14 | 0.8625 |
> | 11 | 0.96 | 0.89 | 1.02 | 0.93 | 0.2821 |
> | 12 | 0.91 | 1.16 | 1.20 | 0.98 | 0.1874 |
> | 13 | 1.20 | 0.84 | 0.84 | 1.19 | 1.3753 |
> | 14 | 1.09 | 1.02 | 1.07 | 0.81 | 0.3519 |
> | 15 | 0.82 | 0.98 | 0.92 | 1.01 | 0.1168 |

> [!example]- 129038 — Te_top (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 0.84 | 0.90 | 1.18 | 0.86 | 0.7708 |
> | 1 | 1.07 | 1.00 | 0.91 | 1.09 | 0.1694 |
> | 2 | 1.14 | 0.82 | 1.02 | 0.94 | 0.0531 |
> | 3 | 0.96 | 1.12 | 0.89 | 1.12 | 1.5275 |
> | 4 | 0.92 | 0.87 | 0.98 | 1.18 | 0.9636 |
> | 5 | 1.19 | 1.17 | 1.11 | 0.96 | 0.0842 |
> | 6 | 1.01 | 0.95 | 0.82 | 1.00 | 0.2686 |
> | 7 | 0.89 | 1.05 | 1.10 | 0.83 | 0.4858 |
> | 8 | 0.86 | 0.84 | 0.83 | 1.00 | 0.1566 |
> | 9 | 1.03 | 1.14 | 1.06 | 1.17 | 0.5493 |
> | 10 | 1.17 | 0.93 | 0.97 | 0.82 | 1.7514 |
> | 11 | 0.95 | 1.03 | 1.14 | 1.04 | 0.0789 |
> | 12 | 0.98 | 0.98 | 1.04 | 1.06 | 0.1252 |
> | 13 | 1.10 | 1.08 | 0.85 | 0.88 | 1.1052 |
> | 14 | 1.10 | 0.89 | 1.16 | 1.14 | 0.3463 |
> | 15 | 0.82 | 1.19 | 0.95 | 0.91 | 0.2481 |

> [!example]- 129038 — steep (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 1.15 | 0.90 | 1.07 | 1.08 | 0.1465 |
> | 1 | 0.95 | 1.00 | 0.97 | 0.97 | 1.3061 |
> | 2 | 0.84 | 0.98 | 1.20 | 1.17 | 0.3185 |
> | 3 | 1.04 | 1.12 | 0.90 | 0.88 | 0.0896 |
> | 4 | 1.08 | 0.91 | 0.95 | 0.82 | 0.0590 |
> | 5 | 0.88 | 1.18 | 1.05 | 1.13 | 0.5274 |
> | 6 | 0.91 | 0.81 | 0.81 | 0.93 | 0.8134 |
> | 7 | 1.12 | 1.09 | 1.11 | 1.02 | 0.2283 |
> | 8 | 1.13 | 0.95 | 0.83 | 1.11 | 1.6996 |
> | 9 | 0.93 | 1.15 | 1.13 | 0.84 | 0.1889 |
> | 10 | 0.86 | 0.87 | 0.91 | 1.05 | 0.1234 |
> | 11 | 1.06 | 1.03 | 1.01 | 0.91 | 0.4364 |
> | 12 | 1.00 | 0.83 | 1.16 | 0.99 | 0.6850 |
> | 13 | 0.80 | 1.06 | 0.86 | 1.06 | 0.0763 |
> | 14 | 0.99 | 0.94 | 1.08 | 0.85 | 0.3151 |
> | 15 | 1.19 | 1.16 | 0.98 | 1.19 | 1.1125 |

> [!example]- 132543 — ne_top (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 0.99 | 1.02 | 1.08 | 1.00 | 0.6430 |
> | 1 | 1.08 | 0.96 | 0.87 | 0.99 | 0.0721 |
> | 2 | 1.12 | 1.11 | 1.14 | 1.16 | 0.1707 |
> | 3 | 0.80 | 0.87 | 0.91 | 0.84 | 1.9176 |
> | 4 | 0.87 | 1.17 | 0.84 | 0.87 | 0.8797 |
> | 5 | 1.16 | 0.81 | 1.01 | 1.13 | 0.2470 |
> | 6 | 1.05 | 1.06 | 0.98 | 0.91 | 0.0823 |
> | 7 | 0.93 | 0.92 | 1.17 | 1.08 | 0.3688 |
> | 8 | 0.92 | 1.14 | 0.97 | 1.10 | 0.0582 |
> | 9 | 1.01 | 0.88 | 1.18 | 0.89 | 0.5188 |
> | 10 | 1.19 | 1.03 | 0.80 | 1.06 | 1.5442 |
> | 11 | 0.89 | 0.99 | 1.05 | 0.94 | 0.1374 |
> | 12 | 0.84 | 1.09 | 1.11 | 0.97 | 0.3062 |
> | 13 | 1.13 | 0.93 | 0.94 | 1.03 | 1.0904 |
> | 14 | 1.06 | 1.18 | 1.07 | 0.82 | 0.4579 |
> | 15 | 0.96 | 0.84 | 0.88 | 1.18 | 0.1022 |

> [!example]- 132543 — Te_top (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 0.88 | 1.14 | 1.11 | 1.18 | 0.0573 |
> | 1 | 1.05 | 0.80 | 0.81 | 0.94 | 1.9807 |
> | 2 | 1.17 | 1.09 | 1.08 | 1.00 | 0.7805 |
> | 3 | 0.99 | 0.96 | 0.99 | 0.87 | 0.1419 |
> | 4 | 0.94 | 1.03 | 0.90 | 0.84 | 0.3152 |
> | 5 | 1.11 | 0.92 | 1.18 | 1.08 | 0.3604 |
> | 6 | 1.00 | 1.18 | 0.92 | 0.97 | 0.9047 |
> | 7 | 0.83 | 0.87 | 1.00 | 1.10 | 0.1224 |
> | 8 | 0.82 | 1.05 | 0.93 | 1.07 | 1.4145 |
> | 9 | 1.05 | 1.00 | 1.04 | 0.81 | 0.0640 |
> | 10 | 1.14 | 1.11 | 0.85 | 1.15 | 0.1601 |
> | 11 | 0.91 | 0.84 | 1.17 | 0.98 | 0.5474 |
> | 12 | 0.96 | 1.17 | 1.06 | 0.91 | 0.4082 |
> | 13 | 1.18 | 0.88 | 0.96 | 1.17 | 0.2218 |
> | 14 | 1.10 | 1.02 | 1.14 | 0.88 | 0.0871 |
> | 15 | 0.87 | 0.94 | 0.83 | 1.05 | 1.0064 |

> [!example]- 132543 — steep (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 1.11 | 0.99 | 0.89 | 0.88 | 0.2041 |
> | 1 | 1.00 | 1.02 | 1.10 | 1.16 | 1.2815 |
> | 2 | 0.89 | 0.88 | 0.94 | 0.95 | 0.6996 |
> | 3 | 1.00 | 1.12 | 1.07 | 1.08 | 0.1115 |
> | 4 | 1.07 | 0.82 | 1.18 | 1.03 | 0.0672 |
> | 5 | 0.83 | 1.18 | 0.81 | 0.90 | 0.4231 |
> | 6 | 0.93 | 0.92 | 1.05 | 1.11 | 0.8208 |
> | 7 | 1.17 | 1.09 | 0.97 | 0.84 | 0.1304 |
> | 8 | 1.19 | 0.86 | 1.00 | 0.95 | 1.7666 |
> | 9 | 0.92 | 1.15 | 0.98 | 1.02 | 0.2812 |
> | 10 | 0.81 | 0.95 | 1.17 | 0.82 | 0.0814 |
> | 11 | 1.08 | 1.04 | 0.84 | 1.14 | 0.5116 |
> | 12 | 1.04 | 0.94 | 0.91 | 1.19 | 0.3269 |
> | 13 | 0.85 | 1.05 | 1.08 | 0.86 | 0.0522 |
> | 14 | 0.96 | 0.85 | 0.87 | 1.07 | 0.1700 |
> | 15 | 1.14 | 1.16 | 1.15 | 1.00 | 1.0654 |

> [!example]- 132588 — ne_top (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 0.89 | 1.11 | 0.97 | 0.83 | 0.2118 |
> | 1 | 1.16 | 0.87 | 1.03 | 1.09 | 0.5308 |
> | 2 | 1.07 | 1.02 | 0.82 | 0.92 | 1.3935 |
> | 3 | 0.98 | 0.95 | 1.18 | 1.17 | 0.0807 |
> | 4 | 0.95 | 1.10 | 1.07 | 1.14 | 0.0639 |
> | 5 | 1.01 | 0.93 | 0.93 | 0.99 | 1.1101 |
> | 6 | 1.12 | 1.18 | 1.12 | 1.00 | 0.4112 |
> | 7 | 0.83 | 0.84 | 0.89 | 0.85 | 0.1724 |
> | 8 | 0.82 | 1.04 | 1.14 | 0.96 | 0.7005 |
> | 9 | 1.13 | 0.99 | 0.86 | 1.11 | 0.2545 |
> | 10 | 1.05 | 1.15 | 1.09 | 0.89 | 0.1088 |
> | 11 | 0.91 | 0.88 | 0.91 | 1.05 | 1.6387 |
> | 12 | 0.97 | 1.17 | 0.85 | 1.06 | 0.8411 |
> | 13 | 1.08 | 0.81 | 1.15 | 0.81 | 0.0531 |
> | 14 | 1.19 | 1.06 | 1.00 | 1.18 | 0.1278 |
> | 15 | 0.85 | 0.92 | 1.00 | 0.93 | 0.3497 |

> [!example]- 132588 — Te_top (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 0.83 | 1.11 | 0.80 | 1.06 | 0.4651 |
> | 1 | 1.17 | 0.80 | 1.04 | 0.80 | 0.1089 |
> | 2 | 1.07 | 1.08 | 0.95 | 1.16 | 0.1301 |
> | 3 | 0.94 | 0.98 | 1.19 | 0.91 | 1.5565 |
> | 4 | 0.98 | 1.00 | 1.08 | 0.97 | 0.8216 |
> | 5 | 1.01 | 0.91 | 0.87 | 1.11 | 0.2458 |
> | 6 | 1.12 | 1.18 | 1.13 | 0.87 | 0.0736 |
> | 7 | 0.88 | 0.88 | 0.92 | 1.01 | 0.6897 |
> | 8 | 0.86 | 1.06 | 1.11 | 1.13 | 0.0821 |
> | 9 | 1.15 | 0.97 | 0.94 | 0.98 | 0.3909 |
> | 10 | 1.05 | 1.14 | 1.06 | 1.03 | 1.8585 |
> | 11 | 0.96 | 0.85 | 0.89 | 0.88 | 0.1731 |
> | 12 | 0.91 | 1.17 | 0.98 | 0.85 | 0.2941 |
> | 13 | 1.09 | 0.86 | 1.17 | 1.09 | 1.0951 |
> | 14 | 1.19 | 1.05 | 0.83 | 0.95 | 0.5193 |
> | 15 | 0.80 | 0.94 | 1.02 | 1.19 | 0.0618 |

> [!example]- 132588 — steep (16 points)
> | pt | Te_ped | Te_width | ne_ped | ne_width | kymin |
> |---:|---:|---:|---:|---:|---:|
> | 0 | 1.12 | 1.19 | 0.96 | 1.04 | 0.0568 |
> | 1 | 0.93 | 0.85 | 1.08 | 0.98 | 1.7721 |
> | 2 | 0.81 | 1.03 | 0.81 | 1.16 | 0.5244 |
> | 3 | 1.03 | 0.92 | 1.14 | 0.82 | 0.1893 |
> | 4 | 1.06 | 1.08 | 1.01 | 0.86 | 0.2216 |
> | 5 | 0.90 | 0.97 | 0.94 | 1.12 | 0.4481 |
> | 6 | 0.95 | 1.14 | 1.17 | 0.95 | 0.9548 |
> | 7 | 1.19 | 0.81 | 0.89 | 1.08 | 0.1055 |
> | 8 | 1.15 | 1.01 | 1.18 | 1.15 | 1.3059 |
> | 9 | 0.99 | 0.94 | 0.85 | 0.89 | 0.0760 |
> | 10 | 0.86 | 1.17 | 1.05 | 1.05 | 0.1444 |
> | 11 | 1.10 | 0.88 | 0.92 | 0.92 | 0.6977 |
> | 12 | 1.01 | 1.12 | 0.83 | 0.96 | 0.3862 |
> | 13 | 0.83 | 0.83 | 1.10 | 1.02 | 0.2608 |
> | 14 | 0.92 | 1.06 | 0.99 | 0.84 | 0.0866 |
> | 15 | 1.13 | 1.00 | 1.07 | 1.18 | 1.1463 |


Switching from survey to targeting remains a **parked hypothesis**, not a production recommendation. A future comparison should fix the switch at a budget fraction before results (candidate: half the total budget), preserve an explicit exploration allocation, and compare against unchanged policies at identical total cost. A surprise-triggered switch would need separate monitor and grading sets and its own validation. No held-out grading information may select regions, tune weights or trigger a switch.

### Deadline order

1. Most feasible before **Oct 28**: audit existing adapter/configuration provenance and score existing designs descriptively with the Stage 1 and reconstruction metrics. Previously inspected designs cannot supply a new confirmatory winner.
2. Conditional on working infrastructure and explicit execution authorization: the registered 20-realization 3D comparison, prioritizing `sglib`, GP uncertainty kernels and the VURS/space-filling references. A reduced field must be frozen before results, never pruned after losses. Missing Matérn-5/2 or sensitivity machinery is a deferral, not a reason to expand code scope now.
3. Park: full 6D confirmation, new switched/region-targeting policies, truth-validated stopping rules, and convergence-failure entropy acquisition. See [[phd-roadmap#Future-work parking lot (do not start before January)]]. Results missing the cutoff stay outside the committee draft regardless of later draft-date discussions.

### MGKDB sampler-by-predictor learning curves (proposed, not launched)

**Update 22:41:** predictor selection is now running as a prerequisite, under the separate bounded protocol below. Sampler learning curves themselves have not started.

User proposal: build nested training sets using each sampler, refit a strong ML predictor at fixed paid-count checkpoints, and compare how quickly held-out growth-rate/frequency error falls. This is a sample-efficiency learning-curve experiment; removal ablation would instead remove selected data from a fixed full dataset. It measures usefulness for a specified predictor and target distribution, not universal information content.

- Select and freeze predictor/preprocessing/hyperparameters on a separate development partition. TabPFN regression is a candidate alongside a fixed GP or tree baseline; classification success is not evidence of regression success. Freeze model seeds and compute settings across sampler designs. A sampler-by-predictor matrix separates sample quality from model compatibility.
- Split by scan/profile/discharge groups before sampling; keep coordinate repeats and related rows together. Use only input features available before the expensive run. No grading labels, fitted scalers or output-derived features may enter acquisition or model selection. MGKDB group holdout measures transfer across its archive, not uniform NSTX-box accuracy.
- Same nested training budgets, shared starts and at least 20 paired acquisition seeds. Freeze checkpoints and the primary error/margin before results. Score normalized RMSE and MAE for $\gamma$ and signed $\omega$ separately, with normalization learned only from development data; any combined score gives both outputs fixed equal weight. Include random/space-filling and full-available-training-pool reference fits. Never train the full-pool reference on grading data.
- If acquisition observes only $\gamma$, say so; success predicting $\omega$ then tests transfer from that design. A joint acquisition objective is a separate registered arm. A simulation providing both responses costs one evaluation, not two. Keep mode-label and failed-run outcomes alongside response scores.
- Genuine sparse grids remain excluded from finite-pool replay when their nodes are absent. A proxy oracle can evaluate their native nodes, but a model fitted to MGKDB is not independent GENE truth. Do not snap nodes or grade a sampler only against the same learned surrogate that generated its responses.
- This differs from the prior experiment that let TabPFN select labels through entropy; its negative result does not answer whether TabPFN predicts $\gamma/\omega$ efficiently from another sampler's designs ([[2026-10-13-tabpfn-vurs-replay#Registered rule (committed before any run)]]).

### Tonight's reduced proxy protocol (fixed before full run)

The user authorized starting existing tests tonight. The implemented 3D runner does not support the full proposed scorecard. Therefore this run is explicitly **development/robustness evidence**, not the held-out-case confirmatory experiment above. No new default will be declared from it.

- Cases: fixed ITG–TEM and ITG–KBM hard-selection 3D $\gamma$ proxies. Seeds 0–19; 256 unique paid points; 16,384 frozen independent evaluation points per case; four single-threaded workers. Twenty seeds vary randomized acquisition, not the underlying physical formulas. Deterministic sparse-grid/grid repeats are not independent evidence.
- Field: grid, `sglib`, tetrahedral refinement, VWRS, VURS, and GP uncertainty/gradient/blend with Matérn-1/2 and -3/2. SG++ is unavailable locally; the runner lacks Matérn-5/2. Neither is replaced by a mock or silently imputed. `sglib` remains the historical zero-tolerance budget adapter, labeled as such. The runner's legacy VURS uses observed tetrahedral gradients; it is not asserted identical to current N-D VURS.
- Primary descriptive score: mean common tetrahedral normalized RMSE at 32, 64 and 98 paid points. Secondary: final global/transition/top-decile RMSE, coverage, native GP error and full learning curves. The existing high-response mask is global, not per mode. No sensitivity-matrix, label-surprise or common-GP-refit result is claimed from this runner.
- Predeclared contrasts: each other arm against VURS within each case. Paired 10,000-resample bootstrap intervals; Holm over 20 primary contrasts if significance is reported; 10% relative error margin. All failures/incomplete trajectories are reported. Results remain conditional on known functions and previously inspected configurations, regardless of p-values.
- A five-arm, 16-point pilot checks execution only; its results were viewed before the full run, no parameter tuned from them. Full settings, hashes and caveats were saved before launch in [the local protocol](C:/Users/joesc/.codex/visualizations/2026/10/21/01a15c9d-02de-7430-95f3-162603401e42/proxy-study-20261020-protocol.md).

### What the proxies can support

These are phenomenological ITG/TEM/KBM formulas with known truth, not GENE solutions or fits validated against experiment. They permit controlled checks of sampling, reconstruction, kinks, branch switches and budget accounting. They omit ETG/MTM, solver failures, realistic convergence noise and self-consistent equilibrium coupling. Conditional 3D slices cannot establish full 5D/6D behavior. Their use is legitimate as numerical unit/stress tests; they cannot alone select the best NSTX campaign sampler.

The committee-facing evidence chain is: controlled proxy behavior, transfer to independent real-GENE MGKDB campaigns, then prospective NSTX grading. A proxy win that reverses on independent real data falsifies transfer. Even agreement does not certify discovery of unseen narrow islands. Any extension beyond existing machinery remains subordinate to the Oct 28 cutoff; the user has authorized this bounded run, not every parked research arm.

### MGKDB predictor selection — fixed first-field protocol

User authorized regression optimization/testing. The local [protocol](C:/Users/joesc/.codex/visualizations/2026/10/21/01a15c9d-02de-7430-95f3-162603401e42/mgkdb-regression-protocol.md) and [runner](C:/Users/joesc/.codex/visualizations/2026/10/21/01a15c9d-02de-7430-95f3-162603401e42/mgkdb_regression.py) were saved before fitting. This is development model selection, not a confirmatory claim that one model or sampler is best.

- Raw MGKDB chunks: Hatch global campaign for development, `pscans3` for final grading. Exclude growth-rate failure sentinel and nonfinite targets; retain signed frequency. Use an explicit allowlist of pre-run equilibrium/species inputs only, excluding labels and output fingerprints. Original archive normalization is retained; transfer across different normalizations is not established.
- Twenty paired grouped development splits (seeds 0–19, 20% groups for validation). Whole directories stay together; directories sharing identical full-input signatures are joined, and any development points exactly matching test inputs are removed. Near duplicates and hidden solver differences remain limitations. Imputation/scaling and target standardization use training folds only.
- Eight fixed configurations: mean baseline; ridge alpha 0.1/10/100; Extra Trees 200 trees with leaf size 1/4; histogram gradient boosting 250 iterations, learning rate 0.05 and 15/31 leaves. Separate models for growth and frequency, equal-weight standardized RMSE for selection, per-target RMSE/MAE/R2 retained. One CPU thread to coexist with proxy work.
- Select by lowest mean development joint normalized RMSE, save selection before final testing, then evaluate the locked model and mean baseline once on `pscans3`. All 20 scores per configuration are required; errors stop selection. Hyperparameter-search validation is not unbiased test evidence. No statistical superiority/default claim follows from the selected minimum.
- Installed TabPFN exposes a regressor, but no regressor checkpoint was found in the searched local cache. It is excluded from this first field without a download; classification results do not substitute for regression evidence. Adding it later requires a declared additional development comparison before another protected test.
- Source repositories are unchanged. The artifact directory contains the reproducible runner, preregistration, source/data hashes, split indices, incremental scores, selected configuration and eventual held-out predictions. New sampler-selection tests need fresh protected grading rather than repeatedly optimizing against the same final test.

### Target-specific predictor selection and ablation — proposed follow-up (2026-10-21)

The first search already fits two independent regressors, one for each response; it selects a shared configuration using their mean normalized error. The user's proposed improvement is to select model family, hyperparameters and response transformation separately for each target, then freeze both predictors before comparing samplers. This follow-up does not replace the completed registration retroactively.

Priority before Oct 28: inspect errors by frequency sign/regime, mode, radius and distance from development support; check units and input completeness; then run a bounded target-specific search with nested grouped validation. Tune tree depth/leaf size/features and boosting learning rate/regularization; compare Matérn-1/2, -3/2 and -5/2 GP regressors if runtime permits. A signed asinh response transform is a candidate for frequency; fit its scale only on training data and score inverse-transformed predictions in original units. Frequency-regime classification followed by regression is a separate candidate, with regimes predicted only from pre-run inputs at evaluation, never supplied as true test labels. TabPFN regression remains conditional on checkpoint availability and a separate declared comparison.

Model selection uses inner grouped folds; outer groups estimate selection performance. At least 20 paired seed/split evaluations remain the minimum for comparative claims, but overlapping splits are not 20 independent campaigns. The already-inspected pscans3 result is a transfer diagnostic, not a fresh final test for a new search. Freeze a previously unused, suitably compatible campaign/group test before further selection, or explicitly limit claims to development validation; do not manufacture fresh evidence by resplitting exposed rows.

After selection, perform two distinct experiments on both frozen predictors: (1) nested sampler-selected training sets, measuring prediction error versus paid count; (2) removal stress tests at equal removed fractions, comparing random removal with omitted complete scan groups, transition neighborhoods and high-growth neighborhoods. Targeted masks and fractions must be fixed before outcomes; targeted removal deliberately changes training support and is not evidence that those points are optimally acquired. Keep one common grading set, starts and paid budgets across samplers. Report gamma and omega error curves separately, plus per-region error and paired uncertainty; avoid declaring one sampler best because a combined score hides a loss on the other response. New predictors may be retained as separate tools rather than forcing one family to serve both.

# Part 2 — Evidence

**Status 2026-10-20:** no NSTX VURS runs yet. 129038 corner runs exist; 129015 corners are running. Neither is a VURS pool.

### Local proxy execution — complete

**2026-10-20 22:31 (America/Guatemala):** launched the reduced 440-trajectory study above (2 fixed cases × 20 seeds × 11 arms), PID 4388. Code `ND_scan_samplers` HEAD `934524d1953db49001a5df2d74d8a39d0db1fbc5`; external `sglib` `d13bc4661cbd3901a70190b52c477e7606576fa3`. No source edits. The interpreter comes from the old checkout's environment; imported code is from `ND_scan_samplers`.

Implementation check: five pilot trajectories (VURS, GP uncertainty under both kernels, grid, sglib), one case, N=16 completed with zero failed trajectories. Pilot is not a ranking. Windows worker pipes required execution outside the sandbox. Full-run output is saved incrementally in `C:/Users/joesc/.codex/visualizations/2026/10/21/01a15c9d-02de-7430-95f3-162603401e42/proxy-study-20261020/results.json`; sibling stdout/stderr logs and PID file record execution. No full-run comparative result is available at launch.

**Verified complete 2026-10-21:** 440/440 trajectories, 109,560 successful checkpoint records, zero failed/unavailable records, report generated. Primary descriptive mean normalized reconstruction RMSE over N=32/64/98:

| Arm | ITG–TEM | ITG–KBM |
|---|---:|---:|
| grid | 0.034585 | 0.050271 |
| GP variance, Matérn-3/2 | 0.035526 | 0.057129 |
| GP variance, Matérn-1/2 | 0.035649 | 0.055274 |
| GP blend, Matérn-3/2 | 0.039049 | 0.055254 |
| VURS | 0.039217 | 0.061802 |
| tetrahedral refinement | 0.040676 | 0.070443 |
| GP gradient, Matérn-3/2 | 0.041586 | 0.055161 |
| VWRS | 0.042510 | 0.075443 |
| GP blend, Matérn-1/2 | 0.042852 | 0.066511 |
| historical sglib adapter | 0.049569 | 0.046967 |
| GP gradient, Matérn-1/2 | 0.056087 | 0.092641 |

VURS does not have the lowest mean on either fixed case. Sparse-grid performance is case dependent; its ITG–KBM early-budget lead does not persist to the lowest final-budget error. These averages are descriptive: the paired interval/multiplicity analysis has not been completed. The registered fixed-function, historical-adapter and legacy-VURS limitations remain. No NSTX/default-method claim follows. Saved [report](C:/Users/joesc/.codex/visualizations/2026/10/21/01a15c9d-02de-7430-95f3-162603401e42/proxy-study-20261020/index.html) and [raw results](C:/Users/joesc/.codex/visualizations/2026/10/21/01a15c9d-02de-7430-95f3-162603401e42/proxy-study-20261020/results.json).

### MGKDB predictor selection — complete

**2026-10-20 22:41:** launched hidden process PID 22092; syntax check passed and grouped fitting is progressing. Audited raw/valid counts: development 1,774/1,482; held-out campaign 1,463/1,153. Development has 28 joined directory groups after exact-duplicate protection; no additional development rows matched held-out full-input signatures. Twenty validation seeds × eight configurations × two response fits are planned. No model ranking inspected or claimed at launch. Results: `C:/Users/joesc/.codex/visualizations/2026/10/21/01a15c9d-02de-7430-95f3-162603401e42/mgkdb-regression-20261020/`; sibling stdout/stderr logs and PID file record execution. Source/data hashes and packages are in `manifest.json`.

**Completion verified 2026-10-21:** finished 2026-10-20 22:42:51, about 82 seconds after launch; all 160 configuration/split evaluations completed (two target fits each). The registered combined development criterion selected `extra1` (Extra Trees, 200 trees, leaf size 1). Independent target fits were already used; only configuration selection was shared.

| Target | Selected-model test RMSE | Mean-baseline test RMSE | Selected-model test MAE | Selected-model test R2 |
|---|---:|---:|---:|---:|
| gamma | 0.331143 | 0.630482 | 0.196334 | 0.364005 |
| omega | 4.270705 | 4.129250 | 2.172157 | -0.106972 |

Frequency baseline MAE is 2.808817: Extra Trees improves MAE but worsens RMSE, so tail/regime error deserves investigation. Gamma transfer improves over mean prediction, but neither these values nor development selection establishes a universally strong surrogate. Raw units are the archive's units. [Held-out scores](C:/Users/joesc/.codex/visualizations/2026/10/21/01a15c9d-02de-7430-95f3-162603401e42/mgkdb-regression-20261020/holdout_scores.json).

**Post-hoc target-specific development reading (2026-10-21, not a new test):** mean fold NRMSE for gamma is lowest for `extra1` (0.473002; `hist31` 0.504969). For omega it is lowest for `extra4` (0.513911), versus `extra1` 0.523016, about 1.7% lower. This is a modest development difference, not evidence that switching leaf size repairs transfer. `extra4` has not been tested on pscans3 in this study. Values come from the existing [development scores](C:/Users/joesc/.codex/visualizations/2026/10/21/01a15c9d-02de-7430-95f3-162603401e42/mgkdb-regression-20261020/development_scores.json); no fits were rerun for this reading.
