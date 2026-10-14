# TabPFN label arms vs VURS on the MGKDB pools

## hatch_pscans_global

| endpoint | comparison | arm | other | d (CI) | wins | Holm p | verdict |
|---|---|---|---|---|---|---|---|
| S1 | A vurs-boundary-pfn - vurs-boundary | 1.490 | 1.474 | +0.017 [-0.012, +0.044] | 7/20 | 0.81 | same speed |
| S1 | B vurs-boundary-pfn - vurs | 1.490 | 1.464 | +0.027 [+0.000, +0.052] | 6/20 | 0.379 | inconclusive |
| S1 | C pfn-entropy - vurs | 1.700 | 1.464 | +0.236 [+0.163, +0.318] | 0/20 | 1.14e-05 | slower |
| S2 | A vurs-boundary-pfn - vurs-boundary | 1.492 | 1.434 | +0.057 [-0.422, +0.525] | 9/20 | 1 | inconclusive |
| S2 | B vurs-boundary-pfn - vurs | 1.492 | 1.358 | +0.133 [-0.221, +0.459] | 7/20 | 1 | inconclusive |
| S2 | C pfn-entropy - vurs | 3.241 | 1.358 | +1.883 [-0.067, +4.907] | 6/20 | 1 | inconclusive |

| secondary | comparison | d (CI) | wins | p |
|---|---|---|---|---|
| S1 band | A | +0.008 [-0.021, +0.038] | 7/20 | 0.498 |
| S1 band | B | +0.017 [-0.008, +0.040] | 7/20 | 0.202 |
| S1 band | C | +0.208 [+0.135, +0.290] | 0/20 | 1.91e-06 |
| S1 band | space-filling vs vurs | +0.004 [-0.023, +0.032] | 10/20 | 0.985 |
| S1 TabPFN scorer | A | -0.010 [-0.043, +0.025] | 11/20 | 0.498 |
| S1 TabPFN scorer | B | -0.030 [-0.080, +0.020] | 10/20 | 0.452 |
| S1 TabPFN scorer | C | +0.256 [+0.100, +0.436] | 3/20 | 0.00271 |
| S1 TabPFN scorer | space-filling vs vurs | +0.028 [-0.013, +0.069] | 8/20 | 0.312 |
| S1 TabPFN scorer, band | A | +0.001 [-0.031, +0.034] | 11/20 | 0.869 |
| S1 TabPFN scorer, band | B | -0.024 [-0.070, +0.022] | 9/20 | 0.622 |
| S1 TabPFN scorer, band | C | +0.239 [+0.090, +0.412] | 4/20 | 0.00271 |
| S1 TabPFN scorer, band | space-filling vs vurs | +0.050 [+0.008, +0.093] | 8/20 | 0.0484 |
| label accuracy | A | +0.001 [-0.009, +0.012] | 10/20 | 0.985 |
| label accuracy | B | -0.005 [-0.013, +0.003] | 8/20 | 0.368 |
| label accuracy | C | -0.039 [-0.065, -0.016] | 4/20 | 0.00558 |
| label accuracy | space-filling vs vurs | +0.014 [+0.006, +0.021] | 16/20 | 0.00315 |
| ctr_macro_nrmse | A | -0.070 [-0.169, +0.030] | 11/20 | 0.261 |
| ctr_macro_nrmse | B | -0.089 [-0.195, +0.005] | 12/20 | 0.143 |
| ctr_macro_nrmse | C | +0.035 [-0.172, +0.249] | 11/20 | 1 |
| ctr_macro_nrmse | space-filling vs vurs | +0.007 [-0.138, +0.142] | 9/20 | 0.784 |
| S1 at N=256 | A | +0.023 [+0.001, +0.045] | 8/20 | 0.0583 |
| S1 at N=256 | B | -0.011 [-0.028, +0.007] | 14/20 | 0.245 |
| S1 at N=256 | C | +0.081 [+0.058, +0.103] | 2/20 | 3.62e-05 |
| S1 at N=256 | space-filling vs vurs | -0.037 [-0.058, -0.016] | 14/20 | 0.00558 |
| S2 at N=256 | A | +0.163 [+0.111, +0.212] | 3/20 | 6.29e-05 |
| S2 at N=256 | B | +0.150 [+0.106, +0.193] | 1/20 | 1.91e-05 |
| S2 at N=256 | C | +0.707 [+0.582, +0.843] | 0/20 | 1.91e-06 |
| S2 at N=256 | space-filling vs vurs | +0.011 [-0.030, +0.050] | 8/20 | 0.571 |

vurs-boundary-pfn: mean realized label surprise 1.324 bits per paid run, median placement 1750 s

pfn-entropy: mean realized label surprise 1.471 bits per paid run, median placement 1606 s

## hatch_pscans3

| endpoint | comparison | arm | other | d (CI) | wins | Holm p | verdict |
|---|---|---|---|---|---|---|---|
| S1 | A vurs-boundary-pfn - vurs-boundary | 1.260 | 1.244 | +0.016 [-0.014, +0.047] | 8/20 | 0.649 | same speed |
| S1 | B vurs-boundary-pfn - vurs | 1.260 | 1.252 | +0.009 [-0.029, +0.042] | 7/20 | 0.649 | same speed |
| S1 | C pfn-entropy - vurs | 1.403 | 1.252 | +0.152 [+0.105, +0.197] | 3/20 | 0.000286 | slower |
| S2 | A vurs-boundary-pfn - vurs-boundary | 2.036 | 1.820 | +0.216 [-0.113, +0.512] | 7/20 | 0.319 | inconclusive |
| S2 | B vurs-boundary-pfn - vurs | 2.036 | 2.057 | -0.021 [-0.421, +0.317] | 7/20 | 0.649 | inconclusive |
| S2 | C pfn-entropy - vurs | 2.257 | 2.057 | +0.200 [-0.384, +0.679] | 4/20 | 0.456 | inconclusive |

| secondary | comparison | d (CI) | wins | p |
|---|---|---|---|---|
| S1 band | A | -0.005 [-0.027, +0.018] | 10/20 | 0.571 |
| S1 band | B | -0.019 [-0.049, +0.008] | 8/20 | 0.756 |
| S1 band | C | +0.103 [+0.063, +0.140] | 3/20 | 8.2e-05 |
| S1 band | space-filling vs vurs | +0.053 [+0.023, +0.084] | 4/20 | 0.00315 |
| S1 TabPFN scorer | A | -0.067 [-0.131, +0.003] | 15/20 | 0.0484 |
| S1 TabPFN scorer | B | -0.072 [-0.145, +0.000] | 11/20 | 0.165 |
| S1 TabPFN scorer | C | -0.005 [-0.082, +0.075] | 8/20 | 1 |
| S1 TabPFN scorer | space-filling vs vurs | +0.077 [+0.010, +0.144] | 7/20 | 0.0441 |
| S1 TabPFN scorer, band | A | -0.093 [-0.144, -0.036] | 15/20 | 0.00639 |
| S1 TabPFN scorer, band | B | -0.113 [-0.176, -0.051] | 17/20 | 0.00199 |
| S1 TabPFN scorer, band | C | -0.058 [-0.126, +0.017] | 15/20 | 0.04 |
| S1 TabPFN scorer, band | space-filling vs vurs | +0.095 [+0.039, +0.151] | 6/20 | 0.00486 |
| label accuracy | A | -0.024 [-0.035, -0.015] | 2/20 | 0.000168 |
| label accuracy | B | -0.023 [-0.033, -0.013] | 3/20 | 8.2e-05 |
| label accuracy | C | -0.025 [-0.040, -0.010] | 5/20 | 0.00486 |
| label accuracy | space-filling vs vurs | +0.003 [-0.004, +0.011] | 11/20 | 0.498 |
| ctr_macro_nrmse | A | -0.002 [-0.049, +0.033] | 9/20 | 0.216 |
| ctr_macro_nrmse | B | +0.024 [+0.010, +0.041] | 4/20 | 0.00315 |
| ctr_macro_nrmse | C | +0.076 [+0.027, +0.155] | 1/20 | 5.72e-06 |
| ctr_macro_nrmse | space-filling vs vurs | +1.408 [+0.008, +4.194] | 5/20 | 0.00831 |
| S1 at N=256 | A | -0.030 [-0.057, -0.004] | 14/20 | 0.0637 |
| S1 at N=256 | B | -0.079 [-0.100, -0.060] | 19/20 | 3.81e-06 |
| S1 at N=256 | C | +0.067 [+0.049, +0.086] | 0/20 | 1.91e-06 |
| S1 at N=256 | space-filling vs vurs | +0.013 [-0.012, +0.037] | 10/20 | 0.368 |
| S2 at N=256 | A | +0.095 [+0.022, +0.162] | 6/20 | 0.0172 |
| S2 at N=256 | B | +0.118 [+0.022, +0.215] | 5/20 | 0.0532 |
| S2 at N=256 | C | +0.592 [+0.454, +0.735] | 0/20 | 1.91e-06 |
| S2 at N=256 | space-filling vs vurs | -0.032 [-0.129, +0.076] | 12/20 | 0.33 |

vurs-boundary-pfn: mean realized label surprise 0.866 bits per paid run, median placement 1116 s

pfn-entropy: mean realized label surprise 0.978 bits per paid run, median placement 1018 s

Claim (TabPFN boundary term helps): False
