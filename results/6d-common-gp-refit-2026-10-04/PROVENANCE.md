# Common-GP refit of the 6D final designs, provenance

- **Run date:** 2026-10-04
- **Source branch:** `codex/benchmark-2d`
- **Code commit:** `4c48870` (`Add a common-GP refit of frozen final designs`)
- **Designs:** final N=512 designs from `results/6d-ionut-spine-2026-09-20` (`98a9247`): eight unbumped Ionut proxies, seeds 0-2
- **Result status:** complete; 6 arms x 8 cases x 3 seeds = 144 designs, each refitted with two kernels

## Command

```bash
PYTHONPATH=. .venv/Scripts/python.exe -m scripts.refit_common_gp --source results/6d-ionut-spine-2026-09-20 --output results/6d-common-gp-refit-2026-10-04 --workers 12
```

## What this tests

Whether VURS's lower native prediction accuracy than the Matern-3/2 uncertainty-only GP comes from where VURS places points or from its Matern-1/2 predictor. Every arm's frozen design is refitted with the same GP (`fit_gp` settings) under Matern-3/2 and Matern-1/2 and scored on 16,384 Sobol reference points.

## Result: ratio to the best design per case, mean / worst over the eight cases

| Design | Matern-3/2 refit | Matern-1/2 refit |
|---|---|---|
| vurs | 1.08 / 1.28 | 1.02 / 1.10 |
| space-filling | 1.09 / 1.34 | 1.14 / 1.39 |
| gpr-var (Matern-3/2 uncertainty) | 1.12 / 1.23 | 1.12 / 1.27 |
| gpr-m05-var (Matern-1/2 uncertainty) | 1.15 / 1.33 | 1.13 / 1.27 |
| vwrs | 1.42 / 2.12 | 1.28 / 1.71 |
| gpr-u50-g50 (Matern-3/2 blend) | 1.63 / 3.43 | 1.14 / 1.43 |

The VURS design gives the lowest mean error under both kernels although it was acquired with Matern-1/2 only, so its native deficit was the predictor, not the placement. Differences among VURS, space-filling and the uncertainty-only designs are within about 10% on average; the large separations are VWRS (no uncertainty term) and the gradient blend under the kernel it was acquired with. Bumped variants were not refitted; one budget, three seeds.
