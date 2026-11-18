# Results: the classic DeepGP comparator

*Aggregates only; tuning set (20,000 participants); every number is a mean AUROC or a per-disease AUROC. The test
set is locked. Updated 18 November 2026.*

## Top rung (215,872 training participants), single runs with the published settings

Two runs of the published architecture (width 32, one layer, attention pooling, batch 64, learning rate 7 × 10⁻⁴,
weight decay 10⁻⁴, dropout 0.2, plain cross-entropy, the 12 covariates), one training each, seed 1, early stopping
on the tuning set after 8 passes without improvement (methods, "The 30-disease variant"):

| run | SNPs | outputs | best mean tuning AUROC | at epoch-equivalents | stopped at |
|---|---|---|---|---|---|
| `classic_i10_n215872_quick` (I10 only) | 28,536 (I10 GWAS p ≤ 0.05, clumped) | 1 | 0.6987 | 4.75 | 6.76 |
| `classic30_n215872` (all 30 diseases) | 179,566 (p ≤ 0.05 in any of the 30 GWAS, clumped once) | 30 | 0.6627 | 7.76 | 9.76 |

Yardsticks on the same tuning set: the covariates alone 0.6627 (I10 0.6990); the logistic ridge on the cross-fitted
block channels 0.6732; the cross-fitted block ridge PRS plus covariates 0.6777 (`widedeepgp-blocks/results/hm3plus/blocks_linear.csv`).
Both classic runs sit at the covariate floor: the 30-output run beats the ridge PRS for none of the 30 diseases, and its
I10 output (0.6990) equals the single-disease run's (0.6987). Multi-disease
training alone does not rescue the published architecture at this size; the two runs are single trainings, not searches.

### Per disease, the 30-output run against the linear comparators

| disease | classic-30 | logistic ridge twin | ridge PRS + covariates |
|---|---|---|---|
| I10 | 0.6990 | 0.7173 | 0.7229 |
| E78 | 0.6916 | 0.7044 | 0.7041 |
| K57 | 0.6250 | 0.6371 | 0.6485 |
| K21 | 0.5897 | 0.6018 | 0.6026 |
| K29 | 0.6001 | 0.6096 | 0.6117 |
| K44 | 0.6161 | 0.6244 | 0.6327 |
| I25 | 0.7307 | 0.7393 | 0.7395 |
| M19 | 0.6602 | 0.6656 | 0.6675 |
| I84+K64 | 0.5358 | 0.5501 | 0.5543 |
| I48 | 0.7470 | 0.7559 | 0.7573 |
| M17 | 0.6293 | 0.6384 | 0.6462 |
| C44 | 0.6680 | 0.6746 | 0.6860 |
| E11 | 0.6691 | 0.6992 | 0.7039 |
| E66 | 0.5541 | 0.5990 | 0.6103 |
| M54 | 0.5873 | 0.5994 | 0.6024 |
| D12 | 0.6233 | 0.6259 | 0.6285 |
| I20 | 0.7040 | 0.7157 | 0.7143 |
| E03 | 0.6805 | 0.7034 | 0.7128 |
| K40 | 0.7793 | 0.7833 | 0.7884 |
| M16 | 0.6591 | 0.6626 | 0.6759 |
| K80 | 0.6094 | 0.6209 | 0.6333 |
| N17 | 0.6973 | 0.7006 | 0.7014 |
| N18 | 0.7247 | 0.7273 | 0.7318 |
| J44 | 0.6875 | 0.6973 | 0.7024 |
| M23 | 0.5726 | 0.5770 | 0.5825 |
| M47 | 0.6408 | 0.6460 | 0.6445 |
| I95 | 0.6911 | 0.6929 | 0.6938 |
| I50 | 0.7178 | 0.7246 | 0.7264 |
| M81 | 0.7609 | 0.7623 | 0.7626 |
| I21 | 0.7310 | 0.7389 | 0.7417 |
| **mean** | **0.6627** | **0.6732** | **0.6777** |
