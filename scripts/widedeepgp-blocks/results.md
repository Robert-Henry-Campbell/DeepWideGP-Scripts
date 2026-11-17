# Results: the closed-form block ridge (methods.md section 11)

Aggregates only; every number before the unlock is a tuning-set number. Per-disease tables:
`results/<rung>/check_block_ridge.csv` and `results/<rung>/blocks_linear.csv`.

## Bottom rung hm3plus_n13199 (17 November 2026, GPU 2, RTX 6000 Ada)

**Build.** 13,199 training rows, folds of 2,640 / 2,640 / 2,640 / 2,640 / 2,639 (permutation position mod 5);
431 Privé blocks; penalties h² 0.05, 0.15, 0.45 (λ = M / h² = 25.2, 8.39, 2.80 million, M = 1,258,200).
Weights for the six models (five fold models and the full model): 2,202 s in all (chromosome 1 234 s,
2 213 s, 3 362 s with its 15,194-SNP block, 6 245 s; chromosomes 15 to 22 under 30 s each), 45 GB of GPU
memory at the peak with `--gram_gb 20`. Scoring: training 69 s, tuning 88 s, related White British 237 s,
Asian 63 s, Black 51 s (508 s; 190 to 240 rows/s, the per-block channel products and the raw channel
writes, not the store reads). Products: 6.5 GB under `blocks/` (six weight files of 453 MB, int8 channels
of 39 KB per participant for the five splits: 0.51 + 0.78 + 2.23 + 0.36 + 0.30 GB, PRS files, manifests).

**Checks** (`check_block_ridge.py`, `results/hm3plus_n13199/check_block_ridge.md`).
- Random block 203 (chromosome 11, 4,420 SNPs): max |stored − numpy| / max |numpy| = 3.5e-8 for the fold-0
  weights and 3.1e-8 for the full weights (float32 storage precision), against an independent float64 ridge
  from dosages read from the store.
- int8 round trip on the block's fold-0 training rows: RMS error 0.0080 SD and maximum 0.0139 SD (the
  uniform rounding error of a 1/36 grid is 0.0080 RMS and 0.0139 maximum), 0.05% of values clipped at
  ±3.5 SD (0.047% for a Gaussian).
- Sum of the dequantised channels against the stored PRS over 2,000 tuning rows: relative RMS error median
  0.013, maximum 0.016 over the 90 channels; every row has at least one clipped channel among its 38,790
  (as 0.05% × 38,790 ≈ 18 per row implies), and the clip, not the rounding, sets this error.
- Out-of-fold training AUROC against tuning AUROC at the chosen penalty, mean over the 30 diseases: 0.5225
  against 0.5207; 27 of 30 diseases agree within 2 bootstrap SE (200 resamples; K44, I25 and M47 do not,
  their out-of-fold training AUROC the higher by 0.028 to 0.044).
- Pseudo-fold variance ratio (tuning-set score variance / training out-of-fold score variance): median
  1.000, range 0.972 to 1.040.
- The three penalties give nearly the same PRS at this N: mean tuning AUROC 0.5204, 0.5203 and 0.5201 for
  h² 0.05, 0.15 and 0.45; chosen 16, 1 and 13 times. With λ = 2.8 to 25 million against N = 13,199 every
  block ridge is close to Z'R / λ, so the penalties differ by little more than a scale factor per block;
  the choice is noise at this rung and should separate at the upper rungs (λ / N falls from 212 to 13 at
  the strongest penalty).

Per disease (tuning set, PRS alone, no covariates):

| label | cases train | cases val | chosen h² | AUROC train (out of fold) | AUROC tuning | variance ratio |
|---|---|---|---|---|---|---|
| I10 | 4199 | 6255 | 0.45 | 0.5378 | 0.5309 | 0.991 |
| E78 | 2181 | 3153 | 0.05 | 0.5243 | 0.5258 | 1.006 |
| K57 | 1895 | 2784 | 0.45 | 0.5239 | 0.5211 | 1.019 |
| K21 | 1535 | 2387 | 0.05 | 0.5165 | 0.5262 | 1.013 |
| K29 | 1523 | 2238 | 0.05 | 0.5258 | 0.5194 | 0.975 |
| K44 | 1566 | 2271 | 0.05 | 0.5492 | 0.5209 | 0.985 |
| I25 | 1484 | 2126 | 0.05 | 0.5578 | 0.5210 | 1.001 |
| M19 | 1359 | 1966 | 0.45 | 0.5306 | 0.5236 | 0.997 |
| I84+K64 | 1230 | 1962 | 0.05 | 0.5019 | 0.5185 | 1.002 |
| I48 | 1152 | 1670 | 0.05 | 0.5203 | 0.5221 | 1.023 |
| M17 | 1168 | 1675 | 0.05 | 0.5152 | 0.5158 | 0.989 |
| C44 | 1079 | 1676 | 0.45 | 0.5161 | 0.5247 | 1.009 |
| E11 | 1032 | 1663 | 0.45 | 0.5232 | 0.5442 | 1.030 |
| E66 | 1058 | 1478 | 0.45 | 0.5239 | 0.5275 | 0.992 |
| M54 | 899 | 1310 | 0.45 | 0.5117 | 0.5269 | 1.018 |
| D12 | 887 | 1309 | 0.05 | 0.5240 | 0.5118 | 1.024 |
| I20 | 911 | 1284 | 0.05 | 0.5053 | 0.5004 | 1.040 |
| E03 | 819 | 1211 | 0.05 | 0.5279 | 0.5277 | 0.972 |
| K40 | 735 | 1140 | 0.45 | 0.5055 | 0.5018 | 1.000 |
| M16 | 802 | 1106 | 0.05 | 0.5369 | 0.5131 | 0.977 |
| K80 | 779 | 1118 | 0.05 | 0.5207 | 0.5390 | 0.992 |
| N17 | 713 | 1007 | 0.05 | 0.5075 | 0.5039 | 1.017 |
| N18 | 659 | 943 | 0.05 | 0.4973 | 0.4965 | 0.986 |
| J44 | 603 | 986 | 0.45 | 0.5186 | 0.5293 | 0.995 |
| M23 | 621 | 872 | 0.15 | 0.5095 | 0.5343 | 0.999 |
| M47 | 583 | 797 | 0.45 | 0.5527 | 0.5090 | 0.999 |
| I95 | 591 | 834 | 0.45 | 0.5267 | 0.5234 | 1.014 |
| I50 | 565 | 814 | 0.45 | 0.5345 | 0.5120 | 0.993 |
| M81 | 532 | 840 | 0.05 | 0.5055 | 0.5089 | 1.014 |
| I21 | 516 | 730 | 0.45 | 0.5253 | 0.5423 | 0.990 |

For context, the PGI arm's tuning-set metrics at this rung (`PGI/results/hm3plus_n13199/tune_metrics.csv`,
means over the 30 diseases): PGI alone 0.519 to 0.525 AUROC for the four methods finished (C+T 0.5248,
lassosum2 0.5213, LDpred2-auto 0.5220, SBayesRC 0.5188), covariates alone 0.6671 through the spline
basis, ΔAUROC 0.0019 to 0.0028. The cross-fitted ridge PRS alone at 0.5207 sits in that range.

**Linear comparators** (`fit_block_linear.py`, `results/hm3plus_n13199/blocks_linear.csv`; tuning set, with
the 12 covariates, linear basis). Mean over the 30 diseases: logistic ridge on the 431 × 90 channels plus
covariates 0.6603 AUROC (training rows, cross-fitted channels: 0.7021); the cross-fitted ridge PRS at the
chosen penalty plus covariates 0.6611. The twin beats the plain PRS for 9 diseases and loses for 21 (mean
difference −0.0008); its penalty sits at the grid's upper edge, λ = 1e6, for 27 diseases (3 to 5 L-BFGS
iterations from the covariates-only start) and at 1e5 for I84+K64, I95 and I21 (20 to 27 iterations). At this
rung the channels carry too little signal for 38,790 free coefficients: the twin collapses towards the
covariates-only model and the plain PRS, a single fixed combination of the same channels, is the better
linear use of them. The first version of the twin, scikit-learn's L2 logistic regression with one penalty on
every coefficient including the covariates, scored 0.52 to 0.55 on the same inputs (methods section 11).
Context: the PGI arm's combined models at this rung reach 0.6690 to 0.6700 with a covariates-only reference
of 0.6671 through its spline basis; the linear covariate basis here gives lower numbers for the same genetic
content, and the shared protocol's refit puts both arms on the spline basis before any comparison. The
fits took 1,620 s for the 30 diseases (five penalties plus the two reference fits each, 8 threads).

**Warm start** (`make_warm_start.py`): `blocks/init_ridge_d16.pt`, 22 tables [L_i, 16] float32 from the
rank-16 SVD of the full model's [1,258,200, 90] weights; the 16 leading singular values keep 74.9% of the
weights' squared norm; partition digest 8953c97f56927860 computed with the model's own function
(`genostore.datamodule.partition_digest`, present in the working tree of DeepWideGP at the time).

**Timing summary for the main session's chains.** Per rung: folds under a minute; the weights about 37
minutes at the bottom rung (27 minutes of eigendecompositions, the same at every rung, plus grams that
scale with N: about 48 minutes of GPU matmul at the top rung in float64); scoring about 200 rows/s; the
comparators about 55 s per disease at the bottom rung, scaling with N; the warm start about a minute.
GPU memory peaked at 45 GB with `--gram_gb 20`; a smaller value trades passes for headroom.
