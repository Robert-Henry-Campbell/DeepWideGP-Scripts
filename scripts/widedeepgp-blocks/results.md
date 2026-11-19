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
lassosum2 0.5213, LDpred2-auto 0.5220, SBayesRC 0.5188), covariates alone 0.6671 through the spline-by-sex
basis that was the protocol's until 18 November 2026 (0.6653 through the linear basis that replaced it, PGI
methods 6.1 and 13), ΔAUROC 0.0019 to 0.0028. The cross-fitted ridge PRS alone at 0.5207 sits in that range.

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
of 0.6671 through the spline-by-sex basis then in force (0.6653 through the linear basis, the protocol since
18 November 2026); the shared protocol's refit puts both arms on the same basis before any comparison. The
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

## Top rung hm3plus: the fold-routed first layer, frozen and trained, three readouts and the recipe (18 November 2026, GPUs 2 to 4)

Single trainings of the S3 transformer (d_model 64, four layers, four heads) from the fold-routed ridge warm start
(`blocks/init_ridge_d16.pt`: five copies of the per-SNP table, one per fold of the block ridge, in the full ridge's rank-16
basis, scaled so the pooled block tokens have unit RMS), with the chain's recipe (learning rate 1e-4, weight decay 1e-3,
batch 64, plateau schedule, 40% input masking and dropout 0.5 unless stated), budget ten epoch-equivalents, validation on
the whole tuning set every quarter epoch-equivalent, patience 16 validations after a 2,000-step burn-in, and no
hyper-parameter search (`runs/blocks_diag/run_diag_folds_s3.sh`, `run_diag_folds_recipe.sh`, `run_diag_readouts.sh`;
methods.md section 12, items 15, 17 and 18). Mean tuning AUROC over the 30 diseases by epoch-equivalent. Yardsticks on the
same tuning set: covariates only 0.6627 (the 12 covariates, linear basis, one unpenalised logistic regression per disease,
fitted for this table), the logistic ridge twin on the 431 × 90 cross-fitted channels plus covariates 0.6732, and the
cross-fitted ridge PRS plus covariates 0.6777 (`results/hm3plus/blocks_linear.csv`).

| run | first layer | readout | mask / dropout | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | best (at epoch-eq) | stopped | I10 at best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frozen mean | frozen | mean | 0.4 / 0.5 | 0.6624 | 0.6704 | 0.6710 | 0.6715 | 0.6714 | 0.6715 | 0.6722 | 0.6700 | 0.6707 | 0.6708 | 0.6722 (7.0) | 10.0 budget | 0.7119 |
| frozen mean, no mask | frozen | mean | 0 / 0.2 | 0.6665 | 0.6708 | 0.6714 | 0.6721 | 0.6724 | 0.6731 | 0.6725 | 0.6729 | 0.6733 | 0.6735 | 0.6735 (10.0) | 10.0 budget | 0.7148 |
| frozen mean, no mask, seed 2 | frozen | mean | 0 / 0.2 | 0.6692 | 0.6705 | 0.6708 | 0.6716 | 0.6720 | 0.6727 | 0.6732 | 0.6728 | 0.6733 | 0.6734 | 0.6737 (8.3) | 10.0 budget | 0.7148 |
| frozen atten | frozen | atten | 0.4 / 0.5 | 0.6615 | 0.6703 | 0.6709 | 0.6715 | 0.6718 | 0.6718 | 0.6723 | 0.6704 | 0.6719 | 0.6706 | 0.6723 (7.0) | 10.0 budget | 0.7122 |
| frozen flat | frozen | flat | 0.4 / 0.5 | 0.6535 | 0.6710 | 0.6719 | 0.6721 | 0.6724 | 0.6727 | 0.6717 | 0.6724 | 0.6721 |  | 0.6728 (5.8) | 9.8 early | 0.7180 |
| flat, no mask | frozen | flat | 0 / 0.2 | 0.6705 | 0.6726 | 0.6731 | 0.6717 | 0.6698 | 0.6679 | 0.6617 |  |  |  | 0.6731 (3.0) | 7.0 early | 0.7190 |
| trained mean, κ 1e-2, ε 1e-8 | trained | mean | 0.4 / 0.5 | 0.6623 | 0.6703 | 0.6708 | 0.6703 | 0.6686 | 0.6664 | 0.6632 |  |  |  | 0.6708 (3.0) | 7.0 early | 0.7099 |

The trained run's per-SNP table, compared with its warm start (RMS entry 0.350; root-mean-square change over all
20 million entries; no entry moved by more than 0.1):

| step | 6,813 | 10,000 | 14,678 | 21,544 |
|---|---|---|---|---|
| epoch-equivalents | 2.0 | 3.0 | 4.4 | 6.4 |
| change, % of the entry scale | 1.4 | 2.1 | 3.2 | 4.3 |
| aimless walk from step 6,813 (√steps) would give | 1.4 | 1.7 | 2.1 | 2.5 |

Per disease, tuning AUROC at each run's best validation (the twin and PRS columns from `blocks_linear.csv`; training
cases from the same file):

| disease | cases (train) | covariates | twin | PRS + cov | frozen mean | no mask | atten | flat | trained |
|---|---|---|---|---|---|---|---|---|---|
| I10 | 68,309 | 0.6990 | 0.7173 | 0.7229 | 0.7119 | 0.7148 | 0.7122 | 0.7180 | 0.7099 |
| E78 | 34,953 | 0.6911 | 0.7044 | 0.7041 | 0.7010 | 0.7036 | 0.7011 | 0.7077 | 0.7010 |
| K57 | 30,506 | 0.6259 | 0.6371 | 0.6485 | 0.6302 | 0.6322 | 0.6304 | 0.6336 | 0.6274 |
| K21 | 25,427 | 0.5893 | 0.6018 | 0.6026 | 0.6035 | 0.6051 | 0.6036 | 0.6026 | 0.6002 |
| K29 | 24,564 | 0.6025 | 0.6096 | 0.6117 | 0.6129 | 0.6134 | 0.6136 | 0.6084 | 0.6111 |
| K44 | 24,297 | 0.6146 | 0.6244 | 0.6327 | 0.6229 | 0.6239 | 0.6239 | 0.6242 | 0.6216 |
| I25 | 23,558 | 0.7312 | 0.7393 | 0.7395 | 0.7395 | 0.7410 | 0.7394 | 0.7417 | 0.7375 |
| M19 | 21,544 | 0.6598 | 0.6656 | 0.6675 | 0.6684 | 0.6677 | 0.6690 | 0.6622 | 0.6657 |
| I84+K64 | 21,296 | 0.5373 | 0.5501 | 0.5543 | 0.5468 | 0.5484 | 0.5467 | 0.5495 | 0.5482 |
| I48 | 18,365 | 0.7471 | 0.7559 | 0.7573 | 0.7495 | 0.7506 | 0.7492 | 0.7520 | 0.7485 |
| M17 | 18,227 | 0.6288 | 0.6384 | 0.6462 | 0.6400 | 0.6412 | 0.6405 | 0.6404 | 0.6384 |
| C44 | 17,671 | 0.6675 | 0.6746 | 0.6860 | 0.6687 | 0.6691 | 0.6691 | 0.6739 | 0.6660 |
| E11 | 17,345 | 0.6697 | 0.6992 | 0.7039 | 0.6876 | 0.6907 | 0.6875 | 0.6985 | 0.6849 |
| E66 | 16,800 | 0.5498 | 0.5990 | 0.6103 | 0.6000 | 0.6079 | 0.6012 | 0.6084 | 0.5994 |
| M54 | 14,550 | 0.5878 | 0.5994 | 0.6024 | 0.6116 | 0.6145 | 0.6120 | 0.6049 | 0.6101 |
| D12 | 14,337 | 0.6201 | 0.6259 | 0.6285 | 0.6273 | 0.6294 | 0.6263 | 0.6265 | 0.6266 |
| I20 | 14,234 | 0.7041 | 0.7157 | 0.7143 | 0.7153 | 0.7175 | 0.7158 | 0.7199 | 0.7166 |
| E03 | 13,242 | 0.6791 | 0.7034 | 0.7128 | 0.6862 | 0.6862 | 0.6854 | 0.6921 | 0.6859 |
| K40 | 12,541 | 0.7788 | 0.7833 | 0.7884 | 0.7778 | 0.7790 | 0.7773 | 0.7789 | 0.7758 |
| M16 | 12,286 | 0.6597 | 0.6626 | 0.6759 | 0.6605 | 0.6598 | 0.6609 | 0.6567 | 0.6597 |
| K80 | 12,194 | 0.6090 | 0.6209 | 0.6333 | 0.6201 | 0.6197 | 0.6201 | 0.6202 | 0.6176 |
| N17 | 11,340 | 0.6986 | 0.7006 | 0.7014 | 0.7026 | 0.7049 | 0.7023 | 0.7014 | 0.7034 |
| N18 | 10,337 | 0.7239 | 0.7273 | 0.7318 | 0.7311 | 0.7319 | 0.7310 | 0.7308 | 0.7295 |
| J44 | 9,884 | 0.6883 | 0.6973 | 0.7024 | 0.7007 | 0.7019 | 0.7009 | 0.7008 | 0.7018 |
| M23 | 9,585 | 0.5712 | 0.5770 | 0.5825 | 0.5821 | 0.5807 | 0.5834 | 0.5741 | 0.5794 |
| M47 | 9,492 | 0.6424 | 0.6460 | 0.6445 | 0.6472 | 0.6469 | 0.6476 | 0.6411 | 0.6431 |
| I95 | 9,153 | 0.6911 | 0.6929 | 0.6938 | 0.6937 | 0.6946 | 0.6939 | 0.6932 | 0.6944 |
| I50 | 8,851 | 0.7214 | 0.7246 | 0.7264 | 0.7264 | 0.7273 | 0.7265 | 0.7257 | 0.7257 |
| M81 | 8,604 | 0.7610 | 0.7623 | 0.7626 | 0.7606 | 0.7613 | 0.7603 | 0.7579 | 0.7571 |
| I21 | 8,090 | 0.7316 | 0.7389 | 0.7417 | 0.7398 | 0.7403 | 0.7395 | 0.7385 | 0.7372 |

Counts over the 30 diseases (a run beats a yardstick when its AUROC is higher):

| run | mean | beats covariates | beats twin | beats PRS + cov | median gap to PRS + cov |
|---|---|---|---|---|---|
| frozen mean | 0.6722 | 28 | 16 | 7 | -0.0019 |
| frozen mean, no mask | 0.6735 | 30 | 17 | 12 | -0.0014 |
| frozen mean, no mask, seed 2 | 0.6737 | 29 | 20 | 10 | -0.0011 |
| frozen atten | 0.6724 | 28 | 17 | 10 | -0.0022 |
| frozen flat | 0.6728 | 27 | 14 | 4 | -0.0040 |
| flat, no mask | 0.6731 | 26 | 14 | 8 | -0.0038 |
| trained mean, κ 1e-2, ε 1e-8 | 0.6708 | 26 | 12 | 4 | -0.0038 |

The flat readout against the frozen mean: median difference +0.0001, higher for 16 of 30 diseases,
correlation of the difference with the log training case count +0.42 (I10 +0.0061,
E78 +0.0067; the nine diseases with at least 20,000 training cases +0.0012 on average); its tuning
cross-entropy is worse (0.2576 against 0.2565). The no-mask recipe against the masked one, both frozen mean: median +0.0011,
higher for 24 of 30. The frozen mean's shortfall against the PRS is largest for the common diseases (-0.0052
on average over the nine with at least 20,000 training cases). Attention and mean differ by at most 0.0003 at any validation.
Seed replicates of the no-mask frozen mean run (seeds 2 and 3) and of the no-mask recipe with the flat readout (seeds 1
and 2) started on 18 November at 21:55Z (`run_diag_seeds.sh`); the first of each finished on 19 November (rows above).
Seed 2 of the frozen mean, no mask, reproduces seed 1: means 0.6737 against 0.6735, per-disease differences 0.0010 on
average and 0.0040 at most (E66), median 0.0000, higher for 15 of 30; the mean of the two seeds is 0.6736 and beats the
covariates for 29 diseases, the twin for 18 and the PRS for 13. The flat readout under the no-mask recipe overfits: it
peaks at epoch-equivalent 3 (0.6731) and falls to 0.6617 by 7, where the early stop ends it, whereas under the masked
recipe the same readout stayed between 0.6717 and 0.6728 from epoch-equivalent 2 to 9.8. At its best it is level with
the frozen mean, no mask (median difference 0.0000, higher for 15 of 30; I10 +0.0042, E78 +0.0026) with a worse tuning
cross-entropy (0.2578 against 0.2564). Seed 3 of the frozen mean and seed 2 of the flat run were still training; their
rows follow when they finish.


## The linear comparators across the ladder, and the warm starts of every rung (18 November 2026)

Means over the 30 diseases of the tuning-set AUROC (`results/<rung>/blocks_linear.csv`). Every model is fitted on the
rung's training rows with the 12 covariates unpenalised; the covariates-only column was fitted for this table by one
unpenalised logistic regression per disease on the same training rows, so it sits below the PGI arm's covariates-only
reference, which is the shared protocol's in-sample refit on the tuning set (0.6653 at the bottom rung, linear basis).
The twin's penalty was chosen on the tuning set over the grid 1e2 to 1e6.

| rung (training rows) | covariates only | logistic ridge twin | twin on its training rows | ridge PRS + covariates | twin − covariates | PRS − covariates | twin beats PRS (of 30) | twin penalty at the grid's upper edge (of 30) |
|---|---|---|---|---|---|---|---|---|
| 13,199 | 0.6598 | 0.6603 | 0.7021 | 0.6611 | +0.0005 | +0.0013 | 9 | 27 |
| 26,287 | 0.6613 | 0.6619 | 0.6932 | 0.6638 | +0.0006 | +0.0025 | 6 | 29 |
| 53,498 | 0.6620 | 0.6633 | 0.6928 | 0.6670 | +0.0013 | +0.0050 | 0 | 28 |
| 107,743 | 0.6625 | 0.6671 | 0.7156 | 0.6718 | +0.0046 | +0.0093 | 2 | 20 |
| 215,872 | 0.6627 | 0.6732 | 0.7286 | 0.6777 | +0.0104 | +0.0149 | 3 | 12 |

The plain cross-fitted PRS beats the twin at every rung, by 0.0008 at the bottom and 0.0045 at the top, and the
twin's penalty sits at the grid's upper edge for most diseases at every rung: 38,790 free coefficients on the channels
are too many at these N, and the single fixed combination of the same channels is the better linear use of them. Both
gains over the covariates grow with N (the PRS from +0.0013 to +0.0149). Rung 26,287's ridge check
(`results/hm3plus_n26287/check_block_ridge.md`, built with its twins on 18 November, 22:08 to 22:59Z, 10 threads):
out-of-fold training AUROC 0.5262 against tuning 0.5272 at the chosen penalty, 27 of 30 diseases within 2 SE,
pseudo-fold variance ratio median 0.999 (0.962 to 1.025).

**Warm starts of every rung.** All five `blocks/init_ridge_d16.pt` files are now the fold-routed kind of methods.md
item 15: five copies of the 22 tables, one per fold of the block ridge, in the full ridge's rank-16 basis, scaled so the
pooled block tokens have unit RMS (entry RMS 0.273, 0.282, 0.297, 0.319 and 0.350 from the bottom rung to the top), made
18 November 21:57 to 22:08Z on GPU 2 (`runs/blocks/run_lin_rest.sh`, two to three minutes a rung, variance kept 0.75).
The earlier single-copy, unscaled files are kept beside them as `init_ridge_d16_unscaled.pt` and are used by nothing.

