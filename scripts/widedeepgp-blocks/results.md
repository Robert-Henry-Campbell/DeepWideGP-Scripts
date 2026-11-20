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
| frozen mean, no mask, seed 3 | frozen | mean | 0 / 0.2 | 0.6687 | 0.6706 | 0.6710 | 0.6716 | 0.6717 | 0.6728 | 0.6730 | 0.6738 | 0.6737 | 0.6738 | 0.6738 (8.0) | 10.0 budget | 0.7167 |
| frozen atten | frozen | atten | 0.4 / 0.5 | 0.6615 | 0.6703 | 0.6709 | 0.6715 | 0.6718 | 0.6718 | 0.6723 | 0.6704 | 0.6719 | 0.6706 | 0.6723 (7.0) | 10.0 budget | 0.7122 |
| frozen flat | frozen | flat | 0.4 / 0.5 | 0.6535 | 0.6710 | 0.6719 | 0.6721 | 0.6724 | 0.6727 | 0.6717 | 0.6724 | 0.6721 |  | 0.6728 (5.8) | 9.8 early | 0.7180 |
| flat, no mask | frozen | flat | 0 / 0.2 | 0.6705 | 0.6726 | 0.6731 | 0.6717 | 0.6698 | 0.6679 | 0.6617 |  |  |  | 0.6731 (3.0) | 7.0 early | 0.7190 |
| flat, no mask, seed 2 | frozen | flat | 0 / 0.2 | 0.6711 | 0.6725 | 0.6723 | 0.6714 | 0.6688 | 0.6638 |  |  |  |  | 0.6725 (2.0) | 6.0 early | 0.7176 |
| trained mean, κ 1e-2, ε 1e-8 | trained | mean | 0.4 / 0.5 | 0.6623 | 0.6703 | 0.6708 | 0.6703 | 0.6686 | 0.6664 | 0.6632 |  |  |  | 0.6708 (3.0) | 7.0 early | 0.7099 |

The trained run's per-SNP table, compared with its warm start (RMS entry 0.350; root-mean-square change over all
20 million entries; no entry moved by more than 0.1):

| step | 6,813 | 10,000 | 14,678 | 21,544 |
|---|---|---|---|---|
| epoch-equivalents | 2.0 | 3.0 | 4.4 | 6.4 |
| change, % of the entry scale | 1.4 | 2.1 | 3.2 | 4.3 |
| aimless walk from step 6,813 (√steps) would give | 1.4 | 1.7 | 2.1 | 2.5 |

Per disease, tuning AUROC at each run's best validation (the twin and PRS columns from `blocks_linear.csv`; training
cases from the same file; `no mask, 3 seeds` is the mean over seeds 1 to 3 of the no-mask frozen mean run, every other
run column seed 1):

| disease | cases (train) | covariates | twin | PRS + cov | frozen mean | no mask | no mask, 3 seeds | atten | flat | trained |
|---|---|---|---|---|---|---|---|---|---|---|
| I10 | 68,309 | 0.6990 | 0.7173 | 0.7229 | 0.7119 | 0.7148 | 0.7154 | 0.7122 | 0.7180 | 0.7099 |
| E78 | 34,953 | 0.6911 | 0.7044 | 0.7041 | 0.7010 | 0.7036 | 0.7043 | 0.7011 | 0.7077 | 0.7010 |
| K57 | 30,506 | 0.6259 | 0.6371 | 0.6485 | 0.6302 | 0.6322 | 0.6316 | 0.6304 | 0.6336 | 0.6274 |
| K21 | 25,427 | 0.5893 | 0.6018 | 0.6026 | 0.6035 | 0.6051 | 0.6038 | 0.6036 | 0.6026 | 0.6002 |
| K29 | 24,564 | 0.6025 | 0.6096 | 0.6117 | 0.6129 | 0.6134 | 0.6130 | 0.6136 | 0.6084 | 0.6111 |
| K44 | 24,297 | 0.6146 | 0.6244 | 0.6327 | 0.6229 | 0.6239 | 0.6250 | 0.6239 | 0.6242 | 0.6216 |
| I25 | 23,558 | 0.7312 | 0.7393 | 0.7395 | 0.7395 | 0.7410 | 0.7410 | 0.7394 | 0.7417 | 0.7375 |
| M19 | 21,544 | 0.6598 | 0.6656 | 0.6675 | 0.6684 | 0.6677 | 0.6693 | 0.6690 | 0.6622 | 0.6657 |
| I84+K64 | 21,296 | 0.5373 | 0.5501 | 0.5543 | 0.5468 | 0.5484 | 0.5494 | 0.5467 | 0.5495 | 0.5482 |
| I48 | 18,365 | 0.7471 | 0.7559 | 0.7573 | 0.7495 | 0.7506 | 0.7500 | 0.7492 | 0.7520 | 0.7485 |
| M17 | 18,227 | 0.6288 | 0.6384 | 0.6462 | 0.6400 | 0.6412 | 0.6415 | 0.6405 | 0.6404 | 0.6384 |
| C44 | 17,671 | 0.6675 | 0.6746 | 0.6860 | 0.6687 | 0.6691 | 0.6699 | 0.6691 | 0.6739 | 0.6660 |
| E11 | 17,345 | 0.6697 | 0.6992 | 0.7039 | 0.6876 | 0.6907 | 0.6920 | 0.6875 | 0.6985 | 0.6849 |
| E66 | 16,800 | 0.5498 | 0.5990 | 0.6103 | 0.6000 | 0.6079 | 0.6040 | 0.6012 | 0.6084 | 0.5994 |
| M54 | 14,550 | 0.5878 | 0.5994 | 0.6024 | 0.6116 | 0.6145 | 0.6143 | 0.6120 | 0.6049 | 0.6101 |
| D12 | 14,337 | 0.6201 | 0.6259 | 0.6285 | 0.6273 | 0.6294 | 0.6290 | 0.6263 | 0.6265 | 0.6266 |
| I20 | 14,234 | 0.7041 | 0.7157 | 0.7143 | 0.7153 | 0.7175 | 0.7177 | 0.7158 | 0.7199 | 0.7166 |
| E03 | 13,242 | 0.6791 | 0.7034 | 0.7128 | 0.6862 | 0.6862 | 0.6866 | 0.6854 | 0.6921 | 0.6859 |
| K40 | 12,541 | 0.7788 | 0.7833 | 0.7884 | 0.7778 | 0.7790 | 0.7795 | 0.7773 | 0.7789 | 0.7758 |
| M16 | 12,286 | 0.6597 | 0.6626 | 0.6759 | 0.6605 | 0.6598 | 0.6594 | 0.6609 | 0.6567 | 0.6597 |
| K80 | 12,194 | 0.6090 | 0.6209 | 0.6333 | 0.6201 | 0.6197 | 0.6199 | 0.6201 | 0.6202 | 0.6176 |
| N17 | 11,340 | 0.6986 | 0.7006 | 0.7014 | 0.7026 | 0.7049 | 0.7048 | 0.7023 | 0.7014 | 0.7034 |
| N18 | 10,337 | 0.7239 | 0.7273 | 0.7318 | 0.7311 | 0.7319 | 0.7319 | 0.7310 | 0.7308 | 0.7295 |
| J44 | 9,884 | 0.6883 | 0.6973 | 0.7024 | 0.7007 | 0.7019 | 0.7018 | 0.7009 | 0.7008 | 0.7018 |
| M23 | 9,585 | 0.5712 | 0.5770 | 0.5825 | 0.5821 | 0.5807 | 0.5829 | 0.5834 | 0.5741 | 0.5794 |
| M47 | 9,492 | 0.6424 | 0.6460 | 0.6445 | 0.6472 | 0.6469 | 0.6489 | 0.6476 | 0.6411 | 0.6431 |
| I95 | 9,153 | 0.6911 | 0.6929 | 0.6938 | 0.6937 | 0.6946 | 0.6946 | 0.6939 | 0.6932 | 0.6944 |
| I50 | 8,851 | 0.7214 | 0.7246 | 0.7264 | 0.7264 | 0.7273 | 0.7270 | 0.7265 | 0.7257 | 0.7257 |
| M81 | 8,604 | 0.7610 | 0.7623 | 0.7626 | 0.7606 | 0.7613 | 0.7611 | 0.7603 | 0.7579 | 0.7571 |
| I21 | 8,090 | 0.7316 | 0.7389 | 0.7417 | 0.7398 | 0.7403 | 0.7403 | 0.7395 | 0.7385 | 0.7372 |

Counts over the 30 diseases (a run beats a yardstick when its AUROC is higher):

| run | mean | beats covariates | beats twin | beats PRS + cov | median gap to PRS + cov |
|---|---|---|---|---|---|
| frozen mean | 0.6722 | 28 | 16 | 7 | -0.0019 |
| frozen mean, no mask | 0.6735 | 30 | 17 | 12 | -0.0014 |
| frozen mean, no mask, seed 2 | 0.6737 | 29 | 20 | 10 | -0.0011 |
| frozen mean, no mask, seed 3 | 0.6738 | 28 | 18 | 15 | -0.0008 |
| frozen mean, no mask, mean of seeds 1 to 3 | 0.6737 | 29 | 18 | 14 | -0.0010 |
| frozen atten | 0.6724 | 28 | 17 | 10 | -0.0022 |
| frozen flat | 0.6728 | 27 | 14 | 4 | -0.0040 |
| flat, no mask | 0.6731 | 26 | 14 | 8 | -0.0038 |
| flat, no mask, seed 2 | 0.6725 | 28 | 14 | 8 | -0.0039 |
| trained mean, κ 1e-2, ε 1e-8 | 0.6708 | 26 | 12 | 4 | -0.0038 |

The flat readout against the frozen mean: median difference +0.0001, higher for 16 of 30 diseases,
correlation of the difference with the log training case count +0.42 (I10 +0.0061,
E78 +0.0067; the nine diseases with at least 20,000 training cases +0.0012 on average); its tuning
cross-entropy is worse (0.2576 against 0.2565). The no-mask recipe against the masked one, both frozen mean: median +0.0011,
higher for 24 of 30. The frozen mean's shortfall against the PRS is largest for the common diseases (-0.0052
on average over the nine with at least 20,000 training cases). Attention and mean differ by at most 0.0003 at any validation.
Seed replicates of the no-mask frozen mean run (seeds 2 and 3) and of the no-mask recipe with the flat readout (seeds 1
and 2) ran on 18 and 19 November (`run_diag_seeds.sh`, GPUs 3 and 4, 21:55Z to 03:32Z; rows above). The three seeds of
the frozen mean, no mask, give 0.6735, 0.6737 and 0.6738: a spread of 0.0003 on the mean (standard deviation 0.0001),
per-disease ranges of 0.0019 on average and 0.0075 at most (E66). The mean of the three seeds, 0.6737, beats the
covariates for 29 of 30 diseases (M16, by -0.0003, is the exception), the twin for 18 (median +0.0017, mean +0.0005)
and the PRS for 14 (median -0.0010, mean -0.0040; -0.0034 over the nine diseases with at least 20,000 training cases),
so the model and its twin are level and the PRS keeps its lead. The flat readout under the no-mask recipe overfits in
both seeds: best at epoch-equivalents 3 and 2 (0.6731, 0.6725), down to 0.6617 and 0.6638 four epoch-equivalents later,
where the early stop ends them, whereas under the masked recipe the same readout stayed between 0.6717 and 0.6728 from
epoch-equivalent 2 to 9.8. At its best the flat readout's seed mean (0.6728) is level with the frozen mean's (median
difference -0.0005, higher for 12 of 30; I10 +0.0028, E78 +0.0017) with a worse tuning cross-entropy (0.2580 against
0.2565 on average). The readout for the grid is therefore the mean and the recipe the no-mask one.


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

## Selected-SNP from-scratch arms (19 November 2026; methods.md section 12, item 19)

Three single S3 trainings of the LD-block model from scratch on GWAS-selected SNPs (p < 0.01 in any of the 30 top-rung
GWAS): `sel_p01anyct_S3_zero` on the 95,642 clumped leads, table from zero, and `sel_p01any_S3_pretrained` on the
448,493 unclumped SNPs, table from one pass of masked-SNP pretraining, and `sel_p01any_S3_zero`, the unclumped set from zero. Un-handicapped table optimiser (κ 0.1, ε 1e-8,
decay 0.01), no-mask recipe, mean readout, ten epoch-equivalents, seed 1. Yardsticks on the tuning set: covariates
0.6627, classic-30 0.6627, full-set ridge PRS + covariates 0.6777, frozen stacked model 0.6735, and each set's own
cross-fitted ridge PRS and twin (`results/p01any_n215872/`, `results/p01anyct_n215872/`). Rows are added as the chain
(`chains/run_selected.sh`, `runs/selected/`) delivers them.

| run | SNPs | start | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | best (at epoch-eq) | own ridge PRS + cov | own twin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sel_p01anyct_S3_zero | 95,642 | zero | | | | | | | | | | | | | |
| sel_p01any_S3_pretrained | 448,493 | masked pretraining | | | | | | | | | | | | | |
| sel_p01any_S3_zero | 448,493 | zero | | | | | | | | | | | | | |

## The no-covariate comparison at the top rung (19 November 2026; methods.md section 12, item 20)

Two single trainings with the covariates removed from the model (`--dm snps`: no covariate encoder, token or vector), each
otherwise its reference run: `blocks_S3_folds_frozen_nomask_nocov` (the best configuration so far, `diag_S3_folds_frozen_nomask`
above, 0.6735 with covariates: S3 transformer, Privé blocks, fold-routed ridge warm start frozen, mean readout, no input mask,
dropout 0.2, all 30 diseases, ten epoch-equivalents, seed 1) and `classic_i10_n215872_nocov` (the classic DeepGP for I10 at the
published settings, 0.6987 with covariates; its table is in `classic-deepgp-comparator/results.md`, "Without covariates").
Yardsticks on the tuning set without covariates: the cross-fitted block ridge PRS alone 0.5666 mean (I10 0.5989); the PGI arm's
scores alone for I10, C+T 0.594, lassosum2 0.604, LDpred2-auto 0.615, SBayesRC 0.616; chance 0.5. Added after the runs: the
linear read of the table itself (`nocov_yardsticks.py`; methods item 20): a logistic regression of each label on the 16
block-summed channels of the frozen table, fitted on the training rows' out-of-fold scores, 0.5716 mean; and the ridge PRS
projected onto the table's 16-dimensional basis, 0.5649. Both runs ended normally on GPU 4 (`runs/nocov/run_nocov.log`):
the classic run 03:34 to 04:47Z, the block run 04:50 to 06:59Z (129 minutes for ten epoch-equivalents, alone on its GPU).

Mean tuning AUROC over the 30 diseases by epoch-equivalent (one validation every quarter; the full curve is the run's
`metrics.csv`):

| run | covariates | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | best (at epoch-eq) | linear read of the table | ridge PRS alone | with covariates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blocks_S3_folds_frozen_nomask_nocov | none | 0.5428 | 0.5456 | 0.5467 | 0.5479 | 0.5482 | 0.5497 | 0.5495 | 0.5501 | 0.5514 | 0.5516 | 0.5517 (9.76) | 0.5716 | 0.5666 | 0.6737 (seeds 1-3) |

Per disease (tuning set; "gain with covariates" is the AUROC over the covariates-only logistic model; the network with
covariates is the mean of seeds 1 to 3 of `diag_S3_folds_frozen_nomask`; table `results/hm3plus/nocov_yardsticks.csv`):

| disease | tuning cases | network, no covariates | linear read of the table | ridge PRS alone | network − linear read | largest block's share of PRS variance | network gain with covariates | PRS gain with covariates |
|---|---|---|---|---|---|---|---|---|
| I10 | 6,255 | 0.5807 | 0.5993 | 0.5989 | -0.0186 | 0.015 | +0.0164 | +0.0239 |
| E78 | 3,153 | 0.5696 | 0.5820 | 0.5705 | -0.0124 | 0.019 | +0.0132 | +0.0130 |
| K57 | 2,784 | 0.5440 | 0.5764 | 0.5834 | -0.0324 | 0.028 | +0.0057 | +0.0226 |
| K21 | 2,387 | 0.5554 | 0.5675 | 0.5530 | -0.0121 | 0.024 | +0.0145 | +0.0133 |
| K44 | 2,271 | 0.5503 | 0.5693 | 0.5658 | -0.0190 | 0.022 | +0.0104 | +0.0181 |
| K29 | 2,238 | 0.5508 | 0.5618 | 0.5455 | -0.0110 | 0.026 | +0.0105 | +0.0092 |
| I25 | 2,126 | 0.5720 | 0.5811 | 0.5660 | -0.0091 | 0.019 | +0.0097 | +0.0082 |
| M19 | 1,966 | 0.5535 | 0.5624 | 0.5508 | -0.0089 | 0.026 | +0.0095 | +0.0077 |
| I84+K64 | 1,962 | 0.5359 | 0.5514 | 0.5424 | -0.0155 | 0.018 | +0.0123 | +0.0172 |
| C44 | 1,676 | 0.5222 | 0.5821 | 0.5839 | -0.0599 | 0.022 | +0.0024 | +0.0185 |
| M17 | 1,675 | 0.5569 | 0.5774 | 0.5678 | -0.0205 | 0.024 | +0.0126 | +0.0173 |
| I48 | 1,670 | 0.5391 | 0.5772 | 0.5798 | -0.0381 | 0.021 | +0.0029 | +0.0102 |
| E11 | 1,663 | 0.5888 | 0.6294 | 0.6191 | -0.0406 | 0.019 | +0.0223 | +0.0342 |
| E66 | 1,478 | 0.5928 | 0.6138 | 0.6001 | -0.0210 | 0.015 | +0.0542 | +0.0605 |
| M54 | 1,310 | 0.5640 | 0.5724 | 0.5557 | -0.0084 | 0.016 | +0.0265 | +0.0146 |
| D12 | 1,309 | 0.5421 | 0.5512 | 0.5506 | -0.0091 | 0.018 | +0.0088 | +0.0083 |
| I20 | 1,284 | 0.5841 | 0.5974 | 0.5714 | -0.0133 | 0.015 | +0.0135 | +0.0101 |
| E03 | 1,211 | 0.5370 | 0.6097 | 0.6195 | -0.0727 | 0.138 | +0.0075 | +0.0337 |
| K40 | 1,140 | 0.5066 | 0.5254 | 0.5705 | -0.0188 | 0.036 | +0.0007 | +0.0096 |
| K80 | 1,118 | 0.5425 | 0.5636 | 0.5765 | -0.0211 | 0.016 | +0.0109 | +0.0243 |
| M16 | 1,106 | 0.5133 | 0.5356 | 0.5748 | -0.0223 | 0.014 | -0.0003 | +0.0162 |
| N17 | 1,007 | 0.5493 | 0.5639 | 0.5407 | -0.0146 | 0.016 | +0.0062 | +0.0028 |
| J44 | 986 | 0.5797 | 0.5837 | 0.5746 | -0.0040 | 0.015 | +0.0136 | +0.0142 |
| N18 | 943 | 0.5565 | 0.5664 | 0.5603 | -0.0099 | 0.024 | +0.0080 | +0.0079 |
| M23 | 872 | 0.5479 | 0.5597 | 0.5446 | -0.0118 | 0.014 | +0.0115 | +0.0111 |
| M81 | 840 | 0.5102 | 0.5397 | 0.5403 | -0.0295 | 0.014 | +0.0001 | +0.0016 |
| I95 | 834 | 0.5375 | 0.5490 | 0.5361 | -0.0115 | 0.013 | +0.0035 | +0.0027 |
| I50 | 814 | 0.5558 | 0.5714 | 0.5568 | -0.0156 | 0.012 | +0.0056 | +0.0050 |
| M47 | 797 | 0.5431 | 0.5515 | 0.5279 | -0.0084 | 0.018 | +0.0065 | +0.0021 |
| I21 | 730 | 0.5681 | 0.5763 | 0.5709 | -0.0082 | 0.019 | +0.0086 | +0.0100 |
| **mean** | | **0.5517** | **0.5716** | **0.5666** | **-0.0199** | | **+0.0109** | **+0.0149** |

Reading.
- **The network's genetic signal is real.** Without covariates it reaches 0.5517, far above chance, and it holds about three
  quarters of the linear signal in both settings: (0.5517 − 0.5) / (0.5716 − 0.5) = 72% of the linear read of its own table
  without covariates, and 0.0109 / 0.0149 = 73% of the ridge PRS's gain over the covariates with them. The with-covariate
  result was not an artefact of the covariates, and the stacked model is not suspect on that ground.
- **The shortfall is the network's, not the table's.** The 16 channels lose almost nothing (the projected PRS 0.5649 against
  0.5666), and a linear read of them reaches 0.5716, above the PRS alone; the network falls below that linear read for all 30
  diseases (median −0.015, mean −0.020). It reads less from its input than a 17-parameter logistic regression does.
- **The same diseases fail with and without covariates.** The per-disease shortfall without covariates (network − linear read)
  correlates 0.81 with the shortfall with them (network gain − PRS gain over the covariates). The largest are E03 (−0.073),
  C44 (−0.060), E11 (−0.041), I48 (−0.038), K57 (−0.032) and M81 (−0.030), on which the table keeps 83% to 97% of the PRS
  weights' variance. Concentration explains E03 only: 14% of its PRS variance sits in one block (HLA), ten times the median
  disease; over the other 29 the largest block's share does not track the shortfall (Spearman −0.15 over all 30). Why the
  mean readout loses C44, I48 and E11 is not established.
- **Still rising at the budget's end**, by about 0.0005 per epoch-equivalent over the last four; at that pace the gap to the
  linear read (0.020) would not close within a feasible budget. One seed; configuration (a)'s seed spread with covariates
  was 0.0002.

## The network on frozen PGI tables (20 November 2026; methods.md section 12, item 22)

Tuning set, 20,000 participants, mean over the 30 diseases. Each score (the network's logit; the PRS) is combined with
the 12 covariates by the PGI arm's own logistic regression on the tuning set (`PGI/methods.md` section 6; the recomputed
PRS AUROCs match the PGI arm's record to 5e-7). Lift = network minus PRS, 95% interval from a paired bootstrap (2,000;
per disease stratified by case status, the mean from one shared resample of participants). Covariates alone: 0.6653.
S3 network, frozen 30-channel table, first layer cached, no-mask recipe, 10 epoch-equivalents, seed 1, no search.
Per-disease rows: `results/pgi_tables/compare.csv`.

| Weights | Arm | Network trained on | PRS + covariates | Network + covariates | Lift | Diseases: lift / no clear difference / worse | Best at (epoch-eq) |
|---|---|---|---|---|---|---|---|
| lassosum2 | as proposed: full rung (in-sample weights) | 215,872 | 0.6829 | 0.6727 | -0.0101 [-0.0116, -0.0085] | 0 / 17 / 13 | 0.25 |
| LDpred2-auto | as proposed: full rung (in-sample weights) | 215,872 | 0.6873 | 0.6760 | -0.0112 [-0.0130, -0.0096] | 0 / 15 / 15 | 0.5 |
| lassosum2 | check: leak-free control | 108,129 | 0.6766 | 0.6711 | -0.0055 [-0.0068, -0.0042] | 0 / 20 / 10 | 9.26 |
| LDpred2-auto | check: leak-free control | 108,129 | 0.6800 | 0.6729 | -0.0071 [-0.0085, -0.0057] | 0 / 19 / 11 | 10.01 |

- **The model as proposed does worse than the PRS it was built from, and overfits.** On the full rung the network is
  below its own PRS + covariates by 0.0101 (lassosum2) and 0.0112 (LDpred2-auto), with 0 of 30 diseases better and 13
  to 15 worse. It peaks at the first or second validation (0.6447 at 0.25 and 0.6546 at 0.5 epoch-equivalents, raw)
  and declines until early stopping: the PGI weights come from GWAS of the same 215,872 training participants, so the
  PRS reads their labels (AUROC 0.887 and 0.898 on 20,000 training participants against 0.572 and 0.582 on the tuning
  set), and the network learns that shortcut.
- **Check, not the headline: without the leak it still does not lift.** With the 107,743 rung's weights and the
  network trained on the other 108,129 (no overlap; the PRS shows no gap, 0.550 / 0.556 and 0.562 / 0.564), it is
  below that PRS + covariates by 0.0055 and 0.0071, 0 of 30 diseases better and 10 to 11 worse. These runs had
  levelled off: over their last two passes (8.26 to 10.01 epoch-equivalents) the raw mean moved by +0.0005 and
  +0.0003, within the run-to-run noise.
- **Where the network loses** is where the ridge-table network lost (item 20): E03 (-0.038 and -0.043 in the controls),
  K80, C44, E11. These runs used the mean readout, but the network can already weight blocks (four layers of
  self-attention across the block tokens), and on the ridge table the per-disease attention readout did not help them
  either (E03 0.6854 against 0.6862 with the mean; item 18), while the flat readout partly did (E03 0.6921, E11 0.6985).
  Why these diseases lose is open (block concentration explained E03 alone in item 20). Closest to a gain:
  M54, N17, N18, I20, I95 (+0.002 to +0.004, every interval includes 0).
