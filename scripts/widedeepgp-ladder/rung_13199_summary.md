# WideDeepGP ladder, first training size: summary

13,199 training participants (median N_eff 3,372), 30 diseases, all 1,258,200 HapMap3+ SNPs.
Mean tuning-set AUROC over the 30 diseases, from the run of 15–16 November 2026 (DeepWideGP
commit 5b472c5). Choices were made on the same tuning set, so the figures are mildly optimistic.

| Model | Mean tuning AUROC |
|---|---|
| Covariates only (year of birth, sex, 10 PCs; same training participants) | 0.6598 |
| Ridge alone (the ablation) | 0.6634 |
| WideDeepGP, ridge + network (the reported model) | 0.6633 |
