# Results: the overnight tweaks of WideDeepGP at the top rung (20 November 2026)

*Aggregates only; the tuning set (20,000 participants), mean AUROC over the 30 diseases unless a disease is named. The
test set is never read. The table is `runs/tweaks/results_table.md`, rebuilt by `python runs/tweaks/summarise.py`; the
yardsticks and the reading are below. Written as the runs finished; the final state is at the end.*

## Yardsticks

| model | tuning AUROC | gain over its covariates |
|---|---|---|
| covariates alone (training-row logistic fit) | 0.6627 | |
| base: S3 on the frozen fold-routed ridge table, 10 epoch-equivalents (`blocks_diag/diag_S3_folds_frozen_nomask`) | 0.6735 (seeds 0.6735, 0.6737, 0.6738) | +0.0108 |
| logistic ridge on the block channels | 0.6732 | +0.0105 |
| block ridge PRS + covariates | 0.6777 | +0.0149 |
| PGI arm (refit on the tuning set; its covariates 0.6653): lassosum2 / LDpred2-auto / SBayesRC | 0.6829 / 0.6873 / 0.6878 | +0.0175 / +0.0219 / +0.0224 |
| naive lassosum2 / LDpred2 + covariates, fitted on the training rows (the in-sample tables) | 0.6341 / 0.6422 | -0.0287 / -0.0205: the tables carry the training labels (PGI AUROC 0.887 / 0.898 on the training rows against 0.572 / 0.582 on the tuning set) |

## The table

(filled from `runs/tweaks/results_table.md` as the runs finish)
