# Methods: the WideDeepGP ladder (training-size runs for the scaling study)

*Results of the first training size and the run folders (`runs/ladder`, `runs/classic_deepgp`) were removed on 16 November 2026 at Robert's request, before code changes; the first size's summary is kept in `rung_13199_summary.md`, and the removed files are in git history (DeepWideGP-Scripts 784009f).*

*Started 15 November 2026. The model and its search are described in `DeepWideGP/methods.md`.
The training sizes and the comparison protocol are those of the linear polygenic-index (PGI) arm,
`PGI/methods.md` sections 2.5, 6 and 8. This document records what is run and how. Results will
go in `results.md`.*

## What is run

WideDeepGP is trained at each of the PGI arm's five training sizes, its run specifications
taken unchanged (Robert's decision of 15 November 2026: "the same in every way"):

| Dataset | Training participants | Median N_eff across the 30 diseases |
|---|---|---|
| `hm3plus_n13199` | 13,199 | 3,372 |
| `hm3plus_n26287` | 26,287 | 6,740 |
| `hm3plus_n53498` | 53,498 | 13,478 |
| `hm3plus_n107743` | 107,743 | 26,955 |
| `hm3plus` | 215,872 | 53,908 |

At every size, both arms share:
- the 30 diseases;
- the 1,258,200 HapMap3+ SNPs;
- the twelve covariates;
- the same 20,000 tuning participants and 40,000 test participants.

The smaller training sets are nested in the larger ones.

## One run per training size

Each run is a whole search inside the run (`DeepWideGP/methods.md` section 8), with seed 1.

- **The linear stage:**
  - one training run of the linear model on one GPU, over nine prior variances (0.003 to 30);
  - each disease keeps the variance and validation pass with its best tuning AUROC;
  - batches of 256, with the learning rates falling on a cosine over 30 epoch-equivalents (at
    least 128,000 participants seen).
- **The network on top:**
  - 12 Optuna trials on the free GPUs minus one, with the linear model's PRS weights fixed;
  - learning rate, weight decay and dropout are searched, with successive-halving pruning;
  - the network is the pilot's default arm: width 32, one encoder layer, four attention heads,
    attention pooling, batches of 16 with four-fold accumulation, bf16 encoders.
- **The run's model** is the best trial of the network on top, whatever its tuning AUROC
  relative to the linear model alone (Robert's decision of 15 November 2026). The linear model
  alone is recorded beside it as an ablation (`linear_only` in the run's `results.json`).

## Order and execution

- **Order:** smallest to largest, one after another, in one detached job
  (`run_ladder_2026-11-15.sh`; run folders under `/mnt/sdg/robert/deepwidegp/runs/ladder/`). It
  does not pause for review between sizes (Robert, 15 November 2026), so each size's results are
  available as soon as it finishes.
- **Start:** the chain starts after the real-label check of `../widedeepgp-gates/` has finished.
  It stops if a run fails.
- **Restarting:** rerunning the script skips finished sizes and continues an unfinished search.
  The linear stage resumes from its last checkpoint, and trials left running by a crash are
  replaced.

## What each run records

Under `runs/ladder/<dataset>/`:
- `results.json`: the chosen trial, every trial's settings, tuning AUROC and cost, and the linear
  model alone;
- `linear/`: the grid run;
- `linear/per_disease/`: the linear model with each disease's choice;
- `joint/`: the Optuna study and one folder per trial, each with its metrics, configuration and
  TensorBoard log.

Only the chosen trial and the linear model keep their weights.

## Evaluation

As in `PGI/methods.md` section 6.1, for both arms:
- **Final prediction:** a logistic regression, fitted on the tuning set, of disease status on
  WideDeepGP's logit for that disease plus the covariate basis. The basis is a natural cubic
  spline of year of birth with four degrees of freedom, separately by sex, plus ten principal
  components.
- **Reference:** the covariates-only regression.
- **The test set:** opened once, for both arms together, after every choice at every size is
  recorded.

## Not part of these runs

The published single-disease DeepGP, as an extra comparator for hypertension (I10) at the same
five sizes (`DeepWideGP/methods.md` section 9). Planned; it waits for the published LD-clumping
parameters.
