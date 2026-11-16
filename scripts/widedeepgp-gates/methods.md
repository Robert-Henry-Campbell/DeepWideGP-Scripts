# Methods: checks of the linear PRS path and the search inside every run (WideDeepGP gates)

*Results (`results.md`, `results/`) and the run folders (`runs/wide_gate`) were removed on 16 November 2026 at Robert's request; the results files are in git history (DeepWideGP-Scripts 784009f), and the gate numbers are recorded in `DeepWideGP/methods.md` section 8.*

*12–15 November 2026. Results in `results.md`; aggregate tables in `results/`. The model and the
search are described in `DeepWideGP/methods.md` (sections 4.10, 5.2, 6, 7 and 8).*

## Purpose

The pilot (`../widedeepgp-pilot/`) found that the WideDeepGP network, given all HapMap3+ SNPs,
learned next to nothing from the genotypes. On 12 November 2026 Robert asked for a linear PRS
inside the model (the linear PRS path) and for every run to choose its own settings (a search
inside every run). Before any full-size run, these checks ask:

- **G1.** Does the linear path, trained by the model's own code, recover what GBLUP recovers on
  simulated traits?
- **G2.** With the network on top of that linear path, is the model at least as good as the
  linear path alone?
- **G3.** On the real labels, does the model improve on the covariates alone, and how does that
  compare with a clumped polygenic score from the same participants?

## Data

All checks use the pilot's proxy dataset (`DeepWideGP-Scripts/scripts/build-ard30-dataset`,
section "Proxy dataset for pilots"):
- **SNPs:** the second eighth of each chromosome's HapMap3+ SNPs, 157,279 in all.
- **Participants:** 40,000 training and 10,000 tuning participants of White-British ancestry
  without relatives in the cohort. The proxy's third table is drawn from the training split, so
  the real test set plays no part.
- **Covariates:** the twelve of the main dataset: year of birth, sex and ten principal components.

Two sets of labels:
- **Simulated traits** (`hm3plus_proxy8_sim`): five traits from the pilot's positive control.
  Each was simulated on these participants' real genotypes, with 20, 200, 2,000 or 20,000
  causal SNPs (one 200-SNP trait enriched in coding and conserved SNPs), a liability-scale
  heritability of 0.3 (the 20-SNP trait) or 0.5, and 20% cases. The covariates have no effect
  on them.
- **Real labels** (`hm3plus_proxy8`): the 30 diseases.

## Models

- **The linear path alone, one setting** (`runs/wide_gate/sim_linear_tau1_k0.1`; DeepWideGP
  9e69019, `--wide only --tune_trials 0`):
  - prior variance τ = 1;
  - κ = 0.1, at a constant learning rate (3 × 10⁻³ for the covariate weights);
  - batches of 256;
  - one validation per epoch-equivalent;
  - 40 epoch-equivalents.
- **The linear stage of a run's search** (DeepWideGP e5cebed; defaults):
  - one training run over a grid of prior variances, each disease keeping its best variance and
    validation pass by tuning AUROC;
  - grid 0.03, 0.1, 0.3, 1, 3, 10 and 30;
  - κ = 0.1, covariate learning rate 10⁻²;
  - batches of 256;
  - the rates on a cosine to zero over 30 epoch-equivalents;
  - early stopping after ten passes without improvement.
  - Runs: `sim_linear_grid` (simulated traits) and `real_linear_grid` (real labels).
  - The default grid was then widened to start at 0.003 (0edb9cb). Its run on the real labels,
    `real_linear_grid9`, was spoilt (see "Incidents").
- **The network on top** (`sim_joint`). The search's second stage: 12 Optuna trials of the
  network on top of the linear stage's per-disease model, with the PRS weights fixed. Searched:
  learning rate, weight decay and dropout, with successive-halving pruning. The model is the
  pilot's default arm:
  - width 32, one encoder layer, four heads, attention pooling;
  - batches of 16 with four-fold accumulation;
  - bf16 encoders.
  - It started from a copy of `sim_linear_grid/linear`. The settings are the same, and the
    stage reads the copy where it lies (DeepWideGP 81ded30).
  - Under the rule of 13 November, the run's model was the joint one only if its mean tuning
    AUROC was strictly higher. Since 15 November (DeepWideGP 5396f44) it is always the joint
    one, and the linear model alone is recorded as an ablation.
- **References:**
  - **covariates alone:** a logistic regression on the twelve standardised covariates, fitted on
    the training participants by `score_snapshots.py`;
  - **from the pilot's positive control:** GBLUP with each trait's true heritability, the
    clumped polygenic score, and the ceiling (the AUROC of the simulated genetic value);
  - **the pilot's network alone** (the model before the linear path, as `--wide off`):
    6 epoch-equivalents with the pilot's fixed settings, best pass.

## Pre-set criteria

Set in the design review before the runs:
- **G1:** every trait within 0.01 of GBLUP, and a mean of at least 0.708.
- **G2:** no lower than the linear stage alone by more than 0.01.
- **G3:** a positive gain over the covariates alone. It is compared with, not required to
  exceed, the +0.0014 a clumped polygenic score added in the pilot.

## Choosing on the tuning set

Each disease's prior variance and pass are chosen on the tuning set. That disease's tuning AUROC
is therefore optimistic, more so for rare diseases (as few as 373 tuning cases among the 30).

So every check also reports a less optimistic figure: the model at the end of training, with the
one prior variance that is best on average across diseases. This is a choice among seven values
for all diseases together.

The test set is not used.

## Compute and execution

- **CPU runs:** 8 threads. The PGI work's allowance is separate.
- **GPU runs:** the free GPUs minus one (Robert's rule).
- **Waiting and launch:** scripts waited for the genotype-store rebuild and for free GPUs, then
  started the runs (`run_scripts/`).
- **GPU test suites:** `run_gpu_suites_when_free.sh` and `rerun_tuner_test_when_free.sh`.
- **Tables:** `make_tables.py` collects them from the run folders. `evaluate_linear_vs_covariates.py`
  scores a linear stage against the covariates alone, per disease.
- **Run folders:** under `/mnt/sdg/robert/deepwidegp/runs/wide_gate/`, outside git.

## Incidents and deviations

- **Store rebuild between runs.** The genotype stores were rebuilt without the 120 withdrawn
  participants at 01:19 UTC on 13 November.
  - The fixed-τ run used the old store, every later run the rebuilt one.
  - No participant of any table was withdrawn, and genotypes were checked identical, so the
    inputs are the same.
  - The SNP statistics were recomputed once for the new store rows. They are identical,
    because the participants are the same.
- **The nine-value real-label run was spoilt.**
  - A session restart on 13 November killed `real_linear_grid9` at pass 28 of 30. It had been
    started as a job tied to the session.
  - On 15 November it was resumed from its last checkpoint. The resumed run repeated pass 28
    on only part of the tuning set, because Lightning restores that pass part-way.
  - Every disease's choice, and the run's best checkpoint, were taken from that partial pass.
  - Found by rescoring: the chosen model scored 0.6573 on the whole tuning set, against 0.6607
    for the covariates alone.
  - Fixed in DeepWideGP 512c755 (every pass now scores the whole tuning set;
    `DeepWideGP/methods.md` section 7), with a test.
  - The real-label network check, which had started from that run, was stopped after four
    minutes. It has not been run.
- **GPU test timeout.** One test of the standalone tuner timed out under the new
  128,000-participant budget floor. It passed with small-data floors (197741a).

## Software

- DeepWideGP branch `widening`: commits 9e69019, e5cebed, 81ded30, c19382a, 0edb9cb, 197741a,
  512c755 (the resume fix) and 5396f44 (the reporting rule).
- Environment: `/home/robert/envs/widedeepgp`, with Python 3.10, PyTorch, PyTorch Lightning,
  Optuna and scikit-learn.
- Seed: 1.
