# DeepGP summary for its authors: methods

**What this is.** A summary of work already done, written on 26 November 2026 at Robert's request for the DeepGP
authors, and restructured the same day to his outline (Tables 1–4, Figure 1, how the over-weighting is measured). No
model was trained for it.
- **`make_summary.py`** collects aggregate AUROCs from the run folders and the PGI arm into `figures/*.csv` and draws
  the figures.
- **One new calculation**: the C+T PRS alone, scored on the code's training and validation parts
  (`prs_alone_by_part.csv`). It reads the per-person PRS files under `data/` and the replication's split rule
  (`inflation.py`: 15% validation, stratified, seed 333). It reproduces the recorded AUROCs on all GWAS people and on
  the test people.
- **Figure 1** is the replication's Figure 3, redrawn without the "PRS + covariates joined outside the GWAS" line and
  with the recipe named "PRS + cov".
- **`results.md`** is the text, with tables filled from the CSVs.
- **`DeepGP_E11_summary.docx`** is the same text, made with pandoc 2.9 and a styled reference document (the builder
  was a session script, not kept). From here on the Word file is meant to be edited by hand.

**Sources**
- Paper numbers: the supplement in `runs/deepgp_replication_e11/paper/` and `../deepgp-replication/`.
- Replication (Tables 2–6):
  - `runs/deepgp_replication_e11`: minor-allele runs and the PRS recipe;
  - `runs/deepgp_no_learning`: risk-allele runs and the test score after every pass;
  - `runs/deepgp_replication_e11_cohort_gwas`: the cohort's own GWAS, LDpred2 and lassosum2, over-weighting;
  - `runs/deepgp_replication`: atrial fibrillation;
  - `runs/tweaks/classic_e11_r07_balanced`: E11 at population prevalence;
  - `runs/nocov`: I10 without covariates.
- Overfitting across training sizes (Table 7): `runs/kelemen_mlp/ladder_prs/check_prs_inputs.json`, the PGI arm's
  LDpred2-auto weights scored on each rung's training people. Top-rung E11: `runs/tweaks/yardsticks/`.
- Five-rung comparison (Tables 8–9):
  - `runs/widedeepgp_ladder/*/scores_val.csv`: WideDeepGP, three seeds, and the covariates-only model;
  - `runs/classic_deepgp_ladder/results/`: classic DeepGP, 150 runs finished 23 November;
  - `PGI/results/<rung>/tune_metrics.csv`: C+T, linear covariate basis;
  - `../widedeepgp-blocks/results/<rung>/blocks_linear.csv`: ridge PRS + covariates. The top rung's file is on disk
    in the working tree but not on branch `widening`.

**Definitions**
- "GWAS people" are the people whose labels made the GWAS (and so the PRS or the SNP list). "New people" are test
  people outside every GWAS.
- **PRS over-weighting.** The PRS's coefficient in the paper's recipe (logistic regression on PRS + covariates,
  standardised, fitted on GWAS people) divided by its coefficient in the same regression fitted on new people.
- **Covariates' share.** The share of the combined score's variance on new people that comes from the covariates.
- **Covariates alone.** For DeepGP and WideDeepGP, a logistic regression on the 12 covariates fitted on the training
  people. For C+T, the shared protocol refits both C+T + covariates and covariates alone on the tuning people, so its
  gains are taken against its own fit (0.6653).
- **Tuning AUROCs** (section 3) are read at each model's best check on the same tuning set, so they are mildly
  optimistic. The locked test set was not read.

**Gaps and caveats**
- No E11 DeepGP run without covariates exists. The only classic no-covariate run is I10 (`runs/nocov`).
- The rerun with the paper's multi-source T2D definition (`runs/deepgp_replication_t2d`) logged its first GWAS step
  on 21 November and nothing after. It has no running process, its data folder is gone and it has no result.
- E11's training-row PRS AUROC was not measured below the top rung.
- Table 4's HapMap3+ and all-SNP rows differ in GWAS size as well as SNP count (215,872 against 34,690 people). Only
  the random-subset rows hold the GWAS fixed. No C+T score was built from the cohort's own HapMap3+ GWAS, which would
  hold the people fixed.
- DeepGP's in-GWAS AUROCs for the model trained to completion come from the training log (averages over batches of
  64). Its genotype files were deleted on 21 November, so the whole parts cannot be re-scored.
- Classic DeepGP's training AUROCs on the ladder were not recorded, only its tuning checks.
- The reason given for classic DeepGP's deficit on the ladder is our reading. It is not tested: the network leans on
  SNPs whose GWAS included its own training people, and so under-uses the covariates.

Usage: `CUDA_VISIBLE_DEVICES= /home/robert/envs/widedeepgp/bin/python make_summary.py`
