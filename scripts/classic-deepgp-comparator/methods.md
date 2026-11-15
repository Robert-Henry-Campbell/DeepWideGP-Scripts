# Methods: the classic DeepGP comparator (hypertension, I10)

*Decided by Robert on 15 November 2026; built and queued the same day. The training sizes, tuning
and test sets and the comparison protocol are those of the PGI arm (`PGI/methods.md` sections
2.5, 6 and 8) and of WideDeepGP (`DeepWideGP/methods.md` sections 2.5 and 9). Results will go in
`results.md`.*

## Purpose

WideDeepGP is the headline model. The study's single-disease deep-learning comparator is the
classic DeepGP: the published single-disease model on GWAS-selected SNPs. It is fitted for one
disease only, hypertension (I10), the disease with the highest effective sample size of the 30
(N_eff 186,775 in the full training set). It is fitted at each of the five training sizes.

For I10 this places three kinds of model side by side at every size:
- the seven linear PGIs;
- the classic DeepGP;
- WideDeepGP.

## SNPs

At each training size the SNPs are those the published pipeline selects from a GWAS of that
size's training participants:
- **The GWAS** is the PGI arm's REGENIE analysis of I10: logistic regression on ALT-allele
  dosage, adjusted for the twelve covariates.
- **Clumping.** plink2 (v2.0.0-a.7.13) clumps the summary statistics at r² 0.1 within 250 kb.
  LD comes from the PGI arm's LD panel, the 20,000 training participants its C+T also clumps on.
- **Selection.** The lead SNPs with p ≤ 0.05 are kept. They are the leads that clumping all
  SNPs would give, as the PGI arm's C+T does, restricted to p ≤ 0.05.
- **The p-value threshold** follows Robert's description of the published pipeline.
- **The clumping parameters** are the standard ones, also the strictest of the PGI arm's C+T
  options, because the published values were not available.
- **Short chromosomes.** A chromosome with fewer than three selected SNPs would be topped up with
  its next strongest SNPs, because the token embedding convolves over three. The build log says
  if this happened.

The selection uses only that size's training participants, so no tuning or test label enters it.

The genotypes of the selected SNPs, for that size's training participants and the tuning and test
participants, are written as a genotype store of their own. This takes one pass over the main
store, chromosome by chromosome, for all five sizes (`build_datasets.py`).
- **Check:** for 200 participants per size, every selected SNP of chromosomes 1, 6 and 22 must
  equal the main store's genotype.
- **Datasets:** `data/datasets/ard30/classic_i10_n<size>`, with the size's sample tables linked.

## Model and training

The published DeepGP model and training:
- model width 32, one encoder layer, four attention heads, attention pooling;
- a two-layer head with 256 hidden units;
- 32-bit arithmetic;
- batches of 64 drawn at random;
- AdamW with learning rate 7 × 10⁻⁴ and weight decay 10⁻⁴, dropout 0.2;
- unweighted cross-entropy;
- the twelve covariates of every other model.

It is run under the study's protocol rather than with the published script, which:
- fixes its own seed (33) and data folder;
- scores the test set at the end of every run;
- writes per-person predictions into the repository.

The model code reproduces the published network: the same output to within 10⁻⁶, and the same
initial weights from the same seed (`DeepWideGP/methods.md` section 12). The run uses one
disease (`--labels I10`), no class weights (`--gamma 0`), shuffled batches and the published
settings.

As for WideDeepGP's network, every training size gets its own search:
- 12 Optuna trials over learning rate, weight decay and dropout;
- the published settings are the first trial (`--tune_first_given`);
- successive-halving pruning;
- early stopping on the tuning set, scored as a whole.

Everything else follows the WideDeepGP protocol:
- seed 1;
- the budget of 30 epoch-equivalents and at least 128,000 participants;
- validation every quarter epoch-equivalent, at most once per 20,000 participants and at least
  once per epoch-equivalent;
- weight snapshots of the chosen trial.

## Execution

- **Scheduling.** The ladder's chained job (`runs/ladder/run_ladder_2026-11-15.sh`) runs the
  comparator at each training size right after WideDeepGP at that size (`run_rung.sh`, from the
  same frozen code), on the free GPUs minus one.
- **If a size's dataset is not ready in time,** the chain logs it and moves on. That size is then
  run afterwards with the same script.
- **Outputs.** Run folders go under `runs/classic_deepgp/`.

## Evaluation

Each size is evaluated with the protocol shared by both arms:
- **Final prediction:** a logistic regression, fitted on the tuning set, of I10 on the model's
  logit plus the covariate basis.
- **The test set:** opened once, for both arms together.
