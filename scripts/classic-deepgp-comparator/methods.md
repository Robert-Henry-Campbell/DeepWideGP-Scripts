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
- the six linear PGIs[^sbrc7m];
- the classic DeepGP;
- WideDeepGP.

[^sbrc7m]: SBayesRC on the 7M imputed set was dropped on 21 Nov 2026 for its compute cost (`PGI/methods.md` §5.6).

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

## The 30-disease variant at the top rung (Robert, 18 November 2026)

A second comparator asks whether the multi-disease output alone, without the LD-block first layer
or any other change of WideDeepGP's architecture, improves prediction. It is the classic DeepGP
network with 30 outputs, trained once at the top rung (215,872 training participants).

- **SNPs.** The classic rule applied to the 30 GWAS together: the SNPs with p ≤ 0.05 in any of the
  30 top-rung REGENIE GWAS (1,034,524 of the 1,258,200, 82%), clumped once at r² 0.1 within 250 kb
  on the LD panel, with each SNP's smallest p over the 30 diseases choosing the index SNPs:
  179,566 lead SNPs (`build_datasets.py --diseases <the 30> --name classic30 --rungs hm3plus`;
  dataset `data/datasets/ard30/classic30_n215872`). For comparison, the per-disease classic sets
  hold 21,328 to 28,536 leads (mean 23,546; I10 28,536), and their union 454,876 SNPs, of which
  171,110 are among the 179,566. One disease gives the classic selection exactly, so the five I10
  datasets are the same builder's default.
- **Model and training.** As the I10 run: width 32, one layer, four heads, attention pooling, a
  256-unit head, fp32, batch 64, AdamW at 7 × 10⁻⁴ with weight decay 10⁻⁴, dropout 0.2, plain
  cross-entropy (uniform shuffled batches, no class weight; the mean over the 30 diseases), the
  twelve covariates, patience 8, the 30-epoch budget. One training with the published settings
  (`--tune_trials 0`, seed 1), no search, matched to the single-disease quick run
  `classic_i10_n215872_quick` (the same frozen code, `runs/blocks/code_0094e74`). Launcher
  `runs/classic_deepgp/run_classic30_n215872.sh`; run folder `runs/classic_deepgp/classic30_n215872/`.
- **Reading.** Per disease, the tuning AUROC beside the covariate-only floor, the PGI arm's methods
  and the LD-block runs (T0), and for I10 beside the single-disease classic run. A caveat recorded
  before the run: the classic network pools the genome into one 32-wide vector before the head, so
  the 30 diseases share a narrow bottleneck; no gain means that multi-disease training does not
  rescue this architecture, not that it cannot help a wider one.
- **Cost.** About 23 GB at batch 64 and about 1.1 h per epoch-equivalent (from the measured
  per-SNP memory and time of the published model), so about 3.5 h for a run that stops on patience
  and 33 h at the budget.

## Balanced batches by oversampling cases (Robert, 18 November 2026)

The published DeepGP trained on balanced case-control sets; the comparator above trains at natural prevalence
on the whole rung with uniform batches. One run asks what balancing the batches does to the published network
at the top rung, for I10, against the covariates alone. The balance comes from showing the cases more often,
not from dropping controls: every training participant stays in.

- **Batches.** The rotation sampler with the one label (`--sampler rotation --labels I10`; DeepWideGP
  `widedeepgp/samplers.py`, methods 5.1): every batch of 64 holds 32 I10 cases and 32 non-cases. The non-cases
  come in shuffled passes without replacement, so each of the 147,563 is seen once per pass of 4,611 batches;
  the 68,309 cases come in passes of their own, 2,135 batches long, so each case is seen 2.16 times per pass over
  the non-cases. Every batch is a function of the seed and the batch number, as in every run.
- **Loss.** Plain cross-entropy on the batch as drawn: no class weight (`--gamma 0`) and no sampling-rate
  correction (`--no_sampling_correction`). The correction (`losses.py`, DeepWideGP methods 5.2) divides each
  person's term by their sampling rate, 1.58 for a case and 0.73 for a non-case here, which weights the cases
  back to the population and would undo the balance; without it the loss is the mean cross-entropy of a 1:1
  batch, the paper's objective. The model's logits are then on the balanced scale (an intercept near 0 for a
  50% batch), not the population's.
- **Everything else as the quick run** `classic_i10_n215872_quick`: the same dataset (`classic_i10_n215872`,
  all 215,872 training participants, 28,536 SNPs), frozen code (`runs/blocks/code_0094e74`), settings, seed,
  budget in draws (30 epoch-equivalents of 215,872 draws), validation every 53,952 draws, patience 8. Launcher
  `runs/classic_deepgp/run_balanced_i10_n215872.sh` (waits for two free GPUs); run
  `classic_i10_n215872_balanced`. Both runs are then scored on the tuning set by `score_snapshots.py`, which
  fits the covariates-only logistic regression on the training split beside each, so the floor comes from one
  code path.
- **What is comparable.** The two runs train on the same participants and are read on the same tuning set, so
  their AUROCs compare directly, with the covariate floor and with every other run; what differs is only how
  often each case enters the loss. Average precision, a rank statistic like AUROC, compares the same way on the
  same tuning set (corrected 18 November, evening: an earlier version listed it among the non-comparable metrics,
  a leftover of the undersampling reading, whose evaluation set changed prevalence). Not comparable: the
  balanced run's cross-entropy and calibration, which sit on the balanced scale (its logits are about
  logit(0.5) − logit(0.316) = 0.77 too high at natural prevalence; the protocol's final logistic refit on the
  tuning set absorbs the shift). Result: results.md, "Balanced batches by oversampling cases".
- **Expectation, stated before the result.** The uniform-batch runs sit at the covariate floor. Balanced
  batches reweight the same information; a run above the floor would say the sampling, not the architecture,
  carries the published result, and a run at the floor that neither does at this size.
- **Record.** The request was first read as undersampling the controls to the case count (a 1:1 training set
  of 136,618 participants). That dataset and the scripts that made and read it were built, committed and
  withdrawn before any training (Robert's clarification, 18 November, evening); nothing of it remains.

## Without covariates (Robert, 19 November 2026)

The three runs above sit at the covariate floor, so every AUROC they reach may be the covariates'. One run asks what the
published network reads from its 28,536 SNPs alone, on a scale where the floor is chance.

- **Model.** `--dm snps` (DeepWideGP `widening`, 19 November 2026; before that a genotype-only model was refused): no
  covariate encoder, so the 22 chromosome vectors alone pass through the attention pooling into the head. Everything else
  as the quick run `classic_i10_n215872_quick`: the dataset `classic_i10_n215872` (all 215,872 training participants, the
  28,536 I10 GWAS SNPs at p ≤ 0.05, clumped), the published settings (width 32, one layer, attention pooling, batch 64,
  learning rate 7 × 10⁻⁴, weight decay 10⁻⁴, dropout 0.2, plain cross-entropy at natural prevalence, constant learning
  rate), uniform batches, seed 1, validation every 53,952 draws, patience 8. Run `runs/nocov/classic_i10_n215872_nocov`,
  launcher `widedeepgp-blocks/chains/run_nocov.sh` (shared with the LD-block model's no-covariate run; design document
  item 20), code frozen in `runs/nocov/code_<commit>`; the best snapshot is then scored on the tuning set by
  `score_snapshots.py`, which fits the covariates-only logistic regression beside it for the record (it is not this
  run's floor).
- **Yardsticks without covariates**, on the tuning set: the cross-fitted block ridge PRS alone for I10, 0.5989 (HM3+
  SNPs, Privé blocks; `widedeepgp-blocks/results/hm3plus/check_block_ridge.md`); the PGI arm's scores alone for I10
  (`PGI/results/hm3plus/tune_metrics.csv`, `auroc_pgi`): C+T 0.594, lassosum2 0.604, LDpred2-auto 0.615, SBayesRC
  0.616; chance 0.5. The classic set's own linear score (a clumping-and-thresholding score on its 28,536 SNPs) is not
  fitted separately; the PGI arm's C+T on the full panel stands in.
- **Expectation, stated before the result.** With covariates the network sits at the floor. Near 0.5 without them says
  that it learned nothing from the SNPs at this size and that its 0.699 was the covariates alone. Clearly above 0.5
  (the tuning standard error for I10 is about 0.004 with 6,255 cases among 20,000) is genetic signal the covariate
  comparison hid, i.e. signal the network reads but cannot add to the covariates. Near the scores alone (0.59 to 0.62)
  would mean the architecture extracts a linear score's worth of signal yet gains nothing when the covariates are
  added, which would point at the optimisation of the with-covariate run, not the architecture. The AUROC without
  covariates compares only with the covariate-free yardsticks, never with the runs above. Result: results.md,
  "Without covariates".

## Ladder runs: the published network with risk-allele dosages (Robert, 21 November 2026)

What the DeepGP replication taught (`scripts/deepgp-replication/results.md`): the published network learns from SNPs
only when each dosage counts the risk allele. With that change, the published network runs on all five rungs and
all 30 diseases: 150 runs, one seed. Earlier classic runs fed ALT dosages, which are as good as random for this,
and sat at the covariate floor.

- **SNPs.** The paper's rule on each rung's own GWAS of the disease (the PGI arm's REGENIE fit on exactly that rung's
  training people): p <= 0.05, clumped at r2 0.7 within 500 kb on the LD panel (`build_lists.py`, which uses
  `build_datasets.leads`). Expected 30,000-58,000 SNPs per run; the E11 top-rung list is the earlier
  `classic_e11_r07_n215872` selection exactly (47,341 SNPs).
- **Risk-allele dosages.** Each SNP is turned to count the allele whose log odds ratio in that GWAS is above 0: a sign
  of -1 on the standardised dosage where the store's REF allele raises risk (about half the SNPs), which equals
  recoding the dosage to 2 - dosage.
- **Genotypes are streamed from the main HapMap3+ store.** Each run's folder holds a SNP list with signs and links to
  the rung's sample tables, nothing else (`data/classic_deepgp_ladder/datasets/<disease>/<rung>/`). The loader
  decodes whole rows and keeps the listed SNPs (DeepWideGP `genostore/datamodule.py`, store.json `"snps"`).
  Per-disease stores would take about 280 GB. Checked: the E11 top-rung list gives the same SNP IDs, labels,
  covariates and dosages for 400 people as the dedicated store.
- **The network and its settings: the paper's type 2 diabetes configuration, for every disease and rung, without a
  search** (Supp. Table 7 = the published `scripts/t2d.sh`): batch 64, one Mamba block per chromosome (the
  one-direction "v1" block), width 32, dropout 0.2, AdamW at 7e-4 with weight decay 1e-4, four attention heads, two
  dense layers, the 3-wide convolutional SNP embedding and attention pooling, base-pair positions in the positional
  code. Why T2D: it is the only configuration the authors published, and the replication reached the paper's DeepGP
  level with it on our data (0.648-0.652 against 0.64). One setting for all diseases is the little-selection figure
  of CLAUDE.md section 6.
- **Code.** DeepWideGP `main_genome.py`, the published training script, lightly edited (commit 50f0ae6, frozen in
  `runs/classic_deepgp_ladder/code_50f0ae6`):
  - the published test scoring is commented out; the best model scores the tuning set instead;
  - `--seed` and `--data_dir` are honoured;
  - validation stays every 300 batches, now counted across epochs. At the smallest rung an epoch holds fewer than
    300 batches, which Lightning refused.
- **Batches: half cases, by recycling the cases** (`--balanced_batches`; the rotation sampler with one label, as the
  section above). Every batch of 64 holds 32 cases and 32 non-cases. An epoch shows every non-case once; the cases
  come round as often as that takes, in reshuffled passes. No participant is dropped. This gives the paper's 1:1
  training batches without throwing away controls.
- **Stopping.** The published code's numbers:
  - validation every 300 batches, stopping after 10 checks without improvement, at most 30 epochs; the best check
    is kept;
  - validation is on the 20,000 tuning people, outside every GWAS, not on the code's 15% of the training people,
    who sit inside it;
  - the metric is AUROC over the whole tuning set. The code averages AUROC over batches of 64, and at population
    prevalence many such batches hold one case or none.
- **Covariates.** The study's 12 (birth year, sex, PC1-10), standardised on the training rows; the paper used 8. In
  the replication the two sets scored the same on covariates alone (0.670 vs 0.669).
- **Before the ladder: a simulated-trait check** (CLAUDE.md section 6). The same pipeline runs on `sim200` of the
  pilot's simulated traits (40,000 training people with its own GWAS, 10,000 tuning people). The ladder starts only
  if its tuning AUROC reaches 0.60; covariates carry no signal for that trait.
- **Order, GPUs and budget** (`runs/classic_deepgp_ladder/schedule.py`):
  - the smallest rung goes first; each rung's first run computes the SNP statistics the rung's 30 runs share;
  - a run starts only on a GPU with nobody else's job, at most two of these runs per GPU and four at once (peak
    about 6.5 GB each, measured), with two loader workers and one maths thread each;
  - the chain stops at the first failed run, or once it has used 336 GPU-hours, the comparator rule's two weeks on
    one GPU;
  - expected about 65 GPU-hours (see the plan's estimate); the per-run results go to
    `runs/classic_deepgp_ladder/results/`.
- **Scoring.** Each run's tuning AUROC and predictions are kept. The comparison with covariates alone and with the
  PGIs follows the shared protocol (`PGI/methods.md` section 6). The locked test set is scored once, with the other
  arms, at the end.
