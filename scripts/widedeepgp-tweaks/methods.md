# Overnight tweaks of WideDeepGP at the top rung (19 to 20 November 2026)

*Design and record of the overnight queue Robert asked for on 19 November 2026 at 23:55 UTC: "try a bunch of tests
and tweaks overnight to the widedeepgp model to maximize lift over the covariates + comparable PGI ... in the
morning I want a table of the things you tried and what helped vs didn't help. If something shows an improvement,
even a tiny insignificant one, keep it." Later the same night: queue everything behind the running jobs, queue the
experiments planned earlier that never ran, and add ideas of my own. Results: `results.md` (the table is produced by
`runs/tweaks/summarise.py`). Code: DeepWideGP branch `tweaks` (a worktree off `widening` 5c47884), frozen copies
`runs/tweaks/code_<commit>`. Aggregates only; the test set is never read.*

## 1. The base configuration and the yardsticks

Every run changes one thing about the best configuration so far, `runs/blocks_diag/diag_S3_folds_frozen_nomask`
(the "base"): the S3 transformer mixer (width 64, four layers, four heads) over the 431 Privé LD blocks of the top rung
(215,872 training participants, 1,258,200 HapMap3+ SNPs, 30 diseases), a frozen fold-routed ridge warm start of the
per-SNP table (16 channels, unit-RMS tokens), the mean readout, no input masking, dropout 0.2, learning rate 1e-4,
weight decay 1e-3, batch 64, the plateau schedule, plain cross-entropy at natural prevalence, bf16, seed 1, 10
epoch-equivalents. Its tuning AUROC (mean over the 30 diseases) is 0.6735 (three seeds 0.6735, 0.6737, 0.6738).
The first layer is fixed, so every run computes the pooled block vectors once (DeepWideGP methods 4.11, "A fixed first
layer is computed once per run"); the queue keeps them on disk (`--first_layer_cache`, below) so a run takes 20 to
40 minutes instead of three hours, and several runs share a GPU.

Yardsticks on the same tuning set (20,000 participants): covariates alone 0.6627; the logistic ridge on the
cross-fitted block channels 0.6732; the cross-fitted block ridge PRS plus covariates 0.6777 (gain over the covariates
+0.0149); the PGI arm's scores combined with the covariates by its own protocol (a refit on the tuning set, whose
covariates-only figure is 0.6653): C+T +0.0165, lassosum2 +0.0175, LDpred2-auto +0.0219, SBayesRC +0.0224. Lift is
read as gain over the covariates-only model fitted the same way as the model under test.

**Lift over the PGI a first layer was built from.** Robert's success metric for a network on a PGI table ("if the
network takes the weights of LDpred2, we want lift over naive LDpred2"). `runs/tweaks/pgi_yardstick.py` recovers each
disease's PGI from the table's pooled vectors (their sum over blocks, undoing the first layer's division by the square
root of the block's SNP count), fits a logistic regression of each disease on [its PGI, the 12 covariates] on the
training rows and scores the tuning set. What it found (section 4) changes how that lift must be read.

## 2. The queue

`runs/tweaks/queue/scheduler.py` runs bash job scripts from `jobs/` in name order as GPU slots free up (a job's
`# slots:` header: 1 for a cached run, 3 for a streamed run that needs a whole GPU, 0 for CPU work; `# after: GLOB`
makes a job wait for others), with the live capacity in `gpu_slots.txt`. GPUs 0 and 1 belong to another user. GPU 4
was drained and GPU 2 held empty between 00:30 and the start of the other agent's chain (session "caching table runs",
which trains the network on the frozen 30-channel lassosum2 and LDpred2 tables under Robert's free-minus-one rule), then
reopened; the queue never launched onto a GPU another session's process was waiting for. Each job writes
`runs/tweaks/<name>/` (the usual run folder), then `score_snapshots.py` re-scores the best snapshot by streaming, an
independent check of every cached run. `queue.log` is the record; `logs/<job>.log` the output.

## 3. What was queued, and why (Robert's numbering)

| Item | Job(s) | What changes | Note |
|---|---|---|---|
| 0 | `budget50` | budget 50 epoch-equivalents (early stopping and the plateau schedule still act) | |
| 9 | `budget20` | budget 20 | |
| 3 | `rot10` | disease-balanced batches: the rotation sampler (each batch half cases of one disease, half its non-cases drawn at random, the diseases in turn, every non-case of a disease shown once per pass, cases cycling), plain cross-entropy on the batch as drawn | Robert's reading of "controls exhausted, then the next epoch" is what `RotationBatchSampler` does per disease; the epoch-equivalent stays the unit of budget. He asked for it as the default without a test; the A/B was cheap, so it ran first and decided the sampler of the searches (section 4) |
| 8 | `labelnorm_rot`, `labelnorm_shuf` | `--label_norm entropy`: each disease's loss divided by the entropy of its training prevalence, so a rare disease weighs as much as a common one (before: equal weights per disease, but a rare disease's cross-entropy and gradient are small) | under both samplers |
| 7 | `perdisease_rot` | `--per_disease_head 32`: the output layer becomes one small two-layer head per disease on the shared hidden layer | |
| 1 | `search_paper` (4 workers, 24 trials) | an Optuna search over the published paper's grid: batch {32, 64, 128}, mixer layers {1, 2, 3}, width {16, 32, 64}, heads {2, 4, 8}, head hidden layers {1, 2, 4}, dropout [0.05, 0.3], learning rate [1e-5, 1e-3], weight decay [1e-5, 1e-3], plus the mixer kind {transformer, bi-Mamba}; successive halving after 2 epoch-equivalents | `--tune_space paper`; the frozen table is not searched |
| 5 | `search_opt256` (4 workers, 16 trials) | batch 256 and a search of the optimiser: learning rate [3e-5, 3e-3], weight decay [1e-6, 1e-2], dropout [0, 0.4], schedule {plateau, cosine, constant}, warm-up {0, 200, 1000} | `--tune_space optimizer` |
| 4 | `e11_build` then `classic_e11_r07_balanced` | the classic DeepGP for type 2 diabetes (E11) with the paper's clumping (r2 0.7 within 500 kb, p <= 0.05; `build_datasets.py --r2 --kb`) instead of the comparator's r2 0.1 / 250 kb: 47,341 SNPs (101,861 with p <= 0.05); published settings, balanced batches as the paper's 1:1 sets; 12 covariates (the paper: age, sex, 6 PCs) | the paper reports T2D 0.64 on a balanced hold-out against 0.60 for its PRS; our yardsticks for E11: covariates 0.6708, ridge PRS + covariates 0.7039, LDpred2 0.7238 (PGI protocol) |
| 2 | `blockattn_lasso`, `blockattn_ldpred` (the proposed method, full-rung tables); secondary: `blockattn_lasso_c107`, `blockattn_ldpred_c107` (half-rung weights), `blockattn_ridge`, `blockattn_ridge_k64` | `--block_attn K`: in every block the K SNPs with the largest rows of the frozen table form a short sequence (token = dosage-scaled table row, the dosage itself and a learned identity per SNP) with the block's pooled vector as one more token; a transformer layer shared by all blocks (self-attention over the pairs of SNPs) runs over each block's sequence and the mean of its outputs is added to the block's token; then the mixer over blocks with the covariate token and the head, as before | K = 32 (431 x 33 tokens per participant), width 32, one layer; the selected dosages are cached in bfloat16 beside the pooled vectors; the lassosum2 and LDpred2 versions use the other agent's 30-channel tables (section 4 on their overfitting), the ridge version the fold-routed 16-channel table (each participant's rows from the copy fitted without them) |
| 6 | `blockmamba_lasso` (proposed); secondary: `blockmamba_lasso_c107`, `blockmamba_ridge` | the same with the published bi-Mamba block within each block, reading its K SNPs in genomic order (local context), the 30 or 16 SNP weights never updating; the mixer over blocks and the head are the "classic second layer" | |
| baselines for 2 and 6 | `pgi_lasso_frozen`, `pgi_ldpred_frozen` (and `_c107` for the secondary check) | the frozen PGI tables under the base recipe | the other agent's chain runs the same tables under the main tree's code; mine give the within-block runs a like-for-like control |

Planned earlier and never run (the LD-block design document, sections 7 and 12, and the chain paused on 18 November):
the top-rung size axis on the frozen table, `size_s1` (16 x 1), `size_s2` (32 x 2), `s4_rot` (128 x 6), `size_s5`
(256 x 8); the second mixer, `mamba_s3` (bi-Mamba, six layers at width 64, matched to S3); the table-width axis,
`table_d4` and `table_d32` (ridge warm starts 4 and 32 wide from `make_warm_start.py`, variance of the ridge kept
0.368 and 0.995 against the base's 16); the attention readout under the new sampler, `atten_rot`. Not queued, and why:
the partition sweep (ldsplit2000, ldsplit5000) needs a block ridge per partition (hours of GPU each) before any warm
start; the masked-pretraining ablation trains the table, so it cannot be cached and takes three hours a run; a
30-disease classic model at the paper's clumping would hold about a million SNPs. The E0 simulated-trait gate was set
aside by Robert on 17 November ("only real diseases").

Added during the night as results came in: `perdisease_shuf`, `covskip_shuf` and `labelnorm_shuf` (items 7 and 8 and
the covariate skip on the base sampler once the rotation proved worse); `budget20_cosine` (the doubled budget with a
cosine schedule); the follow-ups of the 32-wide table, `table_d32_20` (doubled budget), `table_d32_opt256_20` (the
batch-256 optimiser settings) and `table_d32_pdh20` (with the per-disease head); `blockattn_ridge_k64` (64 SNPs, two
layers); the combinations `combo_paper20`, `combo_opt256_20`, `combo_paper20_labelnorm` and `combo_all20` (the search
winners with the doubled budget, with and without the entropy loss and the per-disease head); `yardstick_c107` and
`ensemble_seeds`; and `confirm_seed2`/`confirm_seed3`, which retrain the best combination or follow-up with seeds 2 and 3
(the project rule of 17 November: the chosen configuration is reported as the mean of three seeds with the spread; every
other run keeps seed 1).

Own ideas: `covskip_rot` (`--cov_skip logistic`: a linear map from the covariates straight to the logits, started at
the covariates-only logistic model with the head's output map at zero, so the network begins at the covariate floor
and learns only what the covariates leave; Robert's "over-reliance on covariates" concern of 18 November);
`s4_rot` and `size_s5` (a larger mixer is cheap on a cached first layer); `blockattn_lasso_k64` (64 SNPs, two
layers); `table_d32`; `ensemble_seeds` (the three seeds of the base averaged on the logit scale, what ensembling alone
buys); the attention readout retried.

## 4. Decisions taken during the night

- **The searches run with the base sampler, not the rotation.** `rot10` reached 0.6722 (best at 6.5 epoch-equivalents)
  against the base's 0.6735, and trailed the two budget runs by about 0.0015 at every matched epoch. Robert's rule
  ("keep what improves") read the other way: a decrement is not kept. The items tested on top of the rotation
  (`labelnorm_rot`, `perdisease_rot`, `covskip_rot`, `atten_rot`, `s4_rot`) are judged against `rot10`, and their
  gains, if any, are then confirmed under the base sampler.
- **The PGI weights fitted on the training rows overfit them, and the proposed method is run as specified.** The
  lassosum2 score has AUROC 0.887 on the training rows against 0.572 on the tuning set; LDpred2-auto 0.898 against 0.582
  (`runs/tweaks/yardsticks/*_summary.json`): with 1.2 million weights fitted from the GWAS of 215,872 participants, the
  score fits those participants far better than new ones. A logistic regression of each disease on [its PGI, covariates]
  fitted on the same training rows therefore trusts the score too much and scores 0.634 (lassosum2) and 0.642 (LDpred2)
  on the tuning set, below the covariates alone; the PGI arm's own figures (0.6829, 0.6873) come from a refit on the
  tuning set, which does not see this. A network trained on the training rows with such a table learns the same overfit
  score (the other agent's frozen-table runs peak at their first validation, 0.6447 and 0.6546, and fall), as the
  in-sample ridge warm start did on 18 November. Robert's instruction (03:30 UTC): this is overfitting on the training
  data and a result to report, not something to route around, so items 2 and 6 and their frozen-table controls run on
  the full-rung tables exactly as proposed, and the "naive PGI" yardstick is the PGI arm's protocol figure. As a
  secondary check (the fix tried first), the same architectures also run on the other agent's tables built from the
  107,743-rung's weights for the 108,129 other training participants (dataset `hm3plus_c107743`), where the weights
  were fitted on none of the participants they are applied to, and on the fold-routed ridge table.
- **Pooled vectors kept on disk.** `--first_layer_cache DIR` (DeepWideGP `tweaks`) writes the training and tuning
  splits' pooled vectors (and, with `--block_attn`, the selected dosages) once per table, dataset, partition, folds,
  scaling and device, owner-only under `data/.../blocks/cache/`, and later runs load them (10 to 20 s) instead of
  streaming 1.26 million dosages a participant (2 minutes and a pass over the store per run). The test set is never
  cached. A cached run equals the streamed run to 1e-6 (tests), and every run's best snapshot is re-scored by streaming.

- **Fixes made while the queue ran.** The within-block module's first top-rung runs failed: the fused attention kernels
  refuse more than 65,535 sequences times heads at once (64 participants x 431 blocks x 4 heads), and the bi-Mamba
  kernels' intermediates ran a shared GPU out of memory; the block sequences now go through the mixer in pieces of
  8,192, recomputed in the backward pass for the Mamba variant. The loss's normalisation buffer is derived from the
  counts and not saved, so checkpoints from before the night load (the seed ensemble had failed on the missing key).
  Job scripts first reported exit 0 whatever happened inside; they now propagate the status, and the first attempts'
  failures are in their logs. The on-disk first-layer caches grew to about 200 GB and /mnt/sdg reached 98% at 04:55 UTC
  (the PRSformer session noticed); the caches of finished runs were removed, keeping the three still in use.

## 5. Software

DeepWideGP branch `tweaks` (commits after 5c47884): `--tune_space paper | optimizer`, `--label_norm entropy`,
`--per_disease_head W`, `--cov_skip zero | logistic`, `--first_layer_cache DIR`, `--block_attn K` with
`--block_attn_dim`, `--block_attn_layers`, `--block_attn_heads`, `--block_attn_mixer`; `tests/test_ml_tweaks.py`
(11 tests, CPU; the within-block bi-Mamba path checked on a GPU with the synthetic store). Scripts
`classic-deepgp-comparator/build_datasets.py --r2 --kb`. The queue, the yardstick scripts and the summary are copied
here under `queue/` and `tools/` from `runs/tweaks/`.

## 6. The combined model and the partition test (21 November 2026)

Robert asked: combine every change that tested better overnight, and where two changes are alternatives take the
nominally better one, however small the difference; then test that model on the 431 Privé blocks against the 2,000- and
5,000-block refinements, refitting the block ridge on each partition.

**The combined model.** Each choice, with the comparison that decided it (tuning AUROC, mean of 30 diseases):

| choice | taken | instead of | evidence |
|---|---|---|---|
| table width | 32 | 4, 16, 64 | 0.6753 against 0.6729, 0.6735, 0.6748 |
| recipe | the paper-grid search's: batch 32, transformer width 32 × 2 layers, 4 heads, 2 head layers, lr 7.4e-5, weight decay 1.1e-5, dropout 0.19 | the base recipe (batch 64, S3, lr 1e-4, weight decay 1e-3, dropout 0.2); the batch-256 search's (lr 3.8e-4, weight decay 6.8e-4, dropout 0.12) | 16-wide table, 20 epoch-equivalents: 0.67455 against 0.67405 and 0.67386; 10 epoch-equivalents: 0.6741 against 0.6735 and 0.6740. On the 32-wide table only the other two ran (0.67555, 0.67550) |
| mixer | transformer | bi-Mamba | the paper search: 0.6741 against 0.6738 (at S3 the Mamba led by 0.0001, in a recipe the search ranked lower) |
| schedule | cosine to zero with 200 warm-up steps | plateau | 20 epoch-equivalents: 0.67412 against 0.67405 |
| budget | 20 epoch-equivalents | 10, 50 | 0.67405 against 0.67351 and 0.67404 |
| batches | shuffled, natural prevalence | the disease-balanced rotation | 0.6735 against 0.6722 |
| readout | mean, with a per-disease head of width 32 | the attention readout, which this code cannot combine with the head | head +0.0003 (0.6739); attention +0.0002, on the rotation |
| loss | each disease's loss divided by the entropy of its prevalence | plain cross-entropy | 0.6737 against 0.6735 |

Left out because they tested worse: the covariate skip, the within-block modules, the S1, S4 and S5 mixers, input masking.
The per-disease head and the entropy loss did not add in the two stacks that ran (the 32-wide table with the head 0.67495
against 0.67555 without; the paper recipe with head and loss 0.67451 against 0.67455); Robert's instruction keeps them in.

**The test.** Three trainings of the combined model, identical except the partition: `prive431` (431 blocks of 119 to
15,194 SNPs), `ldsplit2000` (2,000 blocks; 60 to 2,314 SNPs, median 475) and `ldsplit5000` (5,000 blocks; 19 to 1,108,
median 193), the nested partitions built on 17 November. Each partition gets:
- its own block ridge (`build_block_ridge.py`), on the same 215,872 training participants, store, SNP statistics, five folds
  and three penalties as the 431-block ridge. The folders `data/datasets/ard30/hm3plus_ld2000` and `_ld5000` hold only
  these products; their sample tables link to the top rung's;
- its own 32-wide fold-routed warm start (`make_warm_start.py`, unit-RMS tokens);
- its own linear yardstick: that partition's cross-fitted ridge PRS with the covariates (`fit_block_linear.py --prs_only`).

The finer ridges model LD only inside the smaller blocks, so the table changes with the partition as well as the tokens.
Reading each network against its own partition's ridge PRS separates the two. The finer partitions write no channel files
(`--no_channels`: at 5,000 blocks the training split's channels would take 390 GB of scratch disk), so they have no
logistic twin. The first layer is cached in host memory (about 13, 60 and 151 GB), filled in place (DeepWideGP `tweaks`
a74b5e9), never on disk. Each job waits until the host has its memory plus 100 GB to spare and GPU 2 has its VRAM free.
Seed 1 only; four snapshots per run. Queue `runs/partition_test/queue` (frozen code `code_a74b5e9`, scripts
`scripts_e7045c6`); table `runs/partition_test/summary.md`.

**GPU coordination.** GPU 2 was empty; GPUs 0 and 1 belong to another user, 3 and 4 run other sessions' jobs (Delphi;
the Kelemen MLP ladder and scPRS). Every session was asked about its plans first. None clashed. The PGI chain borrows GPU 2
for about five minutes every one to two hours. The PRSformer synthetic chain (session deepwidegp-7c) has waited since
17:17 UTC under the old leave-one-free rule and could start only on GPU 2: which job has GPU 2 is Robert's call.
