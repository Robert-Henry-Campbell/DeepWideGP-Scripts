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
| 2 | `blockattn_lasso`, `blockattn_ldpred`, `blockattn_ridge`, `blockattn_lasso_k64` | `--block_attn K`: in every block the K SNPs with the largest rows of the frozen table form a short sequence (token = dosage-scaled table row, the dosage itself and a learned identity per SNP) with the block's pooled vector as one more token; a transformer layer shared by all blocks (self-attention over the pairs of SNPs) runs over each block's sequence and the mean of its outputs is added to the block's token; then the mixer over blocks with the covariate token and the head, as before | K = 32 (431 x 33 tokens per participant), width 32, one layer; the selected dosages are cached in bfloat16 beside the pooled vectors; the lassosum2 and LDpred2 versions use the other agent's 30-channel tables, the ridge version the fold-routed 16-channel table (each participant's rows from the copy fitted without them) |
| 6 | `blockmamba_lasso`, `blockmamba_ridge` | the same with the published bi-Mamba block within each block, reading its K SNPs in genomic order (local context), the 30 or 16 SNP weights never updating; the mixer over blocks and the head are the "classic second layer" | |
| baselines for 2 and 6 | `pgi_lasso_frozen`, `pgi_ldpred_frozen` | the frozen PGI tables under the base recipe | the other agent's chain runs the same tables under the main tree's code; mine give the within-block runs a like-for-like control |

Planned earlier and never run (the LD-block design document, sections 7 and 12, and the chain paused on 18 November):
the top-rung size axis on the frozen table, `size_s1` (16 x 1), `size_s2` (32 x 2), `s4_rot` (128 x 6), `size_s5`
(256 x 8); the second mixer, `mamba_s3` (bi-Mamba, six layers at width 64, matched to S3); the table-width axis,
`table_d4` and `table_d32` (ridge warm starts 4 and 32 wide from `make_warm_start.py`, variance of the ridge kept
0.368 and 0.995 against the base's 16); the attention readout under the new sampler, `atten_rot`. Not queued, and why:
the partition sweep (ldsplit2000, ldsplit5000) needs a block ridge per partition (hours of GPU each) before any warm
start; the masked-pretraining ablation trains the table, so it cannot be cached and takes three hours a run; a
30-disease classic model at the paper's clumping would hold about a million SNPs. The E0 simulated-trait gate was set
aside by Robert on 17 November ("only real diseases").

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
- **The PGI tables carry the training labels.** The lassosum2 score has AUROC 0.887 on the training rows against 0.572 on
  the tuning set; LDpred2-auto 0.898 against 0.582 (`runs/tweaks/yardsticks/*_summary.json`). A logistic regression of
  each disease on [its PGI, covariates] fitted on the training rows therefore scores 0.634 (lassosum2) and 0.642
  (LDpred2) on the tuning set, below the covariates alone, because the fit trusts the in-sample score. The PGI arm's
  own figures (0.6829, 0.6873) come from a refit on the tuning set, which hides this. A network trained on the training
  rows with such a table reads the labels off it (the warm-start lesson of 18 November, whose signature is a tuning
  AUROC that peaks at the first validations and falls). So: the "naive PGI" yardstick the deep arm's way is below the
  floor and useless; the honest comparison of a PGI-table network is its gain over the covariates against the PGI
  arm's gain over its own covariates (+0.0175 lassosum2, +0.0219 LDpred2); and items 2 and 6 were run twice, on the
  lassosum2 table as asked and, leak-free, on the fold-routed ridge table. The other agent was told at 00:40 UTC.
- **Pooled vectors kept on disk.** `--first_layer_cache DIR` (DeepWideGP `tweaks`) writes the training and tuning
  splits' pooled vectors (and, with `--block_attn`, the selected dosages) once per table, dataset, partition, folds,
  scaling and device, owner-only under `data/.../blocks/cache/`, and later runs load them (10 to 20 s) instead of
  streaming 1.26 million dosages a participant (2 minutes and a pass over the store per run). The test set is never
  cached. A cached run equals the streamed run to 1e-6 (tests), and every run's best snapshot is re-scored by streaming.

## 5. Software

DeepWideGP branch `tweaks` (commits after 5c47884): `--tune_space paper | optimizer`, `--label_norm entropy`,
`--per_disease_head W`, `--cov_skip zero | logistic`, `--first_layer_cache DIR`, `--block_attn K` with
`--block_attn_dim`, `--block_attn_layers`, `--block_attn_heads`, `--block_attn_mixer`; `tests/test_ml_tweaks.py`
(11 tests, CPU; the within-block bi-Mamba path checked on a GPU with the synthetic store). Scripts
`classic-deepgp-comparator/build_datasets.py --r2 --kb`. The queue, the yardstick scripts and the summary are copied
here under `queue/` and `tools/` from `runs/tweaks/`.
