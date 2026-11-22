# Methods: the final WideDeepGP on the five-rung ladder

*Written 22 November 2026. The model is described in DeepWideGP `methods.md` section 4.12; this file records how
the ladder was run. Results: `results.md`.*

**What ran.**
- The final model on the five real rungs: 13,199 to 215,872 training participants, 30 diseases. It is a fixed,
  cross-fitted 32-wide block-ridge table over 431 Privé blocks, a transformer 32 wide with 2 layers, one head per
  disease, and an entropy-normalised loss.
- Seeds 1, 2 and 3 at each rung, with the same settings at every rung.
- Each rung has its own folds, block ridge and table (`make_warm_start.py --embed_dim 32`). The four lower rungs'
  tables were built on 22 November.

**No search per rung.** The settings were chosen on the top rung's tuning set (`widedeepgp-tweaks`). So the top rung's
result is optimistic, and the lower rungs got a setting fixed in advance.

**Three seeds, WideDeepGP only** (Robert, 22 November 2026). They are there to show whether it beats the covariates
alone, with the seed spread as the measure of training chance. Results are the mean of the three with their standard
deviation. The other models run one seed.

**Code and job.**
- One detached chain, `runs/widedeepgp_ladder/chain.sh`, from a frozen copy of DeepWideGP `widening` (f5ebb17).
- It ran on GPU 4 beside scPRS and the Kelemen MLP, as agreed with the coordinating session, with at most three runs
  at once.
- The evaluation batch was 256 instead of 1,024 to save GPU memory: 4.5 GB a run instead of 13.8 GB. No weight
  snapshots were written.

**Checks.**
- Each run's best checkpoint was re-scored on the tuning set with the first layer streamed, not cached, and had to
  give back its training record. All 15 did, exactly.
- Per-person tuning scores are under `data/datasets/ard30/<rung>/scores/widedeepgp_ladder/`, owner-only.
- Only each run's best checkpoint is kept.

**Interruption.**
- At 02:15 UTC on 22 November the chain was stopped while the number of seeds was being decided.
- The third rung's seeds 2 and 3 were stopped at epochs 17.6 and 17.9 and resumed from their last checkpoints. They
  passed the same re-score check.
- The fourth rung's seed 1, still building its first layer, was restarted.

**Yardsticks**, on the same rungs and tuning set:
- **Covariates only:** an unpenalised logistic regression per disease on the rung's training rows, the fit
  `score_snapshots.py` reports.
- **The cross-fitted ridge PRS with the covariates,** and **the logistic ridge twin** on the table's block channels.
  Both are from `widedeepgp-blocks/results.md`.

**The test set was not used.** It is scored once for all models at the end, with the paired bootstrap of
`PGI/methods.md` section 8.
