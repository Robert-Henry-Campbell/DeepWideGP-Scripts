# Results: the overnight tweaks of WideDeepGP at the top rung (20 November 2026)

*Aggregates only; the tuning set (20,000 participants), mean AUROC over the 30 diseases unless a disease is named. The
test set is never read. The table is `runs/tweaks/results_table.md`, rebuilt by `python runs/tweaks/summarise.py`; the
yardsticks and the reading are below. Written as the runs finished; the final state is at the end.*

## Summary (final, 06:20 UTC)

**The proposed method on the full-rung lassosum2 and LDpred2 tables does badly and that is the headline:** the frozen
tables read 0.6451 (lassosum2) and 0.6546 (LDpred2) at their first validations and fall from there, below the covariates
alone (0.6627) throughout and 0.010 to 0.011 below the same scores combined with covariates by the PGI protocol. Items 2
and 6 built on those tables (within-block attention, within-block Mamba) trace the same curves. The weights were fitted
on the training rows and fit them far better than new participants (PGI AUROC 0.89 on the training rows against 0.57 on
the tuning set), so the network learns that overfit score. The half-rung check (weights fitted on other participants) and
the ridge table confirm that the within-block modules themselves add nothing (half-rung tables) or memorise (ridge table).

**What helped, on the fold-routed ridge table (base 0.6735, seed spread 0.0003):**

| change | tuning AUROC | gain | reading |
|---|---|---|---|
| ridge table 32 wide instead of 16 (`table_d32`) | 0.6753 | +0.0018 | the largest gain; better for 27 of 30 diseases; 64 wide gives 0.6748, so the width axis saturates at 32 |
| the same at batch 256 with the optimiser search's settings (`table_d32_opt256_20`) | 0.6755 | +0.0020 | the width gain carries to the four-times-faster recipe |
| the same with the doubled budget (`table_d32_20`) | 0.6755 | +0.0020 | the budget adds 0.0002 to the 32-wide table |
| three-seed ensemble of the base | 0.6746 | +0.0009 | seed averaging alone, at three times the training |
| paper-grid search settings with the doubled budget (`combo_paper20`) | 0.6745 | +0.0010 | the two +0.0006 gains add to about +0.001 |
| doubled budget (items 0 and 9) | 0.6741 | +0.0006 | complete by 15 epoch-equivalents; the 50-epoch run stops itself at 19 |
| paper-grid search (item 1) | 0.6741 | +0.0006 | a width-32, two-layer transformer at batch 32 |
| batch 256 with the optimiser search (item 5) | 0.6740 | +0.0005 | learning rate 3.8e-4, constant schedule |
| per-disease head (item 7) | 0.6739 | +0.0004 | within the seed spread |
| entropy-normalised loss (item 8) | 0.6737 | +0.0002 | within the seed spread |

**No change:** covariate skip, attention readout, cosine schedule, Mamba mixer (0.6736), S2 mixer (0.6738), the
combinations of several small gains on the 16-wide table (all 0.6744 to 0.6745). **Hurt:** balanced batches (item 3,
0.6722), S1 (0.6727), 4-wide table (0.6729), S5 (0.6700 then overfitting), S4 on the rotation (level with its baseline but
overfitting), within-block attention or Mamba on the ridge table (0.664 to 0.666 then falling). **Item 4:** the classic
DeepGP for E11 at the paper's clumping sits at the covariate floor (0.6667 against 0.6697) and overfits after one
epoch-equivalent, exactly as at the comparator's clumping (0.6659); the paper's T2D figures (0.64 against 0.60 for its
PRS) are below our covariates alone, so its headline is not reproduced here.

**Against the linear yardsticks:** the best network (0.6755, +0.0128 over the covariates) still trails the cross-fitted
block ridge PRS plus covariates (0.6777, +0.0149) and every PGI (lassosum2 +0.0175, LDpred2 +0.0219, SBayesRC +0.0224
over their covariates). The night closed about a quarter of the gap to the ridge PRS. Confirmation of the chosen
configuration with seeds 2 and 3: seeds 1, 2 and 3 of `table_d32_opt256_20` give 0.6755, 0.6747 and 0.6749, mean **0.6750** (spread 0.0008), against the base's three seeds 0.6735, 0.6737 and 0.6738, mean 0.6737: a gain of **+0.0014** on three-seed means, larger than either spread, so the 32-wide table is kept as the configuration going forward (with the batch-256 settings, which train it in a quarter of the steps).

## Yardsticks

| model | tuning AUROC | gain over its covariates |
|---|---|---|
| covariates alone (training-row logistic fit) | 0.6627 | |
| base: S3 on the frozen fold-routed ridge table, 10 epoch-equivalents (`blocks_diag/diag_S3_folds_frozen_nomask`) | 0.6735 (seeds 0.6735, 0.6737, 0.6738) | +0.0108 |
| logistic ridge on the block channels | 0.6732 | +0.0105 |
| block ridge PRS + covariates | 0.6777 | +0.0149 |
| PGI arm (refit on the tuning set; its covariates 0.6653): lassosum2 / LDpred2-auto / SBayesRC | 0.6829 / 0.6873 / 0.6878 | +0.0175 / +0.0219 / +0.0224 |
| naive lassosum2 / LDpred2 + covariates, fitted on the training rows (the full-rung tables) | 0.6341 / 0.6422 | -0.0287 / -0.0205: the weights overfit the training rows they were fitted on (PGI AUROC 0.887 / 0.898 there against 0.572 / 0.582 on the tuning set), so a fit on those rows trusts them too much; the PGI arm's refit on the tuning set gives 0.6829 / 0.6873 |

**Secondary check for items 2 and 6, weights fitted on other participants** (dataset `hm3plus_c107743`: the 108,129
training participants outside the 107,743 rung, PGI weights from that rung's GWAS, scored on the usual tuning set;
`runs/tweaks/yardsticks/c107_*`). The proposed method itself, on the full-rung tables, is in the item rows below.

| model fitted on the 108,129 rows | tuning AUROC | gain over its covariates (0.6624) |
|---|---|---|
| naive lassosum2 + covariates (logistic fit; PGI alone 0.556, the same on the training rows: no overfitting to them) | 0.6737 | +0.0113 |
| naive LDpred2-auto + covariates (PGI alone 0.564) | 0.6772 | +0.0148 |
| the other agent's network on the frozen lassosum2 table (`runs/pgi_tables/split_lassosum2`) | 0.6678 | +0.0054 |
| my control on the frozen LDpred2-auto table (`pgi_ldpred_frozen_c107`) | 0.6697 | +0.0073 (the logistic combination: +0.0148) |
| my control on the same table under the base recipe (`pgi_lasso_frozen_c107`); its last two epoch-equivalents moved 0.0001, so a longer budget would not close the gap | 0.6677 | +0.0053 |

So with weights fitted on other participants the frozen-table network trails the plain logistic combination of the same
score by 0.006, as it trails the ridge PRS at the full rung; on this split "lift over naive lassosum2" is read against 0.6737.

## The proposed method on the full-rung PGI tables (items 2 and 6 as specified)

The first layer as proposed: the frozen 30-channel table of the full rung's lassosum2 or LDpred2-auto weights (one copy;
the weights were fitted from the GWAS of the same 215,872 training participants). Tuning AUROC, mean of 30; covariates
alone 0.6627; the PGI arm's protocol figures for the same scores combined with covariates: lassosum2 0.6829, LDpred2-auto
0.6873 (their covariates 0.6653).

| run | what | result | reading |
|---|---|---|---|
| `runs/pgi_tables/top_lassosum2` (the other agent, main-tree code, base recipe) | the frozen lassosum2 table alone | 0.6447 at the first validation (0.25 ee), then falling at every pass to 0.6215 at 4.75 ee, stopped by patience | the network reads the overfit training-row score and gets worse the longer it trains; by the PGI protocol (network logit + covariates refit on the tuning set, `runs/pgi_tables/compare.csv`) it is 0.0101 below lassosum2 + covariates, significantly worse for most diseases |
| `runs/pgi_tables/top_ldpred2_auto` (the other agent) | the frozen LDpred2-auto table alone | 0.6546 at 0.5 ee, then falling to 0.635 at 5 ee | the same; 0.0112 below LDpred2 + covariates by the PGI protocol |
| `pgi_lasso_frozen` (my control, the same recipe as every run here) | the frozen lassosum2 table alone | 0.6451 at the first validation (0.25 ee): 0.6417, 0.6388, 0.6355 ... 0.6216 at 4.75 ee, stopped by patience; re-scored by streaming 0.6451 | reproduces the other agent's run (0.6447); below the covariates alone (0.6627) from the first validation on, 0.038 below lassosum2 + covariates by the PGI protocol |
| `pgi_ldpred_frozen` (my control) | the frozen LDpred2-auto table alone | 0.6546 at 0.50 ee, then falling to 0.6354 at 5.00 ee, stopped by patience | reproduces the other agent's run (0.6546); below the covariates alone from the start |
| `blockattn_lasso` (item 2) | within-block transformer over each block's 32 strongest lassosum2 rows | 0.6458 at 0.5 ee (0.6453 at the first validation), then falling to 0.626 by 4 ee, stopped by patience at 4.75 ee; re-scored by streaming 0.6458 | the same curve as the frozen table alone (0.6451 at 0.25 ee, 0.6216 at the end): the within-block attention neither helps nor hurts; the overfit score dominates |
| `blockattn_ldpred` (item 2 on LDpred2) | the same on the LDpred2-auto table | 0.6558 at 0.50 ee, then falling to 0.6395 at 5.00 ee | the frozen table alone gave 0.6546 and 0.6354: no change |
| `blockmamba_lasso` (item 6) | within-block bi-Mamba in genomic order over each block's 32 strongest lassosum2 rows, the rows never updated | 0.6455 at 0.25 ee, then falling to 0.6224 at 4.75 ee | the same curve as the frozen table alone: local context read by the Mamba block neither helps nor hurts; the overfit score dominates |

Why: the lassosum2 and LDpred2 scores have AUROC 0.887 and 0.898 on the training rows they were fitted on against 0.572
and 0.582 on the tuning set (`runs/tweaks/yardsticks/`). A network trained on those rows therefore sees a near-perfect
predictor and learns to trust it; the tuning set does not have it. Even a logistic regression of [PGI, covariates] on the
training rows scores 0.634 (lassosum2) and 0.642 (LDpred2) on the tuning set, below the covariates alone. This is
overfitting of the weights to the training data, to be reported as such; the PGI arm's protocol avoids it only by
refitting the combination on the tuning set.

## Reading, item by item (updated through the night; the full table is at the end)

Tuning AUROC, mean of 30 diseases, best validation pass; the base is 0.6735 at 10 epoch-equivalents (seed spread
0.0003). A change of 0.0005 or less either way is called no change.

| item | run | result | verdict |
|---|---|---|---|
| 3, balanced batches (rotation sampler, plain cross-entropy as drawn) | `rot10` | 0.6722 (best at 6.5 ee), about 0.0015 below the base at every matched epoch | hurt; not kept as the default. Items 7, 8 and the covariate skip were first run on top of it (below) and are being rerun on the base recipe |
| 0, budget 50 epoch-equivalents | `budget50` | 0.6740 at 14.8 ee; early stopping ended the run at 18.8 ee (16 validations without gain) | helped a little (+0.0005); the curve is flat from 13 ee |
| 9, budget 20 | `budget20` | 0.6741 at 15.5 ee; early stopping ended it at 19.5 ee | helped a little (+0.0006); the same as the 50-epoch run, which stopped itself at 18.8 ee: the gain from training longer is about 0.0005 and is complete by 15 epoch-equivalents |
| 1, the paper's grid searched (24 trials, 2 complete, 22 pruned by successive halving at 2 or 6 ee; 0.7 GPU-hours) | `search_paper` | chosen: batch 32, transformer width 32 with two layers and four heads, two head layers, learning rate 7.4e-5, weight decay 1.1e-5, dropout 0.19: 0.6741 at 8.5 ee; the other complete trial (bi-Mamba width 64, one layer, lr 8.3e-4) 0.6738; pruned trials at 6 ee reached 0.6735 and 0.6733 | helped a little (+0.0006) with a mixer a sixth the size of S3; width, depth, heads and mixer kind barely matter on the frozen table; the Mamba mixer is level with the transformer |
| 5, batch 256 and an optimiser search (16 trials, 4 complete, 12 pruned at 2 ee; 0.3 GPU-hours in all) | `search_opt256` | chosen: learning rate 3.8e-4, weight decay 6.8e-4, dropout 0.12, constant schedule with 200 warm-up steps, 0.6740 at 10 ee; the other complete trials 0.6737 to 0.6738 (learning rates 2e-4 to 7e-4, plateau schedules) | helped a little (+0.0005), and a batch of 256 trains in a quarter of the steps; the optimiser is not the bottleneck, every completed setting lands within 0.0005 of the base |
| own: budget 20 with a cosine schedule to zero (200 warm-up steps) | `budget20_cosine` | 0.6741 at 14.8 ee, stopped at 18.8 | the same as the plateau schedule's 0.6741: the schedule does not matter |
| 4, the classic DeepGP for E11 (type 2 diabetes) at the published paper's clumping (r2 0.7 within 500 kb, p <= 0.05: 47,341 SNPs), published settings, balanced batches | `classic_e11_r07_balanced` | E11 tuning AUROC 0.6667 at 0.75 ee, then falling (0.6484 at 2.75 ee, stopped by patience); covariates alone 0.6697 | at the covariate floor (−0.003) and overfitting from the first epoch-equivalent. The paper's T2D numbers (DeepGP 0.64, its PRS 0.60, LightGBM 0.61 on a balanced hold-out) sit below our covariates-only 0.67, so its headline cannot be matched here: with year of birth, sex and ten PCs the covariates alone carry more than its whole model, and its gain over its PRS is not reproduced on our data. The control at the comparator's clumping (r2 0.1 within 250 kb: 25,714 SNPs, `classic_e11_r01_balanced`) gives 0.6659 at 1.0 ee and overfits the same way, so the clumping change itself is worth nothing (+0.0008, within noise) |
| 2, secondary check: within-block transformer over each block's 32 strongest SNPs, on the lassosum2 table whose weights come from the other half of the rung (108,129 training participants) | `blockattn_lasso_c107` | 0.6676 at 10 ee (458,000 parameters in the within-block module) against its frozen-table control `pgi_lasso_frozen_c107` 0.6677; the two curves lie on top of each other at every epoch | no change: the pairwise SNP interactions within blocks add nothing the pooled linear summary did not already carry; both stay 0.006 below the plain logistic lassosum2 + covariates (0.6737) |
| planned table-width axis: the ridge table 32 wide (99.5% of the ridge's variance against the 16-wide base's) | `table_d32` | **0.6753** at 10 ee, still rising (0.6743 at 6, 0.6748 at 8); ahead of the base at every epoch from the first; better than the base for 27 of 30 diseases | **helped (+0.0018), the largest gain of the night**: the extra channels carry ridge signal the 16 did not; its 20-epoch, batch-256 and per-disease-head follow-ups are running, and the 64-wide table is queued |
| follow-up: the 32-wide table with the doubled budget | `table_d32_20` | 0.6755 at 14.8 ee, stopped at 18.8 | +0.0020: the budget adds nothing to the 32-wide table's 10-epoch result (0.6753) |
| follow-up: the 32-wide table with the per-disease head and the doubled budget | `table_d32_pdh20` | 0.6750 at 17.0 ee | +0.0015: the per-disease head adds nothing to the 32-wide table |
| follow-up: the 32-wide table with the batch-256 optimiser settings and the doubled budget | `table_d32_opt256_20` | 0.6755 at 11.3 ee, stopped at 15.3 | +0.0020: level with the 32-wide table at batch 64 (0.6753) in a quarter of the optimiser steps; the width gain carries over to the faster recipe |
| planned table-width axis: the ridge table 64 wide (all of the ridge's 90 columns' variance but 0.1%) | `table_d64` | 0.6748 at 10.0 ee | +0.0013: level with the 32-wide table (0.6753) at twice its memory; the width axis saturates at 32 |
| planned second mixer: the published bi-Mamba block over the block tokens (six layers at width 64, matched to S3) | `mamba_s3` | 0.6736 at 10 ee | no change: the two mixers are level, as the search also found |
| planned table-width axis: the ridge table 4 wide (36.8% of the ridge's variance) | `table_d4` | 0.6729 at 10 ee | hurt a little (−0.0006): the 16 channels carry signal the 4 do not; the 32-wide table is running ahead of the base |
| combination: the batch-256 optimiser settings with the doubled budget (items 5 + 9) | `combo_opt256_20` | 0.6739 at 10 ee, early-stopped at 14 ee | +0.0004: the budget gain does not add to the optimiser gain at batch 256 |
| 6, secondary check: within-block bi-Mamba in genomic order on the half-rung lassosum2 table | `blockmamba_lasso_c107` | 0.6676 at 10.0 ee against its control 0.6677 | no change |
| 2, secondary check on the half-rung LDpred2-auto table | `blockattn_ldpred_c107` | 0.6696 at 10.0 ee against its frozen-table control `pgi_ldpred_frozen_c107` 0.6697 | no change |
| 2, secondary check on the fold-routed ridge table (full rung) | `blockattn_ridge` | 0.6640 at 0.75 ee, then 0.655 to 0.660 until patience stopped it at 4.75 ee | hurt (−0.0095): with the raw dosages of 13,792 selected SNPs and 458,000 parameters of its own, the within-block module memorises the training rows; the base, which sees only the pooled vectors, does not |
| planned size axis: S1 mixer (width 16, one layer, head 64) | `size_s1` | 0.6727 at 10.0 ee | −0.0008: the smallest mixer trails the base a little; S2, S5 and the Mamba mixer pending |
| planned size axis: S2 mixer (width 32, two layers, head 128) | `size_s2` | 0.6738 at 10.0 ee | +0.0003: level with the base; with the paper search's S2-like winner (0.6741) the size axis is flat from S1 to S3 and falls at S4 and S5 |
| planned size axis: S5 mixer (width 256, eight layers, 7.0 M non-embedding parameters) | `size_s5` | 0.6700 at 1.25 ee, then falling until patience stopped it at 5.3 ee | hurt (−0.0035): the largest mixer memorises the frozen table's pooled vectors; on this first layer the mixer should be small |
| own: ensemble of the base's three seeds (mean of their logits) | `yardsticks/ensemble_seeds.json` | 0.6746 against the members' 0.6735, 0.6737, 0.6738 | +0.0009 over the members' mean: seed-averaging alone buys about as much as any single tweak, at three times the training |
| combination: the paper search's settings with the doubled budget (items 1 + 9) | `combo_paper20` | 0.6745 at 15.5 ee, stopped at 19.5 | +0.0010: the two gains add up to about +0.001 |
| combination of every positive on the 16-wide table: the paper search's settings, the doubled budget, the per-disease head and the entropy loss (items 1 + 9 + 7 + 8) | `combo_all20` | 0.6745 at 15.5 ee | +0.0010: no more than the search settings with the budget alone (0.6745); the small gains do not stack beyond +0.001 |
| combination: the paper search's settings, the doubled budget and the entropy-normalised loss (items 1 + 9 + 8) | `combo_paper20_labelnorm` | 0.6744 at 10.3 ee, stopped at 14.3 | +0.0009: the normalised loss adds nothing to the search's settings |
| 6, secondary check on the fold-routed ridge table | `blockmamba_ridge` | 0.6655 at 0.75 ee, then 0.659 to 0.662 until patience stopped it at 4.8 ee | hurt (-0.0080): as the within-block attention, the Mamba module memorises the raw dosages it is given |
| own: item 2 with 64 SNPs per block and two layers, ridge table | `blockattn_ridge_k64` | 0.6658 at 0.75 ee, then 0.654 to 0.657 | hurt (-0.0077): more SNPs and depth memorise faster |
| confirmation: seeds 2 and 3 of the chosen configuration (`table_d32_opt256_20`) | `table_d32_opt256_20_s2`, `_s3` | 0.6747 (best at 12.8 ee), 0.6749 (at 16.5 ee) | three-seed mean 0.6750 against the base's 0.6737: +0.0014, the gain holds |
| 8, loss normalised by each disease's entropy | `labelnorm_shuf` (base sampler) | 0.6737 at 10 ee | +0.0002, within the seed spread: no change; the ten rarest diseases did not move either |
| 8 on the rotation | `labelnorm_rot` | 0.6718 against `rot10` 0.6722 | no change |
| 7, one small head per disease (width 32) | `perdisease_shuf` (base sampler) | 0.6739 at 10 ee, still rising at the end (0.6731 at 9 ee) | +0.0004, at the edge of the seed spread; kept for the combination runs |
| 7 on the rotation | `perdisease_rot` | 0.6718 against `rot10` 0.6722 | no change |
| own: attention readout (one query per disease) on the rotation | `atten_rot` | 0.6724 against `rot10` 0.6722 | no change (as it was against the mean readout on 18 November) |
| planned size axis, S4 mixer (width 128, six layers) on the rotation | `s4_rot` | 0.6722 against `rot10` 0.6722; falls from 6 ee (0.6717 to 0.6703 at 10) | no change; the larger mixer overfits the frozen table sooner |
| own: covariate skip (logistic start), base sampler | `covskip_shuf` | 0.6732 at 8.5 ee; 0.6707 after one epoch-equivalent against the base's 0.6664 | no change at the end (a faster start only) |
| own: covariate skip (logistic start) | `covskip_rot` | 0.6722 against `rot10` 0.6722; starts at 0.6697 after one epoch-equivalent instead of 0.6638 | no change at the end; a faster start only; base-sampler run pending |

## The table (every run; built by `tools/summarise.py` from the run folders at 06:20 UTC)

Tuning AUROC, mean of 30 diseases (one disease for the E11 classic runs), best validation pass. 'vs reference' is the base (0.6735) for runs on the ridge table at the full rung, the run's own frozen-table control for a run on a PGI table, and the covariates alone for a one-disease run; 'over its PGI' is the lift over the naive score the table was built from (the PGI arm's figure for the full-rung tables; the training-row logistic fit for the half-rung tables).

| run | what | tuning AUROC | at ee | vs reference | reference | re-scored | I10 | E11 | over covariates | over its PGI | min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| atten_rot | Own idea: the attention readout (one query per disease over the blocks) with balanced batches; it equalled the mean read | 0.6724 | 6.5 | -0.0011 (hurt) | the base 0.6735 | 0.6724 | 0.7135 | 0.6890 | +0.0096 |  | 44 |
| blockattn_lasso | Item 2 as proposed: a transformer layer within each block over its 32 strongest lassosum2 rows (self-attention over the  | 0.6458 (stopped early) | 0.5 | +0.0007 (helped) | its frozen-table control pgi_lasso_frozen | 0.6458 | 0.6935 | 0.6645 | -0.0169 | -0.0371 | 46 |
| blockattn_lasso_c107 | Item 2 on the lassosum2 table with weights from the other half of the rung: a transformer layer within each block over i | 0.6676 | 10.0 | -0.0001 (no change) | its frozen-table control pgi_lasso_frozen_c107 | 0.6676 | 0.7065 | 0.6808 | +0.0052 | -0.0061 | 55 |
| blockattn_ldpred | Item 2 on the full-rung LDpred2-auto table. | 0.6558 (stopped early) | 0.5 | +0.0011 (helped) | its frozen-table control pgi_ldpred_frozen | 0.6558 | 0.7003 | 0.6712 | -0.0070 | -0.0315 | 48 |
| blockattn_ldpred_c107 | Item 2 on the LDpred2-auto table with weights from the other half of the rung. | 0.6696 | 10.0 | -0.0001 (no change) | its frozen-table control pgi_ldpred_frozen_c107 | 0.6696 | 0.7092 | 0.6847 | +0.0072 | -0.0075 | 55 |
| blockattn_ridge | Item 2 on the fold-routed ridge table, rerun from the fixed code. | 0.6640 (stopped early) | 0.8 | -0.0095 (hurt) | the base 0.6735 | 0.6640 | 0.7074 | 0.6795 | +0.0013 |  | 65 |
| blockattn_ridge_k64 | Own idea: item 2 on the ridge table with 64 SNPs per block and two layers. | 0.6658 (stopped early) | 0.8 | -0.0077 (hurt) | the base 0.6735 | 0.6658 | 0.7073 | 0.6784 | +0.0031 |  | 135 |
| blockmamba_lasso | Item 6 as proposed: the published bi-Mamba block within each block, reading its 32 strongest SNPs in genomic order with  | 0.6455 (stopped early) | 0.3 | +0.0004 (no change) | its frozen-table control pgi_lasso_frozen | 0.6455 | 0.6938 | 0.6680 | -0.0173 | -0.0374 | 59 |
| blockmamba_lasso_c107 | Item 6 on the lassosum2 table with weights from the other half of the rung, rerun from the fixed code (the first attempt | 0.6676 | 10.0 | -0.0001 (no change) | its frozen-table control pgi_lasso_frozen_c107 | 0.6676 | 0.7064 | 0.6809 | +0.0052 | -0.0061 | 110 |
| blockmamba_ridge | Item 6 on the fold-routed ridge table, from the fixed code (the first attempt started from the old copy and was stopped) | 0.6655 (stopped early) | 0.8 | -0.0080 (hurt) | the base 0.6735 | 0.6655 | 0.7073 | 0.6755 | +0.0028 |  | 131 |
| budget20 | Item 9: the training budget doubled (20 epoch-equivalents), otherwise the base configuration. | 0.6741 (stopped early) | 15.5 | +0.0006 (helped) | the base 0.6735 | 0.6741 | 0.7158 | 0.6923 | +0.0113 |  | 60 |
| budget20_cosine | Own idea: the doubled budget with a cosine schedule to zero (200 warm-up steps) instead of the plateau rule, which rarel | 0.6741 (stopped early) | 14.8 | +0.0006 (helped) | the base 0.6735 | 0.6741 | 0.7161 | 0.6920 | +0.0114 |  | 62 |
| budget50 | Item 0: a 50 epoch-equivalent budget (early stopping and the plateau schedule still act), otherwise the base configurati | 0.6740 (stopped early) | 14.8 | +0.0005 (helped) | the base 0.6735 | 0.6740 | 0.7158 | 0.6917 | +0.0113 |  | 56 |
| classic_e11_r01_balanced | Item 4, the control run: the published network for E11 on the comparator's clumping (r2 0.1, 250 kb), balanced batches,  | 0.6659 (stopped early) | 1.0 | -0.0038 (hurt) | covariates alone | 0.6659 |  | 0.6659 | -0.0038 |  | 51 |
| classic_e11_r07_balanced | Item 4: the published DeepGP network for E11 on the paper's clumping (r2 0.7, 500 kb), published settings (d 32, one bi- | 0.6667 (stopped early) | 0.8 | -0.0030 (hurt) | covariates alone | 0.6667 |  | 0.6667 | -0.0030 |  | 60 |
| combo_all20 | Combination of everything that came out at or above the base: the paper search's chosen settings, the doubled budget, th | 0.6745 (stopped early) | 15.5 | +0.0010 (helped) | the base 0.6735 | 0.6745 | 0.7144 | 0.6922 | +0.0118 |  | 105 |
| combo_opt256_20 | Combination: the batch-256 optimiser search's chosen settings with the doubled budget (items 5 + 9), base sampler. | 0.6739 (stopped early) | 10.0 | +0.0004 (no change) | the base 0.6735 | 0.6739 | 0.7145 | 0.6905 | +0.0111 |  | 43 |
| combo_paper20 | Combination: the paper search's chosen settings with the doubled budget (items 1 + 9), base sampler. | 0.6745 (stopped early) | 15.5 | +0.0010 (helped) | the base 0.6735 | 0.6745 | 0.7155 | 0.6929 | +0.0118 |  | 113 |
| combo_paper20_labelnorm |  | 0.6744 (stopped early) | 10.3 | +0.0009 (helped) | the base 0.6735 | 0.6744 | 0.7137 | 0.6913 | +0.0116 |  | 79 |
| covskip_rot | Own idea: the covariates straight to the logits, starting at the covariates-only logistic model (the head starts at zero | 0.6722 | 8.0 | -0.0013 (hurt) | the base 0.6735 | 0.6722 | 0.7130 | 0.6898 | +0.0094 |  | 32 |
| covskip_shuf | Own idea (covariate skip, logistic start) under the base sampler. | 0.6732 | 8.5 | -0.0003 (no change) | the base 0.6735 | 0.6732 | 0.7148 | 0.6899 | +0.0105 |  | 28 |
| labelnorm_rot | Item 8: each disease's loss divided by the entropy of its prevalence, on top of the balanced batches (item 3). | 0.6718 | 8.5 | -0.0017 (hurt) | the base 0.6735 | 0.6718 | 0.7117 | 0.6882 | +0.0091 |  | 38 |
| labelnorm_shuf | Item 8 under the natural-prevalence batches of the base configuration. | 0.6737 | 10.0 | +0.0002 (no change) | the base 0.6735 | 0.6737 | 0.7137 | 0.6914 | +0.0109 |  | 35 |
| mamba_s3 | Planned second mixer: the published bi-Mamba block over the block tokens, six layers at d 64 (matched to S3's size). | 0.6736 | 10.0 | +0.0001 (no change) | the base 0.6735 | 0.6736 | 0.7160 | 0.6950 | +0.0109 |  | 58 |
| perdisease_rot | Item 7: one small head (width 32) per disease on the shared hidden layer, on top of the balanced batches. | 0.6718 | 8.5 | -0.0017 (hurt) | the base 0.6735 | 0.6718 | 0.7129 | 0.6892 | +0.0090 |  | 33 |
| perdisease_shuf | Item 7 under the base sampler (the rotation turned out worse than the base, so each item is also read on the base recipe | 0.6739 | 10.0 | +0.0004 (no change) | the base 0.6735 | 0.6739 | 0.7151 | 0.6916 | +0.0111 |  | 33 |
| pgi_lasso_frozen | The proposed first layer as specified: the frozen lassosum2 table (30 channels, the full rung's weights, which were fitt | 0.6451 (stopped early) | 0.3 |  | it is the control | 0.6451 | 0.6926 | 0.6688 | -0.0176 | -0.0378 | 20 |
| pgi_lasso_frozen_c107 | Baseline for items 2 and 6 on the lassosum2 table with weights from the other half of the rung (weights from the 107,743 | 0.6677 | 9.3 |  | it is the control | 0.6677 | 0.7070 | 0.6810 | +0.0053 | -0.0060 | 17 |
| pgi_ldpred_frozen | The frozen LDpred2-auto table (full-rung weights) under the base recipe: the control for item 2 on LDpred2. | 0.6546 (stopped early) | 0.5 |  | it is the control | 0.6546 | 0.7024 | 0.6697 | -0.0081 | -0.0327 | 32 |
| pgi_ldpred_frozen_c107 | Baseline on the LDpred2-auto table with weights from the other half of the rung. | 0.6697 | 10.0 |  | it is the control | 0.6697 | 0.7098 | 0.6872 | +0.0073 | -0.0075 | 20 |
| rot10 | Item 3: disease-balanced batches (the rotation sampler: each batch half cases of one disease, half its non-cases, the di | 0.6722 | 6.5 | -0.0013 (hurt) | the base 0.6735 | 0.6722 | 0.7134 | 0.6880 | +0.0094 |  | 27 |
| s4_rot | Own idea: a bigger mixer (S4: d 128, 6 layers, 8 heads) on the frozen table, cheap now that the first layer is cached. | 0.6722 | 8.5 | -0.0013 (hurt) | the base 0.6735 | 0.6722 | 0.7120 | 0.6868 | +0.0095 |  | 48 |
| search_opt256 |  | 0.6740 | 10.0 | +0.0005 (helped) | the base 0.6735 |  |  |  |  |  | 31 |
| | chosen settings: {"lr": 0.00037906794412144944, "l2": 0.0006801289958688564, "dropout": 0.11636189556517773, "lr_schedule": "constant", "warmup_steps": 200} | | | | | | | | | | |
| search_paper |  | 0.6741 | 8.5 | +0.0006 (helped) | the base 0.6735 |  |  |  |  |  | 51 |
| | chosen settings: {"lr": 7.44714490408307e-05, "l2": 1.1268145953419502e-05, "dropout": 0.18741561946967727, "batch_size": 32, "e_layers": 2, "d_model": 32, "n_heads": 4, "n_fcn": 2, "mixer": "transformer"} | | | | | | | | | | |
| size_s1 | Planned size axis at the top rung on the frozen table: S1 (d 16, one layer; head 64). | 0.6727 | 10.0 | -0.0008 (hurt) | the base 0.6735 | 0.6727 | 0.7120 | 0.6886 | +0.0099 |  | 44 |
| size_s2 | Planned size axis: S2 (d 32, two layers; head 128). | 0.6738 | 10.0 | +0.0003 (no change) | the base 0.6735 | 0.6738 | 0.7143 | 0.6905 | +0.0111 |  | 46 |
| size_s5 | Planned size axis: S5 (d 256, eight layers, 8 heads; head 1024). | 0.6700 (stopped early) | 1.3 | -0.0035 (hurt) | the base 0.6735 | 0.6700 | 0.7107 | 0.6848 | +0.0072 |  | 36 |
| table_d32 | Planned table-width axis: the frozen fold-routed ridge table 32 wide (the 16-wide one is the base). | 0.6753 | 10.0 | +0.0018 (helped) | the base 0.6735 | 0.6753 | 0.7176 | 0.6941 | +0.0126 |  | 92 |
| table_d32_20 | Follow-up: the 32-wide ridge table (ahead of the base at matched epochs in table_d32) with the doubled budget. | 0.6755 (stopped early) | 14.8 | +0.0020 (helped) | the base 0.6735 | 0.6755 | 0.7188 | 0.6966 | +0.0128 |  | 110 |
| table_d32_opt256_20 | Follow-up: the 32-wide ridge table with the optimiser search's chosen settings at batch 256 and the doubled budget. | 0.6755 (stopped early) | 11.3 | +0.0020 (helped) | the base 0.6735 | 0.6755 | 0.7177 | 0.6954 | +0.0128 |  | 87 |
| table_d32_opt256_20_s2 |  | 0.6747 (stopped early) | 12.8 | +0.0012 (helped) | the base 0.6735 | 0.6747 | 0.7181 | 0.6946 | +0.0119 |  | 37 |
| table_d32_opt256_20_s3 |  | 0.6749 | 16.5 | +0.0014 (helped) | the base 0.6735 |  |  |  |  |  | 40 |
| table_d32_pdh20 | Follow-up: the 32-wide ridge table (the night's best so far) with the per-disease head and the doubled budget. | 0.6750 | 17.0 | +0.0015 (helped) | the base 0.6735 | 0.6750 | 0.7184 | 0.6950 | +0.0122 |  | 96 |
| table_d4 | Planned table-width axis: the frozen ridge table 4 wide. | 0.6729 | 10.0 | -0.0006 (hurt) | the base 0.6735 | 0.6729 | 0.7147 | 0.6887 | +0.0102 |  | 49 |
| table_d64 | Planned table-width axis: the frozen ridge table 64 wide (its cache holds 26 GB in memory, so one such run at a time). | 0.6748 | 10.0 | +0.0013 (helped) | the base 0.6735 | 0.6748 | 0.7173 | 0.6937 | +0.0121 |  | 81 |


Tuning AUROC (mean of 30) by epoch-equivalent, every validation rounded to the nearest quarter; base run: 1: 0.6664, 2: 0.6709, 3: 0.6714, 4: 0.6721, 5: 0.6724, 6: 0.6731, 7: 0.6730, 8: 0.6729, 9: 0.6733, 10: 0.6735

| run | 0.25 | 0.5 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 15 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| atten_rot | 0.6542 | 0.6584 | 0.6630 | 0.6698 | 0.6708 | 0.6714 | 0.6711 | 0.6715 | 0.6714 | 0.6719 | 0.6713 | 0.6723 |  |  |  |  |  |  |
| blockattn_lasso | 0.6453 | 0.6458 | 0.6398 | 0.6281 | 0.6283 | 0.6268 |  |  |  |  |  |  |  |  |  |  |  |  |
| blockattn_lasso_c107 | 0.6471 | 0.6567 | 0.6593 | 0.6609 | 0.6618 | 0.6635 | 0.6654 | 0.6656 | 0.6663 | 0.6674 | 0.6674 | 0.6676 |  |  |  |  |  |  |
| blockattn_ldpred | 0.6521 | 0.6558 | 0.6539 | 0.6478 | 0.6475 | 0.6402 | 0.6395 |  |  |  |  |  |  |  |  |  |  |  |
| blockattn_ldpred_c107 | 0.6472 | 0.6567 | 0.6593 | 0.6610 | 0.6652 | 0.6678 | 0.6680 | 0.6675 | 0.6685 | 0.6694 | 0.6694 | 0.6696 |  |  |  |  |  |  |
| blockattn_ridge | 0.6528 | 0.6587 | 0.6589 | 0.6547 | 0.6599 | 0.6572 |  |  |  |  |  |  |  |  |  |  |  |  |
| blockattn_ridge_k64 | 0.6538 | 0.6588 | 0.6594 | 0.6535 | 0.6562 | 0.6549 |  |  |  |  |  |  |  |  |  |  |  |  |
| blockmamba_lasso | 0.6455 | 0.6442 | 0.6403 | 0.6272 | 0.6251 | 0.6219 |  |  |  |  |  |  |  |  |  |  |  |  |
| blockmamba_lasso_c107 | 0.6464 | 0.6567 | 0.6594 | 0.6610 | 0.6620 | 0.6635 | 0.6657 | 0.6658 | 0.6661 | 0.6672 | 0.6674 | 0.6676 |  |  |  |  |  |  |
| blockmamba_ridge | 0.6528 | 0.6588 | 0.6642 | 0.6596 | 0.6619 | 0.6591 |  |  |  |  |  |  |  |  |  |  |  |  |
| budget20 | 0.6526 | 0.6587 | 0.6664 | 0.6709 | 0.6714 | 0.6722 | 0.6723 | 0.6732 | 0.6726 | 0.6730 | 0.6733 | 0.6735 | 0.6733 | 0.6738 |  |  |  |  |
| budget20_cosine | 0.6499 | 0.6583 | 0.6660 | 0.6707 | 0.6713 | 0.6721 | 0.6722 | 0.6731 | 0.6725 | 0.6731 | 0.6735 | 0.6738 | 0.6736 | 0.6740 |  |  |  |  |
| budget50 | 0.6526 | 0.6587 | 0.6664 | 0.6709 | 0.6714 | 0.6721 | 0.6723 | 0.6732 | 0.6726 | 0.6731 | 0.6733 | 0.6734 | 0.6734 | 0.6738 |  |  |  |  |
| classic_e11_r01_balanced | 0.6641 | 0.6655 | 0.6659 | 0.6482 | 0.6273 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| classic_e11_r07_balanced | 0.6652 | 0.6664 | 0.6666 | 0.6597 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| combo_all20 | 0.6576 | 0.6603 | 0.6667 | 0.6717 | 0.6721 | 0.6721 | 0.6727 | 0.6727 | 0.6721 | 0.6730 | 0.6738 | 0.6739 | 0.6742 | 0.6742 |  |  |  |  |
| combo_opt256_20 | 0.6415 | 0.6577 | 0.6609 | 0.6708 | 0.6713 | 0.6721 | 0.6724 | 0.6730 | 0.6726 | 0.6727 | 0.6736 | 0.6739 | 0.6733 |  |  |  |  |  |
| combo_paper20 | 0.6522 | 0.6594 | 0.6679 | 0.6712 | 0.6715 | 0.6722 | 0.6726 | 0.6731 | 0.6726 | 0.6735 | 0.6739 | 0.6739 | 0.6741 | 0.6742 |  |  |  |  |
| combo_paper20_labelnorm | 0.6536 | 0.6595 | 0.6669 | 0.6713 | 0.6717 | 0.6722 | 0.6725 | 0.6729 | 0.6726 | 0.6735 | 0.6738 | 0.6741 | 0.6741 |  |  |  |  |  |
| covskip_rot | 0.6620 | 0.6623 | 0.6697 | 0.6712 | 0.6711 | 0.6716 | 0.6714 | 0.6710 | 0.6716 | 0.6722 | 0.6715 | 0.6718 |  |  |  |  |  |  |
| covskip_shuf | 0.6626 | 0.6630 | 0.6707 | 0.6719 | 0.6719 | 0.6723 | 0.6723 | 0.6721 | 0.6728 | 0.6731 | 0.6729 | 0.6731 |  |  |  |  |  |  |
| labelnorm_rot | 0.6551 | 0.6587 | 0.6611 | 0.6696 | 0.6705 | 0.6710 | 0.6707 | 0.6712 | 0.6711 | 0.6716 | 0.6707 | 0.6711 |  |  |  |  |  |  |
| labelnorm_shuf | 0.6541 | 0.6589 | 0.6643 | 0.6707 | 0.6714 | 0.6718 | 0.6720 | 0.6727 | 0.6724 | 0.6729 | 0.6733 | 0.6737 |  |  |  |  |  |  |
| mamba_s3 | 0.6557 | 0.6595 | 0.6640 | 0.6703 | 0.6713 | 0.6721 | 0.6721 | 0.6724 | 0.6718 | 0.6729 | 0.6732 | 0.6736 |  |  |  |  |  |  |
| perdisease_rot | 0.6552 | 0.6591 | 0.6615 | 0.6701 | 0.6708 | 0.6712 | 0.6708 | 0.6709 | 0.6711 | 0.6714 | 0.6711 | 0.6711 |  |  |  |  |  |  |
| perdisease_shuf | 0.6549 | 0.6594 | 0.6662 | 0.6711 | 0.6716 | 0.6722 | 0.6722 | 0.6729 | 0.6728 | 0.6730 | 0.6731 | 0.6739 |  |  |  |  |  |  |
| pgi_lasso_frozen | 0.6451 | 0.6417 | 0.6355 | 0.6239 | 0.6230 | 0.6218 |  |  |  |  |  |  |  |  |  |  |  |  |
| pgi_lasso_frozen_c107 | 0.6467 | 0.6568 | 0.6594 | 0.6609 | 0.6620 | 0.6639 | 0.6663 | 0.6660 | 0.6662 | 0.6673 | 0.6675 | 0.6676 |  |  |  |  |  |  |
| pgi_ldpred_frozen | 0.6530 | 0.6546 | 0.6497 | 0.6454 | 0.6431 | 0.6364 | 0.6354 |  |  |  |  |  |  |  |  |  |  |  |
| pgi_ldpred_frozen_c107 | 0.6463 | 0.6567 | 0.6594 | 0.6611 | 0.6669 | 0.6684 | 0.6685 | 0.6680 | 0.6691 | 0.6695 | 0.6696 | 0.6697 |  |  |  |  |  |  |
| rot10 | 0.6542 | 0.6585 | 0.6638 | 0.6699 | 0.6709 | 0.6712 | 0.6710 | 0.6716 | 0.6713 | 0.6719 | 0.6711 | 0.6721 |  |  |  |  |  |  |
| s4_rot | 0.6566 | 0.6594 | 0.6672 | 0.6703 | 0.6706 | 0.6713 | 0.6710 | 0.6717 | 0.6714 | 0.6707 | 0.6710 | 0.6703 |  |  |  |  |  |  |
| search_opt256 | 0.6415 | 0.6576 | 0.6611 | 0.6707 | 0.6714 | 0.6721 | 0.6725 | 0.6731 | 0.6722 | 0.6729 | 0.6737 | 0.6740 |  |  |  |  |  |  |
| search_paper | 0.6522 | 0.6594 | 0.6679 | 0.6713 | 0.6715 | 0.6722 | 0.6726 | 0.6731 | 0.6725 | 0.6735 | 0.6739 | 0.6739 |  |  |  |  |  |  |
| size_s1 | 0.6335 | 0.6450 | 0.6576 | 0.6643 | 0.6689 | 0.6700 | 0.6705 | 0.6710 | 0.6711 | 0.6719 | 0.6721 | 0.6727 |  |  |  |  |  |  |
| size_s2 | 0.6435 | 0.6563 | 0.6605 | 0.6702 | 0.6708 | 0.6715 | 0.6718 | 0.6724 | 0.6717 | 0.6731 | 0.6735 | 0.6738 |  |  |  |  |  |  |
| size_s5 | 0.6569 | 0.6602 | 0.6686 | 0.6636 | 0.6643 | 0.6641 | 0.6662 |  |  |  |  |  |  |  |  |  |  |  |
| table_d32 | 0.6541 | 0.6593 | 0.6671 | 0.6717 | 0.6723 | 0.6733 | 0.6737 | 0.6743 | 0.6739 | 0.6748 | 0.6745 | 0.6753 |  |  |  |  |  |  |
| table_d32_20 | 0.6542 | 0.6593 | 0.6672 | 0.6717 | 0.6723 | 0.6732 | 0.6736 | 0.6742 | 0.6737 | 0.6744 | 0.6746 | 0.6752 | 0.6750 | 0.6753 |  |  |  |  |
| table_d32_opt256_20 | 0.6426 | 0.6590 | 0.6613 | 0.6720 | 0.6726 | 0.6732 | 0.6731 | 0.6736 | 0.6731 | 0.6740 | 0.6745 | 0.6753 | 0.6751 | 0.6748 |  |  |  |  |
| table_d32_opt256_20_s2 | 0.6454 | 0.6585 | 0.6658 | 0.6711 | 0.6708 | 0.6720 | 0.6725 | 0.6733 | 0.6734 | 0.6730 | 0.6735 | 0.6742 | 0.6741 | 0.6739 |  |  |  |  |
| table_d32_opt256_20_s3 | 0.6432 | 0.6585 | 0.6650 | 0.6714 | 0.6712 | 0.6720 | 0.6722 | 0.6731 | 0.6732 | 0.6736 | 0.6739 | 0.6739 | 0.6740 | 0.6744 | 0.6742 |  |  |  |
| table_d32_pdh20 | 0.6563 | 0.6599 | 0.6686 | 0.6720 | 0.6723 | 0.6729 | 0.6727 | 0.6732 | 0.6731 | 0.6737 | 0.6740 | 0.6747 | 0.6747 | 0.6747 | 0.6747 |  |  |  |
| table_d4 | 0.6522 | 0.6603 | 0.6679 | 0.6704 | 0.6711 | 0.6712 | 0.6716 | 0.6720 | 0.6723 | 0.6725 | 0.6720 | 0.6729 |  |  |  |  |  |  |
| table_d64 | 0.6548 | 0.6597 | 0.6650 | 0.6722 | 0.6726 | 0.6733 | 0.6737 | 0.6742 | 0.6743 | 0.6747 | 0.6746 | 0.6748 |  |  |  |  |  |  |
