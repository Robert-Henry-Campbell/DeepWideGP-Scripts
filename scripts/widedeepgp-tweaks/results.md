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
| 8, loss normalised by each disease's entropy | `labelnorm_shuf` (base sampler) | 0.6737 at 10 ee | +0.0002, within the seed spread: no change; the ten rarest diseases did not move either |
| 8 on the rotation | `labelnorm_rot` | 0.6718 against `rot10` 0.6722 | no change |
| 7, one small head per disease (width 32) | `perdisease_shuf` (base sampler) | 0.6739 at 10 ee, still rising at the end (0.6731 at 9 ee) | +0.0004, at the edge of the seed spread; kept for the combination runs |
| 7 on the rotation | `perdisease_rot` | 0.6718 against `rot10` 0.6722 | no change |
| own: attention readout (one query per disease) on the rotation | `atten_rot` | 0.6724 against `rot10` 0.6722 | no change (as it was against the mean readout on 18 November) |
| planned size axis, S4 mixer (width 128, six layers) on the rotation | `s4_rot` | 0.6722 against `rot10` 0.6722; falls from 6 ee (0.6717 to 0.6703 at 10) | no change; the larger mixer overfits the frozen table sooner |
| own: covariate skip (logistic start), base sampler | `covskip_shuf` | 0.6732 at 8.5 ee; 0.6707 after one epoch-equivalent against the base's 0.6664 | no change at the end (a faster start only) |
| own: covariate skip (logistic start) | `covskip_rot` | 0.6722 against `rot10` 0.6722; starts at 0.6697 after one epoch-equivalent instead of 0.6638 | no change at the end; a faster start only; base-sampler run pending |
