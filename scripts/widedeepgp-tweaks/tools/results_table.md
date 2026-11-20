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
