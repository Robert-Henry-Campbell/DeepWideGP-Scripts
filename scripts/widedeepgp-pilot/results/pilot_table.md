| Arm | Best mean tuning AUROC, by seed | Mean | Cross-entropy | Genotype-dependent part, by seed (95% interval) | Minutes per epoch-equivalent | Peak GPU memory (GiB) |
|---|---|---|---|---|---|---|
| Default | 0.6598 | 0.6598 | 0.2594 | +0.0001 (-0.0001 to +0.0003) | 6.5 | 4.4 |
| Only the target learns | 0.6585 | 0.6585 | 0.2595 | -0.0001 (-0.0004 to +0.0003) | 6.4 | 4.4 |
| Shuffled batches | 0.6601 | 0.6601 | 0.2590 | +0.0000 (-0.0003 to +0.0003) | 6.6 | 4.4 |
| No sampling-rate correction | 0.6573 | 0.6573 | 0.2731 | +0.0003 (-0.0001 to +0.0007) | 6.4 | 4.4 |
| Full class balance | 0.6595 | 0.6595 | 0.2597 | -0.0000 (-0.0002 to +0.0001) | 6.5 | 4.4 |
| bf16 encoders | 0.6598 | 0.6598 | 0.2593 | +0.0001 (-0.0001 to +0.0004) | 5.3 | 2.7 |
| Annotation vectors | 0.6596 | 0.6596 | 0.2594 | +0.0001 (-0.0002 to +0.0004) | 6.7 | 4.8 |
| Mamba-3 block | 0.6593 | 0.6593 | 0.2598 | +0.0000 (-0.0001 to +0.0001) | 10.6 | 9.5 |
| GWAS statistics as annotation | 0.6601 | 0.6601 | 0.2594 | +0.0004 (+0.0001 to +0.0007) | 6.5 | 4.7 |
| *Covariates only (logistic model)* | | 0.6607 | 0.2586 | | | |
