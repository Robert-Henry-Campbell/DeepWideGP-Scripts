# Results: the final WideDeepGP on the five-rung ladder (22 November 2026)

*Aggregates only. The tuning set (20,000 participants), mean AUROC over the 30 diseases. WideDeepGP is the mean ± SD of
seeds 1–3. Methods: `methods.md`. Every number: `ladder_summary.csv`. The test set has not been scored.*

| Training participants | WideDeepGP, seeds 1 / 2 / 3 | Mean ± SD | Covariates alone | WideDeepGP − covariates | Ridge PRS + covariates | WideDeepGP − ridge PRS + cov. | Logistic ridge twin | WideDeepGP − twin |
|---|---|---|---|---|---|---|---|---|
| 13,199 | 0.6587 / 0.6589 / 0.6584 | 0.6587 ± 0.0002 | 0.6598 | −0.0011 | 0.6611 | −0.0024 | 0.6603 | −0.0016 |
| 26,287 | 0.6602 / 0.6608 / 0.6600 | 0.6604 ± 0.0004 | 0.6613 | −0.0010 | 0.6638 | −0.0034 | 0.6619 | −0.0015 |
| 53,498 | 0.6645 / 0.6655 / 0.6645 | 0.6648 ± 0.0006 | 0.6620 | +0.0028 | 0.6670 | −0.0022 | 0.6633 | +0.0015 |
| 107,743 | 0.6690 / 0.6690 / 0.6684 | 0.6688 ± 0.0003 | 0.6625 | +0.0063 | 0.6718 | −0.0030 | 0.6671 | +0.0017 |
| 215,872 | 0.6749 / 0.6744 / 0.6737 | 0.6743 ± 0.0006 | 0.6627 | +0.0116 | 0.6777 | −0.0034 | 0.6732 | +0.0011 |

**Reading.**
- **Against the covariates alone.** WideDeepGP beats them from 53,498 training participants up. The margin is 0.003 at
  53,498, 0.006 at 107,743 and 0.012 at 215,872, far larger than the seed spread (SD 0.0003 to 0.0006). At the two
  smallest rungs it is 0.001 below them in all three seeds.
- **Against the plain cross-fitted ridge PRS with the covariates.** It stays below at every rung, by 0.002 to 0.003.
  But its share of the PRS's gain over the covariates rises with training size: 56% at 53,498, 68% at 107,743 and 78%
  at 215,872.
- **Against the logistic ridge twin** on the same table's block channels, the linear model with the same inputs,
  WideDeepGP is ahead from 53,498 up and behind below.
- **The top rung reproduces.** Its seed 1 gives 0.6749, against 0.6748 for the run that chose the setting. That figure
  is optimistic, because the same tuning set chose the setting. The lower rungs' setting was fixed in advance.
- **What this table does not show.** Uncertainty from which people are in the evaluation set. That comes from the
  paired bootstrap of the test set, scored once for all models at the end (`PGI/methods.md` section 8).
