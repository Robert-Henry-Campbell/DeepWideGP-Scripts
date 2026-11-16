# Results: the WideDeepGP ladder

*Tuning-set figures, written as each training size finishes. They are chosen on the tuning set,
so they are mildly optimistic. The comparison with the PGI arm and the classic DeepGP uses the
shared protocol on the test set after the joint unlock (`methods.md`). Tables: `results/`, made
by `summarise_rung.py`.*

## 13,199 training participants (median N_eff 3,372)

The run: 15 Nov 22:52 UTC to 16 Nov 15:33 UTC.
- **The ridge stage:** 32 minutes.
- **The network stage:** 12 trials, 7 complete and 5 pruned, 18.7 GPU-hours of training. Fixed
  trial shares kept one worker busy for hours after the others had finished (since fixed).

| Mean of 30 diseases (tuning AUROC) | |
|---|---|
| Covariates only (logistic regression, same training participants) | 0.6598 |
| Ridge alone (ablation; prior variance per disease) | 0.6634 |
| **WideDeepGP** (ridge + network; the run's model) | **0.6633** |

- **WideDeepGP against covariates only:** +0.0035. 29 of the 30 diseases are above their
  covariates-only AUROC; the exception is K40 (−0.0002).
- **The largest gains:** E66 (+0.014), K80 (+0.011), M23 (+0.010), I84+K64 (+0.010) and K21
  (+0.007).
- **The network:** adds nothing to the ridge at this size (−0.0001 on average; per disease
  −0.0004 to +0.0003).
- **The ridge's prior variances:** all chosen inside the grid. 16 diseases took 1, 8 took 0.3, 3
  took 0.1, 2 took 3 (E66, K80) and 1 took 0.03 (I20).
- **The chosen network trial:** learning rate 1.2 × 10⁻⁴, weight decay 3.7 × 10⁻⁴, dropout 0.33.

Per disease: `results/rung_13199.csv`.
