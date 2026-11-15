# Results: checks of the linear PRS path and the search inside every run (WideDeepGP gates)

*12–15 November 2026. Methods in `methods.md`. All numbers are tuning-set AUROCs on the pilot's
proxy: 157,279 SNPs, 40,000 training and 10,000 tuning participants. Tables are in `results/`.*

## Summary

- **G1 passes.**
  - On five simulated traits, the linear stage reaches a mean tuning AUROC of 0.7187.
  - GBLUP given each trait's true heritability reaches 0.7132.
  - Four traits are 0.007 to 0.010 above GBLUP. The trait with 20 causal SNPs is 0.0055 below.
  - With one prior variance for all traits at the end of training, the mean is 0.7185, so on
    these traits choosing per trait adds next to nothing.
  - The pilot's network alone reached 0.51.
- **G2 passes.**
  - With the network on top, the best of 12 trials reaches 0.71869, against the linear stage's
    0.71873.
  - Under the rule of that day the linear model was kept. Under the rule decided on 15 November
    (DeepWideGP 5396f44), the network on top is always the run's model, here at 0.71869.
  - On purely additive traits the network has nothing to add, and here it took nothing away.
- **G3, linear half, passes.**
  - On the 30 diseases, the linear model at one prior variance (0.1) reaches 0.6616 against
    0.6607 for the covariates alone (+0.0009). 24 of the 30 diseases are above their
    covariates-only AUROC.
  - With variance and pass chosen per disease on the tuning set: 0.6641, which is optimistic.
  - The pilot's network alone: 0.6593 (−0.0014). The pilot's clumped polygenic score: +0.0014.
  - This eighth of the genome holds little of these diseases' signal, so small gains are
    expected.
- **G3, network half: not run.** Its first attempt was stopped (see `methods.md`, "Incidents").

## Simulated traits (`results/simulated_traits.csv`)

| Trait | Causal SNPs | h² | Ceiling | GBLUP (true h²) | Clumped PRS | Pilot network | Linear, τ 1 | Linear, τ per trait (chosen) | Network on top |
|---|---|---|---|---|---|---|---|---|---|
| sim20 | 20 | 0.3 | 0.783 | 0.671 | 0.751 | 0.509 | 0.668 | 0.666 (3) | 0.665 |
| sim200 | 200 | 0.5 | 0.851 | 0.722 | 0.804 | 0.508 | 0.729 | 0.731 (1) | 0.731 |
| sim2000 | 2,000 | 0.5 | 0.853 | 0.717 | 0.710 | 0.519 | 0.720 | 0.724 (3) | 0.725 |
| sim20000 | 20,000 | 0.5 | 0.855 | 0.724 | 0.629 | 0.517 | 0.727 | 0.730 (3) | 0.730 |
| sim200a | 200, enriched | 0.5 | 0.850 | 0.733 | 0.814 | 0.511 | 0.738 | 0.742 (3) | 0.742 |
| **Mean** | | | 0.839 | 0.713 | 0.742 | 0.513 | 0.717 | 0.719 | 0.719 |

- **The prior-variance grid.** At the end of training the mean AUROC across traits, for each
  prior variance, was:

  | τ | 0.03 | 0.1 | 0.3 | 1 | 3 | 10 | 30 |
  |---|---|---|---|---|---|---|---|
  | Mean AUROC | 0.633 | 0.680 | 0.704 | 0.717 | 0.718 | 0.712 | 0.702 |

  Every trait's best lies inside the grid.
- **Training time.** The linear stage trained for 318 s of a 43-minute run on 8 CPU threads.
  The rest is reading genotypes and scoring the tuning set.
- **The network trials.** The 12 trials of the network on top used 2.4 GPU-hours of training on
  four GPUs. Every trial ended between 0.7180 and 0.7187; 8 were pruned
  (`results/simulated_joint_trials.csv`).

## Real labels: the linear stage (`results/real_diseases_linear.csv`)

Seven-value grid (`real_linear_grid`). For each disease the table gives:
- the covariates-only AUROC;
- the linear model's AUROC at its chosen prior variance and pass;
- its AUROC at τ 0.1 at the end of training.

| | Mean of 30 diseases |
|---|---|
| Covariates alone | 0.6607 |
| Linear, τ 0.1 for all, end of training | 0.6616 (+0.0009; 24 of 30 above) |
| Linear, τ and pass per disease (optimistic) | 0.6641 (+0.0034) |

- **The grid's lower edge.** The chosen prior variances are small: 0.03 to 1, most often 0.1.
  Six diseases chose the lowest value of the grid. The grid was therefore extended to 0.003
  (0edb9cb), so that a disease whose SNPs give the model nothing usable can fall back to its
  covariates.
- **The rerun with that grid.** Its per-disease choices are not valid (see `methods.md`,
  "Incidents"). Its pass-by-pass metrics are valid, because the partial pass was never recorded
  in them, and they show the same pattern. At the end of training the mean AUROC was 0.6603 to
  0.6616 at τ ≤ 0.3 (0.6607 at τ 0.003, the covariates alone in effect), and lower above that.

## Not yet examined

- The network on top on the real labels.
- Any training size other than the proxy's 40,000.
- The full set of SNPs.
- The test set.
