**Figure 4. The part of each model's tuning AUROC that depends on the genotypes.** For every run
the best checkpoint was scored on the 10,000 tuning participants as they are and again after
giving each participant another participant's genotypes (three permutations), keeping
covariates and labels in place. Dots show the resulting loss of AUROC, averaged over the 30
diseases, in thousandths of an AUROC unit, for each seed of an arm; bars are 95% bootstrap
intervals over participants. A model that takes nothing from the SNPs sits at zero.
The dashed line is what a clumped polygenic score built from 165,872 other participants adds to
the covariates on the same tuning participants (1.4 thousandths).
Data: `fig4_genetic_data.csv`.
