**Figure 5. Positive control: traits simulated on the real genotypes.** Each panel is one
simulated trait (prevalence 20%; heritability of liability 0.3 for the trait with 20 causal SNPs
and 0.5 for the others; "enriched" means its causal SNPs were drawn 20 times more often from
coding or conserved SNPs). Dots show the tuning AUROC of the network at the end of training for
each version of its input (rows); training was 6 epoch-equivalents of the 40,000 training
participants, and 30 for the row "trained longer", whose mean AUROC was highest at 13.5
epoch-equivalents and lower after that. The dashed line is the better of two linear methods
fitted to the same training participants (a clumped polygenic score and GBLUP), and the solid
line is the AUROC of the true genetic value, which no method can exceed. An AUROC of 0.5 is
chance. Data: `fig5_control_data.csv`.
