# Replicating DeepGP (Zhu et al., Briefings in Bioinformatics 2026): results

**Not replicated.** The paper reports DeepGP above its PRS baseline (type 2 diabetes, 1:1 test set: DeepGP 0.64, PRS +
covariates 0.60). With the published code and settings on our UK Biobank data, DeepGP at best ties the paper's own
PRS recipe. Two findings explain most of the paper's numbers:

1. **The network learns from SNPs only if each dosage counts the risk allele.** With the minor-allele dosages the paper
   describes, it learns nothing (0.500 with matched controls). With risk-allele dosages it reaches 0.615-0.652, the
   paper's range.
2. **The paper's PRS recipe handicaps the PRS.** Fitting the PRS's logistic step on the people whose GWAS made the PRS
   over-weights the PRS 5-12 times. That pulls PRS + covariates below covariates alone, while the same PRS joined on
   people outside the GWAS adds 0.01-0.03.

No single design of ours gives both the paper's DeepGP (0.64) and its PRS (0.60). Methods: `methods.md`.

## Table 1. Type 2 diabetes (E11), AUROC on test people outside every GWAS (95% interval)

| model | 1:1, matched on age and sex | 1:1, random controls | 1:1, random controls, GWAS of the cohort itself | population prevalence | paper, 1:1 |
|---|---|---|---|---|---|
| DeepGP, minor-allele dosages | 0.500 (0.480-0.518) | 0.625 (0.605-0.644) | | | 0.64 |
| DeepGP, risk-allele dosages | 0.615 (0.596-0.634) | 0.652 (0.633-0.673) | 0.648 (0.629-0.667) | | |
| PRS + covariates, paper's recipe | 0.618 (0.600-0.638) | 0.655 (0.636-0.675) | 0.659 (0.642-0.679) | 0.666 (0.653-0.680) | 0.60 |
| PRS + covariates, joined outside the GWAS | 0.613 (0.594-0.634) | 0.699 (0.681-0.717) | 0.690 (0.673-0.708) | 0.701 (0.688-0.715) | |
| PRS alone | 0.618 (0.599-0.637) | 0.602 (0.583-0.623) | 0.578 (0.560-0.599) | 0.615 (0.601-0.630) | |
| covariates alone | 0.519 (0.498-0.539) | 0.677 (0.659-0.695) | 0.677 (0.659-0.695) | 0.669 (0.656-0.682) | not reported |
| GWAS people / lead SNPs | 215,872 / 47,341 | 215,872 / 47,341 | 34,690 / 98,440 | 215,872 / 47,341 | not stated / ~100,000 |
| training / test people (cases) | 34,690 / 3,326 (17,345 / 1,663) | same | same | 215,872 / 20,000 (17,345 / 1,663) | 44,828 / 7,912 (22,414 / 3,956) |

The paper states minor-allele dosages; its controls are not described. Atrial fibrillation (I48), minor-allele
dosages, GWAS of 215,872 people: DeepGP 0.493 matched and 0.692 random, against covariates alone 0.510 and 0.743
(paper: DeepGP 0.74, PRS 0.70).

## Findings

1. **Minor-allele dosages: no learning** (Fig. 1a). On the matched cohort the training loss stayed at chance for all
   22 passes and every test person got nearly the same score. Atrial fibrillation behaved the same (0.493).
2. **Why: the network has no weight per SNP** (Fig. 1b). Every SNP on a chromosome passes through one shared filter,
   tagged only by a fixed code of its base-pair position, and each chromosome is averaged. Its linear route is one
   weight on summed dosages. The minor allele is the risk allele for 50.7% of the SNPs, so their effects cancel
   (0.506). With risk-allele dosages the same single weight reaches 0.614, against 0.618 for a weight per SNP.
3. **Risk-allele dosages: learning within the first pass** (Fig. 3). A model stopped after one pass scores the same on
   new people as the full run (0.649 against 0.648). DeepGP then ties the recipe in every cohort: −0.003 (−0.014 to
   +0.006) with random controls, −0.012 (−0.020 to −0.003) with the cohort's own GWAS.
4. **The recipe over-weights the PRS** (Fig. 2b). Its PRS weight is 5.5 times what new people support with the GWAS of
   215,872 people, and 11.8 times with the GWAS of the cohort's 34,690 people, the same for 5% to 100% of its lead SNPs.
   The covariates then drive 12-28% of the score's spread on new people instead of 73-83%.
5. **So PRS + covariates falls below covariates alone** (Fig. 2c), in all four settings tested: −0.003 (−0.019 to
   +0.011), −0.022 (−0.044 to +0.001) and −0.018 (−0.037 to +0.002) for type 2 diabetes, −0.039 (−0.057 to −0.020) for
   atrial fibrillation. The same PRS joined outside the GWAS adds +0.013 to +0.032, every interval above zero.
6. **Validation inside the GWAS cannot see the overfitting** (Fig. 2a, Fig. 3). The paper's GWAS included its
   validation people (stated in its limitations), as ours did. Both the recipe and DeepGP score 0.93-0.99 there and
   about 0.65 on new people, from the first check onwards. Validation is so flat that two identical runs kept
   different passes (6 and 12; test 0.652 and 0.643).
7. **Other methods on the cohort's own GWAS overfit the same way.** LDpred2-auto scores 0.95 on its GWAS people and
   0.62 on new people; with covariates (recipe) 0.686, the only genetic model above covariates alone (0.678).
   lassosum2, tuned on validation, scores 0.999 and 0.563.
8. **A fault in the Mamba library, not the cause.** In every Vision Mamba version, including the one the paper used,
   the two-direction block's backward pass drops part of one gradient: for two groups of weights the true slope is
   30% larger than the gradient it reports.
   Correcting it in memory changed the matched minor-allele run from 0.522 to 0.524.

## Questions only the authors can answer

- Which allele did the dosages count?
- How were the 1:1 controls chosen?
- Was the balanced analysis's GWAS run on the balanced training set?
- Was the reported model one network, one fold's network, or an average of five?
- Each disease's covariates-only AUROC (Supp. Fig. 6 does not label diseases).

## Figures (`figures/`, each with its data as CSV)

**Figure 1. The published DeepGP learns from SNPs only with risk-allele dosages.** a, AUROC on test people (95%
bootstrap intervals, 1,000 resamples) of DeepGP with minor-allele dosages (open) and risk-allele dosages (filled), the
paper's PRS recipe and covariates alone, on 1:1 cohorts with controls matched on age and sex or drawn at random (34,690
training and 3,326 test people; SNPs and PRS weights from the GWAS of 215,872 training people). Dashed lines: the
paper's type 2 diabetes results. b, Logistic models of the same 47,341 SNPs on the matched cohort, differing only in
how they may weight SNPs. The line marks 0.5, chance. Data: `fig1.csv`.

**Figure 2. Fitting the PRS recipe on the GWAS people over-weights the PRS and pushes PRS + covariates below covariates
alone.** a, AUROC on the code's training people (85%), validation people (15%), both in the GWAS, and test people
outside it, for the paper's recipe, DeepGP with risk-allele dosages after its first validation check, and covariates
alone; GWAS of the cohort's own 34,690 training people on all 7.8 million imputed SNPs. b, PRS weight in the recipe
divided by the weight fitted on the test people, against the number of SNPs in the PRS; open circles, random 5-100%
subsets of the 98,440 lead SNPs. c, AUROC on test people of covariates alone, the paper's recipe and the same PRS
joined to the covariates on the test people (5 folds), in four settings. Data: `fig2a.csv`, `fig2b.csv`, `fig2c.csv`,
`fig2c_differences.csv`.

**Figure 3. Validation inside the GWAS cannot see the overfitting.** Risk-allele DeepGP on the 1:1 random-controls
cohort (GWAS of 215,872 people), rerun with the test people scored after every validation check. a, Training AUROC
(mean over batches) and validation AUROC, both on people in the GWAS. b, AUROC on test people, with the pass the code
keeps (best validation) and reference lines for covariates alone and both PRS joins. Scoring the test people did not
change training and was never used for a choice. Data: `fig3.csv`, `fig3_references.csv`.
