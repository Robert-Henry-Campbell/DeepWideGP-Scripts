# Replicating DeepGP (Zhu et al., Briefings in Bioinformatics 2026): methods

**Aim.** Reproduce the paper's type 2 diabetes result (balanced 1:1 test set: DeepGP 0.64, LightGBM 0.61, PRS +
covariates 0.60) with its published code, and explain any gap. Run 20-21 November 2026. Results: `results.md`.

**Paper and code.** Zhu, Ghose, Climente-González, Howson, Hu, Nevado-Holgado, Brief. Bioinform. 27(5), bbag550; its
supplement from Europe PMC (PMC13647273). Code: github.com/tndrg/DeepGP at 48e841d, cloned unchanged to
`/mnt/sdg/robert/deepwidegp/DeepGP`. Runs used a frozen copy with three compatibility patches that change no
computation (`runs/deepgp_replication_e11/code_upstream_48e841d`): the code is written for the February 2024 Vision
Mamba library and ours is the September 2024 one (a module and a keyword were renamed). Settings: the published type 2
diabetes configuration (`scripts/t2d.sh`, identical to Supp. Table 7): width 32, one block, four heads, attention
pooling, batch 64, learning rate 7e-4, weight decay 1e-4, dropout 0.2; at most 30 passes, validation every 300
batches on 15% of the training people, early stopping on validation AUROC with patience 10, seed 33.

**People.** Unrelated White British participants of UK Biobank application 15181, without the withdrawn. Training
people come from our training table, test people from our tuning table; the locked test table was never read.

**Outcomes.** Type 2 diabetes: our E11 label (hospital inpatient and death records). Atrial fibrillation: I48. The
paper's multi-source type 2 diabetes definition was not used: 88% of its cases in our data are E11 cases, with the
same age and sex profile.

**Cohorts** (seed 1). Every case of each table, with one control each: matched exactly on sex and year of age (design
M), or drawn at random (design R). Type 2 diabetes: 34,690 training people (17,345 cases) and 3,326 test people (1,663
cases). Design N, for the PRS only: every training (215,872) and tuning (20,000) person at population prevalence.
Atrial fibrillation: 30,862 training and 3,340 test people.

**GWAS and SNPs.** Two GWAS. (1) REGENIE step 2 on all 215,872 training people, 1,258,200 HapMap3+ SNPs, adjusted for
birth year, sex and 10 PCs (the study's shared GWAS). (2) The paper's own `scripts/gwas_plink.sh`, unchanged: PLINK 2
logistic regression with Firth fallback, adjusted for sex, age and PC1-6, on the design-R cohort's 34,690 training
people only, all 7,813,490 imputed SNPs. Clumping as the paper: p <= 0.05, r² 0.7 within 500 kb, giving 47,341 lead
SNPs from (1) and 98,440 from (2) (the paper: about 100,000 from 13.3 million). Both GWAS include the code's
validation people, as the paper's did.

**Leakage check (added 27 Nov 2026, running).**
- **What we saw.** With GWAS (2), the code's validation AUROC was 0.99 from its first check, against 0.648 on the test
  people.
- **Two explanations.** Robert's supervisor suspects data leakage. Robert's explanation: the GWAS that picked the SNPs,
  and each SNP's risk allele, included the code's validation people. A network that largely re-learns the GWAS's linear
  signal then scores those people almost perfectly.
- **The test.** GWAS (2) is refitted on the code's training part only (29,486 people, leaving out its 5,204 validation
  people), and everything else is repeated unchanged. If validation falls to the test level, the gap came from the
  shared GWAS, not from a leak elsewhere.
- **Where.** Design and results in `runs/deepgp_replication_e11_trainonly_gwas`.

**Dosages.** Minor-allele dosages, as the paper states, or risk-allele dosages: each SNP counts the allele whose log
odds ratio in the GWAS that selected it is above 0. After the code's own per-SNP standardisation this flips the sign
of about half the SNPs and changes nothing else.

**PRS comparators.** PRS = log odds ratio x dosage summed over the lead SNPs. The paper's recipe: logistic regression
(scikit-learn defaults) on the PRS, sex, age and PC1-6, standardised and fitted on the training people, all of whom
were in the GWAS. Beside it: the PRS alone; covariates alone, fitted the same way; and the PRS joined to the
covariates on the test people by 5-fold cross-fitting, so that no person whose label shaped the PRS fits the join.
On the cohort's own HapMap3+ GWAS, LDpred2-auto and lassosum2 were fitted with the PGI arm's code and LD reference.

**Diagnostics.** Linear models of the same SNPs that differ only in how they may weight SNPs. A probe of the untrained
network: gradients, the size of each chromosome summary, and the gradient directions of independent batches.
Gradient checks against finite differences. Reruns stopped after one pass, or scoring the test people after every
validation check; these extra scores were never used to choose anything.

**Statistics.** AUROC on the test people; 95% intervals and paired differences from 1,000 bootstrap resamples of the
test people (seed 1). The recipe's over-weighting is its PRS weight divided by the weight a regression fitted on the
test people gives, with every regression standardised on the training people.

**What differs from the paper**
- Outcome definition (above), 77% of its training cases and 42% of its test cases.
- SNPs from 1.26 million HapMap3+ or 7.8 million imputed SNPs, not 13.3 million.
- Matching variables unknown; both extremes run (M, R).
- No five-fold hyperparameter search: the published settings were used as given.
- Newer library versions; DataLoader workers 8 instead of 32; one seed.

**Where everything is.** Scripts and per-step notes in `runs/deepgp_replication` (atrial fibrillation),
`runs/deepgp_replication_e11` (cohorts, minor-allele runs, PRS), `runs/deepgp_no_learning` (risk-allele runs and
diagnostics) and `runs/deepgp_replication_e11_cohort_gwas` (the cohort's own GWAS, LDpred2 and lassosum2,
over-weighting); participant-level files in the matching `data/` folders. Figures: `make_figures.py` and
`recipe_vs_covariates.py` here, reading aggregate results only.
