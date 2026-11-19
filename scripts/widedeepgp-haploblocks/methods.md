# Haploblock pre-test on chromosome 22: would haplotype vectors beat the per-SNP table?

Status: design written 18 November 2026, before any code; results in `results.md` as they arrive.
Branch `widening`. Data under `data/datasets/ard30/hm3plus/haplo/` (owner-only, never in git).

## 1. The question

WideDeepGP's first layer is a per-SNP table pooled by LD block (design document
`../widedeepgp-blocks/methods.md`, section 3). Robert proposed replacing it with haploblock vectors:
partition the genome into about 100,000 short blocks inside which a few common haplotypes account for
nearly all chromosomes, give every common haplotype of every block a learned vector, and pool those.
The arguments for it are not new signal but fewer and better-conditioned parameters: 1,258,200 SNP
profiles whose dosages are collinear within a block become a smaller number of haplotype profiles
whose indicators are mutually exclusive within a block, so stochastic gradient descent estimates a
mean per category instead of crawling along the directions a ridge shrinks. The argument against is
a bound: a haplotype layer can at best match a haplotype ridge, and the haplotype polygenic-score
literature puts the gain of haplotypes over SNPs at a few percent relative.

The pre-test asks two things on one chromosome, before the two-week build (phasing the genome, a
haplotype store, a haplotype input mode, a haplotype warm start):

1. **Arithmetic.** How many haploblocks does the rule give, how many common haplotypes per block,
   what share of chromosomes do they cover, and how many first-layer parameters follow, against the
   per-SNP table (1,258,200 profiles) and the genotype code (3,774,600)?
2. **Signal.** On the same SNPs, the same participants, the same folds, the same residualised labels
   and the same shrinkage grid, does a closed-form ridge on haplotype counts predict the 30 diseases
   out of sample better, worse or the same as the closed-form ridge on SNP dosages? A ridge on
   haplotypes plus SNPs measures what haplotypes carry beyond additive SNP effects.

Why a ridge settles it: at 215,872 participants the first layer is an estimation problem, and the
closed-form ridge of a parametrisation is the best additive estimator of that parametrisation at a
given prior (section 2 of the design document, item 14 of its section 12: no network run has beaten the
ridge of its own inputs yet). If the haplotype ridge does not beat the SNP ridge out of sample, a
network on haplotype vectors will not either; if it does, its margin is roughly what the haplotype
first layer can add.

## 2. Data

- **Rung:** the top rung, `data/datasets/ard30/hm3plus` (215,872 training participants, 20,000 tuning).
- **Chromosome 22** of the HapMap3+ store: 18,173 SNPs (the 17,055 the discussion quoted was the count
  of the held-out groups' participants, not of chr22's SNPs), 15 Privé blocks (ids 417 to 431, 327 to
  2,953 SNPs, 0.4 to 6.8 Mb), 1.44% of the model's SNPs.
- **Participants phased:** the training and tuning rows together (235,872; 471,744 chromosomes).
  Phasing reads genotypes only, so the tuning set's genotypes may enter it, as they enter nothing
  that is fitted; the **test split is not read at all**, not even its genotypes, and the related and
  held-out groups are not needed.
- **Hard calls:** from the store's dosages (0.01 resolution): 0, 1 or 2 when the dosage is within 0.1
  of it (plink2's default threshold), otherwise missing, which the phasing imputes. Measured: 0.65% of
  calls missing, no variant above 5%, one participant above 5% (`haplo/export.json`).
- **Genetic map:** the PGI arm's OMNI interpolated map for chr22 (`data/pgi/ld/genetic_maps/omni/`,
  494,328 positions, 16.05 to 51.24 Mb, covering the store's 16.50 to 51.22 Mb), rewritten in
  SHAPEIT's three-column format.
- **Folds, labels, covariates, standardisation:** the block ridge's (`blocks/folds.npz`: training
  rows by the ladder permutation modulo 5, tuning rows by the pseudo-fold hash; the 12 covariates;
  the cached training-set SNP statistics the ridge used).

## 3. Phasing

SHAPEIT5 `phase_common` (bioconda 5.1.1, installed
into `/mnt/sdg/robert/envs/phasing` with bcftools 1.25), population phasing without a reference
panel, default MCMC and PBWT settings, seed 1, ten threads. The input BCF is made by plink2 from a
PLINK 1 binary file written straight from the store (`export_chr22.py`), with the ALT allele as A1
so that the BCF's genotype 1 is the ALT allele whose dosage the store holds, then given the AC, AN
and AF tags by `bcftools +fill-tags` (phase_common refuses a file without AC and AN); the check is
that the BCF's allele frequencies equal the hard calls' (passed: largest difference 5e-7).
**Measured cost** (18 Nov, 10 threads, 235,872 samples, 18,173 sites): 6.8 hours in all
(24,633 s; about 68 CPU-hours; 15 GB of memory): setup 14 minutes (the PBWT sweep 11.5), then 15 MCMC
iterations whose HMM pass shortened as the haplotypes converged, from 49 minutes in the first burn-in
iteration (265 ± 121 conditioning states, 4.4 Mb windows) to 13 to 19 minutes in the five main
iterations (113 ± 59 states). The first iteration alone had suggested 13 hours. Scaled by sites,
phasing the 1,258,200 SNPs of the store would cost about 4,700 CPU-hours at this sample size with
default settings (about five days on 40 threads, three weeks on ten), before any lighter MCMC scheme
or smaller PBWT depth is tried: a cost the build decision must carry. No accuracy measure exists here
(no trios, no truth haplotypes); for common variants at this sample size the published switch error
is well under one percent, and the rule below cuts blocks far shorter than a switch interval.

## 4. Haploblocks and haplotype features

- **Rule.** Inside each Privé block, from left to right, extend a segment one SNP at a time while the
  haplotypes with frequency at least `f` among the 431,744 **training** chromosomes cover at least
  `c` of them; close the segment one SNP before the first failure and start the next segment there.
  A single SNP is always a valid segment. Primary cell f = 1%, c = 95%; sensitivity grid
  f in {0.5%, 1%, 2%} by c in {90%, 95%, 99%}. The rule is label-free and uses the training
  chromosomes only, so nothing is cross-fitted here; a tuning chromosome whose haplotype in a
  segment is not in the training dictionary gets the "other" code.
- **Codes.** Per segment: common haplotypes 0..H−1 in order of decreasing training frequency, "other"
  = H. Per participant and segment: two codes (one per chromosome).
- **Features for the ridge:** per segment H + 1 counts (copies of each common haplotype and of
  "other", each 0, 1 or 2), standardised with their training mean and SD like the SNP dosages. The
  counts of one segment sum to 2, so one direction per segment is collinear; the ridge's penalty
  handles it.
- **Reported (aggregates):** segments per Privé block and on chr22; SNPs and kb per segment
  (quantiles); H per segment (quantiles, share with H in 2..15); coverage; features
  Σ(H + 1) on chr22 and projected to the genome by SNP share (× 1,258,200 / 18,173), against the
  1,258,200 and 3,774,600 of the SNP codes; all for the nine grid cells.

## 5. The three ridges

All per Privé block, with `build_block_ridge.py`'s machinery (`residual_coefficients`, `solve_block`):
labels residualised on [1, covariates] over each fold model's training rows, per-fold grams in float64
on the CPU, one eigendecomposition per block and model, five fold models and the full model.

- **A. SNP dosages** (18,173 standardised features): the pipeline's ridge restricted to chr22. Check:
  at h² 0.05, 0.15 and 0.45 the fold weights must reproduce the chr22 rows of the stored
  `ridge_weights_fold<k>.npy` (same λ, same residualisation).
- **B. Haplotype counts** (Σ(H + 1) standardised features of the primary cell).
- **C. Both**, A's and B's features in one ridge.

**Penalties.** The pipeline's rule λ = M / h² with M the genome-wide feature count (prior effect
variance h² / M per standardised feature), on the grid h² in {0.0167, 0.05, 0.15, 0.45, 1.35} (the
pipeline's three plus one on each side, since most diseases chose its strongest penalty). For B,
M_hap is the chr22 feature count projected to the genome by SNP share; for C each class keeps its
own M. Implementation: haplotype features are scaled by √(M_snp / M_hap) so that one scalar
λ = M_snp / h² gives the class-specific penalties, and one eigendecomposition serves the whole grid.

**Scores.** A training row's score comes from the fold model that excluded it; a tuning row's from
its pseudo-fold model; summed over the 15 blocks: 30 diseases × 5 penalties per parametrisation.

## 6. Evaluation and decision rule

- **Penalty choice** per disease and parametrisation on the out-of-fold training scores (AUROC of
  the raw score), never on the tuning set, so the tuning numbers are honest. The fixed middle
  penalty h² = 0.15 is reported beside the choice.
- **Metrics,** per disease, on the tuning set (20,000) and on the out-of-fold training rows
  (215,872; the comparator protocol already treats these as out of sample, and they carry about ten
  times the power, which one chromosome needs: chr22 holds about 1.4% of the SNP heritability, so
  its score alone moves AUROC by thousandths):
  (i) the Pearson correlation of the score with the label residualised on the covariates (the
  ridge's own target), (ii) ΔAUROC: the AUROC of covariates plus score over covariates alone, the
  combination a logistic regression fitted on the out-of-fold training rows, applied to the
  evaluated rows.
- **Paired comparisons:** B − A and C − A per disease, with 95% intervals from 2,000 paired
  bootstrap resamples of participants (the same resamples for every disease and method); pooled
  over the 30 diseases as the mean difference with its participant-level bootstrap interval and the
  count of diseases on each side.
- **Decision rule, written before the runs.** The haploblock build is worth two weeks if B − A is
  positive with an interval excluding zero on the out-of-fold training rows and not negative on the
  tuning set, in both metrics. If B − A is indistinguishable from zero or negative, the haplotype
  parametrisation does not add additive signal at this N, and the remaining argument for it is
  optimisation (conditioning), which this test does not measure and which the arithmetic of
  section 4 bounds. C − A, whatever the verdict, is the measured size of haplotype-specific signal
  (tagging and within-block interaction), the ceiling for any gain a haplotype layer could have over
  the SNP table on top of better optimisation.
- **Sensitivity:** B and C at the cells (1%, 90%) and (1%, 99%) beside the primary cell.
- **What the test can resolve (measured before the comparison, from the stored cross-fitted SNP
  ridge restricted to chr22, at the pipeline's chosen penalty).** The chr22 SNP score alone has a mean
  correlation of 0.0060 with the residualised label on the out-of-fold training rows (0.0072 on the
  tuning set; positive for 29 and 25 of the 30 diseases) and a mean ΔAUROC of 0.0002, as a 1.4% share
  of the genome should. The paired comparison therefore resolves relative differences between the
  parametrisations of the order of 15 to 20% pooled over the diseases (two scores correlated at 0.9 on
  215,872 rows give a standard error near 0.001 per disease on the correlation difference), and
  nothing finer: a haplotype gain of a few percent relative, the literature's figure, would read as
  "indistinguishable from zero" here. The pooled difference is also reported relative to A's mean.

## 7. Decisions raised here (defaults taken; object if wrong)

1. Phase training and tuning participants together, the test split untouched.
2. SHAPEIT5 5.1.1 from bioconda, default settings, seed 1; no accuracy estimate.
3. Haploblocks nested in Privé blocks by the coverage rule with f = 1%, c = 95% as primary.
4. The prior-variance rule for the haplotype penalty (λ = M_hap / h²), a five-point h² grid for
   every parametrisation, the penalty chosen on the out-of-fold training rows.
5. The out-of-fold training rows as the second evaluation set, and the pooled interval as the
   decision's basis, because one chromosome's signal is too small for 20,000 tuning rows alone.
6. The within-block "other" bucket as the rare-haplotype fallback (no SNP fallback inside the test).

7. **Clumping and thresholding (19 Nov, section 9):** every haplotype column as a pseudo-SNP, the
   marginal test linear on the residualised label, clumping inside Privé blocks with the PGI arm's
   r² grid and window, its 15 thresholds, the candidate chosen on the out-of-fold training AUROC,
   and a union arm clumped across both feature sets.

## 8. What this test cannot say

Nothing about optimisation: whether a network trains better on orthogonal haplotype indicators than
on collinear dosages is a training question, measured only by building it. Nothing about rare
variation (the "other" bucket). Nothing about phasing error (unmeasured; it attenuates B, not A).
Nothing about blocks that straddle Privé boundaries (none are allowed). One chromosome, one rung.

## 9. The same comparison under clumping and thresholding (19 November 2026)

Robert asked, after the interim reading of the ridges (`results.md`), for the comparison under the
simplest PGI as well: clump and threshold the SNPs, clump and threshold the haploblocks the same
way, score each as the weighted sum of marginal effects, and compare the two scores. The PGI arm's
C+T (`PGI/methods.md` section 5.1) is the template: marginal effects from a GWAS, plink2 clumping
within 250 kb at r² of 0.1, 0.2, 0.5 and 0.8, 15 p-value thresholds log-spaced from 5 × 10⁻⁸ to 1,
the index SNP's BETA as its weight, the best of the 60 candidates chosen afterwards.

**How a haploblock can be clumped and thresholded.** A haploblock with H common haplotypes is a
multi-allelic locus; five codings were considered. (1) Every haplotype as a pseudo-SNP: the copy
count of each haplotype (and of "other") is a column with its own marginal test, and clumping by
r² runs over all columns, within and across haploblocks; two near-complementary haplotypes of one
block are correlated at r² near 1 and the rule keeps the lead one, while a dozen haplotypes at 10%
each sit at r² near 0.01 and all survive. (2) One lead haplotype per haploblock (haplotype against
all others), clumped across blocks: each haploblock a biallelic locus. (3) A block-level omnibus
test on all H columns jointly, thresholded on the block's p, blocks clumped against each other by
LD, each kept block scored with its jointly fitted haplotype effects. (4) A collapsed risk-haplotype
count: within each training fold, haplotypes split into risk and protective by the sign of their
effect, the copies of risk haplotypes summed into one 0/1/2 feature per block, then SNP-style C+T
on those pseudo-SNPs, which adds a selection step the SNP arm lacks. (5) Reference coding: the
commonest haplotype of each block dropped as the baseline, the rest as in (1). **Coding (1) is
primary**, because it treats SNPs and haplotypes identically; the others are not run.

**Design, matching the ridge comparison wherever the two can match.**
- **Features:** the ridge's standardised SNP dosages (A), the standardised haplotype counts of
  section 4 (B, unscaled, since the marginal effect of a standardised column is scale-free), and
  their union (C). A haplotype column's position is its haploblock's midpoint.
- **Marginal effects per fold:** over the fold model's training rows (four fifths of the 215,872),
  the Pearson correlation of each column with the label residualised on [1, covariates] (the
  fold-fitted coefficients the ridge uses), from the same per-block grams, and its two-sided
  p-value from the t statistic. This is the linear score test, not REGENIE's Firth logistic test:
  one matrix product per block and fold instead of 4.7 million logistic fits; at these effect
  sizes the two agree. The PGI arm's REGENIE GWAS is not reused because it was fitted on all
  215,872 training rows, which would make the out-of-fold evaluation set in-sample.
- **Clumping:** greedy by p-value (p1 = p2 = 1) inside each Privé block, within ± 250 kb, at the
  r² grid of the PGI arm, with r² from the fold model's own training rows. Clumps never cross a
  Privé block (the PGI arm's plink2 clumping may, within 250 kb, across the low-LD boundary; not
  material). In C, a haplotype column redundant with a kept SNP is removed and vice versa.
- **Score:** Σ_j β̂_j z_j over the kept columns passing the threshold, with β̂_j the marginal
  correlation and z_j the standardised feature: PLINK `--score` with BETA weights up to scale and
  shift, since BETA on the raw dosage equals the correlation times SD(residual)/SD(dosage). The
  unweighted risk-allele count is not run.
- **Cross-fitting, candidate choice, evaluation, decision rule:** as sections 5 and 6: training
  rows scored by the fold that excluded them, tuning rows by their pseudo-fold; the candidate
  (r², p) per disease and parametrisation chosen on the out-of-fold training AUROC of the raw score
  (the PGI arm chooses on the tuning set, which here is the report set); r with the residualised
  label and ΔAUROC on both evaluation sets; B − A and C − A with the paired bootstrap; the rule of
  section 6. The fixed candidate reported beside the choice is r² 0.1 with p ≤ 1, every clumped
  column with its weight, the least-selected score. The out-of-fold AUROC of every candidate is
  kept (`results/ct_curve_chr22_<cell>.csv`) so that the threshold curves of the two
  parametrisations can be compared without selection.
- **Code:** `compare_ct.py`, sharing the features, grams, routing, bootstrap and evaluation of
  `compare_ridge.py` (whose evaluation stage became the function `evaluate`, parallel over
  diseases, with no change to its numbers); `run_ct_chr22.sh`.

## 10. Files

`export_chr22.py` (hard calls to PLINK 1 binary, the genetic map, an aggregate export record),
`run_phase_chr22.sh` (plink2 to BCF, index, phase_common, index; detached, logs and exit lines),
`haploblocks.py` (segments, dictionary, codes, the grid's aggregate tables), `compare_ridge.py`
(the three ridges, scores, metrics, bootstrap), `run_analysis_chr22.sh` (waits for the phasing
chain, then the nine haploblock cells and the three compared cells; detached, resumable by product),
`make_figure.py` (the figure of section 6 with its data and legend), `test_haploblocks.py` (unit
tests, CPU, synthetic haplotypes and small ridges, the end-to-end run on the synthetic store).
Results: `results/*.csv` (aggregates only), `results/figure_chr22_<cell>.{png,pdf,_data.csv,_legend.md}`
and `results.md`.
