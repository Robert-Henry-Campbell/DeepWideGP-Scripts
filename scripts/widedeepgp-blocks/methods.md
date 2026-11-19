# Methods: WideDeepGP on LD blocks, trained end to end (the deep arm of the scaling study)

*Status: design agreed with Robert on 16 November 2026, amended on 17 November (three seeds per
cell; AUROC as the metric family; the table width as a second size axis; the primary estimands,
decision rules and pre-registered null of section 8; the capped-asymptote bound on the crossover and
the E6 h² scan with its data-limitation projection; then the last open items closed: S1 stays one
layer, no simulated genomes, packages installed as needed, every other default accepted). No
decision is open. **Implemented and started on 17 November 2026** (Robert: "implement the plan ...
then launch it"; GPUs 2, 3 and 4): section 12 lists where the code departs from sections 2 to 7 and
how the chain runs. This document is the durable description of the design and its decisions. The
working plan is `/home/robert/.claude/plans/we-re-just-talking-here-crispy-toucan.md` (revision 7);
results will go in `results.md`. The model's own methods document is `DeepWideGP/methods.md`, whose
section 4.11 describes the code.*

## 1. Why the design changed

- The first training size of the ladder (13,199 participants; `../widedeepgp-ladder/`) showed the
  network adding nothing to the in-model ridge PRS path: ridge alone 0.6634 mean tuning AUROC,
  ridge plus network 0.6633. The simulated-trait gate had shown the same (0.7187 against 0.7187;
  `../widedeepgp-gates/`).
- The pilot (`../widedeepgp-pilot/`) had shown why: with one shared convolution per chromosome, a
  32-number sinusoidal code of the SNP index and a mean over up to 105,260 tokens, the network
  cannot address single SNPs and so cannot learn per-SNP effects (0.51 AUROC on simulated traits
  with ceilings of 0.78 to 0.85). The ridge path held the only per-SNP parameters.
- A ridge inside the model confuses a paper about scaling of deep learning: the headline number was
  a ridge number, and 98% of the parameters were a linear PRS. A first redesign that fed the network
  fixed, cross-fitted ridge scores per LD block was rejected for the same reason: the better the
  linear model one slices, the less is left for the network, and the model stays mixed.
- The decision (Robert, 16 November 2026): one deep network trained end to end from dosages, whose
  first layer is a per-SNP table pooled by LD block. Blocks are a pooling choice inside the network,
  not a linear preprocessing step. The ridge path stays in the code, off by default.

## 2. The question, in Kaplan et al.'s terms

The study measures how genetic prediction scales, styled after Kaplan et al. (2020): performance
against dataset size D, model size N and compute C, with the PGI arm (`PGI/methods.md`) as the
linear reference over the same ladder of five training sizes (13,199 to 215,872 participants, median
N_eff 3,372 to 53,908). The translation:

| Kaplan | Here |
|---|---|
| Loss: cross-entropy in nats | AUROC, in the forms the PGI arm already uses: the gain over the covariate-only model (ΔAUROC) after the shared protocol's tuning-set refit, converted to liability-scale R² for the Daetwyler fit; selection, comparison and the laws all use this family; cross-entropy is a secondary table. The ceiling is close and set by h²_SNP and prevalence (Wray et al. 2010), so curves saturate and the asymptote is fitted |
| D: tokens | Participants as N_eff: the five rungs. SNPs are context, not data |
| N: non-embedding parameters | Two counts: the block MLP, mixer and head (Kaplan's N), and the per-SNP table's width d_e, which holds the SNP effects and most of the parameters and is varied as a second size axis rather than ignored; identity embeddings excluded from both |
| C ≈ 6 N B S | 6 N x tokens processed (participants seen x blocks), plus the transformer's attention term, plus the first layer (6 x M x d_e per participant) and any pretraining, each reported; GPU-hours beside it. Derived afterwards from each run's records, at the batch size used, without Kaplan's critical-batch adjustment. For a fixed recipe C follows from N, D and the number of passes; the compute frontier comes from the learning curves saved along every run |
| Run-to-run variation | Three seeds per cell; the seed spread is the training-noise band on every comparison |
| Context length | The number of blocks (431 to about 5,000): a one-factor side study, as Kaplan treated context and shape |
| Second architecture | Transformer against Mamba over the same tokens |
| Shape; transfer | Three shapes at one size at the top rung; the Asian and Black held-out groups as other distributions (the related White British are a relatedness check, not a shift) |

Curve forms are the PGI arm's (methods 8.4): the Daetwyler expectation, fitted to liability-scale R²
(from AUROC through the Wray and Lee conversions in `PGI/pgi/evaluate.py`) with h² and M_e free, and
a four-parameter sigmoid in log N_eff fitted to ΔAUROC with its asymptote bounded by the heritability
ceiling, so each model family is summarised per disease by an asymptote and a knee; Kaplan's
L(N, D) = [(N_c/N)^(a_N/a_D) + D_c/D]^a_D is fitted to each mixer's grid on the distance to the
ceiling, ceiling − ΔAUROC, which goes to zero as N and D grow as his loss does, with the ceiling from
the project's S-LDSC h² (`PGI/results/hm3plus/sldsc_h2.csv`), fixed and free. Fits are per disease
and their parameters summarised across diseases; the Daetwyler form is descriptive for the deep arm.
With M_e about 60,000, N_eff h²/M_e is 0.36 at the top rung for h² = 0.1 and about 1 for h² = 0.3, so
the ladder sits before or at the knee for most diseases and resolves the asymptote only for the more
heritable ones; the diseases for which it is estimable are listed before the unlock.

## 3. The model

For one participant, in order:

1. **Per-SNP table (first layer).** Every SNP has a learned vector of d_e numbers (16 in the grid;
   4 and 64 on the table axis). The participant's dosage scales the vector and the scaled vectors of
   a block's SNPs are summed. Two forms: `--snp_code dosage` (one vector per SNP, linear in dosage)
   and `--snp_code genotype` (three vectors per SNP for 0, 1 and 2 copies, interpolated for
   fractional dosages: dominance per SNP exact in the first layer, three times the table). With
   d_e = 1 and a summing mixer this is a ridge PRS and weight decay on the table is its penalty, so
   the model contains a ridge as a special case; with d_e < 30 the 30 diseases' effects at a SNP
   share a rank-d_e structure, a learned prior on pleiotropy.
2. **Per-block layer.** Learned identity vectors per block and per chromosome are added (the
   position encoding: with a fixed set of a few thousand tokens, position is identity; no sinusoid),
   then a small MLP shared by all blocks (d_e -> 4 d_e -> d, GELU).
3. **Mixer.** One sequence of all block tokens plus the covariate token, through the existing
   bidirectional Mamba block or a transformer encoder stack (scaled-dot-product attention, O(T)
   memory). `--d_model` and `--e_layers` set N.
4. **Head.** Mean over tokens, concatenated with the covariate vector, then the two-layer head: 30
   logits. The loss and the sampler are those of `DeepWideGP/methods.md` section 5 (since 18
   November 2026 plain cross-entropy at natural prevalence over uniform batches; section 12, item
   12); validation on the whole tuning set, selection on mean tuning AUROC and the search over
   learning rate, weight decay and dropout (12 trials) are unchanged.
5. **Training.** Everything by AdamW on the loss, from dosages, from scratch. The table has its own
   parameter group: learning rate by the old wide path's rule (κ τ √B / √M) and an L2 penalty in the
   loss scaled by 1/(N τ), τ searched; warm-up and cosine decay as the fixed recipe. Snapshots keep
   the table in bfloat16: about 20 per run for the cells that feed the compute figure, six elsewhere.
   The head keeps the published initialisation so gradients reach the table from the first step.
6. **Pretraining (flag).** Masked-SNP prediction within blocks: hide 15% of a block's SNPs per
   participant and predict each hidden genotype class from the block token through a per-SNP output
   vector. No labels, no mixer; per rung on that rung's training rows only, once per partition; the
   supervised model fine-tunes from it. An ablation, not the default; a variant pretrained on all
   training genotypes is one extra cell at the bottom rung.
7. **Warm start (flag).** The dosage-form table initialised from the closed-form ridge of section 6.
   An ablation.
8. **Not modelled:** interactions between distant loci (representable, not identifiable at this N);
   within-block interactions beyond the pooled nonlinearity. No ridge path, no offset, no annotations.

The default configuration for the grid is the plain dosage table (d_e = 16) from scratch; the
genotype code, pretraining and warm start are ablations at a few cells, and the winner on the tuning
set becomes the headline configuration before the grid runs in full.

## 4. Model sizes, and why these

**Mixer axis.**

| Size | Transformer (d x layers) | N (non-embedding, about) | Aspect ratio d/L |
|---|---|---|---|
| S1 | 16 x 1 (one layer kept deliberately: the published DeepGP has one encoder layer, so S1 and S2 bracket it and the classic comparator is compared fairly; Kaplan excluded one-layer models from his fits, so the secondary joint fits are reported with and without S1) | 1e4 | 16 |
| S2 | 32 x 2 | 2.5e4 | 16 |
| S3 | 64 x 4 | 2e5 | 16 |
| S4 | 128 x 6 | 1.2e6 | 21 |
| S5 | 256 x 8 | 6.3e6 | 32 |

N ≈ 12 L d² plus the block MLP and head; `n_heads` = 4 (2 at d = 16). Mamba sizes are matched to the
same N within a factor 1.3 by depth (its block has about 7 d² parameters per layer, so about twice
the layers at each width). The per-SNP table is not part of N.

Justification:
- **Range and spacing.** Five sizes at about a factor 8 each span 2.8 decades, the most the budget
  allows (section 7). Kaplan used finer spacing over six decades; with five points, wide spacing
  covers the range rather than resolving curvature within it.
- **The lower end** is a near-linear model: one small layer over the first layer's tokens, which
  anchors the curve where the network can add little beyond its table. The published DeepGP
  configuration (width 32, one layer, about 11,000 parameters per encoder layer) sits at S2, so the
  classic comparator has a size-matched point.
- **The upper end** is set by two things. Data: 215,872 participants with 30 labels is a small
  dataset by Kaplan's standards, and Kaplan's own overfitting law (Figure 9) says performance at a
  fixed D stops improving and then worsens as N grows; the grid must extend past the optimum to
  measure that bend, which is the point of fitting L(N, D) and reporting N_opt(D). S5 is there to be
  on the far side of it. Compute: S5 as a transformer over 2,000 tokens costs about 7 hours per
  top-rung training; the next step (38 million parameters) would cost about 40 and is excluded.
- **Shape.** All sizes keep the aspect ratio d/L between 16 and 32, inside the region where Kaplan
  found loss insensitive to shape (Figure 5); the shape check at the top rung tests that this holds.

**Table axis.** The mixer's N leaves out the table, which holds the SNP effects and most of the
parameters, so the table's width is varied as a second size axis: d_e in {4, 16, 64}, that is 5, 20
and 80 million entries, at S3 across all five rungs for each mixer. Every figure carries both counts,
mixer parameters and total parameters. This asks whether more participants support a wider genetic
layer, the size question a geneticist cares about.

## 5. Partitions

Tokens are LD blocks in genome order. Three nested partitions: Privé's 431 HM3+ blocks (`block_id`
in `data/reference/hm3plus_in_ukb_pgen.tsv`; 119 to 15,194 of our SNPs each, median 1,636) and two
finer ones of 2,000 and 5,000 blocks, built with bigsnpr `snp_ldsplit` inside each Privé block on
the PGI arm's LD matrices (`data/pgi/ld/bigsnpr/chr*.corr.rds`, 20,000-person panel). Partitions are
label-free (built from LD, genotypes only, as every LD reference is), recorded by digest in every
run's `config.json`, and chosen on the tuning set. Following Kaplan, the block count is not an axis
of the joint law: a one-factor sweep at the fixed model size across the rungs gives the optimal block
count as a function of participants, B_opt(D), and the full-sample optimum as its top slice; the
top-rung slice runs first and chooses the grid's partition. Recombination-hotspot haploblocks (about
38,000 genome-wide, about 30 HM3+ SNPs each) are too fine for this SNP set and cohort and are not
used.

**Built on 17 November 2026** with `make_partition.R`, `check_partition.py` and
`run_make_partition.sh` in this directory (tests: `test_blocks.py`). Each partition lives in
`data/reference/partitions/<name>/`: `partition.npz` (the global 0-based block id of every variant,
per chromosome, in store order; the store's chromosome names and variant counts; a sha256 digest of
the variant ids), `partition.json` (aggregates and provenance), and for the two finer ones the
builder's `blocks.tsv` and `make_partition.json`. Per-chromosome aggregates: `partitions.csv` beside
this document.

| Partition | Blocks | SNPs per block: min / q25 / median / q75 / max | Mean |
|---|---|---|---|
| prive431 | 431 | 119 / 631 / 1,636 / 4,010 / 15,194 | 2,919 |
| ldsplit2000 | 2,000 | 60 / 280 / 476 / 826 / 2,314 | 629 |
| ldsplit5000 | 5,000 | 19 / 105 / 193 / 339 / 1,108 | 252 |

**5.1 Privé's blocks (`prive431`).** The `block_id` column of the reference table, read after
checking row by row that the store's variant ids equal the table's ID column (1,258,200 rows, 22
chromosomes). All 431 blocks of the HM3+ map keep SNPs in our set; their ids are non-decreasing in
store order (position order within a chromosome) and every block lies within one chromosome (11 to
34 blocks per chromosome), so the partition is Privé's, renumbered 0 to 430.

**5.2 The finer partitions (`ldsplit2000`, `ldsplit5000`).** Every Privé block is split into k₁
blocks and each of those into k₂ blocks, so the three levels nest by construction (and are checked).

- *LD input.* The PGI arm's sparse correlation matrices (bigsnpr `snp_cor`, 3 cM windows, the
  20,000-person White British LD panel, unthresholded), matched to the store's variants by rsid
  through `chr<c>.snpinfo.tsv`: the matrices' SNP order is the store's for every chromosome and no
  variant is absent. A variant absent from its chromosome's matrix would join the block of its
  nearest present neighbour by position; the count is recorded in `make_partition.json` (0).
- *How many blocks each Privé block gets.* Largest-remainder apportionment of the target (2,000,
  then 5,000) among the 431 blocks in proportion to their SNP counts, every block getting at least
  one (at the second level, at least its first-level count); inside a Privé block the second-level
  count is apportioned among its first-level blocks by size in the same way. Consequences: the totals
  are exactly 2,000 and 5,000; the mean block holds 629 and 252 SNPs; the smallest Privé blocks stay
  whole (68 blocks of at most 385 SNPs at the 2,000 level, 20 of at most 218 SNPs at the 5,000
  level); and sizes vary less than Privé's (interquartile ranges 280 to 826 and 105 to 339 SNPs,
  against 631 to 4,010).
- *Where the cuts fall.* `snp_ldsplit` (bigsnpr 1.12.21) chooses, for a given number of blocks K,
  the contiguous split that minimises the sum of squared correlations between SNPs in different
  blocks, counting pairs with r² ≥ `thr_r2` = 0.02 (Privé's threshold, which ignores noise-level
  correlations; on the chr22 block we probed, 0.05 gave the same cuts and 0.01 different ones).
  Block sizes are bounded to between a quarter and three times the mean: `min_size` = max(1,
  ⌊m/4K⌋) and `max_size` = min(m, ⌈3m/K⌉) for a block of m SNPs, which keeps every K feasible and
  rules out both singleton blocks carved off to make up the count and one block holding most of the
  region. `max_K` = 2K and `max_cost` = ∞ make the function report every K from ⌈m/max_size⌉ to
  min(2K, ⌊m/min_size⌋); we take the lowest-cost candidate at exactly K, or at the nearest K offered
  if K were missing, which never happened (all 431 blocks at both levels have K equal to their
  target). `max_r2` = 1 removes the function's hard cap on the squared correlation of a pair placed
  in different blocks (default 0.3). The cap belongs to LD references, where a split pair breaks the
  method; here blocks are pooling groups, and a partition finer than LD must cut strong pairs. With
  the cap the range of K on offer shrinks (block 427 of chr22: 3 to 8 blocks against 2 to 10
  without it) and in long-range-LD regions the target can fall outside it; the cost is minimised
  either way, so cuts fall at the weakest LD within the block.
- *What the cuts cost.* Summed over Privé blocks, the cross-block r² of the first level is 139,701,
  of which 74% comes from the MHC block (chr6, 22.1 to 41.4 Mb, 13,398 SNPs into 18 blocks) and
  15% from the chr8 inversion region (4.8 to 13.0 Mb, 4,966 SNPs into 7); the second level adds
  280,336 inside the first-level blocks (MHC 30%, chr8 15%, chr11 44.7 to 69.5 Mb 9%). The median
  Privé block costs 4.4 at the first level and 46.7 more at the second. This is the price of the
  finer resolution in LD terms; the sweep (section 7, E3) measures what the network makes of it.
- *Checks and run.* `check_partition.py` refuses a partition whose rows are not the store's
  variants in store order, whose blocks are not contiguous runs or span two chromosomes, or that is
  not nested in its parent, and writes the npz with the digest of the variant list, so a partition is
  tied to the variants rather than to a store path (the three share one digest). The chain prive431
  → ldsplit2000 → ldsplit5000 passed. `test_blocks.py` reproduces hand-worked splits of synthetic
  LD clusters (cost 0 between clusters, 0.27 for the one cheapest cut inside one), the nearest-
  neighbour rule for absent variants, resumption per chromosome and every refusal. The build took
  6 min 51 s on 8 workers (chromosome 1 longest at 270 s); the per-chromosome work files make it
  resumable.

## 6. Linear comparators and ablation inputs: the closed-form block ridge

The information-matched linear comparators at every rung (fairness rule: whatever the network
sees, a linear model sees too), the E0 references and two ablations need a ridge over the same
SNPs that no participant's own label shaped:
- per rung and fold, a closed-form ridge per Privé block of the label residualised on the 12
  covariates on standardised dosages (the rung's cached training-set SNP statistics), at three
  penalties, from per-block X'X accumulated in one GPU pass over the store; per-SNP weights per fold;
- five folds inside each rung's training table (fold = position mod 5 in the rung's seed-1 order;
  balanced across diseases, nested across rungs); training participants scored by the fold that
  excluded them, tuning, test and held-out participants by one fold model each, chosen by a seeded
  hash of the identifier, so their scores carry the training scores' noise; the tuning and test
  sets never enter a fit;
- products: the cross-fitted ridge PRS per participant and disease (the simplest comparator), and
  Privé-level block channels (431 x 90, int8, about 17 GB at the top rung, deleted after use) for a
  logistic ridge on all channels plus covariates (per-block and cross-trait coefficients) and for the
  frozen-first-layer ablation (stacking against end-to-end training). Both comparators are fitted on
  the rung's training participants with hyper-parameters on the tuning set and written as per-person
  score files in the PGI arm's format for the shared evaluation.

Arrays live under `data/datasets/ard30/<rung>/blocks/`, owner-only, keyed by identifier, never in
git, declared in a `blocks.json` beside `store.json` (never inside it: the PGI fingerprint reads
`store.json`).

## 7. Experiments and budget

**Seeds.** Each cell: one 12-trial search with seed 1, then the chosen configuration retrained with
seeds 2 and 3 (`--tune_trials 0` with the chosen settings), so a cell costs its search plus two
trainings, about a sixth more than the search alone. Every curve and table is the mean of the three
with the seed spread as the training-noise band. Every number before the joint test-set unlock is a
tuning-set number.

**Measured on 16 November 2026 (22:40 UTC; GPU 4; store warm in the page cache; machine load 79
on 64 threads with the PGI arm running):** the existing loader with GPU decode streams the full
1,258,200-SNP dosages at 1,288 rows/s with 8 workers at batch 256 (883 with 4 workers; no gain at
16; 1,586 at batch 1,024, GPU peak 11 GiB). One pass over the 215,872 training rows takes about 3
minutes; a partly cold run gave 1,045 rows/s. The first layer's arithmetic is negligible, so
training is loader-bound: about 3 minutes per pass at the top rung, 10 seconds at the bottom, 20
passes per training; S4 about doubles the top-rung pass, S5 as a transformer over 2,000 tokens is
about 20 minutes per pass.

| Id | What | Cells | GPU-h (rough, seeds included) |
|---|---|---|---|
| E0 | Gate on simulated traits on real genotypes (all SNPs, bottom-rung size): additive traits plus planted per-locus heterogeneity, cross-disease sharing, dominance and a few between-block interactions; references: the cross-fitted ridge PRS, GBLUP, the block-linear comparator, the ceiling. Pass criteria (decided): within 0.01 AUROC of the cross-fitted ridge PRS on additive traits; at least half of the planted gap above it on the non-additive traits | 3 partitions x 2 mixers x 2 sizes, plus the ablation flags | 25 |
| T0 | Pre-test: S1, S3 and S5 trained once each at the top rung (settings from the ablations); train-against-tuning curves inspected to see whether the mixer's overfitting bend lies inside the ladder | 3 runs | 10 |
| E3-top | Partition sweep at the top rung, size S3 | 3 x 2 mixers, 12-trial searches | 50 |
| Abl | Genotype code; pretraining (and the all-genotypes pretraining variant at the bottom rung); warm start; frozen first layer; the rotation sampler with the square-root class weight (the objective until 18 November 2026; one cell, transformer S3, top rung, section 12 item 12); at S3, bottom and top rung | about 12 | 65 |
| E1/E2 | The grid: 5 mixer sizes x 5 rungs at the chosen partition and configuration, 12-trial searches plus two seed retrains, rungs smallest first, each mixer's S3 row and top-rung column first | 25 x 2 mixers | 230 per mixer |
| Tab | Table width d_e in {4, 64} at S3 across the five rungs (d_e = 16 is the grid's S3 row) | 10 x 2 mixers | 90 |
| E2b | Shape: three depth/width shapes at S3's N, top rung | 3 x 2 | 50 |
| E3 | Partition sweep at S3 at the other four rungs | 8 x 2 | 70 |
| Pre, Lin | Pretraining per rung and partition; the closed-form ridge per rung | 15; 5 | 20 |
| E4, E5 | Compute at fixed data (tuning-set curves from every trial's validation passes; test-set curves from the kept snapshots at the unlock); transfer to the held-out groups | scoring | small |
| E6 | h² scan on real genotypes: one phenotype table of 30 simulated traits (3 h²_50 x 3 architectures x replicates; section 8) at the five rungs; the deep envelope (S3 and the top-rung best size) with both mixers and three seeds; the linear methods on the same phenotypes through the PGI positive-control pipeline (CPU, after that arm's real stages) | about 20 deep cells | 80 to 120, plus CPU |
| E7 | Stratification checks: a sibling-difference regression of phenotype on score difference in the related White British at the top rung (deep envelope and each PGI method); the scores regressed on birthplace coordinates and assessment centre on the tuning set | scoring only | small |
| C | Classic single-disease DeepGP (I10, selected SNPs, SNP tokens; datasets built) | 5 rungs | 100 |

About 1,050 to 1,200 GPU-hours in all (seed replicates about 130, the table axis about 90, E6 about
100, the pre-test 10): two to three weeks of packed compute on the three free GPUs, four to five
weeks of wall time. Constraints: CPU
is the shared resource (8 loader workers per training; the PGI arm holds 50 threads until about 30
November), so two trainings at a time until then; about 20 bfloat16 snapshots per run for the cells
that feed the compute figure and six elsewhere (about 100 GB over the grid); batch 64 to 128 or
activation checkpointing for S5 at 2,000 tokens; no training arrays on disk. Runs are launched
detached from a frozen export of `widening`, with every flag explicit.

## 8. Primary estimands, decision rules and the pre-registered null

What UK Biobank alone can answer: the measured range is median N_eff 3,400 to 54,000, before or at
the knee of the Daetwyler curve for most diseases (x = N_eff h²_50 / M_e from about 0.02 to 1, M_e =
60,000 as a unit). In that range the data identify sample efficiency, the trend of the deep-minus-
linear gap with participants and with x, and the model size the participants support. They do not
identify a capacity power law in model size (every size is data-limited), a compute-optimal frontier
(data is not a free variable), or the crossover N* except as a bound, save for diseases whose curves
bend inside the ladder. The claims below are drawn accordingly. Everything is computed on the tuning
set until the unlock and on the test set after it; "interval" is the 95% paired bootstrap interval
over participants with the PGI arm's shared resamples (2,000; stratified within a disease; one
unstratified set across diseases), on the seed-mean predictions; "seed agreement" means the three
seeds' estimates share the claimed sign; a claim passes only if its interval excludes the null value
and the seeds agree; per-disease statements are Holm-adjusted across the 30 diseases.

Notation: for disease j, rung r, family m, G_jrm = ΔAUROC over the covariate-only model through the
shared protocol, and ΔR²_jrm the liability-scale variance explained beyond the covariates (Wray 2010,
Lee 2012). Families: each PGI method; the best linear method per disease ("lin", PGI methods 6.3's
strongest-linear rule); the block-linear comparators ("twin"); the deep envelope ("deep": the mixer
size and table width pre-registered per rung from the tuning set), and each size. Gap Δ_jr =
G_jr,deep − G_jr,lin. x_jr = N_eff,jr h²_50,j / M_e with h²_50,j from
`PGI/results/hm3plus/sldsc_h2.csv`.

- **P1. Sample efficiency and the trend of the gap.** Per disease and family, the pre-knee slope
  s_jm of ΔR²_jrm against N_eff,jr (least squares through the origin; Daetwyler's small-x limit,
  slope A/K); the sample-efficiency ratio ρ_j = s_j,deep / s_j,lin, pooled as a random-effects mean
  and the median. Per disease, the slope b_j of Δ_jr against log N_eff,jr; pooled b from
  Δ_jr = a_j + (b + u_j) log N_eff,jr + e_jr. Decision: "faster" if the pooled b's interval excludes
  zero from above with seed agreement and the pooled ρ's interval excludes 1 from above; "slower"
  symmetrically; "the same" otherwise.
- **P2. The gap against the information per locus.** Δ_jr = c + d log x_jr + u_j + e_jr over the 150
  disease-rung points, a random intercept per disease; sensitivities: a three-knot spline in log x,
  and h² perturbed within its S-LDSC standard error. Decision: "the gap grows with x" if d's interval
  excludes zero from above with seed agreement; then x* is the root of c + d log x with its interval
  and N*_j = x* M_e / h²_50,j (independent of the unit chosen for M_e). An x* inside the observed
  range is an estimate, outside it an extrapolation reported as such.
- **The bound on N* when no crossing is seen (the capped-asymptote bound).** If d's interval includes
  zero, or the gap is negative at every rung, N* is a lower bound built from what the ladder
  measures. The deep envelope's pre-knee slope s_deep (P1, with its interval) fixes A/K; for any
  asymptote A its Daetwyler curve is s_deep N / (1 + s_deep N / A), and a larger A lies above at every
  N. The earliest crossing consistent with the data is that of the curve with A set to a declared cap
  C_max against the theory-constrained linear curve; N_min is the lower confidence limit of that
  crossing over the bootstrap and seed intervals of the slopes. The cap is the disease's additive
  heritability ceiling times a non-additive allowance (1.25 primary; 1.0 and 1.5 as sensitivities,
  reported as a curve of N_min against the cap); a larger cap gives an earlier, weaker bound, so the
  allowance is argued from the share of non-additive variance compatible with P2's real-data curve.
  The statement is "under a deep asymptote no greater than C_max, any crossover lies above N_min".
  The gap's widening or narrowing inside the ladder is reported but not read as evidence for or
  against a crossover beyond it: before the knee a higher asymptote with a later knee shows as a
  widening gap, and a lower asymptote looks the same. "No evidence of a crossover" is never claimed;
  the claims are "no crossing within the ladder" and the bound. The envelope's slope is used, not a
  single size's, since larger models become usable as D grows.
- **The data-limitation projection (the deliverable of E6).** For the additive part the participants
  needed to reach a fraction q of the asymptote are N_q = [q/(1 − q)] M_e / h²_50 (Daetwyler; 9 M_e/h²
  at q = 0.9: about 1.8 million at h² = 0.3, 5.4 million at 0.1, 11 million at 0.05 with M_e =
  60,000), the standard projection (Dudbridge 2013; Chatterjee et al. 2013), and the floor for the
  deep model's first layer. E6 supplies the correction: thirty traits simulated on the real genotypes
  in one phenotype table (three h²_50 values x three architectures x replicates; A: additive with
  polygenicity matched to the real diseases' LDpred2-auto and SBayesRC estimates; B: A plus 10%
  dominance variance and pleiotropy with shared causal loci; C: B plus 5% between-block interaction
  variance), trained at the five rungs for the deep envelope (S3 and the top-rung best size, both
  mixers, three seeds), with the linear methods on the same phenotypes through the PGI arm's
  positive-control pipeline. From the high-h² traits, whose x passes the knee inside the ladder, come
  the knee ratio κ = K_deep / K_lin and the asymptote ratio with intervals; the projection for a real
  disease is N_q,deep = κ [q/(1 − q)] M_e / h²_50,j, reported with κ's interval and against the
  architecture; where the simulated curves cross, N*_proj = x*_sim M_e / h²_50,j, compared with P2's
  x* where both exist. Five pre-registered conditions tie it to the data: (a) real genotypes, so M_e
  and LD are measured; (b) h² and polygenicity matched to the real diseases' estimates; (c) the
  non-additive share as a range, with the share compatible with P2's real-data curve stated; (d) the
  real diseases' gaps lie on the simulated gap-against-x curve at their own x within their intervals
  (residuals reported; on failure the projection is not made); (e) phrasing as a projection under
  architectures consistent with the observed curves, theory as the anchor, simulation as the
  correction.
- **P3. The capacity the participants support.** Per rung and mixer, N_opt(D) is the mixer size with
  the highest seed-mean AUROC; d_e_opt(D) and B_opt(D) likewise. "Size matters at rung r" if the best
  size's gain over the smallest exceeds its paired interval and the largest within-size seed range;
  "N_opt grows with D" if the argmax is non-decreasing over the rungs and the top rung's optimum beats
  the bottom rung's optimum at the top rung by a difference passing that rule. Every grid, table-axis
  and sweep cell's seed-mean model is scored once on the test set at the unlock (pre-registered list;
  nothing chosen from it).
- **P4. The deep model against its linear twin.** Per rung, G_jr,deep − G_jr,twin on the same inputs;
  passing requires the pooled interval to exclude zero with seed agreement; read with the
  frozen-first-layer ablation.
- **Secondary: curve fits.** The Daetwyler form on ΔR² (the linear arms with M_e shared across
  diseases and methods and h² free as primary, the free fit as sensitivity; the deep arm
  descriptively); the bounded sigmoid on ΔAUROC; the Chinchilla-style E + B/D^β with β shared across
  diseases for the deep envelope; Kaplan's joint form on the mixer grid as a description. No exponent
  is claimed as a capacity or data exponent. Asymptotes, knees and per-disease N* only for **eligible
  diseases**: x at the top rung ≥ 0.5 (listed from the S-LDSC table before the runs) and a
  leave-top-rung-out refit that predicts the top rung within its interval; elsewhere N* is a bound
  (the largest N_eff at which the gap's interval lies below zero, or "no crossing within the ladder"
  with the top-rung gap and its interval).
- **Relabelling.** The size axes are reported as N_opt(D) and d_e_opt(D), not as a power law in N;
  the compute figure as "performance against compute at fixed data" (215,872 participants, batch 256,
  data re-used over passes), not a frontier; N* as a bound except for eligible diseases and P2's x*
  with its flag.
- **The pre-registered null.** "Deep learning does not scale differently from linear methods at
  biobank size" is the finding if all of: P1's pooled b interval includes zero (or the seeds disagree)
  and the pooled ρ interval includes 1; P2's d interval includes zero; P3 finds no rung at which size
  matters, or no growth of N_opt with D; P4 includes zero at every rung. Under the null the paper
  reports the measured curves, the bound on N*, the ablations and the seed noise, and states that at
  this cohort size the deep arm is as sample-efficient as its linear twin and no more. A positive
  finding is any primary claim passing its rule; mixed outcomes are reported claim by claim; no claim
  is promoted after the unlock.

Figures at Nature sizes with CSV and legend: ΔAUROC and ΔR² against N_eff by family, per disease and
pooled; the gap against x with the P2 fit; ΔAUROC against mixer size per rung with N_opt(D), and the
table axis; performance against compute at fixed data; B_opt(D); the ablation table; transfer offset
against D; cross-entropy versions of the main tables as an appendix. The per-rung read-out stays the
PGI arm's paired ΔAUROC with its pre-specified claims. Before the unlock, this document carries the
chosen configuration, partition, sizes, seeds, the per-rung envelope, the list of every model to be
scored on the test set, the eligible-disease list and the rules above, and the linear baselines'
positive-control failures are resolved or labelled; the test set is then scored once, for both arms.

## 9. Decisions

Decided by Robert, 16 and 17 November 2026: the Kaplan framing; this end-to-end model as the deep
arm; the ridge path off by default; no SNP-token WideDeepGP arm; the classic I10 DeepGP comparator
kept; both mixers; three seeds per cell (the search once with seed 1, seeds 2 and 3 retraining its
choice); AUROC as the metric family for selection, comparison and the laws, cross-entropy secondary;
the table width as a second size axis; mixer size at all five rungs; the partition as a one-factor
side study; the primary estimands, decision rules, eligibility rule, relabelling and pre-registered
null of section 8; the capped-asymptote bound on N* in place of any "no evidence of a crossover"
claim; E6 with the data-limitation projection and its five anchoring conditions as its deliverable;
the 30 binary diseases only, no continuous traits; the superseded plans deleted (implemented on 17
November: section 12).

Decided on 17 November 2026, closing the list: S1 stays one layer (fair comparison with the published
DeepGP; joint fits reported with and without it); the three-size pre-test (T0) runs before the grid;
no simulated (mosaic) genomes, so the data-rich size law and the tandem law are out of scope; the
cap allowance 1.25 with 1.0 and 1.5 as sensitivities; E6's architectures A, B and C; the E0 pass
criteria as stated; both stratification checks (E7); the tag `pre-blocks` on the pre-blocks code;
packages installed as needed (statsmodels for the random-effects fits, recorded where installed);
the project rules' wording stands.

Settled in the plan, open to objection: d_e = 16 in the grid; the shared block MLP; pretraining per
rung on that rung's rows; the plain dosage table from scratch as the default configuration until
the ablations say otherwise; twelve trials per cell; warm-up plus cosine; the snapshot policy; two
concurrent trainings while the PGI arm runs; the closed-form ridge solved at the Privé level with
pseudo-fold scoring for its fixed-input uses; the related White British as a relatedness check.

No decision is open. What remains before the unlock is the pre-registration of section 8, written
here as the runs settle the configuration, partition, sizes, the per-rung envelope and the eligible
diseases.

## 10. Out of scope

Continuous traits; phased or haplotype-cluster features; attribution (post hoc on the kept
checkpoints, with path-integrated gradients); a multi-task axis; learned partitions; within-block
sequence models; batch-size (critical batch) studies; model size in SNP mode; any new UK Biobank
data; simulated (mosaic) genomes, and with them the data-rich size law and the tandem law
N_opt(C), D_opt(C).

## E0 gate: simulated traits on the real genotypes

*Added 17 November 2026; revised the same day after review (the planted components concentrated, the
additive ceiling as the planted gap's reference, the ridge read from the block ridge's score file, the gate
judged per mixer). Implemented in this folder: `simulate_traits.py` (the dataset and the references),
`e0_summary.py` (the read-out and the pass criteria), `test_e0.py`, `run_simulate_traits.sh`.*

**Purpose.** Before the ablations and the grid, the block model must show on traits of known
architecture that it matches a ridge PRS where the truth is additive and finds planted signal a ridge
cannot represent. Simulating the traits on the real genotypes keeps the real LD, allele frequencies
and M_e, so the references are the ones the real diseases face. E0 is a capability gate at the bottom
rung's size; realistic non-additive architectures are E6's business.

**Dataset** (`data/datasets/ard30/hm3plus_n13199_sim/`, owner-only, never in git). The bottom rung's
participants and tables in their original row order (training 13,199; tuning 20,000; related White
British 57,445), all 1,258,200 SNPs as whole chromosomes (a source that selects `regions` is refused),
the covariates copied verbatim (the tables are read as text, so an `NA` cell stays `NA`), the 30
disease columns replaced by the eight traits below. `store.json` keeps the rung's store and `pos_mode`
and names the traits; `ladder.json` is copied, so the rung's fold rule (position mod 5 in the training
table) and its training-set digest still hold. The locked test table is copied with its identifiers and
covariates and **empty trait columns**: its genotypes are never read (the data module reads an empty
cell as "not applicable"). Standardisation uses the rung's cached training-set SNP statistics
(`snp_stats_train_<hash>.npz`; the data module's hash of the training rows and the regions), copied into
the new dataset so the deep runs recompute nothing; were the cache absent it would be computed from the
training rows with the data module's formula and saved under the same name. One streaming pass per
chromosome reads each participant's genotypes once (decoded on the GPU), gathers the causal SNPs'
columns and accumulates the training GRM and the training-by-tuning cross-GRM; about seven minutes on
one GPU.

**Traits** (liability-threshold model; prevalence 0.2 unless stated; no covariate effect, so every
AUROC is genetic). Causal SNPs are drawn uniformly from all SNPs with seed 1, independently per trait;
effects on the standardised dosage are N(0, 1); each genetic component is scaled to its stated variance
over the simulated participants; noise brings the liability variance to 1; cases are the top
`prevalence` share of the liability. Every planted code is residualised, over the simulated
participants, on the dosages of the SNPs it is made from and scaled to unit variance, so a planted
component has no marginal effect and its variance is purely non-additive.

| Trait | Causal SNPs | h² (liability) | Construction |
|---|---|---|---|
| add_lo | 2,000 | 0.3 | additive |
| add_hi | 20,000 | 0.5 | additive |
| shared_a, shared_b | 2,000 each, 1,500 in common | 0.3 each | effects at the shared SNPs correlated at ρ = 0.7 × 2000/1500 = 0.933, which gives a genetic correlation of about 0.7 under independent SNPs; the remaining 500 are each trait's own |
| dom | 2,000 | 0.2 additive + 0.1 planted | at 20 of the causal SNPs a dominance code: the heterozygote weight 1 − \|x − 1\| (the model's genotype interpolation) residualised on the dosage; effects N(0, 1) |
| epi | 2,000 | 0.2 additive + 0.1 planted | 20 pairs of causal SNPs in different Privé blocks; the code is the product of the two standardised dosages residualised on both; effects N(0, 1) |
| hetero | 2,000 | 0.2 additive + 0.1 planted | at 20 of the causal SNPs a switch: the standardised dosage times an indicator of carrying at least one copy of the minor allele of a modifier SNP in the same Privé block (not causal, MAF ≥ 0.1 by the store's allele frequencies; any other SNP of the block if none qualifies), residualised on both SNPs' dosages; effects N(0, 1) |
| null | 0 | 0 | prevalence 0.1; no genetic effect |

**Why the planted components are concentrated (settled by me on 17 November after review; object if
wrong).** The first build spread the planted variance over all 2,000 causal SNPs (a dominance code at
each; 200 interaction pairs) or made every causal effect conditional (hetero). Its references showed the
gate had no power at the bottom rung: ceiling 0.77 against GBLUP 0.52 to 0.53 for the three planted
traits, with the additive traits' GBLUP at 0.54 to 0.57 (a liability R² of about 0.01, which is what
Daetwyler's formula gives for N = 13,199, h² = 0.3 and M_e of about 60,000). A non-additive component of
variance 0.1 spread over 2,000 loci is learnable to an R² of about 0.001 at this N by any method (the
same formula with h² = 0.1): an AUROC gain of about 0.003, within the noise of 20,000 tuning rows, so no
criterion could tell a model that learns non-additive structure from one that does not. Concentrating
the 0.1 at 20 loci or pairs gives each code a variance of 0.005 (an expected chi-square of about 66 at
N = 13,199, a z of about 8), so the planted part is learnable at the bottom rung while the additive
background stays polygenic, and the gate asks what it was meant to ask: can the model find and use
non-additive structure a ridge cannot represent at all. The alternative not taken, keeping the spread
architectures and running E0 at a larger rung, costs more GPU hours and still leaves a spread dominance
component undetectable below about 100,000 participants.

**References on the tuning set** (`sim_log.json`). The ceiling: the AUROC of the true genetic
liability. The additive ceiling: the AUROC of the trait's best additive model, the in-sample
least-squares fit of the genetic value on a constant and the standardised dosages of the trait's own
SNPs (causal, plus the modifiers) over the simulated participants; its R² is logged as
`additive_r2_in_sample` (1 for the additive traits, about two thirds for the planted ones by
construction; inflated by about k/n of the residual variance, 2% for 2,000 SNPs). GBLUP: the best
linear predictor from all SNPs through the training-set GRM, (K + λI)α = y − ȳ with K = ZZ'/M over the
training rows, λ = (1 − h²)/h² from the trait's liability h² (floored at 0.01 for the null trait), scored
as K_tune,train α. The cross-fitted ridge PRS: `e0_summary.py` computes its tuning AUROC per trait from
the block ridge's `blocks/prs_val.npz` (section 11; `score` at each trait's chosen penalty, `y`,
`labels`), which exists once the block-ridge builder has run on the simulated dataset; it is the
information-matched linear twin of the first layer and the criteria's reference. The block-linear
comparators' tuning AUROC, from `scores/blocks_linear/<method>/val_best.npz`, are shown beside it and
not judged. Also logged per trait: the realised h² (variance of the genetic value over that of the
liability) and the realised share of each component, prevalence and case counts per split, and the
realised genetic correlation of the shared pair.

**Pass criteria** (decided 17 November as "within 0.01 AUROC of the cross-fitted ridge PRS on additive
traits; at least half of the planted gap above it on the non-additive traits"; `e0_summary.py`). Per run
(a plain run's `results.json`, or a search's chosen trial through `best_trial_dir`), on the tuning set:
the additive traits (add_lo, add_hi, shared_a, shared_b) no further than 0.01 AUROC below the ridge; the
planted traits (dom, epi, hetero) at least half of the planted gap above the ridge, the planted gap
being ceiling − additive ceiling: the AUROC the planted structure puts beyond the best possible additive
model (about 0.05 at h² 0.2 + 0.1 and prevalence 0.2). At N = 13,199 the ridge sits far below the
additive ceiling, so in R² terms the bar is lenient (learning about a tenth of the planted variance
clears it), but no additive learner can clear it, because the planted codes have no marginal effect.
The first build's criterion, half of ceiling − ridge, was withdrawn after review: at this N that gap is
almost entirely the sampling limit of the additive part (a threshold of about 0.65, a liability R² of
about 0.1, a third of the heritability from 13,199 people), so no model could pass. The null trait
within [0.40, 0.60] is a leakage check I added, not a decided criterion (the band is about 14 SE wide).
The gate passes when every mixer among the runs has a run that passes every criterion (the plan's "for
each mixer"); an ablation flag that fails a trait shows as FAIL in its row and is information, not a
gate failure, while the mixer's headline run passes. The mixer is read from the run's `config.json`
(`model.mixer`; the chosen trial's for a search). Exit status 0 on a pass, 1 otherwise; an undetermined
criterion (no ridge file yet; a trait a run lacks) is not a pass; a `sim_log.json` without the additive
ceiling (the superseded first build) is refused.

**What the simulation does not do, and consequences.** The realised h² differs from the target by the
sampling of the noise variance (about ±0.005 at this n). The planted SNPs also carry additive effects
(they are drawn from the causal set), so each planted gap is the planted share alone. GBLUP's penalty
uses the liability h² on 0/1 labels: a reference, not a tuned comparator; the ridge PRS is tuned on the
tuning set as the real comparators are. The test rows have no simulated labels, and `sim_truth.npz`
holds the draws (causal indices, effects, planted loci, pairs and modifiers) and the per-person genetic
values of the simulated splits, not the liability thresholds, the component scalings or the
residualisation coefficients: labels for the test rows would need a replay of the build at the unlock,
and E6 builds its own datasets from its own trait specifications rather than reusing this one. GBLUP's
GRM is n_train² in float64 (1.4 GB here, 373 GB at the top rung), so the simulator as it stands is a
bottom-rung tool; a build at a larger rung must drop the GRM reference. The simulator is a standalone
script rather than an extension of `widedeepgp-pilot/simulated_control.py` (plan step 8): that script is
bound to the proxy dataset's third table and its own select/oracle/GWAS flow; what the two share
(`auroc`, the standardisation, owner-only files) the simulator takes from `ridge_common.py` here.

**Checks** (`test_e0.py`, 36 tests on a synthetic store, CPU): realised h², prevalence and ceiling; the
additive ceiling equals the ceiling for the additive traits and the AUROC of the additive component for
the planted ones, below the ceiling; the additive and planted variance components recovered by
regressing the genetic value on additive codes and the orthogonal dominance, product and switch codes;
the shared pair's genetic correlation; GBLUP against a direct solve; missing genotypes as the mean (zero
once standardised); the saved SNP statistics are the file the data module looks for, with its numbers;
the test split's rows are never read while every other row is; the log holds aggregates only; a source
with `regions`, an identifier absent from the store and a second build are refused; `NA` cells copied
verbatim; the read-out's criteria pass and fail each way, the ridge AUROC from a score file with a known
answer, the block-linear twin shown, the per-mixer verdict, a missing ridge file, a trait a run lacks and
a stale `sim_log.json`. The GPU build of the synthetic store reproduced the CPU build (identical labels;
genetic values within 5e-7).

**Build record.** The first build (17 November 2026, 01:37 to 01:44 UTC, GPU 3, 393 s; log
`runs/e0/build_hm3plus_n13199_sim.log`, aggregates only) used the spread specification and is
superseded; its references, the evidence for the decision above (tuning-set AUROC): add_lo ceiling
0.776, GBLUP 0.556; add_hi 0.853, 0.569; shared_a 0.777, 0.548; shared_b 0.771, 0.542; dom 0.771,
0.534 (additive R² 0.676); epi 0.769, 0.521 (0.674); hetero 0.767, 0.533 (0.558); null 0.500, 0.496;
realised genetic correlation of the shared pair 0.681. The dataset is to be rebuilt with the current
specification by `run_simulate_traits.sh` (detached, launched by the main session after moving the
superseded directory aside; before the chain's `lin` stage, which needs it), and the rebuilt dataset's
aggregates table (`sim_log.json`) is to be copied here as the bar every E0 run is read against.

## 11. The closed-form block ridge as built (step 7, 17 November 2026)

*Implemented in this folder: `ridge_common.py` (shared pieces), `make_folds.py`, `build_block_ridge.py`,
`check_block_ridge.py`, `fit_block_linear.py`, `make_warm_start.py`, `test_block_ridge.py` (14 tests on a
synthetic store). Products live under `data/datasets/ard30/<rung>/blocks/` (0700/0600) and are listed
with digests in `<rung>/blocks.json`; aggregates go to `results/<rung>/` and `results.md`.*

**Folds (`make_folds.py`).** A training participant's fold is their position in the ladder's permutation
modulo 5, not their row number in the rung's table. PGI's `make_ladder.py` writes a rung's table as the
parent table filtered to the first n of `default_rng(seed).permutation(sorted parent training IIDs)`, so
the rows keep the parent's order and "row mod 5" would give one participant different folds in different
rungs (checked on the real tables: no rung is a row-prefix of the next, and the top rung's table is not in
permutation order either). The permutation is rebuilt from `ladder.json` (parent, seed; the top rung is its
own parent), the rung's set is checked to be its prefix, and the fold is the position mod 5: balanced to
within one participant (2,640 x 4 and 2,639 at the bottom rung) and nested across rungs by construction.
Participants outside the training table (tuning, related White British, Asian, Black, and the test set
after the unlock) get a pseudo-fold from sha256("<IID>\t1"), its first 8 bytes as a big-endian integer,
mod 5 (3,959 to 4,060 per fold among the 20,000 tuning participants). `folds.npz` holds `iid_<split>` and
`fold_<split>` per split. The test split is written only with `--unlock_test`, and every such call appends
a line to `blocks/test_access.log`.

**The ridge (`build_block_ridge.py`).** Per rung, for each model m in {fold 0..4, full}: Z = dosages
standardised with the rung's cached training-set statistics (the data module's
`snp_stats_train_<hash>.npz`, found by recomputing its hash of the training rows and regions, computed and
cached under that name when absent so the deep runs reuse it; mean and SD over all training rows, missing
genotypes excluded; a missing genotype becomes the mean, z = 0; a SNP without variation has z = 0 and so
weight 0); R = Y − [1, C] B_m with B_m the least-squares coefficients of the 30 labels on the intercept and
the 12 covariates over the model's training rows (fold f excluded; every row for the full model); per Privé
block b, W_b = (Z_b'Z_b + λ I)^-1 Z_b'R at λ = M / h² with M = 1,258,200 and h² in {0.05, 0.15, 0.45}
(λ = 25.2, 8.39 and 2.80 million). This is the LDpred-inf estimator: on the correlation scale
β = (R_LD + (M / (N h²)) I)^-1 β̂ with β̂ = X'y / N, and multiplying through by N gives
(X'X + (M / h²) I)^-1 X'y, so h² is the SNP heritability the infinitesimal prior assumes, M / h² the inverse
prior variance per standardised SNP, and the three values bracket the S-LDSC estimates of these diseases.
Solved per Privé block, as section 6 says. One pass over the training rows accumulates per fold g the
grams G_g = Z_g'Z_g (float64, on the GPU) and Z_g'[Y, 1, C]; model m's gram is the sum over its folds, and
Z'R = Z'Y − (Z'[1 C]) B_m exactly, because the residualisation is linear, so the five fold models and the
full model come from the same pass. Per block and model one eigendecomposition (`torch.linalg.eigh`,
float64, cuSOLVER) gives the weights at every penalty, W = Q diag(1 / (Λ + λ)) Q' Z'R. Weights are stored
as float32 [M, 90] per model, column 3 j + p for disease j and penalty p.

Memory and passes: the five per-fold grams of one chromosome's blocks need up to 40 GB (chromosome 3,
whose largest block has 15,194 SNPs), more than the 48 GB card leaves beside the eigh workspace, so the
blocks of a chromosome are taken in greedy groups whose grams fit `--gram_gb` (20 GB) and each group gets
its own pass over its variant range; the store is block-major, so this reads every variant once, as one
pass per chromosome would. Reading: `GenoStore.read_q` in batches of 512 rows, decoded in numpy on at
most 8 threads and prefetched; standardisation and products on the GPU. Resumable per chromosome: the six
models' weights land in `blocks/partial/weights_chr<c>_<model>.npy`, a finished chromosome is skipped on
a restart, the per-model `ridge_weights_<model>.npy` are assembled at the end and the partial files removed.

Cost (RTX 6000 Ada, FP64 at about 1.2 TFLOPS): the eigendecompositions are the fixed cost, Σ_b n_b³ =
7.1e13 over the 431 blocks, 13.5 s for the 15,194-SNP block, about 27 minutes per rung for the six models
and the same at every rung; the grams scale with N, about 3 minutes at 13,199 and about 48 at 215,872
training rows. Noted, not adopted: a Cholesky factorisation per penalty would cost about a fifth of the
eigendecomposition for three penalties (2.9 against 13.5 s for the largest block); eigh was specified and
keeps the penalty list free to grow without a new factorisation.

**Scores and channels.** Every split is scored in one pass per split, each row with its fold model
(training rows: the model that excluded them; others: the pseudo-fold model): the block channels Z_b W_b
(float32 [n, 431, 90]) and their sum over blocks, the PRS at every penalty. The penalty per disease is the
one with the highest tuning AUROC of the PRS and nothing else, recorded in `ridge_manifest.json` with the
AUROC at every penalty and in `blocks/ridge_summary.csv` (columns `label`, `auroc_val`, `chosen_h2`,
`cases_val`, `auroc_val_h2_<h2>`; what `e0_summary.py` reads). `prs_<split>.npz`: `iid`, `labels`, `score`
[n, 30] at the chosen penalties, `score_all` [n, 90], `fold`, and `y` (added so the checks and comparators
need not re-read the tables). `channels_<split>.npy`: int8 [n, 431, 90], standardised per channel with the
mean and SD over the training rows' cross-fitted channels, clipped at ±3.5 SD, stored as round(x × 36)
(so ±126); `channels_manifest.json` carries the mean and SD per channel and the files' digests. Scoring
order: training first (its scale), tuning next (the choice), then the related White British and the
held-out Asian and Black groups, whose tables and store come from the sibling dataset `hm3plus_asian_black`
(same variants, checked). The test split is scored only with `--unlock_test` and logged. Channels are
written for every scored split (about 39 KB per participant: 4.2 GB at the bottom rung for the five splits,
12 GB at the top), to be deleted after use as section 6 says. `blocks.json` beside `store.json` lists every
product with its sha256; `store.json` is never touched.

**Checks (`check_block_ridge.py`).** On a built rung: (1) one random block's fold-0 and full weights
against an independent numpy float64 ridge from dosages read from the store (same statistics,
residualisation and penalties), and its stored int8 channels against the exact block scores of the fold-0
training rows; (2) the PRS at every penalty against the sum of the dequantised channels over all blocks on
2,000 tuning rows, with the share of rows that have a clipped channel; (3) per disease the out-of-fold
training AUROC against the tuning AUROC with bootstrap standard errors (200 resamples), and the pseudo-fold
variance ratio, the tuning-set score variance over the training out-of-fold score variance, near one when
the tuning scores carry the training scores' noise. Results in `results/<rung>/check_block_ridge.{csv,md}`.

**Linear comparators (`fit_block_linear.py`).** The linear twin: per disease a logistic ridge on the
431 × 90 cross-fitted channels (int8 / 36, standardised units) with the 12 covariates, standardised on the
training rows, entered unpenalised, fitted on the rung's training rows and its penalty λ chosen on the
tuning AUROC over {1e2, 1e3, 1e4, 1e5, 1e6}. The objective is Σ_i log(1 + exp(−s_i η_i)) + (λ / 2) Σ_channels
β², minimised by L-BFGS (scipy) along the penalty path from the strongest λ, started from the covariates-only
fit; the first version used scikit-learn's L2 logistic regression with one penalty on every coefficient,
and on the real bottom rung that twin scored 0.52 to 0.55 AUROC while the PRS-plus-covariates comparator
scored 0.70: a shared penalty strong enough to hold 38,790 noisy channels shrinks the 12 covariates with
them, and their signal (0.667 AUROC alone) is lost. scikit-learn has no per-coefficient penalty and no
offset, so the solver is written out (15 lines) and tested against scikit-learn where the two coincide
(every coefficient penalised; nothing penalised). The plain comparator: the same solver without a penalty
on the standardised cross-fitted PRS at the chosen penalty plus the covariates (the raw PRS is tiny, so a
penalised fit at scikit-learn's default would shrink its coefficient to nothing; found by the test on the
rare synthetic trait). Both are written as per-person tuning-set logits in the deep arm's npz format
(`iid`, `labels`, `logit_population`, `y`) under `<rung>/scores/blocks_linear/{logistic_ridge,ridge_prs}/
val_best.npz`, with aggregates in `results/<rung>/blocks_linear.csv` (the chosen λ, whether it sits at a
grid edge, iterations, training and tuning AUROC). The covariates enter both logits so that
`logit_population` is a population logit comparable to the deep arm's; the shared protocol's tuning-set
refit removes the difference in any case. Cost: the feature matrix is float32 in RAM (2 GB at the bottom
rung, 33 GB at the top) and never copied; about 55 s per disease at the bottom rung for the five penalties
and the two reference fits.

**Warm start (`make_warm_start.py`).** The dosage-form table for `--init_encoder`: the rank-d_e SVD
U_k S_k of the full model's [M, 90] weights (every disease and penalty), split per chromosome as
`tables.<i>` [L_i, d_e] float32, manifest kind "ridge" with `snp_code` dosage, `embed_dim`, `n_blocks`,
`seq_len`, and the source (rung, training-set digest, weights digest, penalties, singular values, variance
kept). `partition_digest` is computed with `genostore.datamodule.partition_digest` over the per-chromosome
compact block ids when that function imports (it does in the current working tree of DeepWideGP);
otherwise it is null and the source dict carries a TODO.

**Tests (`test_block_ridge.py`, 14, CPU or GPU).** On a synthetic store from DeepWideGP's generator (4,000
participants, 220 SNPs, 66 blocks of 4, 4 and 2 SNPs): the training folds follow the permutation, nest
into a sub-rung, and a table that is not a permutation prefix is refused; the pseudo-fold is the documented
hash, deterministic and balanced; every block's weights for three models equal `numpy.linalg.solve` on the
same standardised data; flipping fold 2's labels leaves the fold-2 rows' scores and the fold-2 weights
bit-identical while every other model and row moves; heritable traits predict (AUROC > 0.6 out of fold
and on the tuning rows), the null trait does not, a causal block's channel predicts and quiet blocks do
not; the int8 round trip (rounding within 0.5 / 36 SD, clipping exact, channels standardised on the
training rows, their sum equal to the PRS); a zero-variance SNP gets weight 0 and a missing genotype the
mean; the manifest hashes verify, `blocks.json` lists every product, every file is 0600 and `store.json`
is untouched; the test split is refused without `--unlock_test` and logged with it; the build resumes per
chromosome after an interruption and recomputes nothing when complete; a partition for other variants or
with non-contiguous blocks is refused; the warm start is the truncated SVD; the comparators' files have the
deep arm's format and predictive logits; the check script finds agreement.

**Bottom-rung check** (hm3plus_n13199, GPU 2, 17 November 2026): see `results.md`.

**Deviations from the contract.** (i) Fold = permutation position mod 5, not table row mod 5 (above).
(ii) `prs_<split>.npz` carries `y`; `ridge_weights_full.npy` and `ridge_summary.csv` are extra products.
(iii) `test_access.log` sits in `blocks/`. (iv) Channels are written for the held-out groups too. (v) The
simulated E0 dataset has its traits in no held-out table: build it with `--splits train val wb_related`.

## 12. Implementation (17 November 2026) and where it departs from sections 2 to 7

Everything of plan section 7 was built on 17 November 2026 (DeepWideGP `widening` from dcf07e1, tag
`pre-blocks` = 7dc184d marking the code before; this repository's `scripts/widedeepgp-blocks/` and
`scripts/widedeepgp-scaling/`; PGI `pgi/curves.py`) and the chain was started the same day
(`chains/run_all.sh`, records under `/mnt/sdg/robert/deepwidegp/runs/blocks/`). DeepWideGP `methods.md`
section 4.11 describes the code; the departures from the design above, each an engineering finding or a
simplification, are:

1. **No linear readout; the first layer may not train at small N.** During the implementation a linear
   readout of the table (a map from the table summed over all SNPs to the 30 logits, added to the head's)
   was tried and then removed at Robert's instruction on 17 November 2026: the model is exactly the design
   of section 2, with every prediction passing through the block MLP, the mixer and the head. The reason
   the readout was tried is a finding the reader of the results must know. On a synthetic dataset (4,000
   participants, 132 SNPs, six liability-threshold traits with two shared and two trait-specific causal
   SNPs each) the model of section 2 learned the three shared-SNP traits only after a phase transition at
   the fourth pass and never the two trait-specific ones (tuning AUROC 0.48 and 0.39 against ceilings 0.78
   and 0.97, in every variant tried: a tenth of the table learning rate, two mixer layers, shuffled batches
   without class weights, a narrower head); with one SNP per block it learned nothing at all. The cause is
   that a table starting at zero takes its gradient direction from a random, fast-moving downstream network,
   so its Adam steps do not accumulate; with the readout every trait was learned after one pass. Robert's
   position is that the model was never expected to work at a few thousand participants and that the
   question is whether it works at 200,000; the architecture is therefore kept as designed and the E0 gate
   and the top-rung pre-test decide. The ridge is contained in the first layer (with embed_dim 1 the pooled
   values are a ridge PRS's per-block sums), not as an exact special case of the whole model.
2. **One flag for the first layer's initialisation.** `--init_encoder <file>` loads tables (and whatever
   else the file holds) from a file whose manifest says what it is (`kind` masked or ridge), with
   `--freeze_encoder` to keep the loaded parameters fixed; the flags `--pretrain` and `--init_table` of
   sections 2 and 7 do not exist. `pretrain_blocks.py` writes the masked-pretraining file (each SNP of a
   participant hidden independently with probability 0.15, not "15% of a block's SNPs"; the hidden
   genotype class is predicted from the block's hidden layer through a per-SNP output vector of embed_dim
   entries); `make_warm_start.py` writes the ridge warm start (sections 6 and 11).
3. **One partition per file**; no `--block_level`. `--partition` names `partition.npz` or its folder
   (section 5's layout) or `uniform:<k>` for tests. The nesting of the finer partitions is checked when
   they are built (`check_partition.py --parent`), not at run time.
4. **The table's penalty was first an L2 term in the loss**, Σ T² / (2 N τ), with τ = `--table_var` searched
   log-uniformly in [0.01, 100] beside the learning rate, weight decay and dropout, and the table's own
   learning rate κ τ √B / √M (`--table_kappa` 0.1) without the class-weight factor of the old PRS path.
   Superseded the same evening by item 9.
5. **The frozen cross-fitted first layer** (the stacking ablation of section 7's table) is not
   implemented: a model reading cross-fitted block scores needs a third input mode. The question it
   served, whether end-to-end training adds anything over a linear use of the same inputs, is answered by
   P4 against the linear twin (`fit_block_linear.py`), and by the ridge warm start with and without
   `--freeze_encoder` (the frozen in-sample ridge first layer is leaky and is not run).
6. **FLOPs are derived, not logged.** `config.json` holds the counts (`blocks` section: partition and
   digest, tokens, parameters by module, embedding and non-embedding totals, the init file's digest);
   `scripts/widedeepgp-scaling/compute.py` turns them and the participants seen into compute.
7. **Model sizes as pinned by the chain:** transformer S1 to S5 = d 16 × 1, 32 × 2, 64 × 4, 128 × 6, 256 × 8
   (mixer parameters 3.3k, 25k, 200k, 1.19M, 6.32M); Mamba depths 1, 2, 6, 9, 14 at the same widths (3.9k,
   22k, 209k, 1.09M, 6.25M, each within a factor 1.3); heads of width 4 d (n_heads d / 16) so that the head
   does not outweigh a small mixer; batch 256 (128 × 2 for S4 and S5 at 2,000 or more tokens, where the S5
   transformer at 2,001 tokens peaks at 22.8 GiB and the matched Mamba at 25.2 GiB in bf16 at batch 128);
   20 epoch-equivalents at most, four validations per epoch-equivalent, patience 8, warm-up 200 steps and a
   cosine; 20 snapshots for the S3 row and the top-rung column, six elsewhere.
8. **The chain** (`chains/run_chain.py`, started detached by `chains/run_all.sh` from a frozen export of
   `widening`) runs, in order: the linear comparators of the simulated dataset and the bottom rung
   (`make_folds.py`, `build_block_ridge.py`, `check_block_ridge.py`, `fit_block_linear.py`,
   `make_warm_start.py`; the other rungs follow in the background on GPU 2); E0 as 18 searches on
   `hm3plus_n13199_sim` (3 partitions × 2 mixers × S1 and S3, and at S3 on prive431 the genotype code, the
   masked pretraining and the ridge warm start), then `e0_summary.py` as the gate (the chain stops on a
   failure); T0 (S1, S3, S5 once each at the top rung); E3-top, which chooses the grid's partition by the
   seed-mean tuning AUROC averaged over the two mixers; the ablations at S3 at the bottom and top rung
   (dosage, genotype code, pretraining, the top rung's pretraining applied at the bottom rung, ridge warm
   start), which choose the headline code and initialisation by the same rule at the top rung; the grid
   (S3 row and top-rung column first, then the rest, rungs smallest first); the table axis (embed_dim 4
   and 64); the shapes (32 × 16 and 128 × 1 for the transformer, 32 × 20 and 128 × 2 for Mamba); the
   partition sweep at the other rungs; and the scoring of every chosen model and seed retrain on the
   tuning set with per-person scores saved under the data directory. Every search uses GPUs 2, 3 and 4
   (`--tune_gpus 2,3,4`; Robert, 17 November: exactly these three); the seed-2 and seed-3 retrains run on
   GPUs 2 and 3 beside the next search. Decisions are written to `runs/blocks/decisions.json` with the
   numbers they rest on; a finished cell is skipped, so rerunning continues. Not in the chain: the
   scoring of the held-out groups (E5; they live in a second genotype store, so a scoring step that
   reuses a run's standardisation is needed), the h² scan (E6; its trait table is not simulated yet), the
   stratification checks (E7) and the test unlock, which is never automatic.
9. **The table's optimiser, after the first top-rung runs (17 November 2026, evening).** The first
   relaunch (no simulated gate; T0 then the grid from the top rung down) showed the plain model learning
   nothing from the genotypes at 215,872 participants: S1, S3 and S5 once each at fixed settings reached
   mean tuning AUROCs of 0.6610, 0.6624 and 0.6619 against 0.6627 for the covariates alone (here and below: an
   unpenalised logistic regression on the 12 covariates fitted on the rung's training split and scored on the tuning
   set, never a network's covariate-only output), and a 12-trial
   search at S1 0.6626. The curves showed Adam noise accumulating in the table at S3 (training loss rising
   after the first third of the run while the table's sum of squares grew a thousandfold) and the table's
   gradient collapsing at S5. Robert had the chain stopped and its outputs wiped, and three changes
   applied before a second relaunch: (a) the table's AdamW group gets a large epsilon (`--table_eps`
   10⁻⁷, about eight times the measured per-entry gradient of a null SNP, 1.3 × 10⁻⁸ at batch 256 on the
   top rung) and a per-step weight decay (`--table_decay`, default 1, searched 0.3 to 3) in place of the
   loss penalty Σ T² / (2 N τ) of item 4, which was far too weak; `--table_var` no longer exists;
   (b) each block's sum is divided by the square root of its SNP count, so the first layer's output has
   one scale whatever the partition; (c) the table's pace κ drops to 0.03 and is searched (0.01 to 0.03)
   in place of τ, the pruner may act only after eight validation passes (`--tune_min_validations 8`) and
   early stopping waits sixteen (`--stop_patience 16`), 20 epoch-equivalents at most. Searches take two
   to three times longer. The programme (item 8) is otherwise unchanged; it starts again from the top
   rung, and the first question it answers is whether these settings let the first layer learn at all.
10. **The recipe of Sigurdsson et al. (2023) and the ridge warm start as the default (17 November, night).**
    Robert asked for the ridge warm start to be every run's starting point and for the training recipe of the
    closest published system, the genome-local networks of Sigurdsson et al. (2023, Nucleic Acids Research; UK
    Biobank, 383,000 training participants, 662,000 SNPs): learning rate 10⁻⁴, weight decay 10⁻³, dropout 0.5,
    batch 64, 40% of the SNPs set to missing on every training step (`--input_mask 0.4`, the rest rescaled), a
    plateau schedule (×0.2 after ten validation passes without improvement), early stopping after sixteen
    passes once a burn-in of 2,000 optimiser steps is done. These are `main_multilabel.py`'s defaults now and
    the chain passes them explicitly; the table's epsilon was re-measured under them (3.3 × 10⁻⁸ per null
    entry at batch 64 with masking; `--table_eps` 2.5 × 10⁻⁷). The per-cell search tunes the table's pace and
    decay only (`--tune_space table`, six trials); the network's settings stand at the recipe's. In the chain
    the comparators of every rung are built first (the top rung on one GPU, the others in turn on a second)
    because every cell starts from its rung's warm start; the no-initialisation ablation at S3 runs meanwhile,
    and "no initialisation" replaces "ridge" in the ablation list. DeepWideGP `methods.md` section 4.11 carries
    the sources and the four other systems read for comparison (Bellot et al. 2018; van Hilten et al. 2021;
    G2PT and SNPBERT, preprints), none of which trains a free table over a million SNPs.
11. **Synthetic-store bands.** The smoke run of `scripts/multilabel_synthetic.sh` keeps the published
   SNP-token model (it now passes `--input_mode snps --lr_schedule constant --warmup_steps 0`); the
   blocks-mode arms (`blocks_transformer`, `blocks_bimamba1`, five seeds each, `uniform:5` blocks) have
   their own rows in `multilabel_synthetic_seeds.csv`, and `check_synthetic_run.py` picks a run's arm from
   its configuration.
12. **The objective: plain cross-entropy at natural prevalence (Robert, 18 November 2026).** Every run of the
   chain draws uniform shuffled batches and uses no class weight (`--sampler shuffled --gamma 0`, explicit in
   `flags()`; the same values are now `main_multilabel.py`'s defaults), so each disease's term is its mean
   cross-entropy and the loss their mean over the 30 diseases: the objective of the published DeepGP, of the
   linear comparators and of PRSformer (Dibaeinia et al., 2025). The rotation of disease-balanced batches with
   its sampling-rate correction and the square-root class weight, the objective of every run before, including
   the v2 and v3 reference runs, is one paired ablation cell at the top rung
   (`transformer_S3_prive431_dosage_ridge_hm3plus_rotation`, from the warm start, beside the grid's cell of
   the same size). Reasons (`DeepWideGP/methods.md` section 5.2): at batch 64 and prevalences of 3.7% and above
   only 9% of uniform batches lack a case of the rarest disease, so the rotation solves a problem of the
   published batch of 21 that no longer exists, and its correction costs a fifth of the draws (Kish); the class
   weight departs from maximum likelihood, inflates a rare disease's loss curvature by 1.11 to 3.82 against a
   table weight decay shared by all diseases, and pulls the warm-started table away from the population-weighted
   ridge it starts from; the one credible deep-learning win over LDpred2 (PRSformer, 3.8 million people) used
   plain summed cross-entropy at natural prevalence and found focal, task-uncertainty and averaged losses worse,
   while the studies that balanced cases and controls (Kelemen et al., 2025, oversampling; Zhu et al., 2026 and
   Yang et al., 2025, undersampling controls) either lost to the linear baseline or weakened it. The
   no-initialisation ablation that had started under the old objective was stopped after 1.5 hours and is rerun
   under the new one.

13. **The comparators in two phases, so that the warm start does not wait for the linear twins (18 November
   2026).** The chain had T0 and a rung's grid wait for the rung's whole comparator pipeline, whose last two
   steps (the ridge check and the per-disease logistic ridge twin, about 19 minutes per disease at the top rung)
   run on the CPU and are not needed for the warm start; two GPUs sat reserved and idle for hours while the top
   rung's twin ran. `chains/run_chain.py` now runs each rung's folds, block ridge and warm start as one GPU phase
   (the GPU reserved for the build, released when the phase ends) and the ridge check and the linear twins as a
   CPU phase beside the training; T0 and the grid wait for the warm-start file only, and every step is skipped
   where its product exists (`blocks/folds.npz`, `blocks.json`, `blocks/init_ridge_d16.pt`,
   `results/<rung>/check_block_ridge.md`, `results/<rung>/blocks_linear.csv`). `fit_block_linear.py` resumes per
   disease (`blocks/blocks_linear_partial.npz`, removed when its table is written). The chain was relaunched
   with these changes in the night of 17 to 18 November (`chain.log`); the two twin fits then running
   restarted (the block ridges and the ridge checks were kept), and the no-initialisation ablation search
   resumed its finished trials. At that relaunch the chain still ran the two no-initialisation ablation searches
   before T0, an order chosen when the warm start was a day away; a second relaunch minutes later put T0 first,
   each of its three runs on a GPU of its own as one frees (`free_gpus(exclusive=True)`), and moved the
   no-initialisation cells to the side studies beside the other ablations. The transformer no-initialisation
   cell's first trial had reached 8 epoch-equivalents at the covariate floor (mean tuning AUROC 0.6628 against
   0.6627 for the covariates alone) and resumes there from its checkpoint. A third relaunch (01:30Z) moved the
   lower rungs' GPU phases after T0 as well: their first step, the SNP statistics over the rung's training rows,
   runs on the CPU for tens of minutes while the GPU sits reserved, and their warm starts are needed only when the
   grid reaches those rungs, days later; T0's second run now takes that GPU at once. Single runs (T0 and the seed
   retrains) pass `--resume` when a stopped attempt left a checkpoint. The fitted models and the comparators do
   not change: the same steps run, in another order.

14. **T0 at the covariate floor, and why: the warm start never reached the network (18 November 2026).** The three
   warm-started pre-test runs ended at mean tuning AUROC 0.6629 (S5, 7.8 epoch-equivalents), 0.6633 (S3, 15.5) and
   0.6631 (S1, 8.5) against 0.6627 for the covariates alone, while the linear twins on the same block information
   reached 0.6732 (logistic ridge on the channels) and 0.6777 (ridge PRS plus covariates), ahead of the best deep run
   for every disease (median 0.011, up to 0.057 for E66). Two faults, both in how the ridge warm start meets the
   network. (i) Scale: the ridge weights are of order 1e-5 in standardised-dosage units (table entry RMS 3.6e-5), so a
   block token formed from them has RMS 1.0e-4, four orders of magnitude below the block and chromosome identity
   embeddings (N(0, 1)) it is added to; the block MLP cannot read it. (ii) Erasure: the table's learning rate,
   κ √B / √M = 2.1e-4 at κ 0.03, is six times the entry RMS, so Adam's near-sign steps overwrite any entry with a
   gradient within a few steps, and AdamW decay 1.0 shrinks the whole table by e every 4,674 steps (1.4
   epoch-equivalents); S3's final table has cosine −0.0007 with its warm start. Every warm-started run was therefore a
   from-scratch run, and from scratch sits at the floor (the no-initialisation S3: 0.6628 after 8 epoch-equivalents).
   The grid's six-trial search (κ 0.01 to 0.03, decay 0.3 to 3) could not keep a warm start either, so Robert paused
   the grid after its first cell had started (the lower rungs' comparator builds and the twin fits ran on). The fix:
   `make_warm_start.py` scales the table so that the block tokens have unit RMS over a sample of 2,048 training
   participants (`--token_rms 1`; the factor is recorded in the manifest: 9,924 at the top rung, where the unscaled
   token RMS was 1.01e-4); `tune_optuna.py` gives a warm-started table its own search range, κ 3e-4 to 1e-2 and decay
   1e-3 to 3e-2 (log-uniform), the from-scratch range staying 0.01 to 0.03 and 0.3 to 3; the chain's single-run
   defaults become κ 1e-3 and decay 0 (the table lr 7e-6 per step against entries of order 1 after scaling). Two
   diagnostics at S3 from the rescaled top-rung warm start, ten epoch-equivalents each, run before the grid is
   relaunched (`runs/blocks_diag/`, 11:15Z): the table frozen (`--freeze_encoder`: does the network read a ridge
   table at all? the twin is the yardstick) and the table trained at κ 1e-3 without decay (does end-to-end training
   add to it?). The unscaled warm starts and the three T0 runs are kept beside the new ones as `_unscaled` records.

15. **The warm start leaks the training labels; the first layer becomes fold-routed (18 November 2026).** The two
   diagnostics of item 14 (S3, the in-sample ridge table rescaled to unit-RMS tokens) read the table at once: mean
   tuning AUROC 0.660 and I10 0.704 at the first validation (0.25 epoch-equivalents), above the floor and above
   anything T0 reached. From then on both fell, the frozen table as fast as the trained one: 0.650, 0.623, 0.623,
   0.626 at 0.5, 0.75, 1.0 and 1.25 epoch-equivalents (I10 0.689, 0.644, 0.656, 0.657), levelling at 0.62 to 0.63
   (stopped by hand at 2.0 and 3.25). The cause, measured: the full-training ridge PRS at the chosen penalty has
   in-sample AUROC 0.957 on average over the 30 diseases (I10 0.849, I50 0.999) on 4,096 training participants and
   0.568 on 4,096 tuning participants (the cross-fitted PRS 0.568 and 0.564 on the same rows). A training
   participant's tokens therefore carry their own labels, and the network learns to read them off, a shortcut worth
   nothing out of sample. The remark of item 5, that a frozen in-sample first layer is leaky, holds for the
   trainable one as well. Robert's decision: the first layer is **fold-routed**. The table has one copy per fold of
   the block ridge (five); a training participant is pooled with the copy fitted without them (`blocks/folds.npz`,
   `fold_train`), and every other participant with the copy of their pseudo-fold (a seeded hash of the IID modulo
   five, the comparators' rule), so no participant's tokens were fitted on their label. The copies are the fold
   ridges' weights [M, 90] in the d_e-dimensional basis of the full ridge's right singular vectors (so that the five
   copies share one coordinate system for the network; the basis is 90 × 16 aggregate numbers), scaled by one
   factor so that the cross-fitted tokens have unit RMS over a sample of training participants (10,200 at the top
   rung). Frozen, this first layer is the stacking ablation of item 5; trained, each copy learns end to end from its
   own fifth of the rows on top of a ridge from the other four fifths. Code: DeepWideGP `--folds` (the data module
   reads each participant's fold; `BlockEncoder(n_folds)` pools by copy; the init file's fold count is checked;
   `n_folds` in the run's settings), `make_warm_start.py` (fold copies by default, `--in_sample` for the record),
   the chain's `flags()` (`--folds` for every warm-started cell). Quick test, 12:03Z, S3, ten epoch-equivalents,
   from `runs/blocks/code_11c24e3`: `runs/blocks_diag/diag_S3_folds_frozen` (copies fixed) and `diag_S3_folds_warm`
   (κ 10⁻³, no decay). The in-sample files are kept as `init_ridge_d16_insample.pt` (scaled) and
   `init_ridge_d16_unscaled.pt`; T0 and the item-14 diagnostics stand as records of the in-sample warm start.

16. **A pre-test of the haploblock first layer on chromosome 22 (18 November 2026).** Robert proposed replacing the
   per-SNP table with one vector per common haplotype of about 100,000 short haploblocks: not for new signal but for
   fewer, orthogonal parameters within a block (haplotype indicators are mutually exclusive where dosages are
   collinear). Before the two-week build (genome-wide phasing, a haplotype store, an input mode, a haplotype warm
   start), one chromosome answers two questions: the arithmetic (how many blocks, how many common haplotypes each, how
   many first-layer parameters against the 1,258,200 SNP profiles) and the signal (does a closed-form ridge on
   haplotype counts beat the ridge on SNP dosages out of sample, on the same SNPs, participants, folds, residualised
   labels and shrinkage grid; a ridge on both bounds the haplotype-specific signal). The ridge of a parametrisation is
   what a network on it can at best match at this N (items 14 and 15), so a haplotype ridge that does not beat the
   SNP ridge rules out the layer. Design, decisions and the pre-stated decision rule: `../widedeepgp-haploblocks/methods.md`
   (SHAPEIT5 5.1.1 on the training and tuning rows' hard calls, the test split never read; haploblocks nested in the
   Prive blocks by a coverage rule, f = 1%, c = 95%, with a nine-cell sensitivity grid; penalties λ = M_class / h² on a
   five-point h² grid chosen on the out-of-fold training rows; paired participant bootstrap on the tuning set and the
   out-of-fold training rows). Launched 15:36Z as two detached chains (`run_phase_chr22.sh`, `run_analysis_chr22.sh`);
   results in that folder's `results.md` when they arrive. Section 10 puts phased features out of scope on the
   ground that the server had no phasing tool; that ground is gone (bioconda installs SHAPEIT5 and bcftools), and
   whether a haplotype data product is worth building is what this pre-test decides. **Outcome (19 November 2026,
   `../widedeepgp-haploblocks/results.md`).** Phasing took 6.8 h on ten threads (68 CPU-hours; about 4,700
   genome-wide). The median haploblock has 14 SNPs and 12 common haplotypes, not 3 to 10, yet 0.74 features per SNP
   (about 926,000 vectors genome-wide), so the table would have been affordable. The haplotype ridge carries 23% less
   additive signal than the SNP ridge on the same blocks (pooled r with the residualised label −0.0015 [−0.0021,
   −0.0008] on the 215,872 out-of-fold training rows, 22 of 30 diseases below zero, none above; ΔAUROC −0.00008
   [−0.00014, −0.00002]), and the ridge on both feature sets adds at most 1.4% of the SNP ridge's correlation. Under
   the pre-stated rule the haploblock first layer is not built; the two sensitivity cells were stopped at Robert's
   request. A second comparison under clumping and thresholding (every haplotype as a pseudo-SNP, the PGI arm's
   grid) was run at Robert's request (that methods' section 9; results in its `results.md` section 5).

17. **Block weighting at the readout, and a table that never moved (18 November 2026, afternoon).** The fold-routed
   quick test: the frozen copies rose for two epoch-equivalents and plateaued at 0.671 to 0.672 (best 0.6722 at 7.0,
   budget 10; I10 0.713), the first LD-block run above the covariates, 0.001 below the logistic ridge twin on the same
   channels; the two trained-copy runs (κ 10⁻³ and 10⁻²) sat on the frozen trajectory to the fourth decimal. The
   trained tables, compared with their warm start, had moved by 1% of their own scale after 14,700 steps even at
   κ 10⁻²: the table's Adam epsilon (2.5 × 10⁻⁷, set for a table at scale 10⁻⁴ starting from zero) throttles a table at
   scale 0.35, so those runs did not test end-to-end training at all. Robert then asked whether the 431 blocks have no
   weights of their own at the readout. They did not: the tokens pass a LayerNorm, which fixes each token's length, and
   are averaged, so a block cannot raise its share and every disease reads one 64-number mean (the published DeepGP
   pooled its 22 chromosome vectors with a learned attention; PRSformer found mean pooling over its variants
   catastrophic). Two readouts added to blocks mode (`--final_pool`): `atten`, one learned query per disease over the
   431 block tokens, each disease's logit from the shared head applied to its own pooled vector (uniform attention at
   the start, so it begins as the mean; about 4,000 parameters at S3); `flat`, PRSformer's readout, the 432 tokens
   flattened into one linear map to the 30 diseases (829,000 weights at S3, under the network's weight decay). Three
   runs from the fold-routed warm start at S3, ten epoch-equivalents each, single trainings with the recipe's settings
   (no search): the copies frozen with `atten` and with `flat`, against the frozen `mean` run above, and the copies
   trained with `mean`, κ 10⁻² and epsilon 10⁻⁸, so that the table can move. The two throttled trained runs were
   stopped at 7.5 and 6 epoch-equivalents and kept as records.

18. **The readouts, the trained table and the recipe, judged (18 November 2026, evening; tables in `results.md`).**
   Attention over the 431 block tokens is indistinguishable from the mean (within 0.0003 at every validation; best
   0.6723 against 0.6722): with LayerNormed tokens, a disease-specific weighting of whole blocks adds nothing the shared
   head did not have. The flat readout gains on the common diseases (I10 +0.006, E78 +0.007 over the mean) and loses on
   the rare ones (its gain correlates with the log case count, r = 0.42), for +0.0006 on the mean and a worse tuning
   cross-entropy: 829,000 weights overfit the rare diseases at 216,000 participants. The trained copies with the
   epsilon that lets the table move (κ 10⁻², ε 10⁻⁸) tracked the frozen run to three epoch-equivalents and then fell,
   to the covariate floor by the seventh (0.6632, stopped early), while the frozen run with the same mixer held 0.670
   to 0.672. The table's distance from its warm start grew by 1.4%, 2.1%, 3.2% and 4.3% of its scale at steps 6,813,
   10,000, 14,678 and 21,544: linearly in steps after the first epoch, where an aimless walk grows as the square root.
   Once each participant comes round again their label pushes the same entries the same way, so the table memorises
   the training labels and the tuning AUROC falls as it does. A step small enough to avoid this in ten passes cannot
   move the table either (κ 10⁻³: 0.4%). At this rung, then, end-to-end training of the full per-SNP table is either
   idle or harmful, and the first layer that works is the cross-fitted ridge, frozen: the stacking ablation of
   section 5 rather than the end-to-end model of section 2, which has now failed at the top rung in every form tried
   (from zero, warm-started and erased, warm-started and leaking, warm-started and throttled, warm-started and free).
   The recipe check (`run_diag_folds_recipe.sh`: input mask 0, dropout 0.2, everything else the same) gave the best
   run so far, 0.6735 at ten epoch-equivalents and still rising, equal to the twin (0.6732), above the covariates for
   all 30 diseases and above the masked frozen run for 24 of 30: masking 40% of a fixed linear first layer's inputs
   puts noise of a token's own size on the training tokens that the clean tuning tokens never see. **Launched** to put
   seed spreads under these differences (`run_diag_seeds.sh`, GPUs 3 and 4, 21:55Z, about seven hours): seeds 2 and 3
   of the no-mask frozen mean run, and the untested combination of the no-mask recipe with the flat readout (seeds 1
   and 2). **First round (19 November, 00:55Z):** seed 2 of the frozen mean reproduces seed 1 (0.6737 against 0.6735;
   per-disease differences 0.0010 on average, 0.0040 at most), so the seed spread of this configuration at the top
   rung is about 0.0002 on the mean and its 0.0003 lead over the twin is noise. The flat readout under the no-mask
   recipe overfits (0.6731 at epoch-equivalent 3, 0.6617 at 7, stopped early): the 829,000-weight head needs the
   heavy masking, under which it reaches 0.6728. On this evidence the grid's readout is the mean and its recipe the
   no-mask one; only the first layer is open. **For Robert:** whether the ladder's deep arm is this frozen stacked model, or a first layer with far fewer
   trainable parameters (one gain per SNP multiplying its ridge profile, started at one: 1.26 million numbers whose
   gradient sums over the 16 dimensions, learned per-locus shrinkage; a day's work), and which recipe and readout the
   grid runs.

19. **The selected-SNP from-scratch arms (19 November 2026; plan agreed with Robert in the morning).** The question
   is whether the per-SNP table learns anything by gradient descent on real data once the conditions of the synthetic
   probe (item 1) are restored: far fewer SNPs, chosen by a linear step on the training labels; the table's optimiser
   un-handicapped; a non-zero start where one is available. Every real run so far (items 14 to 18) had all 1,258,200
   SNPs, the throttled optimiser of item 9 and a zero or ridge start, and the classic architecture on C+T SNPs (the
   comparator's two top-rung runs) has no per-SNP parameters at all, so the question is open. Robert's design, after
   discussion: GWAS selection is allowed as preprocessing; the LD blocks, the 16-wide dosage table and the S3 mixer
   stay; no ridge initialisation. Haploblock inputs were considered and dropped: on chr22 the haplotype ridge carries
   23% less signal than the SNP ridge (item 16's pre-test, 19 November), and chr22's haploblocks have a median of 12
   common haplotypes per 14-SNP block, so they would not have reduced the table either.
   - **SNP sets**, from the 30 top-rung REGENIE GWAS (PGI arm, N 215,872): each SNP's smallest p over the 30. (a)
     `p01any`: p < 0.01 in any disease, not clumped: 448,488 SNPs (35.6% of the store; PRSformer's rule, p < 10⁻²
     union over its 18 diseases, 137,245 variants at N 3.8 million, pruning tried and found harmful). (b) `p01anyct`:
     the same, clumped once on the smallest p at r² 0.1 within 250 kb on the PGI LD panel, as the comparator's
     datasets: 95,642 lead SNPs (measured before the build). For reference, p ≤ 0.05 in any disease keeps 82% of the
     SNPs and 179,566 leads, p < 0.001 keeps 99,147 SNPs and 23,555 leads. Stores and datasets `<set>_n215872` from
     the comparator's `build_datasets.py` (new options `--p_max`, `--no_clump`, `--source`; the clumped store is
     built from the unclumped one, a subset). Privé partitions `prive_<set>` from the full reference table
     (`check_partition.py --from_reference` now takes a store of selected SNPs); blocks with no selected SNP vanish.
   - **Two arms**, one training each, seed 1, no search, from a frozen export of DeepWideGP `widening`
     (`runs/selected/code_<commit>`): **A** `sel_p01anyct_S3_zero`, the clumped set, table from zero; **B**
     `sel_p01any_S3_pretrained`, the unclumped set, the table, identity vectors and block MLP hidden layer started
     from one pass of masked-SNP pretraining (`pretrain_blocks.py`, 15% of SNPs hidden, AdamW 1e-3, no labels; its
     first run at scale) and then trained end to end. Pretraining is confined to the unclumped set on purpose: a
     masked lead among leads at r² < 0.1 within 250 kb has next to nothing to be predicted from. The arms therefore
     differ in two things at once; the run that separates them, the unclumped set from zero, is one command on the
     same store and is queued for the next free GPU rather than run now (Robert, 19 November).
   - **Settings.** S3 transformer (d 64, 4 layers, 4 heads), mean readout, the no-mask recipe of item 18 (input mask
     0, dropout 0.2), lr 1e-4, weight decay 1e-3, batch 64, bf16, plateau schedule, plain cross-entropy at natural
     prevalence, ten epoch-equivalents, validation every quarter. The table: κ 0.1 (step κ√B/√M: 1.2e-3 per entry per
     step on 448,488 SNPs, 2.6e-3 on 95,642), Adam ε 1e-8 (the ordinary value, not item 9's throttle), decoupled
     decay 0.01 (an e-fold in about 25 passes: nominal). Entries: 7.2 million and 1.5 million against 20 million.
   - **Yardsticks**, all on the tuning set: covariates alone 0.6627; classic-30 0.6627 (no per-SNP parameters); the
     cross-fitted block ridge PRS plus covariates and the logistic-ridge twin fitted on each set's own SNPs
     (`build_block_ridge.py`, `fit_block_linear.py` on the new datasets; products under `<dataset>/blocks/`,
     aggregates in `results/<set>_n215872/`); the full-set ridge PRS 0.6777 and the frozen stacked model 0.6735.
     Reading: an arm at the floor means the table does not learn from real signal at this N under these conditions
     either; an arm between the floor and its own ridge means it learns, and the gap to the ridge is the cost of
     gradient descent; an arm at or above its ridge is the first end-to-end result worth a grid. The curve per
     epoch-equivalent is reported, as always; a rise then a fall is the memorisation signature of item 18.
   - **Chain** `chains/run_selected.sh` (detached; log `runs/selected/chain.log`): build (a) from the main store
     (about three hours), partition, folds, block ridge (GPU), pretraining (GPU), arm B; build (b) from store (a),
     partition, folds, block ridge, arm A; the two twins on the CPU; every step skipped by its product. GPU steps take
     a GPU among 2 to 4 only when at least two are free. Results: `results.md`, "Selected-SNP from-scratch arms".

## References

Kaplan J, McCandlish S, Henighan T, et al. Scaling laws for neural language models. arXiv
2001.08361 (2020). Daetwyler HD, Villanueva B, Woolliams JA. Accuracy of predicting the genetic
risk of disease using a genome-wide approach. PLoS ONE 2008;3:e3395. Lee SH, Goddard ME, Wray NR,
Visscher PM. A better coefficient of determination for genetic profile analysis. Genet Epidemiol
2012;36:214. Wray NR, Yang J, Goddard ME, Visscher PM. The genetic interpretation of area under the
ROC curve in genomic profiling. PLoS Genet 2010;6:e1000864. Privé F, Arbel J, Aschard H,
Vilhjálmsson BJ. Identifying and correcting for misspecifications in GWAS summary statistics and
polygenic scores. HGG Advances 2022 (the HapMap3+ map and its LD blocks).
