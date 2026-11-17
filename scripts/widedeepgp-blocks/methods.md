# Methods: WideDeepGP on LD blocks, trained end to end (the deep arm of the scaling study)

*Status: design agreed with Robert on 16 November 2026, amended on 17 November (three seeds per
cell; AUROC as the metric family; the table width as a second size axis; the primary estimands,
decision rules and pre-registered null of section 8; the capped-asymptote bound on the crossover and
the E6 h² scan with its data-limitation projection; then the last open items closed: S1 stays one
layer, no simulated genomes, packages installed as needed, every other default accepted). No
decision is open; nothing implemented. This
document is the durable description of the design and its decisions. The working plan with the
implementation sequence is `/home/robert/.claude/plans/we-re-just-talking-here-crispy-toucan.md`
(revision 4). It will be brought up to the implementation as the code lands; results will go in
`results.md`. The model's own methods document is `DeepWideGP/methods.md`, whose status note of 16
November points here.*

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
   logits. The loss (class-weighted, sampling-corrected cross-entropy of `DeepWideGP/methods.md`
   section 5), the rotation sampler, validation on the whole tuning set, selection on mean tuning
   AUROC and the search over learning rate, weight decay and dropout (12 trials) are unchanged.
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
finer ones of about 2,000 and about 5,000 blocks, built with bigsnpr `snp_ldsplit` inside each Privé
block on the PGI arm's LD matrices (`data/pgi/ld/bigsnpr/chr*.corr.rds`, 20,000-person panel).
Partitions are label-free (built from LD, genotypes only, as every LD reference is), recorded by
digest in every run's `config.json`, and chosen on the tuning set. Following Kaplan, the block count
is not an axis of the joint law: a one-factor sweep at the fixed model size across the rungs gives
the optimal block count as a function of participants, B_opt(D), and the full-sample optimum as its
top slice; the top-rung slice runs first and chooses the grid's partition. Recombination-hotspot
haploblocks (about 38,000 genome-wide, about 30 HM3+ SNPs each) are too fine for this SNP set and
cohort and are not used.

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
| Abl | Genotype code; pretraining (and the all-genotypes pretraining variant at the bottom rung); warm start; frozen first layer; at S3, bottom and top rung | about 11 | 60 |
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
the 30 binary diseases only, no continuous traits; the superseded plans deleted; nothing implemented
yet.

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

## References

Kaplan J, McCandlish S, Henighan T, et al. Scaling laws for neural language models. arXiv
2001.08361 (2020). Daetwyler HD, Villanueva B, Woolliams JA. Accuracy of predicting the genetic
risk of disease using a genome-wide approach. PLoS ONE 2008;3:e3395. Lee SH, Goddard ME, Wray NR,
Visscher PM. A better coefficient of determination for genetic profile analysis. Genet Epidemiol
2012;36:214. Wray NR, Yang J, Goddard ME, Visscher PM. The genetic interpretation of area under the
ROC curve in genomic profiling. PLoS Genet 2010;6:e1000864. Privé F, Arbel J, Aschard H,
Vilhjálmsson BJ. Identifying and correcting for misspecifications in GWAS summary statistics and
polygenic scores. HGG Advances 2022 (the HapMap3+ map and its LD blocks).
