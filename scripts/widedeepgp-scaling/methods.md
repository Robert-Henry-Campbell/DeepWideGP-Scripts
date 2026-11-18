# Methods: estimands, curve fits, compute accounting and figures of the scaling study

*Status: written 17 November 2026 with the code (plan section 7, step 9) and revised the same day
after an independent review (the real ablation flags, the twins' rows, the null's requirements, the
test split's lock and log, the raw PGI key, the shared M_e, the duplicate guards, a faster exact
resampled AUROC); no deep run exists yet, so everything here has been exercised on synthetic inputs
only, except the protocol's reproduction of the PGI arm's tuning metrics. The estimands, decision rules and
the pre-registered null are those of the design document, `../widedeepgp-blocks/methods.md`
section 8 (plan section 6), and are not restated; this document records how they are computed, the
choices made in computing them, and what each file produces. Results will go in `results.md`.*

## 1. Files

| File | Does | Writes |
|---|---|---|
| `PGI/pgi/curves.py` (repository PGI, branch `widening`) | Every estimand: P1 to P4, the capped-asymptote bound, the secondary fits, eligibility, Holm's adjustment, the null; the shared protocol and the participant bootstrap on per-person scores | DataFrames, one per result (section 4) |
| `compute.py` | FLOPs per participant from a run's `config.json`; cumulative training compute per snapshot and per validation pass | `<out>/<rung>__<cell>__seed<s>_compute.csv` |
| `fit_curves.py gather` | The tidy table from the deep runs' `scores_<split>_summary.csv` and the PGI arm's `tune_metrics.csv`, with N_eff, h² and K per disease and rung; the manifest of per-person score files | `tidy.csv`, `scores_manifest.csv` |
| `fit_curves.py fit` | `curves.analyse` on the table (point estimates) or on the table plus the per-person scores (shared-protocol values and intervals); `--split test` only with `--unlock_test`, every such call appended to `<out>/test_access.log` | one CSV per result table |
| `make_figures.py` | The figures of the design document's section 8, through `PGI/pgi/report.py` (89 or 183 mm, ≥ 7 pt, data CSV and legend beside each) | `<name>.png/.pdf`, `<name>_data.csv`, `<name>_legend.md`, `ablations.md` |
| `test_scaling.py`; `PGI/tests/test_curves.py` | Tests on synthetic inputs (section 7) | |

Run from the Scripts repository with `/home/robert/envs/widedeepgp/bin/python`; `PGI_REPO` overrides
the PGI location (default `/mnt/sdg/robert/deepwidegp/PGI`).

## 2. The tidy table (`fit_curves.py gather`)

One row per disease, rung, model family, seed and split. Columns: `disease`, `rung`, `n_train`,
`n_eff`, `h2_50`, `h2_50_se`, `h2_liab`, `K`, `family`, `seed`, `split`, `auroc`, `auroc_cov`,
`delta_auroc`, `delta_r2`, and for the deep cells `mixer`, `d_model`, `e_layers`, `embed_dim`,
`partition`, `n_blocks`, `n_mixer_params`, `n_non_embedding_params`, `n_params`, `run_dir`,
`score_path`.

- **Deep rows** come from every `scores_<split>_summary.csv` found under the run roots, with the
  run's `config.json` beside it (the chosen trial's folder, as `score_snapshots.py` writes it). The
  family name is built from the configuration, so runs of one cell type share it:
  `deep:<mixer>:d<d_model>x<e_layers>:de<embed_dim>:<partition>`, plus each ablation off its
  default as the model records it (`train.py`: `model.init_encoder`, `model.freeze_encoder`,
  `blocks.snp_code`): `:snp_code=genotype`; `:init=<stem of the --init_encoder file>` (the chain's
  `init_ridge_d16` for the ridge warm start, `pretrained_<partition>_<code>_e<d_e>` for masked
  pretraining), with `@<rung>` appended when the file comes from another rung's dataset directory
  (the "preall" cells, which use the top rung's pretraining); and `:frozen` for `--freeze_encoder`.
  Every split scored is gathered (`val`; the held-out groups for the transfer figure; `test` only
  after the unlock). Two runs whose rows would land in one cell (family, rung, seed, split), such as
  a search trial gathered beside its seed-1 retrain, are refused by name; `curves.seed_summary`
  refuses duplicate (disease, rung, family, seed) rows again at fit time, and cells without `n_eff`
  or `h2_50` (a disease missing from the power or S-LDSC tables).
- **Linear rows** come from `PGI/results/<rung>/tune_metrics.csv`, cells `ok` or `not_estimable` (the
  latter at gain 0, the pipeline's rule); the family is the method name. A table holds the shared
  protocol's one covariate basis (the 12 covariates entered linearly since 18 November 2026, PGI methods
  6.1 and 13; before, a spline of year of birth by sex). A table with two bases is refused, and so are
  rungs whose tables differ in basis, which happens only while the PGI arm regenerates them.
- **N_eff** of a disease at a rung is that rung's training cases and controls,
  4 / (1/cases + 1/controls), from `PGI/results/ladder/rung_power.csv` (`neff_train` at M_e 60,000).
  **h²** is the project's own S-LDSC estimate (`PGI/results/hm3plus/sldsc_h2.csv`: `h2_obs50` with
  its SE, `h2_liab`), never Pan-UKB's; **K** is the training prevalence (`K_train`).
- **ΔR²** (liability-scale variance explained beyond the covariates, the quantity the Daetwyler form
  is fitted to) is `r2_from_auc(AUROC_combined, K) − r2_from_auc(AUROC_covariates, K)` with the Lee
  2012 inverse of Wray's AUROC (`PGI/pgi/evaluate.py`). Decision: this conversion is the one the
  design document names ("liability-scale R² converted from AUROC"); it depends on the AUROCs and
  K only, so both arms and every split are converted identically.
- **The deep rows' `delta_auroc` in the gathered table** is `auroc_gain` of the summary file: the
  network's AUROC minus that of the deep arm's own covariates-only model (a logistic regression on
  the 12 standardised covariates fitted on the training split, `score_snapshots.covariates_only`).
  This is not the shared protocol (PGI methods 6.1: a refit on the tuning set with its covariate basis). The
  table alone therefore gives a quick look with no intervals; the reported numbers come from `fit`
  with the per-person scores, which recomputes every family through the shared protocol.

**The manifest** lists, per family, rung, seed and split (`val`, and `test` for the unlock), where
the per-person scores are expected: the deep arm's
`<dataset>/scores/<experiment>/<exp_name>/<split>_<snapshot>.npz`, the PGI arm's
`<dataset>/scores/pgi/<fingerprint>/<method>/<split>.npz` (found from the 16-digit fingerprint in
`tune_metrics.csv`) and the block-linear comparators' `<dataset>/scores/blocks_linear/<method>/<split>_best.npz`
(family `twin:<method>`, which needs no tidy rows: its values come from the scores, and its N_eff, h²
and K from the table's rows of the same disease and rung). `exists` says whether the file is there
(the test paths do not exist before the unlock). Nothing participant-level is written: the manifest
holds paths only.

## 3. Computing the estimands (`PGI/pgi/curves.py`)

**Inputs.** The tidy table; with per-person scores, a `Person`: the tuning table's labels and
covariates (`samples_val.tsv` of a rung's dataset directory, the same for every rung) and each
family's raw per-person scores aligned by identifier and label name (`read_scores`). The PGI arm's
files (`pipeline.py`: `iid`, `labels`, `pgi`, `y`, `logit_population`, `status`, ...) are read at
their raw `pgi`, not at the pipeline's combined `logit_population` beside it, so the protocol is
refitted from the PGI as it is for the other arms; the deep arm's and the block comparators' files
(`iid`, `labels`, `logit_population`, `y`) at their population logit.

**The shared protocol.** For every family, rung and disease, `pgi.combine.Combined` is fitted on
the tuning set exactly as the PGI pipeline's `tune_eval` stage does, with the protocol's covariate basis
(`methods.yaml combine.basis`: the 12 covariates entered linearly since 18 November 2026; the spline of year
of birth by sex plus the PCs that preceded it is now the robustness check, `basis="spline"`): per seed, and
on the seed-mean raw score. A fit that cannot be used (constant
score, no convergence, separation) counts as the covariate-only model, the pipeline's not-estimable
rule. The cells' point values are those of the seed-mean predictions (the design document's rule);
the per-seed values give the seed range and seed agreement. On the PGI arm's own scores the
protocol reproduces `tune_metrics.csv`: to 1e-12 in the tests, and on the real bottom-rung files
(LDpred2-auto, 30 diseases, 17 Nov) to 1.8e-8 in ΔAUROC and 4.7e-7 in the covariates' AUROC, the
float32 storage of the scores.

**The robustness check** (`fit --basis spline`, into an output folder of its own) repeats the
per-person fit with the flexible covariate basis for every family. It needs the manifest and a
dataset directory: the table's values were made with the protocol's own basis, so the table-only
quick look refuses `--basis`.

**"lin", the strongest linear method.** With per-person scores, PGI methods 6.3's rule: the tuning
participants are split in two halves by one seeded permutation (seed 1, unstratified); per disease
and rung the methods are ranked by their gain on one half, the best is judged on the other, the
halves are swapped, and "lin" is the mean of the two half values; the two winners are recorded. The
choice is held fixed over the bootstrap resamples (which resample within each half), so the
interval covers sampling noise, not selection noise. What the rule removes is the choice among
methods: the PGIs themselves are the pipeline's final ones, whose within-method hyper-parameters
(the C+T threshold, the lassosum2 grid, SBayesRC's choices) were chosen on the whole tuning set
(PGI methods 6.2), so the judged half is not free of that selection and "lin" stays optimistic by
that amount; PGI methods 6.3's "hyperparameters chosen on the other half" is not what is computed.
From the table alone, "lin" is the method with the largest gain per disease and rung, flagged as
optimistic.

**The deep envelope.** Per rung, the deep family with the highest mean gain over the 30 diseases on
the tuning set, or the pre-registered mapping (`chosen`) once it is frozen in the design document;
its rows are relabelled `deep` and keep `envelope_of`.

**Intervals.** 95% percentile intervals over the PGI arm's shared resamples
(`evaluate.bootstrap_indices`, seed 1, B = 2,000): per disease a set stratified by case status (per-
disease statements), and one unstratified set shared by all diseases (pooled statements). Every
cell's AUROC, gain and ΔR² is resampled once (`resampled_auroc`: one sort per score, then O(B n) in
float32 over the resample-count matrix, each case's drawn weight times the drawn control weight at
its midrank, the mean of the cumulative control weight below its tie group and up to its end; exact,
since every weight and partial sum is an integer or half-integer below 2²⁴, and equal to
`evaluate.auroc` on each resample to 1e-12 in the tests; 1.6 s per [2,000 × 20,000] call at 8
threads, against 10.6 s for the earlier `reduceat` form on continuous scores), and each estimand
is recomputed on the resampled values. Two-sided bootstrap p-values (2 min of the tail shares) are
given where a per-disease statement is Holm-adjusted (ρ_j, b_j).

**Runtime.** With the per-person scores the fit is hours, not minutes: about 0.4 s per combined
fit at 20,000 participants (four per disease for a three-seed cell, one for a one-seed family) and
about 1.6 s per resampled AUROC (two per cell, plus the cached covariate-only AUROC per disease and
resample set), so roughly 2.5 minutes per three-seed family-rung and 1 minute per one-seed
family-rung, of the order of 8 hours for the whole study's cells at 8 threads. `fit` prints a line
per family and rung of the protocol and per disease of the bootstrap, has no resume, and must be
launched detached with a log (CLAUDE.md section 1).

**Mixed models** (P1 ii and P2) are statsmodels `MixedLM`, REML, L-BFGS with BFGS and Powell as
fallbacks (a near-exact fit makes L-BFGS singular), on the response scaled to unit variance (gains of
a few thousandths otherwise break the optimiser; β, SE and covariance are scaled back). Their
bootstrap refits the fixed effects per resample by GLS with the variance components held at the
full-data estimates (an exact refit of β given the components; a full refit per resample would cost
2.6 s each). Wald intervals are reported beside the bootstrap ones; the decision rules use the
bootstrap intervals.

**Seed agreement** is computed from the per-seed pooled statistics: each seed's deep rows (with the
one-seed linear families) give b_s, d_s, the pooled log ρ_s and P4's pooled differences; the seeds
agree when every seed's value has the sign of the seed-mean estimate.

**P1.** Slopes through the origin of ΔR² on N_eff per disease and family (`slope_origin`); ρ_j from
the deep envelope and "lin"; pooled ρ as the DerSimonian-Laird random-effects mean of log ρ_j with
per-disease variances from the stratified draws (equal weights without draws; ratios of slopes of
opposite sign have no log and are left out, their number reported) and as the median; b_j by least
squares per disease, pooled b from Δ = a_j + (b + u_j) log N_eff + e (random intercept and slope).

**P2.** Δ = c + d log x + u_j + e (random intercept); x* = exp(−c/d) with the interval of the
resampled roots, flagged an estimate inside the observed range of x and an extrapolation outside it
(and whether the interval is finite); N*_j = x* M_e / h²_50,j. Sensitivities: a natural cubic
spline in log x with three interior knots at the quartiles (`pgi.combine._NaturalSpline`, df 4),
reported as its fitted curve and the contrast f(x_max) − f(x_min); and d refitted with every
disease's h²_50 moved by ±1 S-LDSC SE.

**The capped-asymptote bound.** With s the deep envelope's slope, A_lin and K_lin the "lin" curve's
Daetwyler parameters from the theory-constrained fit (one M_e over all linear families, K = M_e /
h²_50,j) and C_max = allowance × h²_liab,j (allowance 1.25; 1.0 and 1.5), the crossing is
N* = C_max (A − s K) / (s (C_max − A)): infinite when C_max ≤ A (no crossover possible) or s ≤ 0;
zero when s K ≥ A (the deep curve lies above throughout). N_min is the smaller of the 2.5th
percentile (an order statistic, so infinite draws are handled) of N* over the resampled slopes and
linear parameters, and the minimum of N* over the seeds' slopes. The linear parameters are
resampled with the per-disease stratified draws of every linear curve at once. **A known bias,
inherited from the design (for the main session to rule on):** s is the n²-weighted least-squares
slope through the origin over all five rungs, which for a concave curve lies below the initial
slope A/K; on an exact Daetwyler curve it is 0.55 A/K when the top rung sits at N_eff = K
(x_top = 1), 0.71 A/K at x_top = 0.5, 0.83 A/K at 0.25 and 0.92 A/K at 0.1. Since N* decreases in
s, an underestimated s pushes N_min up, so for diseases near the knee the bound "any crossover
lies above N_min" is not conservative. For those diseases (x_top ≥ 0.5) the asymptotes and N* are
reported from the eligible-disease fits rather than from the bound; for the rest the bias is
below about 20%. The alternatives are to take s from the pre-knee rungs only (x ≪ 1), or from the
deep curve's own free Daetwyler fit (A h²₅₀ / M_e); neither is implemented, the design's wording
being followed as written.

**P3.** For each axis (mixer size by `n_mixer_params`, table width by `embed_dim`, blocks by
`n_blocks`) and each mixer, the other two axes are held at their modal values among the deep cells
(the grid's values), and the per-rung values are the means over diseases of the gain per seed.
"Size matters at rung r" when the best size's gain over the smallest has its paired-interval lower
limit above zero (shared draws of the mean AUROC) and exceeds the largest within-size seed range;
"N_opt grows" when the argmax is non-decreasing over the rungs and the top rung's optimum beats
the bottom rung's optimum at the top rung by that rule. `fit` writes `size.csv` (per rung) and
`size_summary.csv` (per axis and mixer).

**P4.** For every family `twin:*`, per rung the mean over diseases of G_deep − G_twin with the
shared-draw interval and seed agreement; per disease DeLong's paired test on the seed-mean combined
logits (`evaluate.delong_paired`), Holm-adjusted within each rung.

**Secondary fits.** Daetwyler, ΔR² = A / (1 + M_e / (N_eff h²_50)): A is closed-form given M_e,
M_e is profiled on a log scale (bounded scalar minimisation for the point estimate; a 400-point
grid from 1e2 to 1e8 for the resamples), shared over the PGI methods' curves (theory-constrained;
the "lin" curve takes that M_e and gets its own A but does not enter the profile, since it
summarises those curves and would otherwise count one method twice) or per curve (free); the deep
arm descriptively. The bounded sigmoid on ΔAUROC, G = L + (A − L) /
(1 + exp(−(log N − m)/w)) with 0 ≤ L ≤ A ≤ ceiling, parametrised as (A, L/A, m, w) under box
bounds. The floored Chinchilla form on the distance to the ceiling, ceiling_j − G_jr = E_j +
B_j N_eff^−β with β shared across diseases (profiled; (E_j, B_j) linear least squares; N_eff
scaled by 10⁴ inside the fit, B reported in N_eff's units). Kaplan's L(N, D) = [(N_c/N)^(a_N/a_D)
+ D_c/D]^a_D on the distance to the ceiling over each mixer's size × rung grid per disease, by
least squares in log parameters from eight starts, with and without the smallest size (S1, one
layer, which Kaplan excluded). The ceiling is the gain a perfect additive PGI would add to the
covariates: Wray's AUROC at R²(covariates) + h²_liab minus the covariates' AUROC, the latter taken
from a PGI method's cell of the disease where one exists (the tuning-set refit; with per-person
scores every family shares it). None of these exponents is claimed as a capacity or data exponent.

**Eligibility.** Per disease and family (deep, lin): x at the top rung ≥ 0.5, and the free
Daetwyler fit to the other rungs predicting the top rung's ΔR² inside its interval. Where a
disease is not eligible, `gap_bounds.csv` gives the largest N_eff at which the gap's interval lies
below zero, or "no crossing within the ladder" with the top rung's gap and interval.

**The null** is evaluated from the pooled table by `null_result`, with P3 read on the mixer-size
axis (N_opt(D), the clause the design names; any mixer) and P4 over every twin and rung present.
It is `None` (undetermined) while any interval it needs is missing, while no mixer-size axis has
been measured (one deep size only) or while no twin family has scores: a null needs every clause
measured, so the first fits on a partial grid cannot declare it, and the table-only path never
does. The CLI prints the clause-by-clause verdict (`p1_b`, `p1_rho`, `p2_d`, `p3`, `p4`, `null`).

**The table-only path** mixes two covariate references (the deep arm's own covariates-only model
against the tuning-set refit of the PGI arm, section 2), so its gaps and decisions are not
reportable; the CLI marks its output "TABLE ONLY, a quick look". Reported numbers come from the
per-person path.

## 4. Result tables written by `fit`

`cells` (per disease, rung, family: seed-mean values, seed count and range, intervals; the derived
`deep` and `lin` rows follow the concrete cells and share their source's `cell` number), `envelope`,
`slopes`, `rho`, `gap` (with per-rung intervals), `gap_slopes` (b_j), `pooled` (the claims: pooled
ρ, b, d, x*, the spline contrast, each with estimate, interval, Wald interval, seed agreement and
decision), `nstar`, `gap_bounds`, `bound` (per disease and allowance), `size`, `size_summary`,
`p4`, `p4_disease`, `daetwyler`, `sigmoid`, `chinchilla`, `kaplan`, `eligibility`, `spline`,
`h2_sensitivity`, `null` (one row: each clause True, False or empty for "not measured", and
`null`). All aggregate.

## 5. Compute accounting (`compute.py`)

Forward FLOPs per participant from `config.json` (contract, "What a deep run records"): table
2 M codes d_e (codes 1 for the dosage code, 3 for the genotype code); block MLP 2 K (4 d_e² +
4 d_e d); mixer 2 N_mixer T, with N_mixer the mixer's parameter count and T = n_tokens; plus the
transformer's attention 4 L T² d, or bimamba1's scan 2 directions × T × 2d × 16 × 18 per layer.
Training costs three forward passes and tokens processed are participants seen, so the cumulative
compute of a snapshot or validation pass is C = 3 × forward × samples_seen. The head and the
covariate path are left out (thousands of FLOPs per participant); masked pretraining has its own
run records and is accounted separately when those exist. Per run the table has one row per
snapshot (`snapshots.csv`: step, samples seen, train seconds, so GPU-hours) and per validation pass
(`metrics.csv`, the mean tuning AUROC), with the model's counts. No critical-batch correction is
made (plan section 1).

## 6. Figures (`make_figures.py`)

Through `pgi.report.new_figure` and `save`, which refuse a figure that is not 89 or 183 mm wide or
has text below 7 pt: `curves_by_family` (gain and ΔR² against median N_eff, pooled over diseases,
the envelope with its seed band), `curves_per_disease`, `gap_vs_x` (points, the P2 line, x* with
its interval, the spline), `size_axis`, `table_axis` and `blocks_axis` (gain against the axis per
rung, N_opt, d_e_opt and B_opt starred; an axis with one value is skipped), `compute_fixed_data`
(mean tuning AUROC against cumulative training FLOPs at the top rung, one curve per size and mixer:
"performance against compute at fixed data", not a frontier), `transfer_offset` (tuning-set minus
held-out AUROC against D for the groups `asian`, `black` and `wb_related`; a `test` split in the
table is not a transfer group) and the ablation table (`ablations.md`, deep families by rung with
seed ranges). Cross-entropy versions are not drawn here; the secondary cross-entropy tables are
the deep arm's own.

## 7. Tests

`PGI/tests/test_curves.py` (35 tests): a simulator builds ladders of known parameters (Daetwyler
curves with a shared M_e, a deep envelope with chosen ρ and gap trend b, three seeds) and synthetic
resample draws stand in for the bootstrap, so the slopes, ρ, b and d are recovered with intervals
that cover the truth; x* and N* where the gap crosses; the N_opt rule fires only beyond both noise
bands; eligibility rejects pre-knee diseases and failed predictions; the null is declared on a
null simulation and not on a positive one, is undetermined without intervals, without a mixer-size
axis and without a twin; the capped bound lies below the true crossing when the true asymptote is
below the cap, and the crossing formula matches a hand-worked case; Holm against a hand-worked
case; the GLS refit reproduces the mixed model's fixed effects; the Daetwyler, sigmoid, Chinchilla
and Kaplan fits recover exact curves, and a distorted "lin" curve leaves the shared M_e where the
methods put it; duplicate rows and cells without their disease tables are refused with counts; and
on a simulated tuning set in the real layout (five rungs; PGI files in the pipeline's format with
both `pgi` and `logit_population`, rows in another order; deep files with two seeds and the labels
in the other order; the twin's score files with no rows in the tidy table) `read_scores` aligns rows
and columns, takes the raw `pgi` and refuses missing people with counts only, the shared protocol
reproduces the pipeline's metric, the split-half rule picks the better method and averages the
halves, `resampled_auroc` equals the AUROC of each resample to 1e-12 with and without ties and with
half-masks, and the whole per-person path runs with covering intervals, finite twin rows in every
table (slopes, Daetwyler, sigmoid) and the null undetermined for want of a size axis.
`test_scaling.py` (7 tests): FLOPs against a hand count for a tiny configuration of each mixer; the
compute table and CLI on a fake run; gathering, the manifest (both splits) and the family names on
a fake run tree with a fake PGI results folder; the family names of the ablations as the model
records them (SNP code, own and foreign `--init_encoder` file, `--freeze_encoder`, flags in `model`
only); the refusal of two runs in one cell; the table-only fit writing every result table and every
figure drawn from it and checked by `report.save`; and the per-person fit on a simulated tuning and
test set written in the real file layout (deep, PGI and twin score files per rung, one PGI method
with rows but no files), the test split refused without `--unlock_test` and logged with it, and the
table-only CLI marked.

## 8. Software

statsmodels 0.15.0 was installed on 17 November 2026 into `/home/robert/envs/widedeepgp` with pip
(permitted by Robert, 17 Nov), pulling patsy 1.0.3, formulaic 1.2.2, interface-meta 2.0.1, narwhals
2.28.0 and wrapt 2.5.2; numpy 1.26.4, scipy 1.15.3 and pandas 2.3.3 were unchanged. It is used only
through `MixedLM` (array interface; no formulas). Also recorded in `PGI/methods.md` section 14.

## 9. Not done, and open points

- No deep run exists, so nothing has been fitted to real curves; the fake-tree tests define the
  expected layout (`<root>/<rung>/<cell>/seed<s>/` or any tree in which a chosen trial's folder
  holds `config.json` and the summary file). The score files of the block-linear comparators
  (Scripts step 7) and the deep arm (`score_snapshots.py --save_scores`) are read when present.
- The per-rung comparison read-out (paired differences in ΔAUROC against each PGI method, the
  intersection-union claim) stays the PGI arm's report and is not duplicated here.
- The test-set path (`fit --split test --unlock_test`) reads `samples_test.tsv` and the manifest's
  test score files and appends each call to `<out>/test_access.log`; it is refused without the flag
  and is to be run once, at the joint unlock, on the pre-registered list. Its table rows are the
  deep arm's `scores_test_summary.csv` rows (the PGI arm's tuning metrics have no test rows); the
  PGI and twin values come from their test score files through the protocol.
- The capped-asymptote bound's slope is biased low near the knee (section 3); whether to take the
  slope from the pre-knee rungs or the deep curve's fitted A/K is the main session's decision.
- `fit` has no resume; a failed hours-long run restarts from the beginning.
- Pretraining compute is not yet accounted (no run records of that kind exist).
