# Results: the haploblock pre-test on chromosome 22

Status, 19 November 2026 (02:00 UTC): phasing and the nine haploblock cells complete; the three
ridges compared on the primary cell (f = 1%, c = 95%); the two ridge sensitivity cells not run
(stopped at Robert's request at 00:58 UTC, after the primary cell's interim reading); the
clumping-and-thresholding comparison of methods section 9 running (section 5 below, to be filled).
Every number is an aggregate over participants; the test split was never read. Methods: `methods.md`.

## 1. Phasing

SHAPEIT5 `phase_common` on the 235,872 training and tuning participants' hard calls at the 18,173
HapMap3+ SNPs of chromosome 22: 6.8 hours on 10 threads (24,633 s; about 68 CPU-hours; 15 GB). The
HMM pass shortened from 49 minutes (265 ± 121 conditioning states) in the first burn-in iteration to
13 to 19 minutes (113 ± 59 states) in the five main iterations. Scaled by sites to the store's
1,258,200 SNPs: about 4,700 CPU-hours, five days on 40 threads, three weeks on ten. The allele
frequencies of the BCF equalled the hard calls' (largest difference 5 × 10⁻⁷).

## 2. Haploblocks: the arithmetic (all nine cells)

Haploblocks are nested in the 15 Privé blocks of chromosome 22 by the coverage rule of methods
section 4 (extend while the haplotypes with training frequency ≥ f cover ≥ c of training
chromosomes). H is the number of common haplotypes of a haploblock; features = Σ(H + 1), one column
per common haplotype plus "other"; the genome projection scales by the store's SNP count
(1,258,200 / 18,173 = 69.2).

| cell (f, c) | haploblocks | SNPs per block, median | kb, median | H median (10th to 90th pct) | H max | share with 3 ≤ H ≤ 10 | features | per SNP | genome: haploblocks | genome: features |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.005, 0.9 | 529 | 29 | 42 | 26 (20 to 34) | 48 | 1% | 14,568 | 0.80 | 36,625 | 1,008,609 |
| 0.005, 0.95 | 780 | 19 | 28 | 19 (14 to 25) | 35 | 2% | 15,536 | 0.85 | 54,003 | 1,075,628 |
| 0.005, 0.99 | 1,950 | 8 | 10 | 8 (5 to 13) | 20 | 74% | 19,131 | 1.05 | 135,007 | 1,324,527 |
| 0.01, 0.9 | 684 | 21 | 31 | 16 (12 to 21) | 29 | 5% | 11,624 | 0.64 | 47,356 | 804,783 |
| 0.01, 0.95 (primary) | 1,038 | 14 | 20 | 12 (8 to 16) | 27 | 34% | 13,373 | 0.74 | 71,865 | 925,874 |
| 0.01, 0.99 | 2,574 | 6 | 7 | 6 (4 to 10) | 18 | 93% | 19,168 | 1.05 | 178,210 | 1,327,088 |
| 0.02, 0.9 | 1,037 | 15 | 21 | 9 (6 to 12) | 17 | 78% | 10,114 | 0.56 | 71,796 | 700,239 |
| 0.02, 0.95 | 1,686 | 9 | 12 | 7 (4 to 9) | 15 | 95% | 13,046 | 0.72 | 116,729 | 903,234 |
| 0.02, 0.99 | 5,492 | 2 | 2 | 3 (1 to 6) | 14 | 61% | 23,859 | 1.31 | 380,236 | 1,651,868 |

**The 3-to-10 assumption was low.** At the primary cell the median haploblock has 14 SNPs over
20 kb and 12 common haplotypes (10th to 90th percentiles 8 to 16; one haploblock of a single SNP);
34% of haploblocks have 3 to 10. Haploblocks with H in that range appear only
where the coverage is loose (c = 90%) or the frequency floor high (f = 2%), and then the blocks are
longer and the "other" bucket larger. **The table arithmetic holds:** at the primary cell the
haplotype table would have 0.74 vectors per SNP, about 926,000 vectors genome-wide against the
per-SNP table's 1,258,200, over about 72,000 haploblocks; every cell lies between 0.56 and 1.31
vectors per SNP. The design was affordable. The per-segment table of the primary cell is
`haploblocks_chr22_f0.01_c0.95_segments.csv`; the grid, `haploblocks_grid.csv`.

## 3. The three ridges on the primary cell

Per Privé block, cross-fitted over five folds: A on standardised SNP dosages, B on standardised
haplotype counts, C on both; λ = M/h² with h² chosen per disease on the out-of-fold training AUROC
(methods sections 5 and 6). A's fold weights reproduce the stored cross-fitted ridge of the blocks
pipeline (largest relative error 2.3 × 10⁻⁹ over 75 block-folds, `check_snp_weights_chr22.csv`).
Haplotype features 13,373 (scale 1.166), M_hap 925,874.

### 3.1 Pooled over the 30 diseases

Mean of the per-disease differences, 95% paired participant bootstrap (2,000 resamples), and the
count of diseases on each side. The fixed penalty (h² = 0.15 for every disease and parametrisation)
is the little-selection reading.

| metric | set | comparison | chosen penalties: mean [95% interval] | diseases below / above zero | fixed h² = 0.15: mean |
|---|---|---|---|---|---|
| r with residual label | out-of-fold training | B − A | -0.00146 [-0.00206, -0.00082] | 22 / 8 | -0.00127 |
| r with residual label | out-of-fold training | C − A | -0.00016 [-0.00039, +0.00009] | 16 / 14 | -0.00040 |
| ΔAUROC | out-of-fold training | B − A | -0.00008 [-0.00014, -0.00002] | 22 / 8 | -0.00007 |
| ΔAUROC | out-of-fold training | C − A | -0.00001 [-0.00003, +0.00001] | 18 / 12 | -0.00002 |
| r with residual label | tuning | B − A | +0.00029 [-0.00182, +0.00236] | 13 / 17 | +0.00020 |
| r with residual label | tuning | C − A | +0.00046 [-0.00035, +0.00123] | 10 / 20 | +0.00046 |
| ΔAUROC | tuning | B − A | -0.00002 [-0.00021, +0.00018] | 16 / 14 | -0.00003 |
| ΔAUROC | tuning | C − A | +0.00005 [-0.00003, +0.00012] | 12 / 18 | +0.00003 |

### 3.2 Levels

| set | covariates-only AUROC | mean r: A, B, C | mean ΔAUROC: A, B, C | B / A (r) | C / A (r) |
|---|---|---|---|---|---|
| out-of-fold training (215,872) | 0.6588 | 0.0063, 0.0048, 0.0061 | +0.00024, +0.00016, +0.00023 | 0.77 | 0.97 |
| tuning (20,000) | 0.6627 | 0.0067, 0.0070, 0.0072 | +0.00006, +0.00004, +0.00011 | 1.04 | 1.07 |

The chromosome-22 SNP ridge alone reaches r = 0.0063 with the residualised label on the
out-of-fold rows, as its 1.4% share of the SNP heritability implies; this is the scale against which
the differences read. The haplotype ridge chose the strongest shrinkage (h² = 0.0167) for
26 of 30 diseases, the SNP ridge for 18.

### 3.3 Per disease

Out-of-fold training rows unless stated; intervals from the same paired bootstrap.

| disease | training cases | h² chosen A / B / C | r, out-of-fold: A, B, C | B − A, out-of-fold [95%] | C − A, out-of-fold [95%] | ΔAUROC tuning: A, B, C | B − A ΔAUROC tuning [95%] |
|---|---|---|---|---|---|---|---|
| I10 | 68,309 | 0.15 / 0.0167 / 0.0167 | 0.0135, 0.0130, 0.0145 | -0.0005 [-0.0032, +0.0022] | +0.0010 [+0.0000, +0.0020] | +0.0001, +0.0003, +0.0002 | +0.0002 [-0.0000, +0.0004] |
| E78 | 34,953 | 1.35 / 0.45 / 1.35 | 0.0123, 0.0116, 0.0126 | -0.0007 [-0.0032, +0.0018] | +0.0003 [-0.0015, +0.0022] | +0.0005, +0.0004, +0.0005 | -0.0001 [-0.0004, +0.0002] |
| K57 | 30,506 | 0.15 / 0.0167 / 0.0167 | 0.0092, 0.0069, 0.0090 | -0.0023 [-0.0051, +0.0003] | -0.0002 [-0.0012, +0.0007] | +0.0002, +0.0003, +0.0003 | +0.0001 [-0.0003, +0.0005] |
| K21 | 25,427 | 0.15 / 0.0167 / 0.0167 | 0.0064, 0.0047, 0.0062 | -0.0017 [-0.0044, +0.0011] | -0.0003 [-0.0012, +0.0007] | -0.0000, +0.0002, +0.0001 | +0.0002 [-0.0003, +0.0008] |
| K29 | 24,564 | 0.0167 / 0.0167 / 0.0167 | 0.0006, -0.0005, 0.0001 | -0.0011 [-0.0040, +0.0019] | -0.0005 [-0.0015, +0.0006] | +0.0001, -0.0001, +0.0000 | -0.0002 [-0.0003, -0.0000] |
| K44 | 24,297 | 0.05 / 0.0167 / 0.0167 | 0.0115, 0.0071, 0.0107 | -0.0044 [-0.0073, -0.0013] | -0.0007 [-0.0017, +0.0002] | -0.0003, -0.0000, -0.0002 | +0.0003 [-0.0005, +0.0012] |
| I25 | 23,558 | 0.0167 / 0.0167 / 0.0167 | 0.0097, 0.0057, 0.0088 | -0.0040 [-0.0069, -0.0010] | -0.0009 [-0.0019, +0.0001] | +0.0001, +0.0002, +0.0002 | +0.0001 [-0.0002, +0.0004] |
| M19 | 21,544 | 0.0167 / 0.0167 / 0.0167 | 0.0121, 0.0082, 0.0115 | -0.0039 [-0.0069, -0.0010] | -0.0007 [-0.0017, +0.0003] | +0.0005, +0.0004, +0.0006 | -0.0002 [-0.0008, +0.0005] |
| I84+K64 | 21,296 | 0.45 / 0.15 / 0.45 | 0.0057, 0.0076, 0.0070 | +0.0019 [-0.0008, +0.0046] | +0.0014 [-0.0002, +0.0029] | -0.0008, -0.0012, -0.0010 | -0.0004 [-0.0018, +0.0009] |
| I48 | 18,365 | 0.45 / 0.0167 / 0.0167 | 0.0079, 0.0087, 0.0084 | +0.0008 [-0.0019, +0.0033] | +0.0005 [-0.0010, +0.0019] | -0.0003, -0.0004, -0.0003 | -0.0000 [-0.0003, +0.0002] |
| M17 | 18,227 | 0.0167 / 0.0167 / 0.0167 | 0.0041, 0.0009, 0.0032 | -0.0032 [-0.0062, +0.0000] | -0.0009 [-0.0019, +0.0001] | +0.0002, +0.0001, +0.0002 | -0.0001 [-0.0006, +0.0003] |
| C44 | 17,671 | 0.0167 / 0.0167 / 0.0167 | 0.0034, 0.0025, 0.0033 | -0.0010 [-0.0040, +0.0020] | -0.0001 [-0.0012, +0.0009] | +0.0002, -0.0001, +0.0001 | -0.0002 [-0.0004, -0.0000] |
| E11 | 17,345 | 0.0167 / 0.0167 / 0.0167 | 0.0106, 0.0106, 0.0111 | +0.0000 [-0.0029, +0.0030] | +0.0005 [-0.0005, +0.0016] | +0.0006, +0.0008, +0.0007 | +0.0002 [-0.0004, +0.0009] |
| E66 | 16,800 | 0.0167 / 0.0167 / 0.0167 | 0.0102, 0.0089, 0.0104 | -0.0014 [-0.0044, +0.0017] | +0.0001 [-0.0009, +0.0012] | -0.0002, +0.0002, -0.0001 | +0.0003 [-0.0028, +0.0035] |
| M54 | 14,550 | 0.45 / 0.0167 / 0.15 | 0.0069, 0.0089, 0.0078 | +0.0020 [-0.0006, +0.0045] | +0.0009 [-0.0002, +0.0020] | -0.0018, -0.0022, -0.0020 | -0.0004 [-0.0017, +0.0009] |
| D12 | 14,337 | 0.0167 / 0.0167 / 0.0167 | 0.0057, 0.0030, 0.0050 | -0.0027 [-0.0058, +0.0005] | -0.0007 [-0.0018, +0.0005] | -0.0003, +0.0000, -0.0002 | +0.0003 [-0.0003, +0.0009] |
| I20 | 14,234 | 0.15 / 1.35 / 0.05 | 0.0065, 0.0046, 0.0063 | -0.0019 [-0.0053, +0.0015] | -0.0001 [-0.0011, +0.0008] | -0.0001, +0.0000, -0.0000 | +0.0001 [-0.0004, +0.0005] |
| E03 | 13,242 | 0.0167 / 0.0167 / 0.0167 | 0.0147, 0.0077, 0.0133 | -0.0070 [-0.0103, -0.0040] | -0.0014 [-0.0025, -0.0004] | +0.0015, +0.0009, +0.0015 | -0.0006 [-0.0018, +0.0006] |
| K40 | 12,541 | 0.0167 / 0.0167 / 0.0167 | 0.0018, 0.0007, 0.0014 | -0.0011 [-0.0040, +0.0019] | -0.0003 [-0.0014, +0.0007] | +0.0001, -0.0000, +0.0001 | -0.0001 [-0.0002, +0.0000] |
| M16 | 12,286 | 0.05 / 0.0167 / 0.0167 | 0.0096, 0.0094, 0.0100 | -0.0002 [-0.0030, +0.0026] | +0.0005 [-0.0005, +0.0014] | +0.0006, +0.0005, +0.0007 | -0.0002 [-0.0010, +0.0006] |
| K80 | 12,194 | 0.0167 / 0.0167 / 0.0167 | 0.0020, 0.0018, 0.0020 | -0.0002 [-0.0033, +0.0028] | +0.0000 [-0.0011, +0.0011] | +0.0004, -0.0003, +0.0002 | -0.0006 [-0.0010, -0.0003] |
| N17 | 11,340 | 0.0167 / 0.0167 / 0.0167 | 0.0013, 0.0032, 0.0021 | +0.0019 [-0.0011, +0.0049] | +0.0008 [-0.0003, +0.0018] | +0.0001, +0.0002, +0.0002 | +0.0001 [-0.0002, +0.0003] |
| N18 | 10,337 | 1.35 / 0.0167 / 0.0167 | 0.0019, -0.0001, 0.0008 | -0.0020 [-0.0045, +0.0005] | -0.0011 [-0.0032, +0.0009] | +0.0001, -0.0000, +0.0000 | -0.0001 [-0.0003, +0.0001] |
| J44 | 9,884 | 0.45 / 1.35 / 1.35 | 0.0029, 0.0032, 0.0033 | +0.0003 [-0.0027, +0.0035] | +0.0003 [-0.0020, +0.0028] | +0.0002, -0.0002, -0.0001 | -0.0004 [-0.0007, -0.0001] |
| M23 | 9,585 | 0.0167 / 0.0167 / 0.0167 | 0.0069, 0.0005, 0.0051 | -0.0063 [-0.0094, -0.0032] | -0.0018 [-0.0028, -0.0008] | -0.0004, +0.0004, +0.0009 | +0.0007 [-0.0027, +0.0039] |
| M47 | 9,492 | 0.0167 / 0.0167 / 0.0167 | 0.0049, 0.0019, 0.0041 | -0.0030 [-0.0061, +0.0003] | -0.0008 [-0.0019, +0.0003] | +0.0003, +0.0004, +0.0005 | +0.0001 [-0.0008, +0.0009] |
| I95 | 9,153 | 0.0167 / 0.0167 / 0.0167 | -0.0048, -0.0036, -0.0047 | +0.0013 [-0.0020, +0.0043] | +0.0001 [-0.0011, +0.0012] | -0.0001, -0.0002, -0.0002 | -0.0001 [-0.0006, +0.0004] |
| I50 | 8,851 | 0.0167 / 0.0167 / 0.0167 | 0.0032, -0.0005, 0.0019 | -0.0036 [-0.0065, -0.0006] | -0.0012 [-0.0023, -0.0001] | -0.0001, -0.0000, -0.0000 | +0.0001 [-0.0003, +0.0005] |
| M81 | 8,604 | 0.0167 / 0.0167 / 0.0167 | 0.0066, 0.0058, 0.0068 | -0.0008 [-0.0039, +0.0023] | +0.0002 [-0.0009, +0.0013] | +0.0002, +0.0006, +0.0004 | +0.0003 [-0.0001, +0.0008] |
| I21 | 8,090 | 0.0167 / 0.0167 / 0.0167 | 0.0021, 0.0029, 0.0024 | +0.0008 [-0.0021, +0.0038] | +0.0004 [-0.0007, +0.0015] | +0.0000, +0.0000, +0.0000 | -0.0000 [-0.0003, +0.0003] |

Intervals excluding zero on the out-of-fold rows: B − A below zero for 6 diseases on r and
2 on ΔAUROC, above zero for none on either; C − A below zero for 3 and above zero for
1 on r. Figure: `figure_ridge_chr22_f0.01_c0.95.{png,pdf}` with `_data.csv` and `_legend.md`.

### 3.4 Verdict under the pre-stated rule

The rule (methods section 6): build only if B − A is positive with an interval excluding zero on
the out-of-fold training rows and not negative on the tuning set, in both metrics. **B − A is
negative with an interval excluding zero on the out-of-fold rows in both metrics**
(r -0.00146,
ΔAUROC -0.00008), 22 of 30 diseases
below zero, no disease above it. **The haploblock build is not worth two weeks:** the haplotype
parametrisation carries 23% less additive signal than the SNP dosages on the same blocks at
this sample size, and the haplotype-specific signal that the union could add is bounded above at
+1.4% of the SNP ridge's correlation (C − A upper interval limit, out-of-fold rows). The
tuning set's intervals are ten times wider and include zero everywhere, as the resolving-power
paragraph of methods section 6 predicted; they neither support nor contradict the out-of-fold
reading. The fixed penalty gives the same picture.

**Why this is plausible.** A SNP column pools the SNP's carriers across every haplotype background;
a haplotype column splits them, so each of a block's twelve effects is estimated from a fraction of
the carriers, and the haplotype ridge needs stronger shrinkage to hold its noise. The haplotype
coding is richer (within a block it nests the dosage coding when every haplotype is common), so with
enough participants it cannot do worse; at 172,000 training rows per fold model it does. Phasing
error attenuates B, not A, and is unmeasured here; the published switch error for common variants at
this sample size is well under 1%, far too small to explain a 23% loss.

**What this does not say** (methods section 8): nothing about optimisation inside a network, about
rare haplotypes (the "other" bucket) or about other chromosomes. One rung (the top), one
chromosome, one cell; the sensitivity cells (c = 90% and 99%) were not run.

## 4. Cost of a build, had it passed

Genome-wide phasing about 4,700 CPU-hours (section 1), then the haploblock cut (170 s per cell on
chromosome 22, about 3.5 hours genome-wide) and the model change: about two weeks, as proposed.

## 5. Clumping and thresholding (methods section 9)

Running; to be filled.
