# Methods: the 30-disease dataset for DeepWideGP

*Analysis `build-ard30-dataset` (DeepWideGP; DPhil chapter 6). Run on 11 November 2026.*

## Objective

Assemble, for the participants that DeepWideGP trains and is evaluated on, the inputs of a
multi-disease genomic risk model: the list of SNPs, a label for each of 30 age-related
diseases, the covariates, a fixed training / tuning / test split, and a functional-annotation
vector for every SNP. The model itself is described in `DeepWideGP/methods.md`.

## Inputs

| Input | Source |
|---|---|
| Phenotypes | UK Biobank application 15181, local extract of 18 November 2024 (502,399 participants) |
| Withdrawals | The application's list of 25 April 2024 |
| Genotypes | The laboratory's PLINK2 files of UK Biobank imputed data v3 (GRCh37; bi-allelic SNPs; filtered on minor-allele frequency, Hardy–Weinberg equilibrium and missingness): 7,813,490 SNPs for 333,444 participants of White-British genetic ancestry, and a second set with the same SNPs for 17,055 Asian and Black participants |
| SNP set | The HapMap3+ map of LDpred2 (Privé et al., 2023; figshare file 37802721): 1,444,196 variants |
| Diseases | The 30 phenotypes selected by `find-ukb-ard-phenos` (`results/ukb_ard_selected_phenos_overview.csv`) |
| Annotations | `annot_baseline2.2.txt` distributed with SBayesRC (Zheng et al., 2024): 96 annotations of the baseline-LD model v2.2 (Gazal et al., 2017) for 8,140,663 SNPs |

## SNPs (`make_snp_list.R`)

The HapMap3+ map was matched to the variant lists of the genotype files by chromosome and
position, and a match was kept only if the two alleles agreed, in either order. 1,258,200 of
the 1,444,196 variants are present (87.1%): 1,012,953 of the 1,054,330 HapMap3 variants and
245,247 of the 389,866 added ones. The variant identifiers agree for every match. Of the
185,996 absent variants, 116,513 have a minor-allele frequency below 1% in UK Biobank (the
lowest frequency among the variants present is 0.98%) and 69,483 are more frequent and were
removed by the other filters applied when the genotype files were built. Chromosome 2 has the
most SNPs (105,260) and chromosome 21 the fewest (17,844).

The genotypes of these SNPs were then converted for all 333,444 participants into the
project's compact on-disk format (`DeepWideGP/genostore`, `convert.py --extract`), reading only
the listed variants' records from the source files. Eight participants' genotypes per
chromosome were captured directly from the decoder during conversion and compared afterwards
with what the store returns: all 8 × 1,258,200 genotypes agreed. No genotype is missing, and
the store takes 116 GB (0.28 bytes per genotype).

The same list was converted in the same way for the 17,055 participants of the second genotype
set (7.6 GB; 0.35 bytes per genotype, because more of their genotypes are imputed with
fractional dosages). All 1,258,200 SNPs are present there too, in the same order, so one model
input serves both sets; the audit again agreed for every chromosome.

## Participants

Participants on the withdrawal list were removed. The modelling population is the 275,872
genotyped participants of White-British genetic ancestry (field 22006) for whom UK Biobank's
kinship inference found no relative (field 22021 = 0), as defined in `find-ukb-ard-phenos`,
whose code this analysis reuses.

Two further groups were given labels and covariates by the same rules. They are held out and
take no part in training, tuning or testing:

- 57,445 genotyped White-British participants with an identified relative;
- 9,408 Asian and 7,631 Black participants of the second genotype set (membership from the
  laboratory's ancestry-specific sample lists, which between them cover all 17,055 of its
  participants), after removing 16 withdrawals. None lacks a covariate, and the second set
  shares no participant with the first.

## Labels

For each phenotype, a participant is a case if at least one record carries an ICD-10 code whose
first three characters are one of the phenotype's categories, in hospital inpatient diagnoses
(field 41270, any position), the death registry (fields 40001 and 40002) or the cancer
registry (field 40006), at any date; everyone else is a non-case. One phenotype, haemorrhoids,
unites two categories (I84 and K64). The records are extracted and cleaned by the functions of
`find-ukb-ard-phenos`, so the definition is the code that produced the selection.

**Check.** The build stops unless the number of cases of every phenotype equals the count in
the selection table, in each of that table's three populations (all participants, genotyped
White British, unrelated White British). All 90 counts agreed.

## Covariates

- **Year of birth.** Field 34 is not in the extract. It was reconstructed as the date of the
  first assessment (field 53, as a decimal year) minus age at recruitment (field 21022, in
  completed years) minus one half. The date of birth lies somewhere in the year that ends at
  "assessment date minus age", and the reconstruction is the midpoint of that year, so its
  error is at most half a year.
- **Sex** (field 31; 0 female, 1 male).
- **Genetic principal components 1 to 10** (field 22009).

No participant of the modelling population lacks a covariate.

## Split

The 275,872 participants were sorted by identifier and permuted with a fixed seed (20261111,
NumPy `default_rng`); the first 40,000 form the test set, the next 20,000 the tuning set and
the remaining 215,872 the training set. The split is independent of every label and covariate.

| Group | Participants | Use |
|---|---|---|
| Training | 215,872 | Fitting the model |
| Tuning | 20,000 | Stopping, choice of snapshot, hyper-parameters |
| Test | 40,000 | Scored after those choices are fixed |
| Related White British | 57,445 | Held out |
| Asian | 9,408 | Held out |
| Black | 7,631 | Held out |

Cases by group for the modelling population and the related participants
(`results/ard30_cases_by_split.csv`):

| Phenotype | ICD-10 | Training (%) | Tuning | Test | Related, held out |
|---|---|---|---|---|---|
| Essential (primary) hypertension | I10 | 68,309 (31.6) | 6,255 | 12,729 | 19,056 |
| Disorders of lipoprotein metabolism and other lipidaemias | E78 | 34,953 (16.2) | 3,153 | 6,378 | 9,708 |
| Diverticular disease of intestine | K57 | 30,506 (14.1) | 2,784 | 5,624 | 8,460 |
| Gastro-oesophageal reflux disease | K21 | 25,427 (11.8) | 2,387 | 4,671 | 7,152 |
| Gastritis and duodenitis | K29 | 24,564 (11.4) | 2,238 | 4,512 | 7,110 |
| Diaphragmatic hernia | K44 | 24,297 (11.3) | 2,271 | 4,524 | 6,804 |
| Chronic ischaemic heart disease | I25 | 23,558 (10.9) | 2,126 | 4,299 | 6,498 |
| Other arthrosis | M19 | 21,544 (10.0) | 1,966 | 4,006 | 6,274 |
| Haemorrhoids | I84, K64 | 21,296 (9.9) | 1,962 | 3,989 | 5,890 |
| Atrial fibrillation and flutter | I48 | 18,365 (8.5) | 1,670 | 3,346 | 5,029 |
| Gonarthrosis [arthrosis of knee] | M17 | 18,227 (8.4) | 1,675 | 3,476 | 5,193 |
| Other malignant neoplasms of skin | C44 | 17,671 (8.2) | 1,676 | 3,303 | 4,819 |
| Non-insulin-dependent diabetes mellitus | E11 | 17,345 (8.0) | 1,663 | 3,143 | 4,823 |
| Obesity | E66 | 16,800 (7.8) | 1,478 | 2,999 | 4,711 |
| Dorsalgia | M54 | 14,550 (6.7) | 1,310 | 2,631 | 4,045 |
| Benign neoplasm of colon, rectum, anus and anal canal | D12 | 14,337 (6.6) | 1,309 | 2,684 | 4,044 |
| Angina pectoris | I20 | 14,234 (6.6) | 1,284 | 2,609 | 4,082 |
| Other hypothyroidism | E03 | 13,242 (6.1) | 1,211 | 2,383 | 3,782 |
| Inguinal hernia | K40 | 12,541 (5.8) | 1,140 | 2,224 | 3,294 |
| Coxarthrosis [arthrosis of hip] | M16 | 12,286 (5.7) | 1,106 | 2,287 | 3,429 |
| Cholelithiasis | K80 | 12,194 (5.6) | 1,118 | 2,253 | 3,388 |
| Acute renal failure | N17 | 11,340 (5.3) | 1,007 | 2,043 | 3,006 |
| Chronic renal failure | N18 | 10,337 (4.8) | 943 | 1,911 | 3,011 |
| Other chronic obstructive pulmonary disease | J44 | 9,884 (4.6) | 986 | 1,906 | 2,751 |
| Internal derangement of knee | M23 | 9,585 (4.4) | 872 | 1,821 | 2,564 |
| Spondylosis | M47 | 9,492 (4.4) | 797 | 1,722 | 2,766 |
| Hypotension | I95 | 9,153 (4.2) | 834 | 1,639 | 2,487 |
| Heart failure | I50 | 8,851 (4.1) | 814 | 1,548 | 2,450 |
| Osteoporosis without pathological fracture | M81 | 8,604 (4.0) | 840 | 1,615 | 2,581 |
| Acute myocardial infarction | I21 | 8,090 (3.7) | 730 | 1,541 | 2,234 |

## The label matrix and disease-balanced batches (`describe_dataset.py`)

The model is trained on batches balanced on one disease at a time, in rotation
(`DeepWideGP/methods.md`, section 5). What that does depends on how the 30 phenotypes co-occur,
so the training split was summarised (`results/ard30_sampling_summary.json`,
`results/ard30_sampling_by_disease.csv`):

- **Phenotypes per participant.** Mean 2.51 (the sum of the 30 prevalences), median 2; 31.8% of
  training participants have none, 17.3% have one, 20.2% have five or more and 1% have 12 or more.
- **Case share in batches.** A batch balanced on phenotype *i* draws half its participants from
  the cases of *i* and half from its non-cases, so the expected share of cases of phenotype *j*
  in it is P(*j* | *i*)/2 + P(*j* | not *i*)/2, computed from the co-occurrence counts. Averaged
  over the 30 batches of a rotation, cases of the rarest phenotype (I21, 3.7% of the split) are
  7.8% of the participants drawn, against 5.3% if the phenotypes were independent; the median
  across phenotypes is 11.3% (prevalence 6.7%) and the largest 44.8% (I10, prevalence 31.6%).
  Between 76% and 96% of a phenotype's case draws arrive in batches balanced on another
  phenotype.
- **Sampling rate.** A participant's sampling rate ω is how often the rotation draws them
  relative to uniform drawing (mean 1 by construction; computed with the model code's own
  function). It is 0.55 for a participant with none of the phenotypes, 0.80 at the median, 1.77
  at the 90th percentile and 3.10 at the 99th. Weighting draws by 1/ω, as
  the training loss does, leaves an effective sample of 80.0% of the draws (Kish's formula).
- **Class weights.** Bringing each phenotype to parity between cases and non-cases would need
  weights on cases from 2.2 (I10) to 25.7 (I21); their square roots, which the model uses by
  default, run from 1.5 to 5.1.
- **Empty batches under uniform drawing.** A batch of 21 participants drawn uniformly holds no
  case of I21 with probability 0.45, and no case of the median phenotype with probability 0.23.

## Annotations

The annotation table was matched to the SNP list by variant identifier. 1,250,863 of the
1,258,200 SNPs have a row (99.4%); the others were given zeros, and a 97th channel marks
them. The table's intercept column was dropped. Matrices are stored per chromosome in the order
of the genotype store; the model standardises each channel over its SNPs.

## Proxy dataset for pilots (`make_proxy_dataset.py`)

Training time is proportional to participants times SNPs, and one pass over the full training
split costs several GPU-hours. Pilot comparisons and the first stage of the hyper-parameter
search therefore use a reduced dataset with the same 30 labels and 12 covariates:

- **SNPs.** One contiguous stretch of every chromosome: the SNPs from one eighth to two eighths
  of the way along it, by SNP index (157,279 SNPs; between 2,231 on chromosome 21 and 13,158
  on chromosome 2). A contiguous stretch keeps the local linkage disequilibrium that the full
  model sees, which thinning would not. The second eighth was chosen, before any model was
  trained, because of 20 well-established loci for these diseases that were looked up it holds
  the most (five: 9p21, the HLA region, *LDLR*, *PCSK9* and *GCKR*); the choice makes the proxy
  more likely to carry genetic signal and applies equally to every arm of a comparison. No
  genotype is copied: the proxy's `store.json` names the stretches as regions of the same store.
- **Participants.** 40,000 drawn at random from the training split and 10,000 from the tuning
  split (seed 20261112). The rarest phenotype (I21) has 1,557 cases among the 40,000 and 373
  among the 10,000.
- **No test participants.** The loader expects a test table; the proxy's is a further 10,000
  drawn from the *training* split, disjoint from the proxy's 40,000. The real test set is not
  part of the proxy dataset, so a pilot cannot read it.

Results on the proxy rank training schemes; they are not estimates of the full model's
accuracy.

## Outputs

Under `/mnt/sdg/robert/deepwidegp/data/` (participant-level; outside any git repository):

- `reference/hm3plus_in_ukb_pgen.tsv` and `.json`: the SNP list and its counts;
- `genostore/hm3plus/` and `genostore/hm3plus_asian_black/`: the two genotype stores;
- `datasets/ard30/hm3plus/`: `samples_train.tsv`, `samples_val.tsv` (the tuning set),
  `samples_test.tsv`, `samples_wb_related.tsv`, `store.json`, `labels.json`,
  `annotations.npz` and `build_log.json`;
- `datasets/ard30/hm3plus_asian_black/`: `samples_asian.tsv`, `samples_black.tsv` and a
  `store.json` naming the second store;
- `datasets/ard30/hm3plus_proxy8/`: the proxy dataset (`store.json`, the three sample tables,
  `labels.json`, `proxy_log.json`).

`build_log.json` records the participant flow, the counts by group, the covariate summary, the
co-occurrence of the 30 phenotypes in the training set, the checks and the checksums of the
inputs. It holds aggregate numbers only.

In this folder, `results/` holds the aggregate tables written by `describe_dataset.py`: cases by
split, per-phenotype sampling figures and the sampling summary. Counts for the Asian and Black
groups, some of which are small, are kept in `build_log.json` and not in the repository.

## Software

R 4 with data.table (SNP list); Python 3.10 with pandas 2.3.3 and NumPy 1.26.4 (dataset);
pgenlib 0.95.2 (genotype conversion). Unit tests on made-up records: `test_build_dataset.py`
(labels, year of birth, split, tables, the description of sampling, the proxy's stretches and
draws).

    Rscript scripts/build-ard30-dataset/make_snp_list.R
    python -m genostore.convert --pgen-dir <pgen dir> --pattern "UKB_chr{chrom}" --chroms $(seq 1 22) \
        --extract <data>/reference/hm3plus_in_ukb_pgen.tsv --out <data>/genostore/hm3plus   # in DeepWideGP
    python scripts/build-ard30-dataset/build_dataset.py
    python scripts/build-ard30-dataset/describe_dataset.py
    python scripts/build-ard30-dataset/make_proxy_dataset.py

## Limitations

- The labels inherit the limitations listed in `find-ukb-ard-phenos/methods.md`: hospital-based
  ascertainment, a partial cancer registry, no ICD-9 records, and withdrawals as of April 2024.
- Year of birth is reconstructed, with an error of up to half a year.
- 13% of the HapMap3+ variants are absent from the genotype files, almost two thirds of them
  because they are rarer than 1%.
- The held-out groups were prepared but not examined: their covariates, in particular the
  principal components of participants of other ancestries, lie outside the range the model is
  trained on.

## References

- Gazal S, Finucane HK, Furlotte NA, et al. Linkage disequilibrium-dependent architecture of
  human complex traits shows action of negative selection. *Nat Genet* 2017;49:1421–1427.
- Privé F, Albiñana C, Arbel J, Pasaniuc B, Vilhjálmsson BJ. Inferring disease architecture and
  predictive ability with LDpred2-auto. *Am J Hum Genet* 2023;110:2042–2055.
- Zheng Z, Liu S, Sidorenko J, et al. Leveraging functional genomic annotations and genome
  coverage to improve polygenic prediction of complex traits within and between ancestries.
  *Nat Genet* 2024;56:767–777.
