# Held-out ancestry evaluation: every model on the Asian and Black groups

Robert, 22 November 2026: "build the infrastructure to test the black subset and the Asian subset for all these
models. Try to be as standardised as you can be across the different models."

## 1. Question

How well does each model, trained on White British participants at each training size, rank disease risk in UK
Biobank's Asian and Black participants? Every model is scored and evaluated by one rule, and compared with the
covariates alone in the same people.

## 2. The groups

- **People.** 9,408 Asian and 7,631 Black participants: the laboratory's second genotype set (17,055 genotyped;
  16 on the withdrawal list removed). How the laboratory assigned the two groups is not documented here.
- **Tables.** `data/datasets/ard30/hm3plus_asian_black/samples_{asian,black}.tsv`, built with the main dataset
  (DeepWideGP-Scripts `build-ard30-dataset`): the same 30 disease definitions, year of birth, sex and principal
  components 1-10. Cases per disease: 38 to 3,801 (Asian) and 18 to 3,077 (Black); skin cancer (C44) has the fewest.
- **Genotypes.** `data/genostore/hm3plus_asian_black`: the same 1,258,200 HapMap3+ variants as the main store, in
  the same order, with the same alleles. The variant tables of all 22 chromosomes are byte-identical, and the code
  checks position, ID, alleles, order and block width again whenever it reads the groups.
- **Held out.** No training, tuning, test-set scoring or choice uses them. They are scored once a model's ladder has
  finished, and evaluated for all models together.

## 3. Scoring: one rule for every model

**The rule.** A finished run is applied to a group exactly as it was applied to its tuning set: the run's own
training statistics (SNP standardisation, covariate scaling), its SNP list and signs, its PGI weights (Kelemen MLP,
Delphi), its fold routing (WideDeepGP: the pseudo-fold of a participant outside the training table), and its best
checkpoint. Nothing is estimated on the groups.

**How each model reads the groups.**
- *The shared data module* (DeepWideGP `genostore/datamodule.py`, commit c90e9ed): a split that a run's dataset
  lacks is read from the sibling dataset `hm3plus_asian_black` in its own store, after the variant check. Every such
  read is logged in that dataset's `access.log`. WideDeepGP, Kelemen MLP, Delphi, GLN, PRSformer and classic DeepGP
  all read through it. So they need no group-specific code beyond saving each person's ID with their score.
- *scPRS* reads its own genotype cache. A second cache, with the same SNP columns, holds the groups' dosages from
  the held-out store (`data/scprs/cache_heldout`).
- *Classic DeepGP* datasets are not beside the held-out dataset, so its scorer (DeepWideGP `score_genome.py`)
  points the data module at it. It rebuilds each cell's settings exactly as the ladder launched it and refuses any
  difference from the checkpoint's. Robert allowed edits to classic DeepGP's code for this, but not to its
  architecture (22 Nov 2026).
- *The PGI arm* already scores the groups in its score stage, for every method and training size it has finished.
- *WideDeepGP's linear comparators.* The cross-fitted ridge PRS already exists for the groups
  (`<rung>/blocks/prs_{asian,black}.npz`). The logistic ridge on block channels kept no coefficients, so scoring
  it needs a refit (section 6).

**The check before a group is scored.** In the same job, the run's tuning set is scored again by the same frozen
code. Its AUROCs must equal the run's record:
- within 1e-3 per disease, for scores made on a GPU in low precision;
- for classic DeepGP and scPRS, their own scorers also compare each person's score.

If the check fails, no group is scored for that run, and the job stops.

**Outputs.**
- *Per-person scores:* `data/datasets/ard30/<rung>/scores/<model>/.../<group>.npz`, beside the run's tuning-set
  file. They are owner-only, with keys `iid`, `labels`, `y` and the model's score.
- *The run's own folder is never written.* Scoring runs in a shadow folder of links
  (`runs/heldout_ancestry/shadow/`), with a `heldout.json` record per run.
- *Code:* every job runs from a frozen copy of the committed code of all repositories (`freeze.sh`, which writes
  `COMMITS.txt`).

## 4. Evaluation: one rule for every model (`evaluate.py`)

**The combination.** Each model is combined with the covariates by the study's comparison protocol (PGI
`methods.md` 6.1), in the PGI arm's own code (`pgi.combine.Combined`, frozen at b73a629):
- per disease, a logistic regression on the 20,000 tuning participants of the model's score plus the 12 covariates
  (year of birth, sex, ten principal components), each entered linearly;
- applied unchanged to the group.

**The covariate-only reference** is the same regression without the score.

**Why it is fitted on the tuning set, and not refitted in the group.** The question is whether a fixed model
transfers. Refitting in the group would use the group's own labels.

**Per model run, disease and group, it reports:**
- the combined model's AUROC, with a DeLong 95% interval (`pgi.evaluate`);
- the covariate-only AUROC;
- the gain over the covariates: the paired DeLong difference, with its standard error and p;
- the AUROC of the model's score alone;
- the score's tuning AUROC, for reference.

Counts of 1 to 9 are written `<10`. Runs with no group file yet are listed as "not scored", so coverage is visible.

**Checks, made without computing any held-out metric.**
- On the tuning set, the evaluation reproduces the PGI arm's recorded tuning AUROCs (combined, covariate-only and
  PGI alone) in all 179 (method, disease) cells of the smallest rung, to 5e-7, which is the CSV's rounding.
- From the same PGIs, it reproduces the PGI arm's own held-out logits in all 358 (method, disease, group) cells, to
  1.9e-6 (the arm stores float32).

## 5. What to keep in mind when reading the results

- **Principal components.** They are computed across UK Biobank, and the Asian and Black participants lie far
  outside the White British training range. The covariates' linear effects extrapolate there. A within-group AUROC
  uses only within-group variation, but the deep models take the components as inputs, so they see values they
  never saw in training.
- **Rare diseases have few cases in the groups,** so their intervals are wide.
- **WideDeepGP's three seeds** are reported as the mean and spread, as in the main results.

## 6. Decisions taken here, for Robert to confirm

1. **No lock, but a log.** Every read of the groups is logged, and they are scored only after a ladder has
   finished; they are never used for a choice. The alternative is a flag like the test set's `--unlock_test`.
2. **One combination for every model,** the protocol of section 4, fitted on the tuning set.
3. **Only the Asian and Black groups.** The related White British (57,445) are not included; adding them is one
   argument.
4. **The logistic-ridge twin of WideDeepGP is not scored yet.** It needs `fit_block_linear.py` to keep its
   coefficients and a refit, which takes hours per rung on the CPU. The ridge PRS comparator is evaluated already.

## 7. How to run

- **Score a model's finished runs.** Detached (CLAUDE.md 1); it waits for a free GPU and skips runs already done:

  `cd scripts/heldout-ancestry && setsid nohup bash run_heldout.sh <model>... > /dev/null 2>&1 < /dev/null &`

  The models are `kelemen delphi gln prsformer widedeepgp classic scprs`. The log is in `runs/heldout_ancestry/logs/`.
- **See what would be done:** `python score_heldout.py --model <model> --code $(bash freeze.sh) --dry_run`.
- **Evaluate everything scored so far:** `python evaluate.py`, which writes `results/heldout_auroc.csv`. Run it on
  the CPU with `/home/robert/envs/widedeepgp/bin/python`.
- **Tests:** `python -m pytest tests -q`.
