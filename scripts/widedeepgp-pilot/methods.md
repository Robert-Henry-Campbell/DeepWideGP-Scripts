# Methods: pilot of the multi-disease DeepGP (WideDeepGP)

*Analysis `widedeepgp-pilot` (DeepWideGP; DPhil chapter 6). Run on 11–12 November 2026. The
model is described in `DeepWideGP/methods.md` and the data in
`scripts/build-ard30-dataset/methods.md`.*

## Objective

Before any full-size training, check on real data that the multi-disease model trains as
designed, and settle the design choices that were left to a pilot:

1. whether all 30 outputs should learn from every batch, or only the output of the disease a
   batch was balanced on;
2. whether disease-balanced batches train better than uniformly drawn ones;
3. whether the sampling-rate correction and the square-root class weights behave as intended;
4. whether running the encoders in bf16 changes accuracy;
5. whether per-SNP annotation vectors help;
6. whether Mamba-3 can replace the current sequence block.

Two further checks support these: the largest batch, memory and speed of each configuration at
the full dataset's size (`run_capacity.py`), and a dry run of the full dataset.

## Data

The proxy dataset of `build-ard30-dataset` (`hm3plus_proxy8`): the 30 phenotypes and 12
covariates of the full dataset, 157,279 SNPs (the second eighth of every chromosome, as
contiguous stretches), 40,000 training participants and 10,000 tuning participants, all from
the unrelated White-British modelling population. It contains no test participant: its third
table, used below to choose thresholds, is a further 10,000 participants drawn from the full
training split, as its `proxy_log.json` records. A check of the participant lists found the
three tables disjoint, none of their participants in the real test set, and the GWAS sample of
`gwas_annotation.py` disjoint from all of them and from the full tuning and test splits.
Training time is proportional to participants times SNPs, so one pass over the proxy's training
participants costs about a fortieth of one pass over the full training split.

## Arms

Every arm trains the same network (model width 32, one encoder layer per chromosome, four
attention heads) with AdamW (learning rate 7 × 10⁻⁴, weight decay 10⁻⁴, embedding dropout 0.2:
the published single-disease settings), batches of 16 participants with gradients accumulated
over 4 batches (64 participants per update, the published batch size), for 6 epoch-equivalents
(240,000 participants drawn; 3,750 updates), without early stopping, so that arms are compared
at equal amounts of training. The tuning participants are scored twice per epoch-equivalent.

| Arm | Differs from the default in |
|---|---|
| Default | — (rotation of disease-balanced batches; all outputs learn; sampling-rate correction; square-root class weights; 32-bit; no annotations; bidirectional Mamba block) |
| Only the target learns | the loss: only the output of the batch's target disease is trained |
| Shuffled batches | the sampler: participants drawn uniformly, same loss |
| No sampling-rate correction | the loss is not divided by each participant's sampling rate |
| Full class balance | class-weight exponent 1 instead of ½ |
| bf16 encoders | encoders in bf16 (pooling, head and loss stay 32-bit) |
| Annotation vectors | 96 functional annotations per SNP as a gain and offset on its token |
| Mamba-3 block | the sequence block (two Mamba-3 blocks per layer, one per direction) |

Each arm was run with two seeds (1 and 2), which set the initial weights and the order of
batches. Runs were executed one per GPU (NVIDIA RTX 6000 Ada, 48 GB).

## Read-out

- **Primary:** the best mean tuning AUROC of a run: the mean over the 30 diseases of the AUROC
  on the 10,000 tuning participants, at the validation pass where that mean was highest.
- **Seed-to-seed spread:** the range of the two seeds of an arm. An arm is said to differ from
  the default when the difference between the two arms' means exceeds the larger of their two
  seed ranges. With two seeds this is a coarse yardstick, not a test.
- **Equal GPU time:** arms that run at a different speed from the default (bf16, annotations,
  Mamba-3) are also compared at the GPU time that both had reached, using the best mean AUROC
  each had attained by then.
- **Covariates-only baseline:** a logistic model per disease on the 12 covariates, fitted on the
  proxy's training participants, gives the AUROC that needs no genotype. Every run and arm is
  also given as its gain over this model (`results/pilot_runs.csv`, `results/pilot_arms.csv`).
- Per-disease AUROCs, cross-entropy on the population scale, training time and peak GPU memory
  are tabulated for every run.

## The part of the AUROC that comes from the genotypes (`genetic_signal.py`)

Year of birth and sex predict these diseases well by themselves, so a model's AUROC says
little about what it learned from the SNPs. Each run's best checkpoint was therefore scored on
the tuning participants twice: as they are, and after giving every participant the genotypes
of another tuning participant while keeping their own covariates and labels (three random
permutations, averaged). The AUROC lost when the genotypes are permuted is the part that
depends on a person's own genotypes; a model that ignores the SNPs, or draws only noise from
them, loses nothing. The interval is a bootstrap over participants (300 resamples) of the mean
loss over the 30 diseases.

## Positive control: simulated traits on the real genotypes (`simulated_control.py`)

Added during the pilot, after the read-out above showed that no arm had learned anything
detectable from the genotypes on the real labels. That result has two possible readings: the
proxy holds too little genetic signal for any method to find, or the network cannot find signal
that is there. To separate them, five traits were simulated on the proxy's genotypes, so that
the signal is present by construction and its size is known:

| Trait | Causal SNPs | Heritability of liability | Causal SNPs drawn |
|---|---|---|---|
| sim20 | 20 | 0.3 | uniformly |
| sim200 | 200 | 0.5 | uniformly |
| sim2000 | 2,000 | 0.5 | uniformly |
| sim20000 | 20,000 | 0.5 | uniformly |
| sim200a | 200 | 0.5 | 20 times more often from coding or conserved SNPs (7% of the SNPs) |

Causal SNPs were drawn from the proxy's 157,279 SNPs and given normally distributed effects on
the standardised genotype. The genetic value was scaled to the stated heritability, normal
noise was added to a liability of variance one, and the 20% of participants with the highest
liability were called cases. The covariates have no effect on these traits, so their AUROC is
entirely genetic. Participants, SNPs, covariates and splits are those of the proxy
(seed 20261113).

Three references were computed from the same 40,000 training participants and scored on the
10,000 tuning participants: the true genetic value (the ceiling); a polygenic score from
marginal effects, without and with clumping (the strongest SNP of each cluster with
*r*² > 0.1 within 500 SNPs, among SNPs with *p* < 10⁻³), with the *p*-value threshold chosen
on the proxy's third table; and GBLUP with the true heritability, the best linear predictor
when every SNP has a small effect.

The default arm, the Mamba-3 arm and the annotation arm were then trained on the five traits
with the pilot's settings (one seed each). The enriched trait gives the annotation arm
something real annotations can help with: two of the 97 channels mark where its causal SNPs
concentrate.

Further runs of the default configuration change what the network is given, or how long it
is trained, to locate the obstacle:

- **GWAS-selected SNPs.** The same traits and participants, with the input reduced to the
  2,476 SNPs that are a clumped lead SNP (*p* < 10⁻³ in the training participants) for at
  least one trait: between 26 and 267 per chromosome. This is the kind of input published
  DeepGP receives. The selection uses training participants only.
- **Oracle annotation.** All 157,279 SNPs, with five annotation channels, one per trait, that
  mark its causal SNPs, used through the model's annotation gain and offset. The mark says
  where the causal SNPs are, not in which direction each acts.
- **Association statistics as annotation.** All 157,279 SNPs, with five annotation channels
  holding every SNP's signed marginal *z*-score for each trait, computed in the training
  participants. Through the annotation gain the network can then weight each SNP's dosage by
  the size and sign of its association, which is the per-SNP information a network with shared
  weights does not otherwise have.
- **Clumped association statistics as annotation.** As the previous run, but each trait's
  channel keeps the *z*-score only at that trait's clumped lead SNPs (all of them with
  *p* < 10⁻³, 243 to 660 per trait) and is zero elsewhere: the candidate weights of a clumped
  polygenic score, given to the network as annotation. The score itself keeps only the leads
  beyond a threshold chosen on the third table (55 to 660 SNPs).
- **Longer training.** The run on GWAS-selected SNPs continued from 6 to 30 epoch-equivalents:
  its folder was copied (`pilot_sim/selected_s1` to `pilot_sim/selected_long_s1`) and the copy
  resumed from its last checkpoint with a limit of 30. Its `checkpoints/` folder therefore still
  holds the copied run's `best.ckpt` and `last.ckpt` (both from the end of the first 6
  epoch-equivalents) beside the continued run's `best-v1.ckpt` and `last-v1.ckpt`. Its
  `results.json` names the right files, and `genetic_signal.py` and `score_snapshots.py`
  (`--snapshots best`) follow it, but a script that opens `checkpoints/best.ckpt` or
  `checkpoints/last.ckpt` by name, or a `--resume` with the pilot's code (8f03ca6, which starts
  from `last.ckpt`), would use the copied run's weights. The current code resumes such a folder
  from `last-v1.ckpt`, and a run directory copied and resumed with it carries on in `best.ckpt`
  and `last.ckpt` with the copied run's best score.
- **Other optimisation settings.** An Optuna search of eleven trials (learning rate, weight
  decay, dropout, accumulation, warm-up; four epoch-equivalents per trial, successive halving)
  on the GWAS-selected SNPs, with the mean AUROC of the five traits as objective. Trial 0 was
  stopped by hand after five validation passes, to free its GPU for `gwas_annotation.py`, and
  is recorded as failed; the study then continued in a new worker on another GPU with sampler
  seed 2 instead of 1, and that worker's log replaced the first one's. Of the other ten
  trials, nine were pruned and one completed.

Every run of the control is read at the end of its training (Figure 5 and the control table of
`results.md`). The run continued to 30 epoch-equivalents is also given at its best validation
pass; that pass is chosen on the tuning participants, so its AUROCs are optimistic.

## Association statistics as annotation on the real labels (`gwas_annotation.py`)

Added after the positive control, in which the network learned from the SNPs once each SNP came
with a signed association statistic. The same input was built for the 30 diseases without
letting any participant inform their own annotation:

- **GWAS sample.** The 165,872 participants of the full training split that are in none of
  the proxy's tables. They share no participant with the proxy's training or tuning
  participants or with the test set.
- **Statistics.** For every proxy SNP and disease, the *z*-score of the SNP in a linear
  regression of case status on the standardised dosage, after year of birth, sex and the ten
  principal components had been regressed out of case status.
- **Arm.** The default configuration with these 30 *z*-scores per SNP as its annotation
  vectors (in place of the functional annotations), trained on the proxy as every other arm.
- **Yardstick.** What a linear method draws from the same statistics: for each disease a
  clumped polygenic score (lead SNPs with *p* < 10⁻³; SNPs with *r*² > 0.1 within 500 SNPs of
  a stronger lead removed, linkage disequilibrium from the proxy's training participants;
  *p*-value threshold chosen on the proxy's third table), added to the 12 covariates in a
  logistic model fitted on the proxy's training participants and scored on its tuning
  participants.
- **Alignment of labels and genotypes.** The GWAS reads genotypes through the same map from
  participant to genotype row as training does, so it also shows that the two are joined
  correctly: its strongest associations lie at textbook loci, 9p21 for the coronary diseases,
  *LDLR* for lipid disorders, *ABCG8* for gallstones and the MHC for hypothyroidism.

## Decision rules, fixed before the pilot

- **Mamba-3** replaces the current block only if it passes its gates (below) and its mean
  tuning AUROC exceeds the default's by more than the seed-to-seed spread at equal GPU time.
  A tie goes to the current block.
- **Annotation vectors** enter the main model only if they raise the mean tuning AUROC by more
  than the seed-to-seed spread. A model without annotations is kept in any case, as the
  like-for-like comparator of polygenic scores that use none.
- **bf16** is adopted for full-size runs if its mean tuning AUROC is within the seed-to-seed
  spread of the 32-bit default.
- The other arms are comparisons, not candidates: they test whether the default's design
  choices (all outputs learn; rotation; correction; square-root weights) hold up.

## Mamba-3 gates

Mamba-3 (upstream `mamba_ssm` 2.3.2, single-input mode, state size 16) runs in one direction;
the model wraps two blocks per layer, one per direction, with separate weights and summed
outputs. Before it entered the pilot it had to pass, in order:

1. a forward and backward pass on a 2,048-token input in 32-bit and bf16;
2. tests of the wrapper: one block reads in one direction only; the wrapped pair reads both
   ways; reversing the input and exchanging the two blocks' weights reverses the output;
3. a training step at full length (all 22 chromosomes, 1,258,200 SNPs, the longest 105,260)
   with finite outputs and gradients;
4. capacity: largest batch, memory and speed at full length;
5. the six-trait synthetic run of `DeepWideGP/scripts/multilabel_synthetic.sh`, held to the
   same bands as the current block (record:
   `/mnt/sdg/robert/deepwidegp/runs/gate_evidence/multi6_mamba3_gate5`).

## Capacity (`run_capacity.py`, `capacity_probe.py`)

For each configuration (sequence block × precision × checkpointing, plus forward-only passes
and the annotated model) the batch size was doubled from 8 until a training step no longer
fitted on one 48 GB GPU and then bisected. A trial is a separate process that builds the model
at the full dataset's shape, runs four steps on random inputs of the right shape (no
participant data) and reports peak allocated memory and the median time of the last three
steps.

The search ran on a GPU of its own. One trial was discarded and repeated because a pilot run
had been started on the same GPU while it ran (the first run of the shuffled arm shared its GPU
for about a minute as a result, which affects that run's timing slightly and nothing else).

## Dry run at full size

Two hundred training steps at a batch of 18 on the full dataset (215,872 training participants,
1,258,200 SNPs), followed by one validation pass over the 20,000 tuning participants, to confirm
memory, throughput and the cost of validation before a full-size run is committed to
(`/mnt/sdg/robert/deepwidegp/runs/dryrun/fp32_b18`). It was launched on 11 November at 22:50
from the repository's working tree, and training began at 23:00, after the SNP statistics had
been computed. That tree was committed at 23:09, in the commits ending at 8f03ca6, after four
files the run had already loaded were edited: help texts in `args_generator.py`, Mamba-3's
activation checkpointing in `widedeepgp/model.py`, and the steps at which training metrics are
logged in `widedeepgp/callbacks.py` and `train.py`; none of these edits changes this
configuration's training or what is compared below. It was repeated on 12 November with bf16
encoders and otherwise the same settings (the same seed, 33, and so the same batches; a batch of
18), from the export of 8f03ca6 described under Software
(`/mnt/sdg/robert/deepwidegp/runs/dryrun/bf16_b18`), to check bf16 at the full sequence length,
which the pilot's arms, at most 13,158 SNPs per chromosome, do not reach. The training loss
(logged every 50 steps), the validation metrics, the step time and the peak memory of the two
runs are compared. The two runs shared the machine with different other jobs, so the ratio of
their step times is approximate; the capacity trials, each on a GPU of its own, measure it
cleanly.

## Software

`DeepWideGP` at branch `widening`; every training run of the pilot's arms and of the control,
and the bf16 repeat of the dry run (not the 32-bit dry run; see above), used an export of commit
8f03ca6 (`/mnt/sdg/robert/deepwidegp/runs/pilot_code_8f03ca6`), so that later changes to the
repository could not make runs differ. The scripts below load the model code from `DEEPGP_REPO`,
which defaults to the live repository: set
`export DEEPGP_REPO=/mnt/sdg/robert/deepwidegp/runs/pilot_code_8f03ca6` to rerun them on the
code the runs used. Python 3.10, PyTorch 2.10, PyTorch Lightning 2.0.8; the Vision-Mamba fork of
`mamba_ssm` 1.1.1 for the current block and upstream `mamba_ssm` 2.3.2 in a second environment
for Mamba-3. `gwas_annotation.py` needs one GPU for a few minutes; it and `simulated_control.py`
stop unless the dataset's third table is recorded as a draw from the training split, because
they read that table without the test lock of the model repository. Two checks run on the CPU:
`check_control_data_path.py` recomputes the control's ceilings through the model's loader, and
`position_code_capacity.py` fits true effect profiles on the position code. Unit tests on
made-up numbers: `test_pilot.py` and `test_pilot_checks.py`.

    python scripts/build-ard30-dataset/make_proxy_dataset.py
    python scripts/widedeepgp-pilot/run_pilot.py            # 16 runs, one per GPU
    python scripts/widedeepgp-pilot/run_capacity.py
    python scripts/widedeepgp-pilot/genetic_signal.py       # scores with permuted genotypes (one GPU)
    python scripts/widedeepgp-pilot/simulated_control.py build
    python scripts/widedeepgp-pilot/run_pilot.py --experiment pilot_sim --rd-mode hm3plus_proxy8_sim \
        --arms default mamba3 annot --seeds 1
    python scripts/widedeepgp-pilot/simulated_control.py select    # then train with --rd_mode hm3plus_proxy8_sim_selected
    python scripts/widedeepgp-pilot/simulated_control.py oracle    # then train with --rd_mode hm3plus_proxy8_sim_oracle --annot
    python scripts/widedeepgp-pilot/simulated_control.py gwas      # then train with --rd_mode hm3plus_proxy8_sim_gwas --annot
    python scripts/widedeepgp-pilot/simulated_control.py summarise
    python scripts/widedeepgp-pilot/check_control_data_path.py     # the control's ceilings through the loader (CPU)
    python scripts/widedeepgp-pilot/position_code_capacity.py      # what the position code can represent (CPU)
    python scripts/widedeepgp-pilot/gwas_annotation.py             # z-scores of the 30 diseases as annotation
    python scripts/widedeepgp-pilot/run_pilot.py --arms gwas_annot
    python scripts/widedeepgp-pilot/analyze_pilot.py        # tables and figures in results/

## Departures from the plan

The plan specified the eight arms, the capacity measurements and the dry run. Three analyses
were added while the pilot ran, because the planned read-out turned out not to bear on the
question: the scoring with permuted genotypes, the positive control on simulated traits with its
variants, and the arm with association statistics as annotation on the real labels. Each was
designed before its own result was seen but in the light of the results before it: the variants
of the control were chosen one after another to locate the obstacle. They are an adaptive
sequence of experiments, not a pre-specified one, and should be read as such.

One check was added after the pilot: the bf16 repeat of the dry run (12 November, above),
because the bf16 arm had been compared only at the proxy's sequence lengths.

## Limitations

- **The positive control is at pilot scale**: 40,000 training participants, 157,279 SNPs, a
  model of width 32 with one layer, at most 30 epoch-equivalents. It shows that the network,
  given all the SNPs, learns next to no SNP effects there while linear methods learn them; it
  does not show that a larger model trained for longer on five times the participants could not.
- **The simulated traits are simpler than real ones**: additive, with normally distributed
  effects, no covariate effects and no dominance or interaction, which a network could in
  principle exploit and a linear method could not.
- **The control's runs have one seed each**, and the association statistics given to the
  network in the control were computed from its own training participants. Tuning participants
  played no part in them, so the reported AUROCs are honest, but the network was partly fitting
  its own labels' noise.
- **The account of why** (a position code too coarse to address single SNPs) is an argument
  consistent with the results, not something the experiments isolate.
- **The read-out with permuted genotypes cannot see a SNP path that hurts**: a path that only
  adds noise adds the same noise when the genotypes are permuted, so it shows no drop. Every
  network of the pilot ended below the covariates-only model, which this read-out cannot
  explain; the gain over that model is therefore reported beside it.
- The proxy has an eighth of the SNPs and under a fifth of the training participants, and runs
  for six epoch-equivalents. On the real labels it can rank the training machinery (speed,
  memory, calibration, and how well a training scheme fits what the covariates carry, as AUROC
  differences) but not designs by what they learn from the SNPs, of which it holds very little
  (section 5 of the results); it does not estimate the accuracy of the full model, and a
  scheme's advantage could change with more data or longer training.
- With so little genetic information, most of the AUROC comes from the covariates. Arms that
  differ only in how the genotypes are used (annotations, the sequence block) have little room
  to separate, so a tie in the pilot is weak evidence of no difference at full size.
- Two seeds per arm give a coarse measure of seed-to-seed variation.
- Hyper-parameters were not tuned for any arm; all arms use the published single-disease
  settings.
