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
the unrelated White-British modelling population. It contains no test participant. Training
time is proportional to participants times SNPs, so one pass over the proxy's training
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
  proxy's training participants, gives the AUROC that needs no genotype.
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

Added during the pilot, after the read-out above showed that no arm had learned anything from
the genotypes on the real labels. That result has two possible readings: the proxy holds too
little genetic signal for any method to find, or the network cannot find signal that is there.
To separate them, five traits were simulated on the proxy's genotypes, so that the signal is
present by construction and its size is known:

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
  channel keeps the *z*-score only at that trait's clumped lead SNPs and is zero elsewhere:
  the weights of a clumped polygenic score, given to the network as annotation.
- **Longer training.** The run on GWAS-selected SNPs continued from 6 to 30 epoch-equivalents.
- **Other optimisation settings.** A ten-trial Optuna search (learning rate, weight decay,
  dropout, accumulation, warm-up; four epoch-equivalents per trial, successive halving) on the
  GWAS-selected SNPs, with the mean AUROC of the five traits as objective.

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
   same bands as the current block.

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

Two hundred training steps at a batch of 18 on the full dataset (215,872 training
participants, 1,258,200 SNPs), followed by one validation pass over the 20,000 tuning
participants, to confirm memory, throughput and the cost of validation before a full-size run
is committed to.

## Software

`DeepWideGP` at branch `widening`; Python 3.10, PyTorch 2.10, PyTorch Lightning 2.0.8; the
Vision-Mamba fork of `mamba_ssm` 1.1.1 for the current block and upstream `mamba_ssm` 2.3.2 in
a second environment for Mamba-3.

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
    python scripts/widedeepgp-pilot/gwas_annotation.py             # z-scores of the 30 diseases as annotation
    python scripts/widedeepgp-pilot/run_pilot.py --arms gwas_annot
    python scripts/widedeepgp-pilot/analyze_pilot.py        # tables and figures in results/

## Limitations

- The proxy has an eighth of the SNPs and under a fifth of the training participants, and runs
  for six epoch-equivalents. It ranks training schemes; it does not estimate the accuracy of
  the full model, and a scheme's advantage could change with more data or longer training.
- With so little genetic information, most of the AUROC comes from the covariates. Arms that
  differ only in how the genotypes are used (annotations, the sequence block) have little room
  to separate, so a tie in the pilot is weak evidence of no difference at full size.
- Two seeds per arm give a coarse measure of seed-to-seed variation.
- Hyper-parameters were not tuned for any arm; all arms use the published single-disease
  settings.
