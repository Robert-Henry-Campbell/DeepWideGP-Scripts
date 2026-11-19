#!/bin/bash
# The no-covariate comparison at the top rung (Robert, 19 November 2026; methods.md item 20 and the classic comparator's
# methods, "Without covariates"): two single trainings on the 215,872 training participants with --dm snps, which drops
# the covariate encoder, token and vector, so that each network's tuning AUROC is what it reads from the genotypes alone.
#   classic_i10_n215872_nocov            the classic DeepGP for I10 at the published settings, as runs/classic_deepgp/
#                                        classic_i10_n215872_quick (0.6987 = the covariate floor), minus the covariates.
#   blocks_S3_folds_frozen_nomask_nocov  WideDeepGP as the best run so far, runs/blocks_diag/diag_S3_folds_frozen_nomask
#                                        (0.6735; seed 2 0.6737): S3 transformer on the Prive blocks, the fold-routed ridge
#                                        warm start of the per-SNP table frozen, mean readout, no input mask, dropout 0.2,
#                                        all 30 diseases, ten epoch-equivalents, seed 1, minus the covariates.
# Yardsticks without covariates (tuning set): the cross-fitted block ridge PRS alone, 0.5666 mean over the 30 diseases and
# 0.5989 for I10 (data/datasets/ard30/hm3plus/blocks/ridge_summary.csv); the PGI arm's scores alone for I10 (C+T 0.594,
# lassosum2 0.604, LDpred2-auto 0.615, SBayesRC 0.616); chance 0.5.
# GPUs: each run takes one of GPUs 2 to 4 once at least two of them are free (CLAUDE.md section 2), the highest free index
# (the selected chain takes the lowest); the short classic run goes first. Ten threads for the blocks run, four for the
# classic one, as their reference runs. A relaunch resumes an unfinished run from its last checkpoint and skips a run whose
# results.json exists; each best snapshot is then scored on the tuning set beside the covariates-only logistic model
# (score_snapshots.py), the floor the runs do not have. Code: a frozen export of DeepWideGP widening, pinned at the first launch.
# Detached:  setsid nohup bash run_nocov.sh > /dev/null 2>&1 < /dev/null &
set -u
ROOT=/mnt/sdg/robert/deepwidegp
PY=/home/robert/envs/widedeepgp/bin/python
OUT=$ROOT/runs/nocov
DATA=$ROOT/data/datasets
BLOCKS=$DATA/ard30/hm3plus/blocks
L=$OUT/run_nocov.log
mkdir -p "$OUT/logs"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }

[ -e "$OUT/code_commit" ] || git -C "$ROOT/DeepWideGP" rev-parse --short widening > "$OUT/code_commit"
H=$(cat "$OUT/code_commit"); CODE=$OUT/code_$H
if [ ! -d "$CODE" ]; then
    rm -rf "${CODE:?}.tmp" && mkdir -p "$CODE.tmp" && git -C "$ROOT/DeepWideGP" archive "$H" | tar -x -C "$CODE.tmp" && mv "$CODE.tmp" "$CODE" \
        || { say "export of DeepWideGP $H failed; exit 1"; exit 1; }
fi
say "starting from code_$H, pid $$"

free_gpus() {   # the indices among 2, 3 and 4 with no compute process
    local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | while IFS=, read -r idx uuid; do
        case "$idx" in 2|3|4) grep -q "$uuid" <<< "$busy" || echo "$idx";; esac
    done
}
take_gpu() {    # waits until at least two of the three are free; prints the highest free index
    local f
    while :; do
        f=($(free_gpus))
        [ ${#f[@]} -ge 2 ] && { echo "${f[-1]}"; return; }
        sleep 180
    done
}
train() {       # name threads flags...: one training (resumed if a checkpoint exists), then the tuning-set scoring
    local name=$1 threads=$2; shift 2
    local log=$OUT/logs/$name.log gpu="" resume="" rc
    if [ ! -e "$OUT/$name/results.json" ]; then
        gpu=$(take_gpu); say "train $name: starting on GPU $gpu"
        [ -e "$OUT/$name/checkpoints/last.ckpt" ] && resume="--resume"
        (cd "$CODE" && CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads $PY main_multilabel.py \
            --data_dir "$DATA" --label ard30 --dm snps --wide off --sampler shuffled --gamma 0 --seed 1 --tune_trials 0 \
            --run_dir "$ROOT/runs" --experiment nocov --exp_name "$name" "$@" $resume) >> "$log" 2>&1
        rc=$?; say "train $name: exit $rc"
        { [ $rc -eq 0 ] && [ -e "$OUT/$name/results.json" ]; } || return 1
    fi
    if [ ! -e "$OUT/$name/scores_val_summary.csv" ]; then
        [ -n "$gpu" ] || gpu=$(take_gpu)
        (cd "$CODE" && CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads $PY score_snapshots.py \
            --run "$OUT/$name" --split val --snapshots best --batch_size 1024) >> "$log" 2>&1
        say "score $name: exit $?"
    fi
}

( train classic_i10_n215872_nocov 4 --rd_mode classic_i10_n215872 --labels I10 --input_mode snps --input_mask 0 \
    --lr_schedule constant --warmup_steps 0 --stop_patience 8 --stop_burnin 0 --batch_size 64 --accumulate 1 \
    --lr 7e-4 --l2 1e-4 --dropout 0.2 --d_model 32 --e_layers 1 --n_heads 4 --n_fcn 2 --head_hidden 256 \
    --final_pool atten --snp_embed cov --precision 32 --eval_batch_size 256 --num_workers 4 ) &
sleep 240       # the first run claims its GPU before the second looks
( train blocks_S3_folds_frozen_nomask_nocov 10 --rd_mode hm3plus --input_mode blocks \
    --partition "$ROOT/data/reference/partitions/prive431" --mixer transformer --snp_code dosage --embed_dim 16 \
    --d_model 64 --e_layers 4 --n_heads 4 --head_hidden 256 --n_fcn 2 --final_pool mean --snp_stats train \
    --lr 1e-4 --l2 1e-3 --dropout 0.2 --input_mask 0 --batch_size 64 --accumulate 1 --eval_batch_size 1024 --precision bf16 \
    --num_workers 10 --max_epoch_equiv 10 --val_per_epoch 4 --val_min_participants 20000 --val_max_epoch_equiv 1 \
    --min_participants 128000 --stop_patience 16 --stop_burnin 2000 --tune_min_validations 8 --lr_schedule plateau \
    --warmup_steps 0 --snapshots_per_decade 6 --snapshots_max 20 --folds "$BLOCKS/folds.npz" \
    --init_encoder "$BLOCKS/init_ridge_d16.pt" --freeze_encoder --table_kappa 1e-3 --table_decay 0.0 --table_eps 2.5e-7 ) &
wait
for r in classic_i10_n215872_nocov blocks_S3_folds_frozen_nomask_nocov; do
    say "$r: $(tail -n 1 "$OUT/logs/$r.log" 2>/dev/null | cut -c1-200)"
done
say "exit 0"
