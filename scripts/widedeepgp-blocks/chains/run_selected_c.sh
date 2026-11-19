#!/bin/bash
# The third selected-SNP arm (design document section 12, item 19; Robert, 19 Nov 2026): the unclumped set from zero,
# `sel_p01any_S3_zero`, which separates the SNP set from the start in the other two arms. Waits for run_selected.sh's
# last GPU launch (so the two chains never race for a GPU), then takes a GPU among 2, 3 and 4 once at least two are
# free, and trains with the same settings as run_selected.sh's arms, without --init_encoder.
#   setsid nohup bash run_selected_c.sh > /dev/null 2>&1 < /dev/null &
set -u
ROOT=/mnt/sdg/robert/deepwidegp
PY=/home/robert/envs/widedeepgp/bin/python
DATA=$ROOT/data
PARTS=$DATA/reference/partitions
OUT=$ROOT/runs/selected
LOG=$OUT/chain.log
RUN=sel_p01any_S3_zero
NAME=p01any
N=215872
mkdir -p "$OUT/logs"
[ -e "$OUT/run_selected_c.sh" ] || cp "$0" "$OUT/run_selected_c.sh"
say() { echo "$(date -u +%FT%TZ) [c] $*" >> "$LOG"; }
free_gpus() {
    local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | while IFS=, read -r idx uuid; do
        case "$idx" in 2|3|4) grep -q "$uuid" <<< "$busy" || echo "$idx";; esac
    done
}
take_gpu() {
    local f
    while :; do
        f=($(free_gpus))
        [ ${#f[@]} -ge 2 ] && { echo "${f[0]}"; return; }
        sleep 180
    done
}
[ -e "$OUT/$RUN/results.json" ] && { say "train $RUN: exists"; exit 0; }
say "waiting for the main chain's last GPU launch (train sel_p01anyct_S3_zero) and the $NAME partition, pid $$"
until [ -e "$OUT/logs/train_sel_p01anyct_S3_zero.log" ] && [ -e "$PARTS/prive_$NAME/partition.npz" ]; do
    grep -q "FAILED" "$LOG" 2>/dev/null && { say "the main chain failed; not starting $RUN"; exit 1; }
    sleep 300
done
sleep 300    # the main chain's run has claimed its GPU
CODE=$(ls -d "$OUT"/code_* | head -n 1)
gpu=$(take_gpu); say "train $RUN: starting on GPU $gpu from $(basename "$CODE")"
(cd "$CODE" && CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 $PY main_multilabel.py \
    --data_dir "$DATA/datasets" --label ard30 --rd_mode "${NAME}_n$N" --input_mode blocks --wide off --partition "$PARTS/prive_$NAME" \
    --mixer transformer --snp_code dosage --embed_dim 16 --d_model 64 --e_layers 4 --n_heads 4 --head_hidden 256 --n_fcn 2 \
    --final_pool mean --snp_stats train --lr 1e-4 --l2 1e-3 --dropout 0.2 --input_mask 0 --batch_size 64 --accumulate 1 \
    --eval_batch_size 1024 --precision bf16 --num_workers 10 --max_epoch_equiv 10 --val_per_epoch 4 --val_min_participants 20000 \
    --val_max_epoch_equiv 1 --min_participants 128000 --stop_patience 16 --stop_burnin 2000 --tune_min_validations 8 \
    --lr_schedule plateau --warmup_steps 0 --snapshots_per_decade 6 --snapshots_max 20 --run_dir "$ROOT/runs" --experiment selected \
    --exp_name "$RUN" --seed 1 --tune_trials 0 --sampler shuffled --gamma 0 --table_kappa 0.1 --table_eps 1e-8 --table_decay 0.01) \
    >> "$OUT/logs/train_$RUN.log" 2>&1
rc=$?; echo "exit $rc at $(date -u +%FT%TZ)" >> "$OUT/logs/train_$RUN.log"
say "train $RUN: $(tail -n 1 "$OUT/logs/train_$RUN.log")"
say "[c] exit $rc"
