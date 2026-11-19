#!/bin/bash
# Real-data check of the first-layer cache (plan ~/.claude/plans/first-layer-cache.md, 19 Nov 2026), from a frozen copy
# of DeepWideGP widening: (1) the best snapshot of blocks_diag/diag_S3_folds_frozen_nomask re-scored on the tuning set
# from the cache; (2) the same configuration trained 1 epoch-equivalent with the cache; (3) its best snapshot re-scored
# by streaming; (4) a summary. Each step is skipped when its output exists. Detached:
#   setsid nohup bash run_check.sh > /dev/null 2>&1 < /dev/null &
set -u
ROOT=/mnt/sdg/robert/deepwidegp
PY=/home/robert/envs/widedeepgp/bin/python
OUT=$ROOT/runs/cache_check
ORIG=$ROOT/runs/blocks_diag/diag_S3_folds_frozen_nomask
NAME=S3_folds_frozen_nomask_cached_1ee
L=$OUT/run_check.log
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
run() {         # name command...: one step on a free GPU, logged; stops the chain on failure
    local name=$1 gpu; shift
    gpu=$(take_gpu); say "$name: starting on GPU $gpu"
    (cd "$CODE" && CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 "$@") >> "$OUT/$name.log" 2>&1
    local rc=$?; say "$name: exit $rc"
    [ $rc -eq 0 ] || { echo "exit $rc at $(date -u +%FT%TZ)" >> "$L"; exit $rc; }
}

[ -e "$OUT/check_rescore.json" ] || run rescore $PY "$OUT/check_rescore.py" "$CODE" "$ORIG" "$OUT/check_rescore.json"
[ -e "$OUT/$NAME/results.json" ] || run train $PY main_multilabel.py --data_dir $ROOT/data/datasets --label ard30 \
    --rd_mode hm3plus --input_mode blocks --wide off --partition $ROOT/data/reference/partitions/prive431 --mixer transformer \
    --snp_code dosage --embed_dim 16 --table_eps 2.5e-7 --d_model 64 --e_layers 4 --n_heads 4 --head_hidden 256 --n_fcn 2 \
    --final_pool mean --snp_stats train --lr 1e-4 --l2 1e-3 --dropout 0.2 --input_mask 0 --batch_size 64 --accumulate 1 \
    --eval_batch_size 1024 --precision bf16 --num_workers 10 --max_epoch_equiv 1 --val_per_epoch 4 \
    --val_min_participants 20000 --val_max_epoch_equiv 1 --min_participants 128000 --stop_patience 16 --stop_burnin 2000 \
    --tune_min_validations 8 --lr_schedule plateau --warmup_steps 0 --snapshots_per_decade 6 --snapshots_max 20 \
    --run_dir $ROOT/runs --experiment cache_check --exp_name $NAME --folds $ROOT/data/datasets/ard30/hm3plus/blocks/folds.npz \
    --sampler shuffled --gamma 0 --seed 1 --tune_trials 0 --init_encoder $ROOT/data/datasets/ard30/hm3plus/blocks/init_ridge_d16.pt \
    --freeze_encoder --table_kappa 1e-3 --table_decay 0.0
[ -e "$OUT/$NAME/scores_val_summary.csv" ] || run rescore_streamed $PY score_snapshots.py --run "$OUT/$NAME" --split val \
    --snapshots best --batch_size 1024
(cd "$CODE" && $PY "$OUT/summarise_check.py" "$ORIG" "$OUT/$NAME" "$OUT/check_rescore.json" "$OUT/check_summary.md") >> "$L" 2>&1
echo "exit $? at $(date -u +%FT%TZ)" >> "$L"
