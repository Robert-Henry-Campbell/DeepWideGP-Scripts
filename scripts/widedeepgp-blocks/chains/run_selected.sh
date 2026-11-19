#!/bin/bash
# The selected-SNP from-scratch arms of the LD-block model (design document section 12, item 19; Robert, 19 Nov 2026).
# Two SNP sets from the 30 top-rung GWAS, p < 0.01 in any disease: unclumped (p01any, PRSformer's input rule) and
# clumped once at r2 0.1 within 250 kb (p01anyct). For each: the store, its Prive partition, folds, the closed-form
# block ridge (the linear yardstick on the same SNPs), then one S3 run of the model trained from scratch with the
# un-handicapped table optimiser; the unclumped arm starts from masked pretraining, the clumped arm from zero. Every
# step is skipped when its product exists, so a rerun continues; the chain stops at the first failure. GPUs: among
# 2, 3 and 4, a GPU step starts only when at least two are free and takes one (free minus one). CPU steps: 8 threads.
#   setsid nohup bash run_selected.sh > /dev/null 2>&1 < /dev/null &
# Log: /mnt/sdg/robert/deepwidegp/runs/selected/chain.log; step logs in runs/selected/logs/.
set -u
ROOT=/mnt/sdg/robert/deepwidegp
PY=/home/robert/envs/widedeepgp/bin/python
SCRIPTS=$ROOT/DeepWideGP-Scripts/scripts
BLOCKS=$SCRIPTS/widedeepgp-blocks
DATA=$ROOT/data
DS=$DATA/datasets/ard30
PARTS=$DATA/reference/partitions
REF=$DATA/reference/hm3plus_in_ukb_pgen.tsv
OUT=$ROOT/runs/selected
LOG=$OUT/chain.log
DISEASES="I10 E78 K57 K21 K29 K44 I25 M19 I84+K64 I48 M17 C44 E11 E66 M54 D12 I20 E03 K40 M16 K80 N17 N18 J44 M23 M47 I95 I50 M81 I21"
N=215872
mkdir -p "$OUT/logs"
[ -e "$OUT/run_selected.sh" ] || cp "$0" "$OUT/run_selected.sh"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
fail() { say "FAILED: $*"; say "exit 1"; FAILED=1; exit 1; }
# the script's own errors go to a log, and any exit that is neither the end nor fail() is recorded as a failure,
# so the chain cannot die silently (19 Nov: an unbound variable under set -u killed it unseen) and the waiter stops
exec 2>> "$OUT/logs/chain_stderr.log"
FAILED=0; DONE=0
trap '[ $DONE = 1 ] || [ $FAILED = 1 ] || { say "FAILED: unexpected exit (see logs/chain_stderr.log)"; say "exit 1"; }' EXIT

CODE=$(ls -d "$OUT"/code_* 2>/dev/null | head -n 1)     # the arms share one frozen copy: reuse it on a relaunch
if [ -z "$CODE" ]; then
    H=$(git -C "$ROOT/DeepWideGP" rev-parse --short widening) || fail "git"
    CODE=$OUT/code_$H
    rm -rf "${CODE:?}.tmp" && mkdir -p "$CODE.tmp" && git -C "$ROOT/DeepWideGP" archive widening | tar -x -C "$CODE.tmp" \
        && mv "$CODE.tmp" "$CODE" || fail "code export"
fi
say "starting from $(basename "$CODE"), pid $$"

free_gpus() {   # the indices among 2, 3 and 4 with no compute process
    local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | while IFS=, read -r idx uuid; do
        case "$idx" in 2|3|4) grep -q "$uuid" <<< "$busy" || echo "$idx";; esac
    done
}
take_gpu() {    # waits until at least two of the three are free; prints the lowest free index
    local f
    while :; do
        f=($(free_gpus))
        [ ${#f[@]} -ge 2 ] && { echo "${f[0]}"; return; }
        sleep 180
    done
}
build() {       # name source [--no_clump]
    local name=$1 source=$2; shift 2
    [ -e "$DS/${name}_n$N/build_log.json" ] && { say "build $name: exists"; return; }
    say "build $name: starting from $source"
    (cd "$ROOT/DeepWideGP" && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY -I "$SCRIPTS/classic-deepgp-comparator/build_datasets.py" \
        --threads 8 --rungs hm3plus --diseases $DISEASES --name "$name" --p_max 0.01 --source "$source" "$@") \
        >> "$OUT/logs/build_$name.log" 2>&1 || fail "build $name (logs/build_$name.log)"
    say "build $name: done; $(grep -m1 'SNPs selected' "$OUT/logs/build_$name.log")"
}
partition() {   # name
    local name=$1 part=prive_$1
    [ -e "$PARTS/$part/partition.npz" ] && { say "partition $part: exists"; return; }
    (cd "$BLOCKS" && $PY check_partition.py "$part" --from_reference "$REF" --store "$DATA/genostore/${name}_n$N" \
        --partitions "$PARTS") >> "$OUT/logs/partition_$name.log" 2>&1 || fail "partition $part (logs/partition_$name.log)"
    say "partition $part: $(tail -n 1 "$OUT/logs/partition_$name.log")"
}
folds_and_ridge() {   # name
    local name=$1 gpu; local d=$DS/${1}_n$N part=$PARTS/prive_$1     # one local per line: bash expands before it assigns
    [ -e "$d/blocks/folds.npz" ] || (cd "$BLOCKS" && $PY make_folds.py "$d" --splits train val) >> "$OUT/logs/ridge_$name.log" 2>&1 \
        || fail "folds $name (logs/ridge_$name.log)"
    [ -e "$d/blocks/ridge_manifest.json" ] && { say "ridge $name: exists"; return; }
    gpu=$(take_gpu); say "ridge $name: starting on GPU $gpu"
    (cd "$BLOCKS" && CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY build_block_ridge.py "$d" --partition "$part" \
        --splits train val --threads 8) >> "$OUT/logs/ridge_$name.log" 2>&1 || fail "ridge $name (logs/ridge_$name.log)"
    say "ridge $name: done"
}
pretrain() {    # name -> <dataset>/blocks/pretrained_d16.pt
    local name=$1 gpu; local d=$DS/${1}_n$N
    [ -e "$d/blocks/pretrained_d16.pt" ] && { say "pretrain $name: exists"; return; }
    gpu=$(take_gpu); say "pretrain $name: starting on GPU $gpu"
    (cd "$CODE" && CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 $PY pretrain_blocks.py --out "$d/blocks/pretrained_d16.pt" \
        --epochs 1 --mask_rate 0.15 --log_every 100 --data_dir "$DATA/datasets" --label ard30 --rd_mode "${name}_n$N" \
        --partition "$PARTS/prive_$name" --embed_dim 16 --snp_code dosage --snp_stats train --batch_size 64 --lr 1e-3 \
        --num_workers 10 --seed 1) >> "$OUT/logs/pretrain_$name.log" 2>&1 || fail "pretrain $name (logs/pretrain_$name.log)"
    say "pretrain $name: done; $(grep -m1 accuracy_last "$OUT/logs/pretrain_$name.log" | tr -d ' ,')"
}
train() {       # run dataset-name [extra flags]: in the background, after claiming a GPU
    local run=$1 name=$2; shift 2
    [ -e "$OUT/$run/results.json" ] && { say "train $run: exists"; return; }
    local gpu; gpu=$(take_gpu); say "train $run: starting on GPU $gpu"
    (cd "$CODE" && CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 $PY main_multilabel.py \
        --data_dir "$DATA/datasets" --label ard30 --rd_mode "${name}_n$N" --input_mode blocks --wide off --partition "$PARTS/prive_$name" \
        --mixer transformer --snp_code dosage --embed_dim 16 --d_model 64 --e_layers 4 --n_heads 4 --head_hidden 256 --n_fcn 2 \
        --final_pool mean --snp_stats train --lr 1e-4 --l2 1e-3 --dropout 0.2 --input_mask 0 --batch_size 64 --accumulate 1 \
        --eval_batch_size 1024 --precision bf16 --num_workers 10 --max_epoch_equiv 10 --val_per_epoch 4 --val_min_participants 20000 \
        --val_max_epoch_equiv 1 --min_participants 128000 --stop_patience 16 --stop_burnin 2000 --tune_min_validations 8 \
        --lr_schedule plateau --warmup_steps 0 --snapshots_per_decade 6 --snapshots_max 20 --run_dir "$ROOT/runs" --experiment selected \
        --exp_name "$run" --seed 1 --tune_trials 0 --sampler shuffled --gamma 0 --table_kappa 0.1 --table_eps 1e-8 --table_decay 0.01 "$@" \
        >> "$OUT/logs/train_$run.log" 2>&1; echo "exit $? at $(date -u +%FT%TZ)" >> "$OUT/logs/train_$run.log") &
    TRAIN_PIDS="$TRAIN_PIDS $!"
    sleep 240       # the run claims its GPU before the next acquisition
}
twin() {        # name (CPU): the logistic ridge twin and the ridge PRS on the same SNPs
    local name=$1
    [ -e "$BLOCKS/results/${name}_n$N/blocks_linear.csv" ] && { say "twin $name: exists"; return; }
    say "twin $name: starting"
    (cd "$BLOCKS" && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY fit_block_linear.py "$DS/${name}_n$N") >> "$OUT/logs/twin_$name.log" 2>&1 \
        || fail "twin $name (logs/twin_$name.log)"
    say "twin $name: done"
}

TRAIN_PIDS=""
build p01any "$DATA/genostore/hm3plus" --no_clump
partition p01any
folds_and_ridge p01any
pretrain p01any
train sel_p01any_S3_pretrained p01any --init_encoder "$DS/p01any_n$N/blocks/pretrained_d16.pt"
build p01anyct "$DATA/genostore/p01any_n$N"
partition p01anyct
folds_and_ridge p01anyct
train sel_p01anyct_S3_zero p01anyct
twin p01any
twin p01anyct
for p in $TRAIN_PIDS; do wait "$p"; done
for r in sel_p01any_S3_pretrained sel_p01anyct_S3_zero; do say "train $r: $(tail -n 1 "$OUT/logs/train_$r.log")"; done
DONE=1
say "exit 0"
