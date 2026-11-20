#!/bin/bash
# Frozen PGI tables (DeepWideGP-Scripts design item 22; Robert, 20 Nov 2026): the LD-block network (S3, the frozen no-mask
# recipe) on a 30-channel table of lassosum2 or LDpred2-auto weights (one channel per disease) instead of the ridge, with
# the first layer cached. Full rung as asked (weights from the GWAS of all 215,872, so in-sample for every training
# participant) and a leak-free control (weights of the 107,743 rung; the network trains on the other 108,129). One branch
# per method; each step takes the highest free GPU among 2-4 that the other branch has not claimed, once at least two such
# are free. Steps are skipped when their product exists; a failed step stops its branch. Then the lift over each PGI.
#   cd /mnt/sdg/robert/deepwidegp/runs/pgi_tables && setsid -f nohup bash run_pgi_tables.sh > /dev/null 2>&1 < /dev/null
set -u
ROOT=/mnt/sdg/robert/deepwidegp
PY=/home/robert/envs/widedeepgp/bin/python
OUT=$ROOT/runs/pgi_tables
SRC=$OUT/src
DATA=$ROOT/data/datasets/ard30
L=$OUT/chain.log
mkdir -p "$OUT/logs"
exec 2>> "$OUT/logs/chain_stderr.log"
say() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
trap 'rc=$?; [ $rc -eq 0 ] || say "FAILED: chain shell exited with $rc"' EXIT

[ -e "$OUT/code_commit" ] || git -C "$ROOT/DeepWideGP" rev-parse --short widening > "$OUT/code_commit"
H=$(cat "$OUT/code_commit"); CODE=$OUT/code_$H
if [ ! -d "$CODE" ]; then
    rm -rf "${CODE:?}.tmp" && mkdir -p "$CODE.tmp" && git -C "$ROOT/DeepWideGP" archive "$H" | tar -x -C "$CODE.tmp" && mv "$CODE.tmp" "$CODE" \
        || { say "FAILED: export of DeepWideGP $H"; exit 1; }
fi
export PYTHONPATH=$CODE
rm -f "$OUT"/.claim_*                               # claims of an earlier launch
say "starting from code_$H, pid $$"

free_gpus() {   # the indices among 2, 3 and 4 with no compute process
    local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' ' | while IFS=, read -r idx uuid; do
        case "$idx" in 2|3|4) grep -q "$uuid" <<< "$busy" || echo "$idx";; esac
    done
}
take_gpu() {    # prints the highest free GPU among 2-4 that no step has claimed, once at least two such are free
    local f g
    while :; do
        exec 9> "$OUT/.gpu.lock"; flock 9
        f=()
        for g in $(free_gpus); do [ -e "$OUT/.claim_$g" ] || f+=("$g"); done
        if [ ${#f[@]} -ge 2 ]; then
            g=${f[-1]}; touch "$OUT/.claim_$g"; flock -u 9; exec 9>&-; echo "$g"; return
        fi
        flock -u 9; exec 9>&-
        sleep 120
    done
}
step() {        # name command...: one logged step on a GPU of its own; its exit status is returned
    local name=$1 gpu rc; shift
    gpu=$(take_gpu); say "$name: starting on GPU $gpu"
    (cd "$CODE" && CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 "$@") >> "$OUT/logs/$name.log" 2>&1
    rc=$?; rm -f "$OUT/.claim_$gpu"; say "$name: exit $rc"; return $rc
}
train() {       # name dataset init
    local resume=""
    [ -e "$OUT/$1/checkpoints/last.ckpt" ] && resume=--resume
    step "train_$1" $PY main_multilabel.py --data_dir $ROOT/data/datasets --label ard30 --rd_mode "$2" --input_mode blocks \
        --wide off --partition $ROOT/data/reference/partitions/prive431 --mixer transformer --snp_code dosage --embed_dim 30 \
        --table_eps 2.5e-7 --d_model 64 --e_layers 4 --n_heads 4 --head_hidden 256 --n_fcn 2 --final_pool mean --snp_stats train \
        --lr 1e-4 --l2 1e-3 --dropout 0.2 --input_mask 0 --batch_size 64 --accumulate 1 --eval_batch_size 1024 --precision bf16 \
        --num_workers 10 --max_epoch_equiv 10 --val_per_epoch 4 --val_min_participants 20000 --val_max_epoch_equiv 1 \
        --min_participants 128000 --stop_patience 16 --stop_burnin 2000 --tune_min_validations 8 --lr_schedule plateau \
        --warmup_steps 0 --snapshots_per_decade 6 --snapshots_max 20 --run_dir $ROOT/runs --experiment pgi_tables \
        --exp_name "$1" --sampler shuffled --gamma 0 --seed 1 --tune_trials 0 --init_encoder "$3" --freeze_encoder \
        --table_kappa 1e-3 --table_decay 0.0 $resume
}
branch() {      # method table-flags: full rung, then the control: table -> training -> streamed tuning scores
    local m=$1 flags=$2 arm ds init where
    for arm in top split; do
        if [ $arm = top ]; then ds=hm3plus; where="--in_sample"; else ds=hm3plus_c107743; where="--weights_dataset $DATA/hm3plus_n107743"; fi
        init=$DATA/$ds/blocks/init_pgi_${m}_d30.pt
        [ -e "$init" ] || step "table_${arm}_$m" $PY $SRC/make_pgi_table.py $DATA/$ds --method $m $flags $where --out "$init" || return 1
        [ -e "$OUT/${arm}_$m/results.json" ] || train "${arm}_$m" $ds "$init" || return 1
        [ -e "$OUT/${arm}_$m/scores_val_summary.csv" ] || step "score_${arm}_$m" $PY score_snapshots.py --run "$OUT/${arm}_$m" \
            --split val --snapshots best --save_scores --batch_size 1024 || return 1
    done
    say "branch $m: done"
}

# the control's dataset and its training SNP statistics, before the two branches share them
if [ ! -d "$DATA/hm3plus_c107743" ]; then
    $PY $SRC/make_complement.py $DATA/hm3plus $DATA/hm3plus_n107743 $DATA/hm3plus_c107743 >> "$L" 2>&1 || { say "FAILED: complement"; exit 1; }
fi
if ! ls "$DATA"/hm3plus_c107743/snp_stats_train_*.npz > /dev/null 2>&1; then
    step stats_c107743 $PY -c "from args_generator import args_initial
from widedeepgp.train import build_data
build_data(args_initial(['--label', 'ard30', '--rd_mode', 'hm3plus_c107743', '--data_dir', '$ROOT/data/datasets',
    '--input_mode', 'blocks', '--partition', '$ROOT/data/reference/partitions/prive431', '--snp_stats', 'train', '--wide', 'off']))" || exit 1
fi

( branch lassosum2 --drop_zero || say "FAILED: branch lassosum2" ) &
( branch ldpred2_auto "" || say "FAILED: branch ldpred2_auto" ) &
wait
runs=$(for r in top_lassosum2 split_lassosum2 top_ldpred2_auto split_ldpred2_auto; do
    [ -e "$OUT/$r/scores_val_summary.csv" ] && echo "$OUT/$r"; done)
[ -n "$runs" ] || { say "FAILED: no run finished"; exit 1; }
say "comparing: $(echo $runs | wc -w) runs"
PYTHONPATH=$ROOT/PGI $PY $SRC/compare_pgi_tables.py --out "$OUT/compare" $runs >> "$L" 2>&1
echo "exit $? at $(date -u +%FT%TZ)" >> "$L"
