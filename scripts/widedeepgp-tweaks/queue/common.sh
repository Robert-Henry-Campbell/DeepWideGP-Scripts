# Shared pieces of the overnight tweak jobs (sourced by every job script). A job gets the GPU index as $1 (the scheduler
# also sets CUDA_VISIBLE_DEVICES). Base configuration = the best run so far, blocks_diag/diag_S3_folds_frozen_nomask:
# S3 transformer, frozen fold-routed unit-RMS ridge table (d_e 16), mean readout, no input mask, dropout 0.2, lr 1e-4,
# wd 1e-3, batch 64, plateau schedule, plain cross-entropy at natural prevalence, bf16, 10 epoch-equivalents, seed 1.
ROOT=/mnt/sdg/robert/deepwidegp
PY=/home/robert/envs/widedeepgp/bin/python
DS=$ROOT/data/datasets
BLOCKS=$DS/ard30/hm3plus/blocks
RUNS=$ROOT/runs/tweaks
EXP=tweaks
BASE_FLAGS="--data_dir $DS --label ard30 --rd_mode hm3plus --input_mode blocks --wide off --partition $ROOT/data/reference/partitions/prive431 --mixer transformer --snp_code dosage --embed_dim 16 --table_eps 2.5e-7 --d_model 64 --e_layers 4 --n_heads 4 --head_hidden 256 --n_fcn 2 --final_pool mean --snp_stats train --lr 1e-4 --l2 1e-3 --dropout 0.2 --input_mask 0 --batch_size 64 --accumulate 1 --eval_batch_size 1024 --precision bf16 --num_workers 4 --max_epoch_equiv 10 --val_per_epoch 4 --val_min_participants 20000 --val_max_epoch_equiv 1 --min_participants 128000 --stop_patience 16 --stop_burnin 2000 --tune_min_validations 8 --lr_schedule plateau --warmup_steps 0 --snapshots_per_decade 6 --snapshots_max 20 --run_dir $ROOT/runs --experiment $EXP --folds $BLOCKS/folds.npz --sampler shuffled --gamma 0 --seed 1 --tune_trials 0 --init_encoder $BLOCKS/init_ridge_d16.pt --freeze_encoder --table_kappa 1e-3 --table_decay 0.0"

stamp() { date -u +%FT%TZ; }

train() {   # train CODE NAME [flags...]: one training run from a frozen code copy (resumes if it was interrupted); then its
            # best snapshot scored on the tuning set by streaming (an independent check of a cached run) with the covariates-only floor
    local code=$1 name=$2; shift 2
    local resume=""
    [ -e "$RUNS/$name/results.json" ] && { echo "$name: results.json exists, nothing to train"; return 0; }
    [ -e "$RUNS/$name/checkpoints/last.ckpt" ] && resume="--resume"
    echo "$(stamp) $name: training from $code on GPU ${CUDA_VISIBLE_DEVICES:-?}"
    (cd "$code" && $PY main_multilabel.py $BASE_FLAGS --exp_name "$name" "$@" $resume) || return $?
    echo "$(stamp) $name: trained; exit $?"
}

score() {   # score CODE NAME: the best snapshot on the tuning set, streamed
    local code=$1 name=$2
    [ -e "$RUNS/$name/scores_val_summary.csv" ] && return 0
    (cd "$code" && $PY score_snapshots.py --run "$RUNS/$name" --split val --snapshots best --batch_size 1024)
}

# ---- jobs from the tweaks code (worktree branch tweaks, frozen copy code_<commit>): the frozen table's pooled vectors kept on disk
CODE2=$RUNS/code_$(cat $RUNS/queue/code_commit)
CACHE_FLAGS="--first_layer_cache $BLOCKS/cache"
ROT="--sampler rotation --no_sampling_correction --gamma 0"     # item 3: disease-balanced batches, plain cross-entropy as drawn
SEARCH_SAMPLER=""          # the searches run under the base sampler (shuffled, natural prevalence): rot10 trailed the shuffled runs by about 0.0015 at matched epochs

search_worker() {   # search_worker CODE NAME N_TRIALS INDEX [flags...]: one worker of a run's search (takes trials until N are done or running)
    local code=$1 name=$2 n=$3 i=$4; shift 4
    echo "$(stamp) $name: worker $i from $code on GPU ${CUDA_VISIBLE_DEVICES:-?}"
    (cd "$code" && WIDEDEEPGP_TUNE_STAGE=deep WIDEDEEPGP_TUNE_N=$n WIDEDEEPGP_TUNE_WORKER=$i $PY main_multilabel.py $BASE_FLAGS --exp_name "$name" "$@" --tune_trials $n --tune_worker)
}

search_summary() {  # search_summary CODE NAME N_TRIALS [flags...]: once the workers are done, the run's results.json and the choice
    local code=$1 name=$2 n=$3; shift 3
    (cd "$code" && $PY main_multilabel.py $BASE_FLAGS --exp_name "$name" "$@" --tune_trials $n --tune_gpus none)
}

# ---- the PGI tables (the other agent's init_pgi_<method>_d30.pt: one copy, 30 channels = one method's weights per disease) and
# the within-block models (items 2 and 6), from the frozen copy code_<commit3>
CODE3=$RUNS/code_$(cat $RUNS/queue/code_commit3)
PGI_BASE_FLAGS=${BASE_FLAGS/"--folds $BLOCKS/folds.npz"/}          # no folds: the PGI tables have one copy
train_pgi() {   # train_pgi NAME METHOD [flags...]: a run on the frozen PGI table of METHOD (lassosum2 | ldpred2_auto), cached
    local name=$1 method=$2; shift 2
    local resume=""
    [ -e "$RUNS/$name/results.json" ] && { echo "$name: results.json exists, nothing to train"; return 0; }
    [ -e "$RUNS/$name/checkpoints/last.ckpt" ] && resume="--resume"
    echo "$(stamp) $name: training from $CODE3 on GPU ${CUDA_VISIBLE_DEVICES:-?} (table init_pgi_${method}_d30.pt)"
    (cd "$CODE3" && $PY main_multilabel.py $PGI_BASE_FLAGS --exp_name "$name" --init_encoder $BLOCKS/init_pgi_${method}_d30.pt --embed_dim 30 \
        $CACHE_FLAGS "$@" $resume) || return $?
}

# ---- the leak-free PGI tables (the other agent's chain): dataset hm3plus_c107743 (the 108,129 training participants outside the
# 107,743 rung) with tables built from that rung's PGI weights, so no participant's inputs carry their own label
C107=hm3plus_c107743
C107_BLOCKS=$DS/ard30/$C107/blocks
C107_FLAGS="--rd_mode $C107 --first_layer_cache $C107_BLOCKS/cache"
train_c107() {   # train_c107 NAME METHOD [flags...]: a run on the leak-free frozen PGI table of METHOD on the c107743 dataset, cached
    local name=$1 method=$2; shift 2
    local resume=""
    [ -e "$RUNS/$name/results.json" ] && { echo "$name: results.json exists, nothing to train"; return 0; }
    [ -e "$RUNS/$name/checkpoints/last.ckpt" ] && resume="--resume"
    echo "$(stamp) $name: training from $CODE3 on GPU ${CUDA_VISIBLE_DEVICES:-?} (table $C107_BLOCKS/init_pgi_${method}_d30.pt)"
    (cd "$CODE3" && $PY main_multilabel.py $PGI_BASE_FLAGS --exp_name "$name" $C107_FLAGS --init_encoder $C107_BLOCKS/init_pgi_${method}_d30.pt \
        --embed_dim 30 "$@" $resume) || return $?
}
