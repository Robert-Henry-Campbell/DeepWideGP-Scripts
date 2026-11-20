# slots: 3
# after: 098_e11_build_r01
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 4, the control run: the published network for E11 on the comparator's clumping (r2 0.1, 250 kb), balanced batches, as 021.
CODE=$RUNS/code_5c47884
NAME=classic_e11_r01_balanced
[ -e "$DS/ard30/classic_e11_n215872/build_log.json" ] || { echo "dataset missing"; exit 1; }
if [ ! -e "$RUNS/$NAME/results.json" ]; then
    RESUME=""; [ -e "$RUNS/$NAME/checkpoints/last.ckpt" ] && RESUME="--resume"
    (cd $CODE && OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 $PY main_multilabel.py --data_dir $DS --label ard30 --rd_mode classic_e11_n215872 --labels E11 \
        --input_mode snps --input_mask 0 --wide off --lr_schedule constant --warmup_steps 0 --stop_patience 8 --stop_burnin 0 \
        --sampler rotation --no_sampling_correction --gamma 0 --batch_size 64 --accumulate 1 --lr 7e-4 --l2 1e-4 --dropout 0.2 \
        --d_model 32 --e_layers 1 --n_heads 4 --n_fcn 2 --head_hidden 256 --final_pool atten --snp_embed cov --precision 32 \
        --eval_batch_size 256 --tune_trials 0 --seed 1 --num_workers 6 --run_dir $ROOT/runs --experiment $EXP --exp_name $NAME $RESUME) || { echo "exit $? at $(stamp)"; exit 1; }
fi
score $CODE $NAME
echo "exit $? at $(stamp)"
