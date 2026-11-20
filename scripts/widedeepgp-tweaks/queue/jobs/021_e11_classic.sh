# slots: 3
# after: 020_e11_build
# Item 4: the published DeepGP network for E11 on the paper's clumping (r2 0.7, 500 kb), published settings (d 32, one
# bi-Mamba layer, 4 heads, attention pooling, batch 64, lr 7e-4, wd 1e-4, dropout 0.2, fp32), balanced batches by
# oversampling cases (the paper trained on 1:1 sets; 32 cases + 32 non-cases, plain cross-entropy on the batch as drawn),
# early stopping after 8 passes without gain. Streamed (SNP tokens): a whole GPU. Then the best snapshot scored on the tuning set.
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
CODE=$RUNS/code_5c47884
NAME=classic_e11_r07_balanced
[ -e "$DS/ard30/classic_e11_r07_n215872/build_log.json" ] || { echo "dataset missing"; exit 1; }
if [ ! -e "$RUNS/$NAME/results.json" ]; then
    RESUME=""; [ -e "$RUNS/$NAME/checkpoints/last.ckpt" ] && RESUME="--resume"
    (cd $CODE && OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 $PY main_multilabel.py --data_dir $DS --label ard30 --rd_mode classic_e11_r07_n215872 --labels E11 \
        --input_mode snps --input_mask 0 --wide off --lr_schedule constant --warmup_steps 0 --stop_patience 8 --stop_burnin 0 \
        --sampler rotation --no_sampling_correction --gamma 0 --batch_size 64 --accumulate 1 --lr 7e-4 --l2 1e-4 --dropout 0.2 \
        --d_model 32 --e_layers 1 --n_heads 4 --n_fcn 2 --head_hidden 256 --final_pool atten --snp_embed cov --precision 32 \
        --eval_batch_size 256 --tune_trials 0 --seed 1 --num_workers 6 --run_dir $ROOT/runs --experiment $EXP --exp_name $NAME $RESUME) || { echo "exit $? at $(stamp)"; exit 1; }
fi
score $CODE $NAME
echo "exit $? at $(stamp)"
