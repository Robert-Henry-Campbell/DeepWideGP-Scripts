#!/bin/bash
# Gate G3 on the real labels of the pilot's proxy (30 diseases; 157,279 SNPs; 40,000 train /
# 10,000 tune), as one run's whole search under the final rules (DeepWideGP 5396f44): the linear
# stage (nine-value grid) on the first free GPU, then 12 Optuna trials of the network on top on
# the free GPUs minus one; the run's model is the network on top, the linear model alone its
# ablation. Model as the pilot's default arm with bf16 encoders. Replaces the spoilt
# real_linear_grid9 / stopped real_joint of 15 Nov. Detached (setsid nohup).
set -u
G=/mnt/sdg/robert/deepwidegp/runs/wide_gate
echo "start $(date -u +%FT%TZ), pid $$" > "$G/real_search_wait.log"
cd /mnt/sdg/robert/deepwidegp/DeepWideGP || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout 30h /home/robert/envs/widedeepgp/bin/python main_multilabel.py --data_dir /mnt/sdg/robert/deepwidegp/data/datasets \
    --label ard30 --rd_mode hm3plus_proxy8 --final_pool atten --d_model 32 --e_layers 1 --n_heads 4 --batch_size 16 \
    --accumulate 4 --eval_batch_size 128 --precision bf16 --wide on --tune_gpus auto --num_workers 4 \
    --run_dir /mnt/sdg/robert/deepwidegp/runs --experiment wide_gate --exp_name real_search > "$G/real_search.log" 2>&1
echo "exit $? at $(date -u +%FT%TZ)" >> "$G/real_search_wait.log"
