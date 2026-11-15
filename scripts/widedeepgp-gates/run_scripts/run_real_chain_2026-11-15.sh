#!/bin/bash
# Gate G3 on the real labels of the pilot's proxy, restarted on 15 Nov 2026 after a session
# restart on 13 Nov killed the linear stage (nine-value grid) at pass 28 of 30: the same command
# resumes it from its last checkpoint (CPU, 8 threads); then the network on top of it (Optuna's 12
# trials of the joint stage on the free GPUs minus one; model as the pilot's default arm with bf16
# encoders), starting from a copy of that linear stage. Detached: setsid nohup, log beside it.
set -u
G=/mnt/sdg/robert/deepwidegp/runs/wide_gate
L=$G/real_chain_2026-11-15.log
echo "start $(date -u +%FT%TZ), pid $$" > "$L"
cd /mnt/sdg/robert/deepwidegp/DeepWideGP || exit 1
if [ ! -e "$G/real_linear_grid9/linear/results.json" ]; then
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" timeout 3h /home/robert/envs/widedeepgp/bin/python -c "
import torch; torch.set_num_threads(8)
import sys, main_multilabel
main_multilabel.main(sys.argv[1:])" --data_dir /mnt/sdg/robert/deepwidegp/data/datasets --label ard30 --rd_mode hm3plus_proxy8 \
        --final_pool atten --d_model 32 --e_layers 1 --n_heads 4 --wide only --tune_gpus none --num_workers 4 \
        --run_dir /mnt/sdg/robert/deepwidegp/runs --experiment wide_gate --exp_name real_linear_grid9 >> "$G/real_linear_grid9.log" 2>&1
    echo "linear: exit $? at $(date -u +%FT%TZ)" >> "$L"
fi
if [ ! -e "$G/real_linear_grid9/linear/results.json" ]; then echo "the linear stage did not finish; stopped" >> "$L"; exit 3; fi
mkdir -p "$G/real_joint"
[ -e "$G/real_joint/linear/results.json" ] || cp -a "$G/real_linear_grid9/linear" "$G/real_joint/linear"
echo "real_joint: starting at $(date -u +%FT%TZ)" >> "$L"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
timeout 30h /home/robert/envs/widedeepgp/bin/python main_multilabel.py --data_dir /mnt/sdg/robert/deepwidegp/data/datasets \
    --label ard30 --rd_mode hm3plus_proxy8 --final_pool atten --d_model 32 --e_layers 1 --n_heads 4 --batch_size 16 \
    --accumulate 4 --eval_batch_size 128 --precision bf16 --wide on --tune_gpus auto --num_workers 4 \
    --run_dir /mnt/sdg/robert/deepwidegp/runs --experiment wide_gate --exp_name real_joint > "$G/real_joint.log" 2>&1
echo "real_joint: exit $? at $(date -u +%FT%TZ)" >> "$L"
