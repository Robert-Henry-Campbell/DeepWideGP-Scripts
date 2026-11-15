#!/bin/bash
# Gate G3, first half: the linear stage (grid of prior variances, each disease keeping its best)
# on the real labels of the pilot's proxy (30 diseases; 157,279 SNPs; 40,000 train / 10,000
# tune), on the CPU (8 threads), after the simulated-trait grid gate has finished, so that our
# CPU use stays at about 8 threads. Compared afterwards with the covariates-only logistic model
# (0.6607) and the clumped PRS of the pilot. Started unattended on 13 Nov 2026.
set -u
G=/mnt/sdg/robert/deepwidegp/runs/wide_gate
echo "waiting since $(date -u +%FT%TZ), pid $$" > "$G/real_linear_wait.log"
deadline=$(( $(date +%s) + 12 * 3600 ))
until grep -q "^exit " "$G/grid_gate_wait.log" 2>/dev/null || grep -q "not started\|gave up" "$G/grid_gate_wait.log" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$deadline" ]; then echo "gave up waiting at $(date -u +%FT%TZ)" >> "$G/real_linear_wait.log"; exit 3; fi
    sleep 60
done
if grep -q "not started\|gave up" "$G/grid_gate_wait.log"; then
    echo "the simulated-trait gate did not start (store not swapped?); not started" >> "$G/real_linear_wait.log"; exit 3
fi
echo "starting at $(date -u +%FT%TZ)" >> "$G/real_linear_wait.log"
cd /mnt/sdg/robert/deepwidegp/DeepWideGP || exit 1
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" timeout 4h /home/robert/envs/widedeepgp/bin/python -c "
import torch; torch.set_num_threads(8)
import sys, main_multilabel
main_multilabel.main(sys.argv[1:])" --data_dir /mnt/sdg/robert/deepwidegp/data/datasets --label ard30 \
    --rd_mode hm3plus_proxy8 --final_pool atten --d_model 32 --e_layers 1 --n_heads 4 --wide only \
    --tune_gpus none --num_workers 4 --run_dir /mnt/sdg/robert/deepwidegp/runs --experiment wide_gate \
    --exp_name real_linear_grid > "$G/real_linear_grid.log" 2>&1
echo "exit $? at $(date -u +%FT%TZ)" >> "$G/real_linear_wait.log"
