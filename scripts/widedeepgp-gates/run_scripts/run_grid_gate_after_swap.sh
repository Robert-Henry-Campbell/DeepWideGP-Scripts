#!/bin/bash
# Gate G1 for the linear stage as a run's search now does it (DeepWideGP e5cebed): the grid of
# prior variances in one pass, each simulated trait keeping its best, on the CPU (8 threads),
# on the pilot's simulated-trait proxy (157,279 SNPs; 40,000 train / 10,000 tune). Waits until
# the genotype store rebuild has swapped the store in (a run holding the old store open would
# block the swap) and the fixed-tau gate run has ended. Started unattended on 13 Nov 2026.
set -u
G=/mnt/sdg/robert/deepwidegp/runs/wide_gate
LOG=/mnt/sdg/robert/deepwidegp/data/genostore/rebuild_without_withdrawn.log
echo "waiting since $(date -u +%FT%TZ), pid $$" > "$G/grid_gate_wait.log"
deadline=$(( $(date +%s) + 8 * 3600 ))
until grep -q "hm3plus: rebuilt store in place" "$LOG" && ! pgrep -u "$(id -u)" -f "exp_name sim_linear_tau1_k0.1" > /dev/null; do
    if grep -q -E "hm3plus: (checks failed|still open|convert exit [1-9]|verify exit [1-9])|stopped" "$LOG"; then
        echo "the rebuild did not swap the store in; not started ($(date -u +%FT%TZ))" >> "$G/grid_gate_wait.log"; exit 3
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then echo "gave up waiting at $(date -u +%FT%TZ)" >> "$G/grid_gate_wait.log"; exit 3; fi
    sleep 60
done
echo "starting at $(date -u +%FT%TZ)" >> "$G/grid_gate_wait.log"
cd /mnt/sdg/robert/deepwidegp/DeepWideGP || exit 1
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" timeout 4h /home/robert/envs/widedeepgp/bin/python -c "
import torch; torch.set_num_threads(8)
import sys, main_multilabel
main_multilabel.main(sys.argv[1:])" --data_dir /mnt/sdg/robert/deepwidegp/data/datasets --label ard30 \
    --rd_mode hm3plus_proxy8_sim --final_pool atten --d_model 32 --e_layers 1 --n_heads 4 --wide only \
    --tune_gpus none --num_workers 4 --run_dir /mnt/sdg/robert/deepwidegp/runs --experiment wide_gate \
    --exp_name sim_linear_grid > "$G/sim_linear_grid.log" 2>&1
echo "exit $? at $(date -u +%FT%TZ)" >> "$G/grid_gate_wait.log"
