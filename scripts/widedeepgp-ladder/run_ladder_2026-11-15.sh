#!/bin/bash
# The WideDeepGP ladder: the PGI arm's five training sizes (median N_eff rungs), smallest to
# largest, one after another without pausing for review (Robert, 15 Nov 2026). Each rung is one
# run's whole search (DeepWideGP 5396f44 or later): the linear stage on one GPU, then 12 Optuna
# trials of the network on top on the free GPUs minus one; the run's model is the network on top,
# the linear model alone its recorded ablation. Model: width 32, one layer, four heads, attention
# pooling, batches of 16 x 4, bf16 encoders, seed 1. Starts once the real-label gate
# (runs/wide_gate/real_search) has finished cleanly. Stops if a rung fails; rerunning this script
# continues (finished rungs are skipped, an unfinished search continues where it stopped).
# Detached: setsid nohup; log ladder_chain.log, one log per rung.
set -u
L=/mnt/sdg/robert/deepwidegp/runs/ladder
G=/mnt/sdg/robert/deepwidegp/runs/wide_gate
LOG=$L/ladder_chain.log
echo "waiting since $(date -u +%FT%TZ), pid $$, for the real-label gate (wide_gate/real_search)" >> "$LOG"
until grep -q "^exit " "$G/real_search_wait.log" 2>/dev/null; do sleep 300; done
echo "gate ended: $(grep '^exit ' "$G/real_search_wait.log" | tail -1)" >> "$LOG"
if ! grep -q "^exit 0 " "$G/real_search_wait.log" || [ ! -e "$G/real_search/results.json" ]; then
    echo "the gate did not finish cleanly; ladder not started ($(date -u +%FT%TZ))" >> "$LOG"; exit 3
fi
cd /mnt/sdg/robert/deepwidegp/DeepWideGP || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for ds in hm3plus_n13199 hm3plus_n26287 hm3plus_n53498 hm3plus_n107743 hm3plus; do
    if [ -e "$L/$ds/results.json" ]; then echo "$ds: done already" >> "$LOG"; continue; fi
    echo "$ds: starting at $(date -u +%FT%TZ) (DeepWideGP $(git rev-parse --short HEAD))" >> "$LOG"
    timeout 7d /home/robert/envs/widedeepgp/bin/python main_multilabel.py --data_dir /mnt/sdg/robert/deepwidegp/data/datasets \
        --label ard30 --rd_mode "$ds" --final_pool atten --d_model 32 --e_layers 1 --n_heads 4 --batch_size 16 \
        --accumulate 4 --eval_batch_size 128 --precision bf16 --wide on --tune_gpus auto --num_workers 4 \
        --run_dir /mnt/sdg/robert/deepwidegp/runs --experiment ladder --exp_name "$ds" >> "$L/$ds.log" 2>&1
    rc=$?
    echo "$ds: exit $rc at $(date -u +%FT%TZ)" >> "$LOG"
    if [ "$rc" -ne 0 ] || [ ! -e "$L/$ds/results.json" ]; then
        echo "$ds did not finish; ladder stopped (rerun this script to continue)" >> "$LOG"; exit 1
    fi
done
echo "ladder finished at $(date -u +%FT%TZ)" >> "$LOG"
