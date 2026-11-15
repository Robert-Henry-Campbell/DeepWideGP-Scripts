#!/bin/bash
# The WideDeepGP ladder: the PGI arm's five training sizes (median N_eff rungs), smallest to
# largest, one after another without pausing for review and without any time limit (Robert,
# 15 Nov 2026). Each rung is one run's whole search: the linear stage on one GPU, then 12 Optuna
# trials of the network on top on the free GPUs minus one; the run's model is the network on top,
# the linear model alone its recorded ablation. Model: width 32, one layer, four heads, attention
# pooling, batches of 16 x 4, bf16 encoders, seed 1.
# - Each rung runs from a frozen export of the newest commit of DeepWideGP's `widening` branch
#   (code_<commit>/), so work in progress never reaches a running rung.
# - After each rung, the classic DeepGP comparator (hypertension, I10) runs at the same size if
#   its rung script is ready (runs/classic_deepgp/run_rung.sh); its failure does not stop the ladder.
# - Starts once the real-label gate (runs/wide_gate/real_search) has finished cleanly; stops if a
#   rung fails; rerunning continues (finished rungs are skipped, an unfinished search continues).
# Detached: setsid nohup; log ladder_chain.log, one log per rung.
set -u
L=/mnt/sdg/robert/deepwidegp/runs/ladder
G=/mnt/sdg/robert/deepwidegp/runs/wide_gate
REPO=/mnt/sdg/robert/deepwidegp/DeepWideGP
CMP=/mnt/sdg/robert/deepwidegp/runs/classic_deepgp/run_rung.sh
LOG=$L/ladder_chain.log
echo "waiting since $(date -u +%FT%TZ), pid $$, for the real-label gate (wide_gate/real_search)" >> "$LOG"
until grep -q "^exit " "$G/real_search_wait.log" 2>/dev/null; do sleep 300; done
echo "gate ended: $(grep '^exit ' "$G/real_search_wait.log" | tail -1)" >> "$LOG"
if ! grep -q "^exit 0 " "$G/real_search_wait.log" || [ ! -e "$G/real_search/results.json" ]; then
    echo "the gate did not finish cleanly; ladder not started ($(date -u +%FT%TZ))" >> "$LOG"; exit 3
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for ds in hm3plus_n13199 hm3plus_n26287 hm3plus_n53498 hm3plus_n107743 hm3plus; do
    if [ -e "$L/$ds/results.json" ]; then
        echo "$ds: done already" >> "$LOG"
    else
        H=$(git -C "$REPO" rev-parse --short widening)
        C=$L/code_$H
        if [ ! -d "$C" ]; then
            rm -rf "${C:?}.tmp" && mkdir -p "$C.tmp" && git -C "$REPO" archive widening | tar -x -C "$C.tmp" && mv "$C.tmp" "$C" \
                || { echo "$ds: could not export commit $H; ladder stopped" >> "$LOG"; exit 1; }
        fi
        echo "$ds: starting at $(date -u +%FT%TZ) from code_$H" >> "$LOG"
        (cd "$C" && /home/robert/envs/widedeepgp/bin/python main_multilabel.py --data_dir /mnt/sdg/robert/deepwidegp/data/datasets \
            --label ard30 --rd_mode "$ds" --final_pool atten --d_model 32 --e_layers 1 --n_heads 4 --batch_size 16 \
            --accumulate 4 --eval_batch_size 128 --precision bf16 --wide on --tune_gpus auto --num_workers 4 \
            --run_dir /mnt/sdg/robert/deepwidegp/runs --experiment ladder --exp_name "$ds") >> "$L/$ds.log" 2>&1
        rc=$?
        echo "$ds: exit $rc at $(date -u +%FT%TZ)" >> "$LOG"
        if [ "$rc" -ne 0 ] || [ ! -e "$L/$ds/results.json" ]; then
            echo "$ds did not finish; ladder stopped (rerun this script to continue)" >> "$LOG"; exit 1
        fi
    fi
    if [ -x "$CMP" ]; then
        echo "$ds classic DeepGP: starting at $(date -u +%FT%TZ)" >> "$LOG"
        bash "$CMP" "$ds" >> "$L/classic_$ds.log" 2>&1
        echo "$ds classic DeepGP: exit $? at $(date -u +%FT%TZ)" >> "$LOG"
    else
        echo "$ds classic DeepGP: comparator not ready; to be run later" >> "$LOG"
    fi
done
echo "ladder finished at $(date -u +%FT%TZ)" >> "$LOG"
