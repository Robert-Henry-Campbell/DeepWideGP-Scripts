#!/bin/bash
# Gates G2 and G3, second half: the network on top of the linear model (--wide on: Optuna's 12
# trials of the joint stage, on the free GPUs minus one), first on the simulated traits, then on
# the real labels of the pilot's proxy. Each starts from a copy of the linear stage computed on
# the CPU (sim_linear_grid, real_linear_grid9; same settings, the joint stage reads the copy in
# place). Runs only after the GPU test suites of the code under test have passed. Model as the
# pilot's default arm, with the bf16 encoders adopted for full-size runs. Started unattended on
# 13 Nov 2026.
set -u
G=/mnt/sdg/robert/deepwidegp/runs/wide_gate
T=/mnt/sdg/robert/deepwidegp/runs/gate_evidence/final_tests_2026-11-13
W=$G/joint_gates_wait.log
echo "waiting since $(date -u +%FT%TZ), pid $$" > "$W"
deadline=$(( $(date +%s) + 36 * 3600 ))
wait_for() {   # file pattern
    until grep -q "$2" "$1" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then echo "gave up waiting for $1 at $(date -u +%FT%TZ)" >> "$W"; exit 3; fi
        sleep 120
    done
}
wait_for "$T/gpu_summary.txt" "GPU suites"
# every suite of 81ded30 passed except one test of the quick training suite, the tuner test, which
# timed out under the new budget floor; its rerun with small-data floors (197741a) must pass
wait_for "$T/rerun_summary.txt" "rerun of the tuner test\|gave up"
ok=1
for s in wide model train_slow capacity_probe; do grep -E "^$s:" "$T/gpu_summary.txt" | grep -q "exit 0" || ok=0; done
grep -q "capacity_probe ok lines: 4 of 4" "$T/gpu_summary.txt" || ok=0
if grep -E "FAILED|ERROR" "$T/gpu_summary.txt" | grep -v -q "test_the_tuner_runs_shared_trials_and_prunes"; then ok=0; fi
grep -q "1 passed" "$T/rerun_summary.txt" && grep -q "exit 0" "$T/rerun_summary.txt" || ok=0
if [ "$ok" -ne 1 ]; then
    echo "GPU tests did not all pass; joint gates not started ($(date -u +%FT%TZ))" >> "$W"; exit 3
fi
echo "GPU tests passed (with the tuner test's rerun); $(date -u +%FT%TZ)" >> "$W"
cd /mnt/sdg/robert/deepwidegp/DeepWideGP || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4          # per worker process: the GPUs do the work
joint() {   # name, linear run, variant, the linear run's wait log
    local name=$1 lin=$2 variant=$3 lin_log=$4
    until [ -e "$G/$lin/linear/results.json" ]; do
        if grep -q -E "^exit [1-9]|not started|gave up" "$G/$lin_log" 2>/dev/null; then
            echo "$name: the linear stage $lin did not finish; skipped ($(date -u +%FT%TZ))" >> "$W"; return 1
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then echo "$name: gave up waiting for $lin at $(date -u +%FT%TZ)" >> "$W"; return 1; fi
        sleep 120
    done
    if [ ! -e "$G/$name/linear/results.json" ]; then
        mkdir -p "$G/$name" && cp -a "$G/$lin/linear" "$G/$name/linear" || { echo "$name: copy failed" >> "$W"; return 1; }
    fi
    echo "$name: starting at $(date -u +%FT%TZ)" >> "$W"
    timeout 30h /home/robert/envs/widedeepgp/bin/python main_multilabel.py --data_dir /mnt/sdg/robert/deepwidegp/data/datasets \
        --label ard30 --rd_mode "$variant" --final_pool atten --d_model 32 --e_layers 1 --n_heads 4 --batch_size 16 \
        --accumulate 4 --eval_batch_size 128 --precision bf16 --wide on --tune_gpus auto --num_workers 4 \
        --run_dir /mnt/sdg/robert/deepwidegp/runs --experiment wide_gate --exp_name "$name" > "$G/$name.log" 2>&1
    echo "$name: exit $? at $(date -u +%FT%TZ)" >> "$W"
}
joint sim_joint sim_linear_grid hm3plus_proxy8_sim grid_gate_wait.log
joint real_joint real_linear_grid9 hm3plus_proxy8 real_linear9_wait.log     # the nine-value grid (DeepWideGP 0edb9cb)
echo "finished at $(date -u +%FT%TZ)" >> "$W"
