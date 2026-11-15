#!/bin/bash
# Rerun of the one GPU test that failed on 13 Nov (a timeout: test_the_tuner_runs_shared_trials_and_prunes),
# on an export of DeepWideGP 197741a, which gives the test the small-data floors. Waits until at least
# two GPUs have no compute process (the lab's rule: the free ones minus one) and uses one of them.
# Gives up after 24 hours. Writes rerun_summary.txt beside this script.
L=/mnt/sdg/robert/deepwidegp/runs/gate_evidence/final_tests_2026-11-13
R=$L/code_197741a
PY=/home/robert/envs/widedeepgp/bin/python
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
free_gpus() {
    local busy
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read -r idx uuid; do
        if ! grep -q "$uuid" <<< "$busy"; then echo "$idx"; fi
    done
}
echo "waiting since $(date -u +%FT%TZ), pid $$" > "$L/rerun_wait.log"
deadline=$(( $(date +%s) + 24 * 3600 ))
while true; do
    mapfile -t FREE < <(free_gpus)
    if [ "${#FREE[@]}" -ge 2 ]; then break; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then echo "gave up at $(date -u +%FT%TZ)" | tee -a "$L/rerun_wait.log" > "$L/rerun_summary.txt"; exit 3; fi
    sleep 120
done
echo "free GPUs at $(date -u +%FT%TZ): ${FREE[*]}; using ${FREE[1]}" >> "$L/rerun_wait.log"
(cd "$R" && CUDA_VISIBLE_DEVICES=${FREE[1]} $PY -m pytest tests/test_ml_train.py -k test_the_tuner_runs_shared_trials_and_prunes -v -p no:warnings -p no:cacheprovider > "$L/rerun_tuner_test.log" 2>&1; echo "exit $?" >> "$L/rerun_tuner_test.log")
echo "rerun of the tuner test on 197741a, $(date -u +%FT%TZ): $(grep -E '^(=+ .*(passed|failed|error).*=+|exit [0-9]+)$' "$L/rerun_tuner_test.log" | tr '\n' ' ')" > "$L/rerun_summary.txt"
