#!/bin/bash
# GPU test suites of DeepWideGP commit 81ded30 (branch widening: the linear PRS path, its grid stage and the
# search inside every run), run on an export of that commit (code_81ded30/, outside git, so
# later edits cannot change what is tested). Replaces the job of 12 Nov that waited to test
# commit 2310856, whose code this commit contains.
#
# Robert's rule is to use the free GPUs minus one: this waits until at least two GPUs have no
# compute process, then uses every free GPU but the lowest-numbered one. It gives up after
# 24 hours. Logs and gpu_summary.txt are written beside this script. Started unattended on
# 13 Nov 2026 at 00:42 UTC (first at 00:16 for 9e69019 and 00:39 for e5cebed, superseded), when another user's jobs held all five GPUs.
L=/mnt/sdg/robert/deepwidegp/runs/gate_evidence/final_tests_2026-11-13
R=$L/code_81ded30
PY=/home/robert/envs/widedeepgp/bin/python
PROBE=/mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/scripts/widedeepgp-pilot
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

free_gpus() {
    local busy
    busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read -r idx uuid; do
        if ! grep -q "$uuid" <<< "$busy"; then echo "$idx"; fi
    done
}

echo "waiting since $(date -u +%FT%TZ), pid $$" > "$L/gpu_wait.log"
deadline=$(( $(date +%s) + 24 * 3600 ))
while true; do
    mapfile -t FREE < <(free_gpus)
    if [ "${#FREE[@]}" -ge 2 ]; then break; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "gave up at $(date -u +%FT%TZ): fewer than two GPUs free for 24 hours" | tee -a "$L/gpu_wait.log" > "$L/gpu_summary.txt"
        exit 3
    fi
    sleep 120
done
USE=("${FREE[@]:1}")
echo "free GPUs at $(date -u +%FT%TZ): ${FREE[*]}; using: ${USE[*]}" >> "$L/gpu_wait.log"
date -u +%FT%TZ > "$L/started_gpu.txt"

suite() {   # name gpu
    case "$1" in
        wide) (cd "$R" && CUDA_VISIBLE_DEVICES=$2 $PY -m pytest tests/test_ml_wide.py -v -p no:warnings -p no:cacheprovider > "$L/wide.log" 2>&1; echo "exit $?" >> "$L/wide.log") ;;
        train_quick) (cd "$R" && CUDA_VISIBLE_DEVICES=$2 $PY -m pytest tests/test_ml_train.py -m "not slow" -v -p no:warnings -p no:cacheprovider > "$L/train_quick.log" 2>&1; echo "exit $?" >> "$L/train_quick.log") ;;
        model) (cd "$R" && CUDA_VISIBLE_DEVICES=$2 $PY -m pytest tests/test_ml_model.py -v -p no:warnings -p no:cacheprovider > "$L/model.log" 2>&1; echo "exit $?" >> "$L/model.log") ;;
        train_slow) (cd "$R" && CUDA_VISIBLE_DEVICES=$2 $PY -m pytest tests/test_ml_train.py -m slow -v -p no:warnings -p no:cacheprovider > "$L/train_slow.log" 2>&1; echo "exit $?" >> "$L/train_slow.log") ;;
        probe) (cd "$PROBE" && for repo in "$R" /mnt/sdg/robert/deepwidegp/runs/pilot_code_8f03ca6; do for an in "" "--annot"; do echo "DEEPGP_REPO=$repo $an"; DEEPGP_REPO=$repo CUDA_VISIBLE_DEVICES=$2 $PY capacity_probe.py --batch 2 --proxy-eighth 1 --steps 2 $an 2>&1 | grep -E "RESULT|Error|error" | cut -c1-400; done; done > "$L/capacity_probe.log" 2>&1; echo "exit $?" >> "$L/capacity_probe.log") ;;
    esac
}

n=${#USE[@]}
if [ "$n" -ge 4 ]; then
    suite train_slow "${USE[0]}" & suite wide "${USE[1]}" & (suite train_quick "${USE[2]}"; suite model "${USE[2]}") & suite probe "${USE[3]}" &
    wait
elif [ "$n" -ge 2 ]; then
    suite train_slow "${USE[0]}" &
    (for s in wide train_quick model probe; do suite "$s" "${USE[1]}"; done) &
    wait
else
    for s in wide train_slow train_quick model probe; do suite "$s" "${USE[0]}"; done
fi
date -u +%FT%TZ > "$L/finished_gpu.txt"
{
    echo "GPU suites of commit 81ded30 (export code_81ded30), $(cat "$L/started_gpu.txt") to $(cat "$L/finished_gpu.txt"), GPUs ${USE[*]}"
    for f in wide train_quick model train_slow capacity_probe; do
        echo "$f: $(grep -E '^(=+ .*(passed|failed|error).*=+|exit [0-9]+)$' "$L/$f.log" | tr '\n' ' ')"
    done
    echo "capacity_probe ok lines: $(grep -c '"ok": true' "$L/capacity_probe.log") of 4"
    grep -h -E "FAILED|ERROR" "$L"/wide.log "$L"/train_quick.log "$L"/model.log "$L"/train_slow.log | head -20
} > "$L/gpu_summary.txt"
