#!/bin/bash
# Held-out scoring of the named models' finished runs (methods.md section 3), detached (CLAUDE.md 1):
#   setsid nohup bash run_heldout.sh kelemen delphi gln prsformer widedeepgp classic scprs > /dev/null 2>&1 < /dev/null &
# Freezes the committed code, waits for a free GPU (no compute process on it), then scores each model in turn; a
# model that fails stops the launch. Finished runs are skipped, so it can be launched again when more runs finish.
# Log: runs/heldout_ancestry/logs/<time>_<models>.log, last line "exit <code> at <time>".
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=/mnt/sdg/robert/deepwidegp/runs/heldout_ancestry
mkdir -p "$OUT/logs"
LOG=$OUT/logs/$(date -u +%Y%m%dT%H%M%SZ)_$(echo "$*" | tr ' ' '_').log
exec >> "$LOG" 2>&1
trap 'echo "exit $? at $(date -u +%FT%TZ)"' EXIT
[ $# -gt 0 ] || { echo "name the models"; exit 2; }
CODE=$(bash "$HERE/freeze.sh") || exit 1
echo "started $(date -u +%FT%TZ), pid $$, code $CODE: $(tr '\n' ' ' < "$CODE/COMMITS.txt")"
free_gpu() {
  local busy
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader | while IFS=', ' read -r i u; do
    grep -q "$u" <<< "$busy" || { echo "$i"; break; }
  done
}
until GPU=$(free_gpu) && [ -n "$GPU" ]; do sleep 120; done
export CUDA_VISIBLE_DEVICES=$GPU
echo "GPU $GPU taken at $(date -u +%FT%TZ)"
for m in "$@"; do
  /home/robert/envs/widedeepgp/bin/python "$HERE/score_heldout.py" --model "$m" --code "$CODE" --device cuda || exit 1
done
