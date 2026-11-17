#!/bin/bash
# Start the LD-block study's chain (run_chain.py) detached, from a frozen export of DeepWideGP's `widening`:
#
#   setsid nohup bash run_all.sh > /dev/null 2>&1 < /dev/null &
#
# Rerunning continues: finished cells are skipped. Log: /mnt/sdg/robert/deepwidegp/runs/blocks/chain.log.
set -u
ROOT=/mnt/sdg/robert/deepwidegp
OUT=$ROOT/runs/blocks
REPO=$ROOT/DeepWideGP
mkdir -p "$OUT"
LOG=$OUT/run_all.log
H=$(git -C "$REPO" rev-parse --short widening)
C=$OUT/code_$H
if [ ! -d "$C" ]; then
    rm -rf "${C:?}.tmp" && mkdir -p "$C.tmp" && git -C "$REPO" archive widening | tar -x -C "$C.tmp" && mv "$C.tmp" "$C" \
        || { echo "could not export commit $H at $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }
fi
echo "starting at $(date -u +%FT%TZ) from code_$H, pid $$" >> "$LOG"
BLOCKS_CODE=$C /home/robert/envs/widedeepgp/bin/python "$(dirname "$0")/run_chain.py" >> "$LOG" 2>&1
echo "exit $? at $(date -u +%FT%TZ)" >> "$LOG"
