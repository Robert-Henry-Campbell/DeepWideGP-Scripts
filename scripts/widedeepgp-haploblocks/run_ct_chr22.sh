#!/bin/bash
# Clumping-and-thresholding comparison on chr22 (methods.md section 9), detached:
#   setsid nohup bash run_ct_chr22.sh > /dev/null 2>&1 < /dev/null &
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
ROOT=/mnt/sdg/robert/deepwidegp
DS=$ROOT/data/datasets/ard30/hm3plus
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/robert/envs/widedeepgp/bin/python
PART=$ROOT/data/reference/partitions/prive431
LOG=$DS/haplo/run_ct_chr22.log
CELL=${CELL:-f0.01_c0.95}
echo "[$(date -u +%FT%TZ)] compare_ct $CELL" >> "$LOG"
if [ -s "$HERE/results/compare_ct_chr22_${CELL}_pooled.csv" ]; then echo "[$(date -u +%FT%TZ)] already done" >> "$LOG"; echo "exit 0 at $(date -u +%FT%TZ)" >> "$LOG"; exit 0; fi
$PY "$HERE/compare_ct.py" "$DS" --chrom 22 --partition "$PART" --cell "$CELL" >> "$LOG" 2>&1
echo "exit $? at $(date -u +%FT%TZ)" >> "$LOG"
