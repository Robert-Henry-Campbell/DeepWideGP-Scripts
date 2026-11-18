#!/bin/bash
# Waiter for the chr22 haploblock pre-test (methods.md sections 4 to 6): once the phasing chain has written its exit
# line, cut the haploblocks (the nine grid cells) and run the ridge comparison for the primary cell and the two
# coverage sensitivities. Every step is skipped when its products exist. Detached:
#   setsid nohup bash run_analysis_chr22.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10
ROOT=/mnt/sdg/robert/deepwidegp
DS=$ROOT/data/datasets/ard30/hm3plus
H=$DS/haplo
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/robert/envs/widedeepgp/bin/python
PART=$ROOT/data/reference/partitions/prive431
export PATH=/mnt/sdg/robert/envs/phasing/bin:$PATH
LOG=$H/run_analysis_chr22.log
step() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
fail() { step "FAILED: $*"; echo "exit 1 at $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }

step "waiting for the phasing chain"
until grep -q "^exit " "$H/run_phase_chr22.log" 2>/dev/null; do sleep 120; done
grep -q "^exit 0" "$H/run_phase_chr22.log" || fail "the phasing chain did not finish cleanly"
[ -f "$H/chr22_phased.bcf.csi" ] || fail "no indexed phased BCF"

missing=0
for f in 0.005 0.01 0.02; do for c in 0.9 0.95 0.99; do
  [ -f "$H/chr22_haploblocks_f${f}_c${c}.npz" ] || missing=1
done; done
if [ $missing = 1 ]; then
  step "haploblocks (nine cells)"
  $PY "$HERE/haploblocks.py" "$DS" --chrom 22 --partition "$PART" --grid >> "$LOG" 2>&1 || fail haploblocks
fi
for cell in f0.01_c0.95 f0.01_c0.9 f0.01_c0.99; do
  [ -f "$HERE/results/compare_chr22_${cell}_pooled.csv" ] && continue
  step "compare_ridge $cell"
  $PY "$HERE/compare_ridge.py" "$DS" --chrom 22 --partition "$PART" --cell "$cell" >> "$LOG" 2>&1 || fail "compare_ridge $cell"
done
step "done"
echo "exit 0 at $(date -u +%FT%TZ)" >> "$LOG"
