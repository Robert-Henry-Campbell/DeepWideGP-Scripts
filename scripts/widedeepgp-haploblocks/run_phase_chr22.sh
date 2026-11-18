#!/bin/bash
# Chromosome 22 of the top rung for the haploblock pre-test (methods.md sections 2 and 3): hard calls from the
# store, a BCF through plink2, the allele-frequency check, SHAPEIT5 phase_common, indexes. Every step is skipped
# when its product exists. Detached:  setsid nohup bash run_phase_chr22.sh > /dev/null 2>&1 < /dev/null &
set -uo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
ROOT=/mnt/sdg/robert/deepwidegp
DS=$ROOT/data/datasets/ard30/hm3plus
H=$DS/haplo
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/home/robert/envs/widedeepgp/bin/python
PLINK2=/mnt/sdg/robert/miniforge3/envs/ard-gwas/bin/plink2
export PATH=/mnt/sdg/robert/envs/phasing/bin:$PATH
mkdir -p -m 700 "$H"
LOG=$H/run_phase_chr22.log
step() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }
fail() { step "FAILED: $*"; echo "exit 1 at $(date -u +%FT%TZ)" >> "$LOG"; exit 1; }

step "start"
if [ ! -f "$H/export.json" ]; then
  step "export hard calls"
  $PY "$HERE/export_chr22.py" "$DS" --chrom 22 --threads 8 \
      --map "$ROOT/data/pgi/ld/genetic_maps/omni/chr22.OMNI.interpolated_genetic_map.gz" >> "$LOG" 2>&1 || fail export
fi
if [ ! -f "$H/chr22_unphased.bcf.csi" ]; then
  step "plink2 to BCF"
  $PLINK2 --bfile "$H/chr22_hardcalls" --export bcf --out "$H/chr22_unphased" --threads 4 --memory 16000 >> "$LOG" 2>&1 || fail plink2
  bcftools index -f "$H/chr22_unphased.bcf" || fail index
  chmod 600 "$H"/chr22_unphased.*
  step "check allele frequencies"
  $PY "$HERE/export_chr22.py" "$DS" --chrom 22 --check "$H/chr22_unphased.bcf" >> "$LOG" 2>&1 || fail check
fi
if [ ! -f "$H/chr22_phased.bcf.csi" ]; then
  step "phase_common"
  SHAPEIT5_phase_common --input "$H/chr22_unphased.bcf" --region 22 --map "$H/chr22.gmap.gz" \
      --output "$H/chr22_phased.bcf" --thread 10 --seed 1 --log "$H/phase_common.log" >> "$LOG" 2>&1 || fail phase_common
  bcftools index -f "$H/chr22_phased.bcf" || fail index_phased
  chmod 600 "$H"/chr22_phased.* "$H/phase_common.log"
fi
step "done"
echo "exit 0 at $(date -u +%FT%TZ)" >> "$LOG"
