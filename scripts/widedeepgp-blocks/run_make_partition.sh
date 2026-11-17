#!/bin/bash
# Builds the ldsplit2000 and ldsplit5000 partitions (22 chromosomes, 8 workers; resumable), then checks them
# against the store, nests them (prive431 -> ldsplit2000 -> ldsplit5000) and writes the aggregate table.
# Launch detached:  setsid nohup bash run_make_partition.sh > /dev/null 2>&1 < /dev/null &
set -u
cd /mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts || exit 1
OUT=/mnt/sdg/robert/deepwidegp/data/reference/partitions
PY=/home/robert/envs/widedeepgp/bin/python
CHECK="$PY scripts/widedeepgp-blocks/check_partition.py"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
{
  echo "start $(date -u +%FT%TZ)"
  /usr/bin/Rscript scripts/widedeepgp-blocks/make_partition.R cores=8 \
    && $CHECK ldsplit2000 --tsv $OUT/ldsplit2000/blocks.tsv --parent prive431 --built $OUT/ldsplit2000/make_partition.json \
    && $CHECK ldsplit5000 --tsv $OUT/ldsplit5000/blocks.tsv --parent ldsplit2000 --built $OUT/ldsplit5000/make_partition.json \
    && $CHECK --csv scripts/widedeepgp-blocks/partitions.csv prive431 ldsplit2000 ldsplit5000
  echo "exit $? at $(date -u +%FT%TZ)"
} >> "$OUT/make_partition.log" 2>&1
