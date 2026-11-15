#!/bin/bash
# Build the classic DeepGP comparator's five datasets (detached: setsid nohup). Log beside the data.
L=/mnt/sdg/robert/deepwidegp/data/classic_deepgp_work
mkdir -p "$L" && chmod 700 "$L"
cd /mnt/sdg/robert/deepwidegp/DeepWideGP || exit 1
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /home/robert/envs/widedeepgp/bin/python -I \
    /mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/scripts/classic-deepgp-comparator/build_datasets.py --threads 8 \
    >> "$L/build.log" 2>&1
echo "exit $? at $(date -u +%FT%TZ)" >> "$L/build.log"
