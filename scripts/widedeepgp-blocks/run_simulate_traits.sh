#!/bin/bash
# Build the E0 dataset of simulated traits on the real genotypes of the bottom rung (one GPU; about seven minutes).
# Launched detached by the main session, before the chain's lin stage needs the dataset:
#   setsid nohup bash run_simulate_traits.sh > /dev/null 2>&1 < /dev/null &
# The build refuses to overwrite an existing dataset: move a superseded one aside first.
set -u
LOG=/mnt/sdg/robert/deepwidegp/runs/e0/build_hm3plus_n13199_sim.log
mkdir -p "$(dirname "$LOG")"
cd /mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/scripts/widedeepgp-blocks || exit 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "start at $(date -u +%FT%TZ) on GPU $CUDA_VISIBLE_DEVICES" >> "$LOG"
/home/robert/envs/widedeepgp/bin/python simulate_traits.py --source hm3plus_n13199 --name hm3plus_n13199_sim >> "$LOG" 2>&1
echo "exit $? at $(date -u +%FT%TZ)" >> "$LOG"
