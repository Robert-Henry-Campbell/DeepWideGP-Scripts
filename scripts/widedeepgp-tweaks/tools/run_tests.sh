#!/bin/bash
# The CPU test files touched by the tweaks, from the worktree, CUDA hidden (CLAUDE.md: never let a test take another user's GPU).
cd /mnt/sdg/robert/deepwidegp/runs/tweaks/worktree || exit 1
L=/mnt/sdg/robert/deepwidegp/runs/tweaks/tests_log/cpu_$(date -u +%Y%m%dT%H%M%SZ).log
echo "started $(date -u +%FT%TZ)" > "$L"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/robert/envs/widedeepgp/bin/python -m pytest -q -p no:cacheprovider \
    tests/test_ml_tweaks.py tests/test_ml_blocks.py tests/test_ml_loss.py tests/test_ml_entry_points.py tests/test_ml_training_pieces.py \
    -m "not slow" >> "$L" 2>&1
echo "exit $? at $(date -u +%FT%TZ)" >> "$L"
