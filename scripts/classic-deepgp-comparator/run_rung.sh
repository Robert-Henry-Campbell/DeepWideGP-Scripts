#!/bin/bash
# The classic DeepGP comparator (hypertension, I10) at one training size, called by the ladder
# chain (runs/ladder/run_ladder_2026-11-15.sh) after the WideDeepGP run of that size, with the
# WideDeepGP dataset's name (hm3plus_n13199 ... hm3plus). The published DeepGP model and training
# on its GWAS-selected SNPs (dataset classic_i10_n<size>, built by
# DeepWideGP-Scripts/scripts/classic-deepgp-comparator/build_datasets.py), under the study's
# protocol: seed 1, 12 Optuna trials over learning rate, weight decay and dropout with the
# published settings first, early stopping on the tuning set, no test set. Runs from the same
# frozen export of DeepWideGP's `widening` as the ladder. Rerunning continues an unfinished search.
set -u
rung=${1:?training size, e.g. hm3plus_n13199}
case "$rung" in
    hm3plus_n13199) n=13199 ;; hm3plus_n26287) n=26287 ;; hm3plus_n53498) n=53498 ;;
    hm3plus_n107743) n=107743 ;; hm3plus) n=215872 ;; *) echo "unknown training size $rung"; exit 2 ;;
esac
ds=classic_i10_n$n
D=/mnt/sdg/robert/deepwidegp/data/datasets/ard30/$ds
R=/mnt/sdg/robert/deepwidegp/runs/classic_deepgp
REPO=/mnt/sdg/robert/deepwidegp/DeepWideGP
if [ ! -e "$D/build_log.json" ]; then echo "$ds: dataset not built yet; to be run later"; exit 3; fi
if [ -e "$R/$ds/results.json" ]; then echo "$ds: done already"; exit 0; fi
H=$(git -C "$REPO" rev-parse --short widening)
C=/mnt/sdg/robert/deepwidegp/runs/ladder/code_$H
if [ ! -d "$C" ]; then
    rm -rf "${C:?}.tmp" && mkdir -p "$C.tmp" && git -C "$REPO" archive widening | tar -x -C "$C.tmp" && mv "$C.tmp" "$C" || exit 1
fi
echo "$ds: starting at $(date -u +%FT%TZ) from code_$H"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$C" && /home/robert/envs/widedeepgp/bin/python main_multilabel.py --data_dir /mnt/sdg/robert/deepwidegp/data/datasets \
    --label ard30 --rd_mode "$ds" --labels I10 --wide off --sampler shuffled --gamma 0 --batch_size 64 --accumulate 1 \
    --lr 7e-4 --l2 1e-4 --dropout 0.2 --d_model 32 --e_layers 1 --n_heads 4 --n_fcn 2 --head_hidden 256 \
    --final_pool atten --snp_embed cov --precision 32 --eval_batch_size 256 --tune_trials 12 --tune_first_given \
    --tune_gpus auto --num_workers 4 --run_dir /mnt/sdg/robert/deepwidegp/runs --experiment classic_deepgp --exp_name "$ds"
