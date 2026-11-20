# slots: 0
# Item 4: the classic DeepGP's SNP set for type 2 diabetes (E11) with the published paper's clumping (r2 0.7 within 500 kb,
# p <= 0.05) instead of the comparator's r2 0.1 / 250 kb: one pass over the main store (about 3 h), CPU only.
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
OUT=$DS/ard30/classic_e11_r07_n215872
[ -e "$OUT/build_log.json" ] && { echo "dataset exists"; exit 0; }
cd $ROOT/DeepWideGP-Scripts/scripts/classic-deepgp-comparator && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY build_datasets.py --threads 8 \
    --name classic_e11_r07 --rungs hm3plus --diseases E11 --r2 0.7 --kb 500
echo "exit $? at $(stamp)"
