# slots: 0
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 4, the control: the classic E11 SNP set at the comparator's own clumping (r2 0.1 within 250 kb, p <= 0.05), so the clumping
# change can be read on its own. CPU only, one pass over the main store.
OUT=$DS/ard30/classic_e11_n215872
[ -e "$OUT/build_log.json" ] && { echo "dataset exists"; exit 0; }
cd $ROOT/DeepWideGP-Scripts/scripts/classic-deepgp-comparator && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 $PY build_datasets.py --threads 8 --name classic_e11 --rungs hm3plus --diseases E11
echo "exit $? at $(stamp)"
