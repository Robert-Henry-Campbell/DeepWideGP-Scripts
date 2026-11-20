# slots: 0
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Planned table-width axis: a 32-wide ridge warm start (fold copies, unit-RMS tokens) from the top rung's block ridge.
[ -e $BLOCKS/init_ridge_d32.pt ] && exit 0
cd $ROOT/DeepWideGP-Scripts/scripts/widedeepgp-blocks && OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 DEEPGP_REPO=$CODE3 $PY make_warm_start.py $DS/ard30/hm3plus --embed_dim 32
echo "exit $? at $(stamp)"
