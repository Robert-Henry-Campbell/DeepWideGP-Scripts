# slots: 1
# after: 080_warm_d32
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Planned table-width axis: the frozen fold-routed ridge table 32 wide (the 16-wide one is the base).
train $CODE3 table_d32 $CACHE_FLAGS --init_encoder $BLOCKS/init_ridge_d32.pt --embed_dim 32 && score $CODE3 table_d32
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
