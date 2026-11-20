# slots: 1
# after: 082_warm_d4
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Planned table-width axis: the frozen ridge table 4 wide.
train $CODE3 table_d4 $CACHE_FLAGS --init_encoder $BLOCKS/init_ridge_d4.pt --embed_dim 4 && score $CODE3 table_d4
echo "exit $? at $(stamp)"
