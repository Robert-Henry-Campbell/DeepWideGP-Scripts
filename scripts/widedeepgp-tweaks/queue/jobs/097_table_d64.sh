# slots: 1
# after: 096_warm_d64
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Planned table-width axis: the frozen ridge table 64 wide (its cache holds 26 GB in memory, so one such run at a time).
train $CODE3 table_d64 $CACHE_FLAGS --init_encoder $BLOCKS/init_ridge_d64.pt --embed_dim 64 && score $CODE3 table_d64
echo "exit $? at $(stamp)"
