# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Follow-up: the 32-wide ridge table (ahead of the base at matched epochs in table_d32) with the doubled budget.
train $CODE3 table_d32_20 $CACHE_FLAGS --init_encoder $BLOCKS/init_ridge_d32.pt --embed_dim 32 --max_epoch_equiv 20 && score $CODE3 table_d32_20
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
