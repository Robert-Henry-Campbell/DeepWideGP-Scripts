# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Follow-up: the 32-wide ridge table with the optimiser search's chosen settings at batch 256 and the doubled budget.
F=$(best_flags search_opt256) || exit 1
train $CODE3 table_d32_opt256_20 $CACHE_FLAGS --init_encoder $BLOCKS/init_ridge_d32.pt --embed_dim 32 --batch_size 256 $F --max_epoch_equiv 20 && score $CODE3 table_d32_opt256_20
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
