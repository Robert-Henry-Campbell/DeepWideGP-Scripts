# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Follow-up: the 32-wide ridge table (the night's best so far) with the per-disease head and the doubled budget.
train $CODE3 table_d32_pdh20 $CACHE_FLAGS --init_encoder $BLOCKS/init_ridge_d32.pt --embed_dim 32 --per_disease_head 32 --max_epoch_equiv 20 && score $CODE3 table_d32_pdh20
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
