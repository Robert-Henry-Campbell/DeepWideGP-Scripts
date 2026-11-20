# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 2 on the fold-routed ridge table, rerun from the fixed code.
train $CODE3 blockattn_ridge $CACHE_FLAGS --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer transformer && score $CODE3 blockattn_ridge
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
