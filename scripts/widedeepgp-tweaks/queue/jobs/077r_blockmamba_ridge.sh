# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 6, leak-free on the fold-routed ridge table, from the fixed code (the first attempt started from the old copy and was stopped).
train $CODE3 blockmamba_ridge $CACHE_FLAGS --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer bimamba1 && score $CODE3 blockmamba_ridge
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
