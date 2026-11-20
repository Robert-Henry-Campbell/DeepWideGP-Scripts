# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Own idea: item 2 on the ridge table with 64 SNPs per block and two layers.
train $CODE3 blockattn_ridge_k64 $CACHE_FLAGS --block_attn 64 --block_attn_dim 32 --block_attn_layers 2 --block_attn_heads 4 --block_attn_mixer transformer && score $CODE3 blockattn_ridge_k64
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
