# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 2 as proposed: a transformer layer within each block over its 32 strongest lassosum2 rows (self-attention over the pairs),
# the block's pooled vector as one more token, added to the block token; then the mixer over blocks with the covariate token.
train_pgi blockattn_lasso lassosum2 --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer transformer && score $CODE3 blockattn_lasso
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
