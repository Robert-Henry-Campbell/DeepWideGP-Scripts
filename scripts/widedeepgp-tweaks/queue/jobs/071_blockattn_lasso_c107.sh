# slots: 1
# after: 069_wait_leakfree_tables
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 2 on the leak-free lassosum2 table: a transformer layer within each block over its 32 strongest SNPs (self-attention over the
# pairs), the block's linear summary as one more token, added to the block token; then the mixer over blocks with the covariate token.
train_c107 blockattn_lasso_c107 lassosum2 --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer transformer && score $CODE3 blockattn_lasso_c107
echo "exit $? at $(stamp)"
