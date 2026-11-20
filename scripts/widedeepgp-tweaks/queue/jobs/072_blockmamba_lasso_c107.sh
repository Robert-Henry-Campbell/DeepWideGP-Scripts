# slots: 1
# after: 069_wait_leakfree_log
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 6 on the leak-free lassosum2 table: the published bi-Mamba block within each block, reading its 32 strongest SNPs in genomic
# order with the frozen lassosum2 rows (never updated), feeding the attention mixer over blocks and the head.
train_c107 blockmamba_lasso_c107 lassosum2 --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer bimamba1 && score $CODE3 blockmamba_lasso_c107
echo "exit $? at $(stamp)"
