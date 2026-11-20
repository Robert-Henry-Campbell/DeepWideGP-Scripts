# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 6 on the leak-free lassosum2 table, rerun from the fixed code (the first attempt ran out of GPU memory).
train_c107 blockmamba_lasso_c107 lassosum2 --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer bimamba1 && score $CODE3 blockmamba_lasso_c107
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
