# slots: 1
# after: 069_wait_leakfree_tables
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 2 on the leak-free LDpred2-auto table.
train_c107 blockattn_ldpred_c107 ldpred2_auto --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer transformer && score $CODE3 blockattn_ldpred_c107
echo "exit $? at $(stamp)"
