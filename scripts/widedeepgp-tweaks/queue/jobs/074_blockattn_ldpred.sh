# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 2 on the full-rung LDpred2-auto table.
train_pgi blockattn_ldpred ldpred2_auto --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer transformer && score $CODE3 blockattn_ldpred
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
