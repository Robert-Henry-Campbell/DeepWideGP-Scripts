# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 6 as proposed: the published bi-Mamba block within each block, reading its 32 strongest SNPs in genomic order with the
# frozen lassosum2 rows (never updated), feeding the attention mixer over blocks and the head.
train_pgi blockmamba_lasso lassosum2 --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer bimamba1 && score $CODE3 blockmamba_lasso
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
