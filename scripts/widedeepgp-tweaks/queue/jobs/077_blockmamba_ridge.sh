# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 6, weights fitted on other participants: the within-block bi-Mamba over each block's 32 strongest SNPs in genomic order, on the fold-routed ridge table.
train $CODE3 blockmamba_ridge $CACHE_FLAGS --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer bimamba1 && score $CODE3 blockmamba_ridge
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
