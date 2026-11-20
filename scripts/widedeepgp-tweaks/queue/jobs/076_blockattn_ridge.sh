# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 2, leak-free: the within-block transformer over each block's 32 strongest SNPs on the fold-routed ridge table (16 channels),
# each participant's rows from the copy fitted without them; the PGI tables carry the training labels (yardsticks), this table does not.
train $CODE3 blockattn_ridge $CACHE_FLAGS --block_attn 32 --block_attn_dim 32 --block_attn_layers 1 --block_attn_heads 4 --block_attn_mixer transformer && score $CODE3 blockattn_ridge
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
