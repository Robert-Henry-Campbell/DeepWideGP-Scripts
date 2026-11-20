# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Planned second mixer: the published bi-Mamba block over the block tokens, six layers at d 64 (matched to S3's size).
train $CODE3 mamba_s3 $CACHE_FLAGS --mixer bimamba1 --e_layers 6 && score $CODE3 mamba_s3
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
