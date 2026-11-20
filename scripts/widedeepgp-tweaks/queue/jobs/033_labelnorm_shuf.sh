# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 8 under the natural-prevalence batches of the base configuration.
train $CODE2 labelnorm_shuf $CACHE_FLAGS --label_norm entropy && score $CODE2 labelnorm_shuf
echo "exit $? at $(stamp)"
