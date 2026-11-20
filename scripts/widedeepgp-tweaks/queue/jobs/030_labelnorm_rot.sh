# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 8: each disease's loss divided by the entropy of its prevalence, on top of the balanced batches (item 3).
train $CODE2 labelnorm_rot $CACHE_FLAGS $ROT --label_norm entropy && score $CODE2 labelnorm_rot
echo "exit $? at $(stamp)"
