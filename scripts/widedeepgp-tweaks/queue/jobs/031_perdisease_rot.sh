# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 7: one small head (width 32) per disease on the shared hidden layer, on top of the balanced batches.
train $CODE2 perdisease_rot $CACHE_FLAGS $ROT --per_disease_head 32 && score $CODE2 perdisease_rot
echo "exit $? at $(stamp)"
