# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 7 under the base sampler (the rotation turned out worse than the base, so each item is also read on the base recipe).
train $CODE3 perdisease_shuf $CACHE_FLAGS --per_disease_head 32 && score $CODE3 perdisease_shuf
echo "exit $? at $(stamp)"
