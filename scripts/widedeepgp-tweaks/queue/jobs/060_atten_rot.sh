# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Own idea: the attention readout (one query per disease over the blocks) with balanced batches; it equalled the mean readout under the old objective.
train $CODE2 atten_rot $CACHE_FLAGS $ROT --final_pool atten && score $CODE2 atten_rot
echo "exit $? at $(stamp)"
