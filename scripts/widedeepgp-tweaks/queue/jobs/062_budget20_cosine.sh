# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Own idea: the doubled budget with a cosine schedule to zero (200 warm-up steps) instead of the plateau rule, which rarely fires in 20 epoch-equivalents.
train $CODE3 budget20_cosine $CACHE_FLAGS --max_epoch_equiv 20 --lr_schedule cosine --warmup_steps 200 && score $CODE3 budget20_cosine
echo "exit $? at $(stamp)"
