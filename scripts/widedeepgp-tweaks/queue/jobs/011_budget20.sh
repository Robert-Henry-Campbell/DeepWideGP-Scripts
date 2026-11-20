# slots: 1
# Item 9: the training budget doubled (20 epoch-equivalents), otherwise the base configuration.
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
CODE=$RUNS/code_5c47884
train $CODE budget20 --max_epoch_equiv 20 && score $CODE budget20
echo "exit $? at $(stamp)"
