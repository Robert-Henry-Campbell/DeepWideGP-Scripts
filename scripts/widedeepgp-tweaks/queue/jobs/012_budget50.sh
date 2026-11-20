# slots: 1
# Item 0: a 50 epoch-equivalent budget (early stopping and the plateau schedule still act), otherwise the base configuration.
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
CODE=$RUNS/code_5c47884
train $CODE budget50 --max_epoch_equiv 50 --snapshots_max 60 && score $CODE budget50
echo "exit $? at $(stamp)"
