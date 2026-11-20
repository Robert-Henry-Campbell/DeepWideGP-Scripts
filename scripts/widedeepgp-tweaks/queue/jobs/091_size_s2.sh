# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Planned size axis: S2 (d 32, two layers; head 128).
train $CODE3 size_s2 $CACHE_FLAGS --d_model 32 --e_layers 2 --n_heads 4 --head_hidden 128 && score $CODE3 size_s2
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
