# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Planned size axis at the top rung on the frozen table: S1 (d 16, one layer; head 64).
train $CODE3 size_s1 $CACHE_FLAGS --d_model 16 --e_layers 1 --n_heads 4 --head_hidden 64 && score $CODE3 size_s1
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
