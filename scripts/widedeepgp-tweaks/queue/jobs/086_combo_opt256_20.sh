# slots: 1
# after: 051_search_opt256_summary
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Combination: the batch-256 optimiser search's chosen settings with the doubled budget (items 5 + 9), base sampler.
F=$(best_flags search_opt256) || exit 1
echo "chosen settings: $F"
train $CODE3 combo_opt256_20 $CACHE_FLAGS --batch_size 256 $F --max_epoch_equiv 20 && score $CODE3 combo_opt256_20
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
