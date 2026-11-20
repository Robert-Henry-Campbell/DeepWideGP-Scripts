# slots: 1
# after: 041_search_paper_summary
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Combination: the paper search's chosen settings with the doubled budget (items 1 + 9), base sampler.
F=$(best_flags search_paper) || exit 1
echo "chosen settings: $F"
train $CODE3 combo_paper20 $CACHE_FLAGS $F --max_epoch_equiv 20 && score $CODE3 combo_paper20
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
