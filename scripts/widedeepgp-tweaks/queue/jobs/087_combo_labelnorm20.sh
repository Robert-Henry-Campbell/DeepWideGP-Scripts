# slots: 1
# after: 041_search_paper_summary
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Combination: the paper search's chosen settings, the doubled budget and the entropy-normalised loss (items 1 + 9 + 8).
F=$(best_flags search_paper) || exit 1
train $CODE3 combo_paper20_labelnorm $CACHE_FLAGS $F --max_epoch_equiv 20 --label_norm entropy && score $CODE3 combo_paper20_labelnorm
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
