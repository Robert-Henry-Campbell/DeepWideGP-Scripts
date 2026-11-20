# slots: 1
# after: 041_search_paper_summary
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Combination of everything that came out at or above the base: the paper search's chosen settings, the doubled budget, the
# per-disease head (+0.0004 on the base sampler) and the entropy-normalised loss (+0.0002).
F=$(best_flags search_paper) || exit 1
train $CODE3 combo_all20 $CACHE_FLAGS $F --max_epoch_equiv 20 --per_disease_head 32 --label_norm entropy && score $CODE3 combo_all20
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
