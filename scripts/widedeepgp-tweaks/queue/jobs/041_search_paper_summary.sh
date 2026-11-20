# slots: 1
# after: 040_search_paper_w*
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 1: the search's summary and choice once its workers are done.
search_summary $CODE2 search_paper 24 $CACHE_FLAGS $SEARCH_SAMPLER --tune_space paper
echo "exit $? at $(stamp)"
