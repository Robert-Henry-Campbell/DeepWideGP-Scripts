# slots: 1
# after: 050_search_opt256_w*
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 5: the search's summary and choice.
search_summary $CODE2 search_opt256 16 $CACHE_FLAGS $SEARCH_SAMPLER --batch_size 256 --tune_space optimizer
echo "exit $? at $(stamp)"
