# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 5: batch 256 and a search of the optimiser (lr, weight decay, dropout, schedule, warm-up), 16 trials of 10 epoch-equivalents,
# frozen ridge table, balanced batches. Worker 3 of 4.
search_worker $CODE2 search_opt256 16 3 $CACHE_FLAGS $SEARCH_SAMPLER --batch_size 256 --tune_space optimizer
echo "exit $? at $(stamp)"
