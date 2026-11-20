# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Item 1: an honest hyper-parameter search over the published paper's grid (batch 32/64/128, mixer layers 1-3, width 16/32/64,
# heads 2/4/8, head depth 1/2/4, dropout 0.05-0.3, lr 1e-5..1e-3, weight decay 1e-5..1e-3; plus the mixer kind), 24 trials of
# 10 epoch-equivalents each on the frozen ridge table with balanced batches, successive halving after 2 epoch-equivalents. Worker 2 of 4.
search_worker $CODE2 search_paper 24 2 $CACHE_FLAGS $SEARCH_SAMPLER --tune_space paper
echo "exit $? at $(stamp)"
