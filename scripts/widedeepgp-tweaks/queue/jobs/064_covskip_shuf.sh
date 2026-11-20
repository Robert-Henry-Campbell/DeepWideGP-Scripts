# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Own idea (covariate skip, logistic start) under the base sampler.
train $CODE3 covskip_shuf $CACHE_FLAGS --cov_skip logistic && score $CODE3 covskip_shuf
echo "exit $? at $(stamp)"
