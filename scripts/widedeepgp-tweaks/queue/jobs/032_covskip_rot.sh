# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Own idea: the covariates straight to the logits, starting at the covariates-only logistic model (the head starts at zero),
# so the network learns only what the covariates do not explain; on top of the balanced batches.
train $CODE2 covskip_rot $CACHE_FLAGS $ROT --cov_skip logistic && score $CODE2 covskip_rot
echo "exit $? at $(stamp)"
