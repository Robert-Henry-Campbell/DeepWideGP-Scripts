# slots: 1
# Item 3: disease-balanced batches (the rotation sampler: each batch half cases of one disease, half its non-cases, the
# diseases in turn, non-cases of a disease shown once per pass, cases cycling), plain cross-entropy on the batch as drawn.
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
CODE=$RUNS/code_5c47884
train $CODE rot10 --sampler rotation --no_sampling_correction --gamma 0 && score $CODE rot10
echo "exit $? at $(stamp)"
