# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# The proposed first layer as specified: the frozen lassosum2 table (30 channels, the full rung's weights, which were fitted on the
# training rows: PGI AUROC 0.887 on them against 0.572 on the tuning set) under the base recipe; the control for items 2 and 6.
train_pgi pgi_lasso_frozen lassosum2 && score $CODE3 pgi_lasso_frozen
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
