# slots: 1
# after: 069_wait_leakfree_tables
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Baseline for items 2 and 6 on the leak-free lassosum2 table (weights from the 107,743 rung, trained on the other 108,129 participants).
train_c107 pgi_lasso_frozen_c107 lassosum2 && score $CODE3 pgi_lasso_frozen_c107
echo "exit $? at $(stamp)"
