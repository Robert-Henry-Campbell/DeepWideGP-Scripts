# slots: 1
# after: 069_wait_leakfree_log
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Baseline on the LDpred2-auto table with weights from the other half of the rung.
train_c107 pgi_ldpred_frozen_c107 ldpred2_auto && score $CODE3 pgi_ldpred_frozen_c107
echo "exit $? at $(stamp)"
