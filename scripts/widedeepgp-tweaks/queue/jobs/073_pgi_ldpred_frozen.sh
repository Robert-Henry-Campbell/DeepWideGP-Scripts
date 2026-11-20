# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# The frozen LDpred2-auto table (full-rung weights) under the base recipe: the control for item 2 on LDpred2.
train_pgi pgi_ldpred_frozen ldpred2_auto && score $CODE3 pgi_ldpred_frozen
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
