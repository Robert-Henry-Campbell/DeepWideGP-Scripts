# slots: 1
# after: 069_wait_leakfree_log
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# The naive half-rung-weight lassosum2 and LDpred2 scores plus covariates, fitted on the c107743 training rows and scored on the tuning set:
# the honest "lift over naive PGI" yardsticks for the runs above (and the first-layer cache for them).
for m in lassosum2 ldpred2_auto; do
    [ -e "$RUNS/yardsticks/c107_${m}_summary.json" ] && continue
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 $PY $RUNS/pgi_yardstick.py $CODE3 $C107_BLOCKS/init_pgi_${m}_d30.pt $RUNS/yardsticks/c107_$m --rd_mode $C107 || exit 1
done
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
