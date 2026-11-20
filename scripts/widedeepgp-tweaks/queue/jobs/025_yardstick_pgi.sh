# slots: 1
# The naive lassosum2 and LDpred2-auto scores plus covariates, fitted on the training rows and scored on the tuning set (the
# deep arm's way), from the other agent's 30-channel tables: the yardsticks for lift over the PGI a first layer is built from.
# Leaves each table's pooled vectors in the disk cache for the runs that use it.
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
for m in lassosum2 ldpred2_auto; do
    [ -e "$RUNS/yardsticks/${m}_summary.json" ] && continue
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 $PY $RUNS/pgi_yardstick.py $CODE2 $BLOCKS/init_pgi_${m}_d30.pt $RUNS/yardsticks/$m || exit 1
done
echo "exit $? at $(stamp)"
