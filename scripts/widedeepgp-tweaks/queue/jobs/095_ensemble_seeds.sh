# slots: 1
# Own idea: the three seeds of the base configuration (blocks_diag/diag_S3_folds_frozen_nomask, _s2, _s3) averaged on the logit
# scale and scored on the tuning set: what seed-ensembling alone buys.
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
D=$ROOT/runs/blocks_diag
[ -e $RUNS/yardsticks/ensemble_seeds.json ] || $PY $RUNS/ensemble.py $CODE3 $RUNS/yardsticks/ensemble_seeds.json $D/diag_S3_folds_frozen_nomask $D/diag_S3_folds_frozen_nomask_s2 $D/diag_S3_folds_frozen_nomask_s3
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
