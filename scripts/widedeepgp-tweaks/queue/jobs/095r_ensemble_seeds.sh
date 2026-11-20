# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Own idea, rerun from the fixed code: the three seeds of the base configuration averaged on the logit scale and scored on the tuning set.
D=$ROOT/runs/blocks_diag
[ -e $RUNS/yardsticks/ensemble_seeds.json ] || $PY $RUNS/ensemble.py $CODE3 $RUNS/yardsticks/ensemble_seeds.json $D/diag_S3_folds_frozen_nomask $D/diag_S3_folds_frozen_nomask_s2 $D/diag_S3_folds_frozen_nomask_s3
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
