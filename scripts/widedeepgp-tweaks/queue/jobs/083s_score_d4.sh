# slots: 1
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# The streamed re-score of table_d4 (its first attempt ran out of GPU memory on a crowded GPU), at a smaller scoring batch.
[ -e "$RUNS/table_d4/scores_val_summary.csv" ] && exit 0
(cd "$CODE3" && $PY score_snapshots.py --run "$RUNS/table_d4" --split val --snapshots best --batch_size 256)
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
