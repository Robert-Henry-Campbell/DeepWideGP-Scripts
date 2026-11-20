# slots: 0
# Waits (up to 4 h) for the other agent's leak-free tables: its chain log records "table_split_lassosum2: exit 0" and
# "table_split_ldpred2_auto: exit 0" once each table is closed (the files are written in place, so their existence is not enough).
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
LOG=$ROOT/runs/pgi_tables/chain.log
for i in $(seq 1 120); do
    if grep -q "table_split_lassosum2: exit 0" "$LOG" 2>/dev/null && grep -q "table_split_ldpred2_auto: exit 0" "$LOG" 2>/dev/null \
       && [ -e $DS/ard30/$C107/store.json ] && [ -e $DS/ard30/$C107/samples_val.tsv ] \
       && [ -e $C107_BLOCKS/init_pgi_lassosum2_d30.pt ] && [ -e $C107_BLOCKS/init_pgi_ldpred2_auto_d30.pt ]; then
        echo "tables complete at $(stamp)"; exit 0
    fi
    sleep 120
done
echo "tables not complete after 4 h"; exit 1
