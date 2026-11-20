# slots: 0
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Waits (up to 4 h) for the other agent's dataset and tables with weights from the other half of the rung: hm3plus_c107743 with init_pgi_{lassosum2,ldpred2_auto}_d30.pt.
for i in $(seq 1 120); do
    [ -e $DS/ard30/$C107/store.json ] && [ -e $DS/ard30/$C107/samples_val.tsv ] && [ -e $C107_BLOCKS/init_pgi_lassosum2_d30.pt ] && [ -e $C107_BLOCKS/init_pgi_ldpred2_auto_d30.pt ] && { echo "tables present at $(stamp)"; exit 0; }
    sleep 120
done
echo "tables not found after 4 h"; exit 1
echo "exit $? at $(stamp)"
