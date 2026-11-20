# slots: 1
# after: 08[45678]*
source /mnt/sdg/robert/deepwidegp/runs/tweaks/queue/common.sh
# Confirmation, seed 3 of the best combination run (the base's seeds 2 and 3 exist: 0.6737, 0.6738), so the night's gain is read
# against the seed spread rather than one run.
best=$($PY - <<'PYEOF'
import json, os
root = "/mnt/sdg/robert/deepwidegp/runs/tweaks"
runs = {r: json.load(open(f"{root}/{r}/results.json"))["best_val_auroc_mean"] for r in ("combo_paper20", "combo_opt256_20", "combo_paper20_labelnorm", "combo_all20", "table_d32_20", "table_d32_opt256_20", "table_d32_pdh20") if os.path.exists(f"{root}/{r}/results.json")}
print(max(runs, key=runs.get))
PYEOF
) || exit 1
echo "best combination: $best"
A=$($PY - "$RUNS/$best/config.json" <<'PYEOF'
import json, sys
a = json.load(open(sys.argv[1]))["args"]
keys = ("batch_size", "d_model", "e_layers", "n_heads", "n_fcn", "mixer", "lr", "l2", "dropout", "lr_schedule", "warmup_steps", "max_epoch_equiv", "label_norm", "per_disease_head", "init_encoder", "embed_dim")
print(" ".join(f"--{k} {a[k]}" for k in keys))
PYEOF
)
train $CODE3 ${best}_s3 $CACHE_FLAGS $A --seed 3 && score $CODE3 ${best}_s3
rc=$?; echo "exit $rc at $(stamp)"; exit $rc
