"""Re-score a finished frozen-table run's best checkpoint on the tuning set from the cached first layer and compare
with the AUROCs it recorded when streamed (plan ~/.claude/plans/first-layer-cache.md). Aggregates only.
    python check_rescore.py <code dir> <run dir> <out.json>"""
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

code, run, out_path = sys.argv[1:4]
sys.path.insert(0, code)
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
from widedeepgp import metrics as M  # noqa: E402
from widedeepgp.train import build_data, cache_first_layer, load_model, predict_split  # noqa: E402

config = json.load(open(os.path.join(run, "config.json")))
res = json.load(open(os.path.join(run, "results.json")))
dm = build_data(SimpleNamespace(**config["args"]))
model, _ = load_model(os.path.join(run, "checkpoints", os.path.basename(res["best_checkpoint"])), dm, "cuda")
t0 = time.time()
cache_first_layer(model, dm, "cuda")
t1 = time.time()
logits, y = predict_split(model, dm, "val", "cuda")
t2 = time.time()
m = M.per_label_metrics(logits, y, model.loss_fn.logit_shift.cpu().numpy(), config["model"]["label_names"])
diffs = [float(m["auroc"][j]) - res["val"][name]["auroc"] for j, name in enumerate(m["labels"])]
out = {"run": run, "recorded_auroc_mean": res["best_val_auroc_mean"], "cached_auroc_mean": float(m["auroc_mean"]),
       "max_abs_diff_per_disease": max(abs(d) for d in diffs), "cache_build_seconds": round(t1 - t0, 1),
       "cached_tuning_pass_seconds": round(t2 - t1, 1),
       "cache_gib": round(sum(dm.splits[s]["pooled"].numel() * 4 for s in ("train", "val")) / 2 ** 30, 2)}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out, indent=1), flush=True)
