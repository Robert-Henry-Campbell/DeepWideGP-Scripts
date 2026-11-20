#!/usr/bin/env python3
"""Own idea: an ensemble of trained networks (the mean of their logits per disease) scored on the tuning set, beside each
member alone. Members are run directories whose best checkpoint is scored (streamed, as score_snapshots.py does).
    python ensemble.py <code_dir> <out.json> <run_dir> [<run_dir> ...]
Aggregates only (per-disease and mean AUROC)."""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

code, out, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, code)
os.chdir(code)
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
from widedeepgp import metrics as M                                            # noqa: E402
from widedeepgp.train import build_data, load_model, predict_split            # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
logits, members, y, names = [], [], None, None
dm = None
for run in runs:
    config = json.load(open(os.path.join(run, "config.json")))
    if dm is None:
        cfg = SimpleNamespace(**config["args"])
        cfg.eval_batch_size, cfg.num_workers = 1024, 4
        dm = build_data(cfg)
        names = config["model"]["label_names"]
    best = json.load(open(os.path.join(run, "results.json")))["best_checkpoint"]
    best = os.path.join(run, "checkpoints", os.path.basename(best))
    model, _ = load_model(best, dm, device)
    z, y = predict_split(model, dm, "val", device)
    logits.append(z)
    m = M.per_label_metrics(z, y, None, names)
    members.append({"run": run, "auroc_mean": m["auroc_mean"], "auroc": dict(zip(names, map(float, m["auroc"])))})
    print(f"{os.path.basename(run)}: {m['auroc_mean']:.4f}", flush=True)
    del model
ens = M.per_label_metrics(np.mean(logits, axis=0), y, None, names)
res = {"members": members, "ensemble_auroc_mean": ens["auroc_mean"], "ensemble_auroc": dict(zip(names, map(float, ens["auroc"]))),
       "mean_of_members": float(np.mean([m["auroc_mean"] for m in members])), "n_val": int(len(y))}
json.dump(res, open(out, "w"), indent=1)
print(f"ensemble of {len(runs)}: {ens['auroc_mean']:.4f} against the members' mean {res['mean_of_members']:.4f}")
