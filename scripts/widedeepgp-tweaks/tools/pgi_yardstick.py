#!/usr/bin/env python3
"""The naive polygenic score the deep arm's way: for a frozen per-SNP table whose channels are one PGI method's weights
per disease (the other agent's init_pgi_<method>_d30.pt), each participant's PGI_j is the sum over blocks of the pooled
vectors (times sqrt of the block's SNP count, which the first layer divides by). A logistic regression of each disease on
[its PGI, the 12 covariates] is fitted on the training rows and scored on the tuning set: the yardstick a network on
that table must beat (Robert, 20 Nov 2026: "lift over naive ldpred2"). Also the PGI alone and the covariates alone.
Uses the training code's own first layer and its disk cache (so the cache is ready for the runs). Aggregates only.
    python pgi_yardstick.py <code_dir> <init.pt> <out_prefix> [--cache DIR]
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

a = argparse.ArgumentParser()
a.add_argument("code_dir"); a.add_argument("init"); a.add_argument("out")
a.add_argument("--cache", default=None, help="the disk cache folder (default: <dataset>/blocks/cache)")
a.add_argument("--rd_mode", default="hm3plus", help="the dataset under data/datasets/ard30 whose training rows fit the logistic regressions")
args = a.parse_args()
args.cache = args.cache or f"/mnt/sdg/robert/deepwidegp/data/datasets/ard30/{args.rd_mode}/blocks/cache"
sys.path.insert(0, args.code_dir)
os.chdir(args.code_dir)
from main_multilabel import parse_args                                   # noqa: E402
from sklearn.linear_model import LogisticRegression                      # noqa: E402
from sklearn.metrics import roc_auc_score                                # noqa: E402
from widedeepgp.train import build_data, build_model, cached_first_layer, train_labels   # noqa: E402

ROOT = "/mnt/sdg/robert/deepwidegp"
f = torch.load(args.init, map_location="cpu", weights_only=False)
de = int(f["manifest"]["embed_dim"])
cfg = parse_args(["--data_dir", f"{ROOT}/data/datasets", "--label", "ard30", "--rd_mode", args.rd_mode, "--input_mode", "blocks",
                  "--partition", f"{ROOT}/data/reference/partitions/prive431", "--mixer", "transformer", "--embed_dim", str(de),
                  "--d_model", "64", "--e_layers", "4", "--n_heads", "4", "--input_mask", "0", "--init_encoder", args.init,
                  "--freeze_encoder", "--snp_stats", "train", "--num_workers", "4", "--eval_batch_size", "1024", "--precision", "bf16",
                  "--tune_trials", "0", "--batch_size", "64"])
cfg.first_layer_cache = args.cache
t0 = time.time()
dm = build_data(cfg)
model = build_model(cfg, dm)
model.encoder.load_init(cfg.init_encoder, model.hp["partition_digest"], model.hp["snp_code"], freeze=True)
path = cached_first_layer(model, dm, "cuda" if torch.cuda.is_available() else "cpu", args.cache, cfg)
print(f"first layer cached at {path} ({time.time() - t0:.0f} s)", flush=True)
scale = (1.0 / model.encoder.block_scale.cpu()).reshape(1, -1, 1)       # sqrt of each block's SNP count
pgi = {s: (dm.splits[s]["pooled"].float() * scale).sum(1).numpy().astype(np.float64) for s in ("train", "val")}   # [N, channels]
labels = dm.label_names
if pgi["train"].shape[1] != len(labels):
    raise SystemExit(f"{de} channels for {len(labels)} diseases: the table's channels are not one PGI per disease")
ytr, yva = train_labels(dm), np.asarray(dm.splits["val"]["y"], dtype=np.float32)
ctr, cva = dm.splits["train"]["cov"], dm.splits["val"]["cov"]
rows = []
for j, name in enumerate(labels):
    ok = ~np.isnan(ytr[:, j]); okv = ~np.isnan(yva[:, j])
    sd = pgi["train"][ok, j].std() or 1.0
    xtr = np.column_stack([pgi["train"][ok, j] / sd, ctr[ok]]); xva = np.column_stack([pgi["val"][okv, j] / sd, cva[okv]])
    comb = LogisticRegression(max_iter=3000).fit(xtr, ytr[ok, j]).decision_function(xva)
    cov = LogisticRegression(max_iter=3000).fit(ctr[ok], ytr[ok, j]).decision_function(cva[okv])
    rows.append({"label": name, "n_val": int(okv.sum()), "n_cases_val": int(yva[okv, j].sum()),
                 "auroc_pgi_alone": roc_auc_score(yva[okv, j], pgi["val"][okv, j]),
                 "auroc_pgi_alone_train": roc_auc_score(ytr[ok, j], pgi["train"][ok, j]),
                 "auroc_combined": roc_auc_score(yva[okv, j], comb), "auroc_covariates": roc_auc_score(yva[okv, j], cov)})
    rows[-1]["gain"] = rows[-1]["auroc_combined"] - rows[-1]["auroc_covariates"]
with open(args.out + "_per_disease.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
summary = {"init": args.init, "dataset": args.rd_mode, "method": f["manifest"].get("source", {}).get("method"), "n_train": int(len(ytr)), "n_val": int(len(yva)),
           "mean_pgi_alone_auroc": float(np.mean([r["auroc_pgi_alone"] for r in rows])),
           "mean_pgi_alone_auroc_training_rows": float(np.mean([r["auroc_pgi_alone_train"] for r in rows])),
           "mean_combined_auroc": float(np.mean([r["auroc_combined"] for r in rows])),
           "mean_covariates_auroc": float(np.mean([r["auroc_covariates"] for r in rows])),
           "mean_gain": float(np.mean([r["gain"] for r in rows])), "cache_file": path, "seconds": round(time.time() - t0, 1)}
with open(args.out + "_summary.json", "w") as fh:
    json.dump(summary, fh, indent=1)
print(json.dumps(summary, indent=1))
