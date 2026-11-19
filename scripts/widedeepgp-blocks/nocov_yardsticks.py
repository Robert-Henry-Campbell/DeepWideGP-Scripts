#!/usr/bin/env python3
"""Covariate-free yardsticks for the no-covariate LD-block run (methods.md item 20), per disease on the tuning set:
the cross-fitted ridge PRS alone; that PRS projected onto the 16-dimensional basis the warm-start table keeps
(make_warm_start.py: the top right singular vectors of ridge_weights_full); and the best linear read of the table, a
logistic regression of the label on the 16 block-summed channels fitted on the training rows' out-of-fold scores. The
projection commutes with the sum over blocks, so both come from prs_<split>.npz score_all. Also the share of each
disease's PRS variance in its largest block, beside the network's AUROCs with and without covariates. Aggregates only.

    python nocov_yardsticks.py <dataset_dir> <nocov_run> <withcov_run> [<withcov_run> ...] --out <csv>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("nocov_run")
    ap.add_argument("withcov_runs", nargs="+")
    ap.add_argument("--embed_dim", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    b = os.path.join(a.dataset_dir, "blocks")
    man = json.load(open(os.path.join(b, "ridge_manifest.json")))
    labels, pick, n_pen = man["labels"], man["chosen_penalty_index"], len(man["penalties_h2"])
    w = np.load(os.path.join(b, "ridge_weights_full.npy"), mmap_mode="r")
    gram = sum(np.asarray(w[i:i + 200000], np.float64).T @ np.asarray(w[i:i + 200000], np.float64)
               for i in range(0, w.shape[0], 200000))
    ev, vec = np.linalg.eigh(gram)
    ev, vec = ev[::-1], vec[:, ::-1]
    basis = vec[:, :a.embed_dim]
    print(f"variance kept by {a.embed_dim}: {ev[:a.embed_dim].sum() / ev.sum():.4f}")
    va, tr = np.load(os.path.join(b, "prs_val.npz")), np.load(os.path.join(b, "prs_train.npz"))
    sv, st = va["score_all"].astype(np.float64), tr["score_all"].astype(np.float64)
    sd = np.asarray(json.load(open(os.path.join(b, "channels_manifest.json")))["sd"], np.float64)
    nocov = pd.read_csv(os.path.join(a.nocov_run, "scores_val_summary.csv")).query("snapshot == 'best'").set_index("label")
    withcov = pd.DataFrame({r: {k: v["auroc"] for k, v in json.load(open(os.path.join(r, "results.json")))["val"].items()}
                            for r in a.withcov_runs}).mean(axis=1)
    lin = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", os.path.basename(a.dataset_dir.rstrip("/")),
                                   "blocks_linear.csv")).set_index("label")
    rows = []
    for j, lab in enumerate(labels):
        c = n_pen * j + pick[j]
        yv, yt = va["y"][:, j], tr["y"][:, j]
        okv, okt = ~np.isnan(yv), ~np.isnan(yt)
        fit = LogisticRegression(max_iter=1000).fit(st[okt] @ basis, yt[okt])
        share = sd[:, c] ** 2 / (sd[:, c] ** 2).sum()
        rows.append({"label": lab, "cases_val": int(yv[okv].sum()),
                     "prs_alone": roc_auc_score(yv[okv], sv[okv, c]),
                     "prs_projected": roc_auc_score(yv[okv], (sv @ basis @ basis[c])[okv]),
                     "linear_table": roc_auc_score(yv[okv], fit.decision_function(sv[okv] @ basis)),
                     "top_block_share": share.max(), "network_nocov": nocov.loc[lab, "auroc"],
                     "covariates_only": nocov.loc[lab, "auroc_covariates_only"], "network_withcov": withcov[lab],
                     "prs_plus_covariates": lin.loc[lab, "auroc_val_ridge_prs"]})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(a.out, index=False)
    print(t.drop(columns="label").mean().round(4).to_string())


if __name__ == "__main__":
    main()
