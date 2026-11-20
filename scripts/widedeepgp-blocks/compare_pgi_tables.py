#!/usr/bin/env python3
"""Lift of a network on a frozen PGI table over that PGI itself, on the tuning set, by the comparison protocol of
PGI/methods.md section 6 (design document item 22). Per disease, each score (the network's logit for the disease; the
PGI) is combined with the 12 covariates by the PGI arm's own logistic regression fitted on the tuning set, and the two
AUROCs are compared: paired bootstrap, 2,000 resamples stratified by case status per disease, and one unstratified
resample of participants shared by all diseases for the mean (methods 8.2). Writes aggregates only.

    PYTHONPATH=<PGI> python compare_pgi_tables.py --out <prefix> RUN_DIR [RUN_DIR ...]

Each run's best snapshot must have been scored with score_snapshots.py --save_scores; its init file names the method,
the rung whose weights it holds and the tuning scores of that PGI.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

B, SEED = 2000, 1
PGI_RESULTS = "/mnt/sdg/robert/deepwidegp/PGI/results"


def fast_auroc(s, y):
    """AUROC from ranks (ties broken by order: the scores are continuous)."""
    r = np.empty(len(s))
    r[np.argsort(s, kind="stable")] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def combined_logit(combine, score, cov, y):
    m = combine.Combined().fit(score, cov, y, "linear")
    if not m.result["converged"] or m.result["separation"]:
        return None
    return m.logit(score, cov)


def compare_run(run, combine, evaluate, covs):
    cfg = json.load(open(os.path.join(run, "config.json")))
    res = json.load(open(os.path.join(run, "results.json")))
    man = torch.load(cfg["args"]["init_encoder"], map_location="cpu", weights_only=False)["manifest"]["source"]
    method, wd = man["method"], man["weights_dataset"]
    scores = os.path.join(cfg["dataset_dir"], "scores", cfg["args"]["experiment"], cfg["args"]["exp_name"], "val_best.npz")
    net = np.load(scores, allow_pickle=False)
    pgi = np.load(glob.glob(os.path.join(wd, "scores", "pgi", "*", method, "val.npz"))[0], allow_pickle=False)
    val = pd.read_csv(os.path.join(cfg["dataset_dir"], "samples_val.tsv"), sep="\t", dtype={"IID": str})
    if net["iid"].astype(str).tolist() != val["IID"].tolist() or pgi["iid"].astype(str).tolist() != val["IID"].tolist():
        raise SystemExit(f"{run}: the network's or the PGI's tuning scores are not in the tuning table's order")
    labels = list(cfg["model"]["label_names"])
    if [str(x) for x in net["labels"]] != labels:
        raise SystemExit(f"{run}: the saved scores' labels differ from the model's")
    tm = pd.read_csv(os.path.join(PGI_RESULTS, os.path.basename(wd), "tune_metrics.csv"))
    tm = tm[tm["method"] == method].set_index("trait")
    shared = evaluate.bootstrap_indices(len(val), B, stratified=False, seed=SEED)
    rows, fixed = [], {}
    for j, t in enumerate(labels):
        y = val[t].to_numpy(np.float64)
        has = ~np.isnan(y)
        cov = val.loc[has, covs].reset_index(drop=True)
        yh = y[has]
        s_net, s_pgi = net["logit_population"][has, j].astype(np.float64), pgi["pgi"][has, j].astype(np.float64)
        l_net, l_pgi = combined_logit(combine, s_net, cov, yh), combined_logit(combine, s_pgi, cov, yh)
        a_net, a_pgi = evaluate.auroc(l_net, yh), evaluate.auroc(l_pgi, yh)
        idx = evaluate.bootstrap_indices(int(has.sum()), B, yh, stratified=True, seed=SEED)
        d = np.array([fast_auroc(l_net[i], yh[i]) - fast_auroc(l_pgi[i], yh[i]) for i in idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        fixed[t] = (np.flatnonzero(has), l_net, l_pgi, yh)
        rows.append({"trait": t, "n": int(has.sum()), "n_cases": int(yh.sum()),
                     "auroc_network": evaluate.auroc(s_net, yh), "auroc_network_combined": a_net,
                     "auroc_pgi": evaluate.auroc(s_pgi, yh), "auroc_pgi_combined": a_pgi,
                     "auroc_pgi_combined_recorded": float(tm.loc[t, "auroc_combined"]),
                     "auroc_covariates": float(tm.loc[t, "auroc_covariates"]), "lift": a_net - a_pgi,
                     "lift_lo": lo, "lift_hi": hi,
                     "verdict": "lift" if lo > 0 else ("worse" if hi < 0 else "no clear difference")})
    pos = {t: np.full(len(val), -1, dtype=np.int64) for t in labels}
    for t, (where, *_rest) in fixed.items():
        pos[t][where] = np.arange(len(where))
    means = []
    for b in range(B):
        per = []
        for t, (where, l_net, l_pgi, yh) in fixed.items():
            k = pos[t][shared[b]]
            k = k[k >= 0]
            per.append(fast_auroc(l_net[k], yh[k]) - fast_auroc(l_pgi[k], yh[k]))
        means.append(np.mean(per))
    lo, hi = np.percentile(means, [2.5, 97.5])
    table = pd.DataFrame(rows)
    mean = {"trait": "MEAN", "n": len(val), "n_cases": int(table["n_cases"].sum()),
            **{c: float(table[c].mean()) for c in table.columns if c.startswith("auroc") or c == "lift"},
            "lift_lo": lo, "lift_hi": hi, "verdict": "lift" if lo > 0 else ("worse" if hi < 0 else "no clear difference")}
    table = pd.concat([table, pd.DataFrame([mean])], ignore_index=True)
    table.insert(0, "run", os.path.basename(run))
    table.insert(1, "method", method)
    table.insert(2, "weights_from", os.path.basename(wd))
    table.insert(3, "in_sample", bool(man["training_overlap"]))
    n_train = int(cfg["n_train"])
    curve = [(round(h["samples_seen"] / n_train, 2), round(h["val_auroc_mean"], 4)) for h in res["history"]]
    info = {"run": os.path.basename(run), "method": method, "weights_from": os.path.basename(wd),
            "training_overlap": man["training_overlap"], "n_train": n_train, "best_epoch_equiv":
            round(res["best_samples_seen"] / n_train, 2), "curve": curve, "leak_check": man["leak_check"],
            "snps_kept": man["snps_kept"], "blocks_empty": man["blocks_empty"], "wall_seconds": res["wall_seconds"],
            "check_max_abs_pgi_recorded": float((table["auroc_pgi_combined"] - table["auroc_pgi_combined_recorded"]).abs()[:-1].max())}
    return table, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output prefix: <out>.csv (per disease) and <out>.json (runs)")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    from pgi import combine, config, evaluate
    covs = [str(c) for c in config.methods()["gwas"]["covariates"]]
    tables, infos = [], []
    for run in a.runs:
        t, i = compare_run(os.path.normpath(run), combine, evaluate, covs)
        tables.append(t)
        infos.append(i)
        m = t.iloc[-1]
        print(f"{i['run']}: mean lift {m['lift']:+.4f} [{m['lift_lo']:+.4f}, {m['lift_hi']:+.4f}] over {i['method']} + covariates "
              f"({m['auroc_network_combined']:.4f} vs {m['auroc_pgi_combined']:.4f}); "
              + ", ".join(f"{v} {int((t['verdict'][:-1] == v).sum())}" for v in ("lift", "no clear difference", "worse")),
              flush=True)
    pd.concat(tables, ignore_index=True).to_csv(a.out + ".csv", index=False, float_format="%.6g")
    with open(a.out + ".json", "w") as f:
        json.dump(infos, f, indent=1)


if __name__ == "__main__":
    main()
