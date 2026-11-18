#!/usr/bin/env python3
"""Gathers the deep arm's tuning-set summaries and the PGI arm's tuning metrics into the tidy table
of PGI/pgi/curves.py and runs every estimand of plan section 6 (design document section 8).

    python fit_curves.py gather --runs <root> [--runs <root> ...] --out <dir>
        Walks each root for runs holding config.json and scores_<split>_summary.csv (every split
        found), adds the PGI arm's results/<rung>/tune_metrics.csv, attaches each disease's N_eff per
        rung (PGI results/ladder/rung_power.csv) and S-LDSC h2 (results/hm3plus/sldsc_h2.csv),
        computes delta_r2 and writes <out>/tidy.csv and <out>/scores_manifest.csv (where each cell's
        per-person tuning scores are expected and whether they exist).
    python fit_curves.py fit --table <dir>/tidy.csv --out <dir>
                             [--manifest <dir>/scores_manifest.csv --dataset <rung dir> --bootstrap 2000]
        Point estimates from the table alone; with the manifest and a dataset directory (its
        samples_val.tsv and store.json), the shared-protocol values and bootstrap intervals from the
        per-person scores. Writes one CSV per result table, <out>/<name>.csv (aggregates only).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

PGI = os.environ.get("PGI_REPO", "/mnt/sdg/robert/deepwidegp/PGI")
sys.path.insert(0, PGI)
from pgi import curves  # noqa: E402
from pgi import evaluate as E  # noqa: E402

RUNGS = {"hm3plus_n13199": 13199, "hm3plus_n26287": 26287, "hm3plus_n53498": 53498, "hm3plus_n107743": 107743,
         "hm3plus": 215872}
DATASETS = "/mnt/sdg/robert/deepwidegp/data/datasets/ard30"
SLDSC = os.path.join(PGI, "results", "hm3plus", "sldsc_h2.csv")
POWER = os.path.join(PGI, "results", "ladder", "rung_power.csv")


def disease_table(sldsc=SLDSC, power=POWER, M_e=60_000):
    """Per disease and training size: N_eff of that rung's training cases and controls, the S-LDSC
    h2 on the 50% scale (with its SE) and on the liability scale, and the training prevalence K."""
    s = pd.read_csv(sldsc).rename(columns={"label": "disease", "h2_obs50": "h2_50", "h2_obs50_se": "h2_50_se",
                                           "K_train": "K"})
    p = pd.read_csv(power).rename(columns={"label": "disease", "neff_train": "n_eff"})
    p = p[p.M_e == M_e][["disease", "n_train", "n_eff"]]
    return p.merge(s[["disease", "h2_50", "h2_50_se", "h2_liab", "K"]], on="disease")


def family_of(config):
    """A deep cell's family name, its rung (the dataset directory's name) and its size columns from
    config.json: mixer, shape, table width, partition, and each ablation off its default as the
    model records it: the SNP code, the first layer's --init_encoder file (its stem, with the rung
    it came from when not the run's own) and :frozen for --freeze_encoder."""
    m, a, b = config["model"], config["args"], config.get("blocks")
    rung = os.path.basename(os.path.normpath(config["dataset_dir"]))
    fam = f"deep:{m['mixer']}:d{m['d_model']}x{m['e_layers']}"
    cols = dict(mixer=m["mixer"], d_model=m["d_model"], e_layers=m["e_layers"], n_params=config["n_parameters"])
    if b is None:
        return fam + ":snps", rung, cols
    fam += f":de{b['embed_dim']}:{b['partition']}"
    if b.get("snp_code", "dosage") != "dosage":
        fam += f":snp_code={b['snp_code']}"
    init = m.get("init_encoder") or a.get("init_encoder")
    if init:
        src = os.path.basename(os.path.dirname(os.path.dirname(init)))          # <rung>/blocks/<file>.pt
        fam += f":init={os.path.splitext(os.path.basename(init))[0]}" + ("" if src == rung else f"@{src}")
    if m.get("freeze_encoder") or a.get("freeze_encoder"):
        fam += ":frozen"
    cols.update(embed_dim=b["embed_dim"], partition=b["partition"], n_blocks=b["n_blocks"],
                n_mixer_params=b["n_parameters_by_module"]["mixer"], n_non_embedding_params=b["n_non_embedding_parameters"])
    return fam, rung, cols


def deep_rows(roots):
    """One row per run, split and disease from every scores_<split>_summary.csv under the roots.
    Two runs that would land in one cell (family, rung, seed, split) are refused by name."""
    rows = []
    for root in roots:
        for dirpath, _, files in sorted(os.walk(root)):
            summaries = sorted(f for f in files if f.startswith("scores_") and f.endswith("_summary.csv"))
            if not summaries or "config.json" not in files:
                continue
            with open(os.path.join(dirpath, "config.json")) as f:
                config = json.load(f)
            fam, rung, cols = family_of(config)
            scores = os.path.join(config["dataset_dir"], "scores", config["args"]["experiment"], config["args"]["exp_name"])
            for name in summaries:
                split = name[len("scores_"):-len("_summary.csv")]
                for r in pd.read_csv(os.path.join(dirpath, name)).itertuples():
                    rows.append(dict(disease=r.label, rung=rung, n_train=RUNGS.get(rung, config.get("n_train")), family=fam,
                                     seed=config["args"]["seed"], split=split, snapshot=r.snapshot, auroc=r.auroc,
                                     auroc_cov=r.auroc_covariates_only, delta_auroc=r.auroc_gain, n=r.n, n_cases=r.n_cases,
                                     run_dir=dirpath, score_path=os.path.join(scores, f"{split}_{r.snapshot}.npz"), **cols))
    t = pd.DataFrame(rows)
    cell = ["family", "rung", "seed", "split"]
    runs = t.drop_duplicates(cell + ["run_dir"]) if len(t) else t
    clash = runs.duplicated(cell, keep=False)
    if clash.any():
        raise ValueError("runs that would share a cell:\n" + runs[clash].sort_values(cell)[cell + ["run_dir"]].to_string(index=False))
    return t


def pgi_rows(results_dir):
    """The PGI arm's tuning metrics (cells ok or not estimable, the latter at gain 0). A table holds the shared
    protocol's one covariate basis (PGI methods 6.1: the 12 covariates, linear, since 18 November 2026); a table
    with two bases is refused rather than filtered."""
    rows = []
    for rung, n_train in RUNGS.items():
        path = os.path.join(results_dir, rung, "tune_metrics.csv")
        if not os.path.exists(path):
            continue
        t = pd.read_csv(path, dtype={"fingerprint": str, "pgi_key": str})
        if "basis" in t and t.basis.nunique() > 1:
            raise ValueError(f"{path} holds rows of several covariate bases {sorted(t.basis.unique())}: one protocol per table")
        t = t[t.status.isin(["ok", "not_estimable"]) & np.isfinite(pd.to_numeric(t.delta_auroc, errors="coerce"))]
        for r in t.itertuples():
            rows.append(dict(disease=r.trait, rung=rung, n_train=n_train, family=r.method, seed=1, split="val", status=r.status,
                             auroc=r.auroc_combined, auroc_cov=r.auroc_covariates, delta_auroc=r.delta_auroc, n=r.n,
                             n_cases=r.n_cases, fingerprint=r.fingerprint))
    return pd.DataFrame(rows)


def manifest(deep, pgi, datasets=DATASETS):
    """Where each cell's per-person scores are expected, for the tuning split and the test split
    (present only after the unlock): the deep arm's <split>_<snapshot>.npz, the PGI arm's
    scores/pgi/<fingerprint>/<method>/<split>.npz and the block-linear comparators'
    scores/blocks_linear/<method>/<split>_best.npz (family twin:<method>). Paths only."""
    rows = []
    for split in ("val", "test"):
        if len(deep):
            d = deep[deep.split == split].drop_duplicates(["family", "rung", "seed"])
            rows += [dict(family=r.family, rung=r.rung, seed=r.seed, split=split, path=r.score_path) for r in d.itertuples()]
        if len(pgi):
            for r in pgi.drop_duplicates(["family", "rung", "fingerprint"]).itertuples():
                hits = sorted(glob.glob(os.path.join(datasets, r.rung, "scores", "pgi", f"{r.fingerprint}*", r.family, f"{split}.npz")))
                rows.append(dict(family=r.family, rung=r.rung, seed=1, split=split, path=hits[0] if hits else
                                 os.path.join(datasets, r.rung, "scores", "pgi", f"{r.fingerprint}...", r.family, f"{split}.npz")))
        for rung in RUNGS:
            for path in sorted(glob.glob(os.path.join(datasets, rung, "scores", "blocks_linear", "*", f"{split}_best.npz"))):
                rows.append(dict(family="twin:" + os.path.basename(os.path.dirname(path)), rung=rung, seed=1, split=split, path=path))
    m = pd.DataFrame(rows, columns=["family", "rung", "seed", "split", "path"])
    m["exists"] = [os.path.exists(p) for p in m.path]
    return m


def gather(roots, results_dir, datasets=DATASETS, sldsc=SLDSC, power=POWER):
    """The tidy table and the score manifest. delta_r2 = R2(AUROC combined) - R2(AUROC covariates)
    on the liability scale (evaluate.r2_from_auc, Lee 2012) at the training prevalence K."""
    deep, pgi = deep_rows(roots), pgi_rows(results_dir)
    t = pd.concat([deep, pgi], ignore_index=True)
    t = t.merge(disease_table(sldsc, power), on=["disease", "n_train"], how="left")
    t["delta_r2"] = E.r2_from_auc(t.auroc.to_numpy(float), t.K.to_numpy(float)) - E.r2_from_auc(t.auroc_cov.to_numpy(float), t.K.to_numpy(float))
    return t, manifest(deep, pgi, datasets)


def person_from(table, manifest_path, dataset, split="val"):
    """The tuning set (samples_<split>.tsv of ``dataset``, labels and covariates from its store.json)
    with every per-person score file the manifest lists as existing, as a curves.Person."""
    man = pd.read_csv(manifest_path)
    man = man[(man.split == split) & man.exists]
    with open(os.path.join(dataset, "store.json")) as f:
        store = json.load(f)
    labels, covariates = list(store["labels"]), list(store["covariates"])
    tune = pd.read_csv(os.path.join(dataset, f"samples_{split}.tsv"), sep="\t", dtype={"IID": str})
    K = table.drop_duplicates("disease").set_index("disease").K.reindex(labels).to_numpy(float)
    scores = {(fam, rung): [curves.read_scores(p, tune.IID, labels) for p in g.sort_values("seed").path]
              for (fam, rung), g in man.groupby(["family", "rung"])}
    return curves.Person(y=tune[labels].to_numpy(float), cov=tune[covariates], labels=labels, K=K, scores=scores)


def fit(table_path, out, manifest_path=None, dataset=None, B=2000, split="val", unlock_test=False):
    """curves.analyse on one split's rows of the table, with the per-person scores when a manifest
    and a dataset directory are given (hours at the real size: launch detached, CLAUDE.md section
    1). The test split needs --unlock_test and appends the access to <out>/test_access.log."""
    if split == "test":
        if not unlock_test:
            raise SystemExit("the test split is locked: pass --unlock_test only at the joint unlock (PGI methods 6.4)")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "test_access.log"), "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tfit --split test\ttable={table_path}\tmanifest={manifest_path}\tdataset={dataset}\tB={B}\n")
    table = pd.read_csv(table_path)
    if "split" in table:
        table = table[table.split == split]
    person = person_from(table, manifest_path, dataset, split) if manifest_path and dataset else None
    results = curves.analyse(table, person, B=B)
    os.makedirs(out, exist_ok=True)
    for name, df in results.items():
        df.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gather")
    g.add_argument("--runs", action="append", default=[], help="a root of deep runs (repeatable)")
    g.add_argument("--pgi_results", default=os.path.join(PGI, "results"))
    g.add_argument("--datasets", default=DATASETS)
    g.add_argument("--sldsc", default=SLDSC)
    g.add_argument("--power", default=POWER)
    g.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--table", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--manifest", default=None)
    f.add_argument("--dataset", default=None, help="a rung's dataset directory (samples_val.tsv, store.json)")
    f.add_argument("--bootstrap", type=int, default=2000)
    f.add_argument("--split", default="val")
    f.add_argument("--unlock_test", action="store_true", help="required for --split test; the joint unlock only")
    a = ap.parse_args(argv)
    if a.cmd == "gather":
        t, m = gather(a.runs, a.pgi_results, a.datasets, a.sldsc, a.power)
        os.makedirs(a.out, exist_ok=True)
        t.to_csv(os.path.join(a.out, "tidy.csv"), index=False)
        m.to_csv(os.path.join(a.out, "scores_manifest.csv"), index=False)
        print(f"{len(t)} rows ({t.family.nunique()} families, {t.rung.nunique()} rungs, {t.disease.nunique()} diseases); "
              f"{int(m.exists.sum())} of {len(m)} score files present -> {a.out}")
    else:
        r = fit(a.table, a.out, a.manifest, a.dataset, a.bootstrap, a.split, a.unlock_test)
        if not (a.manifest and a.dataset):
            print("TABLE ONLY, a quick look: the deep gains are against the deep arm's own covariates-only model and the PGI "
                  "gains against the tuning-set refit; no intervals, so no decision below is reportable and the null is undetermined.")
        print(r["pooled"].to_string(index=False))
        print("null:", r["null"].iloc[0].to_dict())


if __name__ == "__main__":
    main()
