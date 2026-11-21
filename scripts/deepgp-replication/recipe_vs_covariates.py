"""Paired bootstrap of the paper's PRS recipe against covariates alone, on the test people of each setting of
Fig. 2c (results.md): AUROC of the recipe (logistic regression on PRS + sex, age, PC1-6 fitted on the training
people, who are in the GWAS), of covariates alone (fitted the same way), and of the same PRS joined to the covariates
on the test people (5 folds); differences from covariates alone with 95% intervals (1,000 resamples, seed 1).
Reads the saved PRS and cohort tables in data/; writes aggregates only to figures/fig2c_differences.csv.
Usage: python recipe_vs_covariates.py"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA = "/mnt/sdg/robert/deepwidegp/data"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
COVS = ["sex", "age"] + [f"PC{i}" for i in range(1, 7)]
SETTINGS = (("Type 2 diabetes, population prevalence; GWAS of the same 215,872 people", "deepgp_replication_e11", "N", "deepgp_replication_e11"),
            ("Type 2 diabetes, 1:1 random controls; GWAS of 215,872 people", "deepgp_replication_e11", "R", "deepgp_replication_e11"),
            ("Type 2 diabetes, 1:1 random controls; GWAS of the cohort's 34,690 people", "deepgp_replication_e11", "R", "deepgp_replication_e11_cohort_gwas"),
            ("Atrial fibrillation, 1:1 random controls; GWAS of 215,872 people", "deepgp_replication", "R", "deepgp_replication"))


def lr(cols, tr, te):
    mu, sd = tr[cols].mean(), tr[cols].std()
    return LogisticRegression(max_iter=1000).fit((tr[cols] - mu) / sd, tr.y).decision_function((te[cols] - mu) / sd)


rows = []
for name, cohort_dir, v, prs_dir in SETTINGS:
    t = pd.read_csv(f"{DATA}/{cohort_dir}/cohort/split_{v}.tsv", sep="\t", dtype={"IID": str})
    tr, te = t[t.split == "train"].copy(), t[t.split == "test"].reset_index(drop=True)
    p = np.load(f"{DATA}/{prs_dir}/prs_{v}.npz")
    tr["prs"], te["prs"] = p["train"], p["test"]
    y = te.y.to_numpy()
    joined = np.zeros(len(te))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=1).split(te, y):
        joined[b] = lr(["prs"] + COVS, te.iloc[a], te.iloc[b])
    s = {"covariates alone": lr(COVS, tr, te), "recipe": lr(["prs"] + COVS, tr, te), "joined outside the GWAS": joined}
    rng = np.random.default_rng(1)
    boot = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    bs = {k: np.array([roc_auc_score(y[i], x[i]) for i in boot]) for k, x in s.items()}
    r = {"setting": name, "test_people": len(y)}
    for k in s:
        r[f"{k}_auroc"] = round(roc_auc_score(y, s[k]), 4)
    for k in ("recipe", "joined outside the GWAS"):
        d = bs[k] - bs["covariates alone"]
        r[f"{k}_minus_covariates"] = round(r[f"{k}_auroc"] - r["covariates alone_auroc"], 4)
        r[f"{k}_minus_covariates_ci_low"], r[f"{k}_minus_covariates_ci_high"] = (round(float(x), 4) for x in np.percentile(d, [2.5, 97.5]))
    rows.append(r)
out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/fig2c_differences.csv", index=False)
print(out.drop(columns=["test_people"]).to_string(index=False))
