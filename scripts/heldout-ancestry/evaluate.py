#!/usr/bin/env python3
"""Every model on the held-out groups by one rule (methods.md section 4): per model run, disease and group, the
comparison protocol of PGI/methods.md 6.1 - a logistic regression on the 20,000 tuning participants of the model's
score plus the 12 covariates entered linearly (the PGI arm's own pgi.combine.Combined), applied unchanged to the
group - beside the covariate-only reference fitted the same way. AUROC with DeLong intervals (pgi.evaluate), and
the paired DeLong difference from the reference. Reads per-person score files only; writes aggregates.

    python evaluate.py [--cohorts asian black] [--rungs ...] [--models ...] [--out results/heldout_auroc.csv]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("DEEPWIDEGP_PROJECT", "/mnt/sdg/robert/deepwidegp")
DATASETS = os.path.join(ROOT, "data", "datasets", "ard30")
HELD_OUT = "hm3plus_asian_black"            # the data module's sibling dataset of the held-out groups
PGI_CODE = os.environ.get("PGI_CODE", os.path.join(ROOT, "data", "pgi", "code_b73a629"))   # frozen: the arm's protocol
if PGI_CODE not in sys.path:
    sys.path.insert(0, PGI_CODE)
from pgi.combine import Combined                     # noqa: E402
from pgi.evaluate import delong_ci, delong_paired    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNGS = ["hm3plus_n13199", "hm3plus_n26287", "hm3plus_n53498", "hm3plus_n107743", "hm3plus"]
# model: (its tuning-set files under a rung's folder, the name of a group's file beside one)
MODELS = {
    "widedeepgp": ("scores/widedeepgp_ladder/*/val_best.npz", "{cohort}_best.npz"),
    "block_logistic_ridge": ("scores/blocks_linear/logistic_ridge/val_best.npz", "{cohort}_best.npz"),
    "block_ridge_prs": ("blocks/prs_val.npz", "prs_{cohort}.npz"),
    "kelemen_mlp": ("scores/kelemen_mlp/*/*/val.npz", "{cohort}.npz"),
    "delphi": ("scores/delphi/*/*/val.npz", "{cohort}.npz"),
    "gln": ("scores/gln/*/*/val.npz", "{cohort}.npz"),
    "prsformer": ("scores/prsformer/*/*/val.npz", "{cohort}.npz"),
    "scprs": ("scores/scprs/*/*/val.npz", "{cohort}.npz"),
    "classic_deepgp": ("scores/classic_deepgp/*/*/val.npz", "{cohort}.npz"),
    "pgi": ("scores/pgi/*/*/val.npz", "{cohort}.npz"),
}
SCORE_KEYS = ("pgi", "logits", "logit_population", "score")   # the first present: a PGI's own index, else the model's


def small(n):
    """A count of 1 to 9 participants is written <10 (CLAUDE.md section 3)."""
    return "<10" if 0 < n < 10 else str(int(n))


def table(rung, split):
    """samples_<split>.tsv as the data module reads it: the rung's own, else the held-out groups' dataset's."""
    path = os.path.join(DATASETS, rung, f"samples_{split}.tsv")
    if not os.path.exists(path):
        path = os.path.join(DATASETS, HELD_OUT, f"samples_{split}.tsv")
    return pd.read_csv(path, sep="\t", dtype={"IID": str})


def load(path, t):
    """A per-person file with its table rows in the file's order (by its iid; a file without one is in the table's
    order). Refuses a file whose labels are not the table's: it would be misaligned."""
    z = np.load(path)
    key = next(k for k in SCORE_KEYS if k in z.files)
    labels, y = [str(s) for s in z["labels"]], np.asarray(z["y"], np.float64)
    if "iid" in z.files:
        rows = t.set_index("IID").loc[z["iid"].astype(str)].reset_index()
    elif len(t) == len(y):
        rows = t
    else:
        raise ValueError(f"{path}: {len(y)} people without IDs, the table has {len(t)}")
    if not np.array_equal(np.nan_to_num(rows[labels].to_numpy(np.float64), nan=-1), np.nan_to_num(y, nan=-1)):
        raise ValueError(f"{path}: its labels are not the table's in its order")
    return labels, np.asarray(z[key], np.float64).reshape(len(y), -1), y, rows


def variant(model, folder, rung, labels):
    """The model's version from its run folder: the disease or rung part dropped (e.g. C44_large_linear -> large_linear)."""
    if model == "block_ridge_prs":
        return "ridge_prs"
    v = re.sub(rf"^({re.escape(labels[0])}|{re.escape(rung)})_?", "", folder) if len(labels) == 1 or folder.startswith(rung) else folder
    return v or model


def evaluate(rungs, models, cohorts):
    rows, ref = [], {}
    val = table(RUNGS[0] if not rungs else rungs[0], "val")
    for rung in rungs:
        tv = table(rung, "val")
        if not tv["IID"].equals(val["IID"]):
            raise ValueError(f"{rung}: another tuning set than {rungs[0]}'s")
        groups = {c: table(rung, c) for c in cohorts}
        for model in models:
            pattern, name = MODELS[model]
            for vpath in sorted(glob.glob(os.path.join(DATASETS, rung, pattern))):
                labels, s_v, y_v, cov_v = load(vpath, tv)
                folder = os.path.basename(os.path.dirname(vpath))
                for c in cohorts:
                    cpath = os.path.join(os.path.dirname(vpath), name.format(cohort=c))
                    got = load(cpath, groups[c]) if os.path.exists(cpath) else None
                    for j, lab in enumerate(labels):
                        row = dict(model=model, variant=variant(model, folder, rung, labels), rung=rung, disease=lab,
                                   cohort=c, file=os.path.relpath(cpath, DATASETS))
                        rows.append(row)
                        if got is None:
                            row["status"] = "not scored"
                            continue
                        ok = np.isfinite(s_v[:, j]) & np.isfinite(y_v[:, j])
                        if not ok.all() or s_v[:, j].std() == 0:
                            row["status"] = "no score on the tuning set"
                            continue
                        clabels, s_c, y_c, cov_c = got
                        jc = clabels.index(lab)
                        if lab not in ref:             # one tuning set for every rung: one reference per disease
                            ref[lab] = Combined(spline_df=4).fit(None, tv, tv[lab].to_numpy(np.float64), "linear")
                        base = ref[lab].logit(None, cov_c)          # in this file's order
                        m = Combined(spline_df=4).fit(s_v[:, j:j + 1], cov_v, y_v[:, j], "linear")
                        logit = m.logit(s_c[:, jc:jc + 1], cov_c)
                        y = y_c[:, jc]
                        a, a_lo, a_hi = delong_ci(logit, y)
                        r_, r_lo, r_hi = delong_ci(base, y)
                        s_, s_lo, s_hi = delong_ci(s_c[:, jc], y)
                        gain, se, _, p = delong_paired(logit, base, y)
                        row.update(status="ok", n=len(y), n_cases=small(int(np.nansum(y))), auroc=a, auroc_lo=a_lo,
                                   auroc_hi=a_hi, auroc_covariates_only=r_, auroc_covariates_only_lo=r_lo,
                                   auroc_covariates_only_hi=r_hi, gain=gain, gain_se=se, gain_p=p, auroc_score=s_,
                                   auroc_score_lo=s_lo, auroc_score_hi=s_hi, auroc_tuning_score=delong_ci(s_v[:, j], y_v[:, j])[0],
                                   converged=bool(m.result.get("converged", True)),
                                   separation=bool(m.result.get("separation", False)))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohorts", nargs="+", default=["asian", "black"])
    ap.add_argument("--rungs", nargs="+", default=RUNGS)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "heldout_auroc.csv"))
    a = ap.parse_args(argv)
    df = evaluate(a.rungs, a.models, a.cohorts)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False, float_format="%.6f")
    done = df[df.status == "ok"] if "status" in df else df
    print(f"{len(df)} rows, {len(done)} evaluated; written {a.out}")
    for (m, c), g in done.groupby(["model", "cohort"]) if len(done) else []:
        print(f"  {m} {c}: {len(g)} (run, disease) rows")
    return df


if __name__ == "__main__":
    main()
