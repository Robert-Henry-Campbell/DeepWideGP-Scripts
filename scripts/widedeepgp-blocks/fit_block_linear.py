#!/usr/bin/env python3
"""The linear twin of the deep model, and the plain ridge PRS, as comparators (methods.md section 11).

Per disease, a logistic ridge on the rung's Prive-level block channels (431 x 90, cross-fitted; int8 / 36 =
standardised units) with the 12 standardised covariates unpenalised, fitted on the training rows by L-BFGS,
its penalty chosen on the tuning set over a log grid; and an unpenalised logistic regression on the
cross-fitted ridge PRS at its chosen penalty plus the covariates. Per-person tuning-set scores in the deep
arm's npz format under <dataset_dir>/scores/blocks_linear/{logistic_ridge,ridge_prs}/val_best.npz
(owner-only); aggregate AUROCs in results/<rung>/blocks_linear.csv beside this script. Resumable per disease
(blocks/blocks_linear_partial.npz, removed at the end).

    OMP_NUM_THREADS=8 python fit_block_linear.py <dataset_dir> [--lam_grid 1e2 1e3 1e4 1e5 1e6]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_common import Rung, auroc, save_npz  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def logistic_ridge(x, y, lam, n_free, max_iter=500, w0=None):
    """Minimises sum_i log(1 + exp(-s_i eta_i)) + lam / 2 ||beta_pen||^2 over eta = x beta + b, s = 2 y - 1, with
    the intercept and the first ``n_free`` columns unpenalised (L-BFGS, warm-started from w0). x is float32 and
    is never copied; returns (w = [beta, b] float64, iterations)."""
    s = 2.0 * y.astype(np.float64) - 1.0
    pen = lam * np.r_[np.zeros(n_free), np.ones(x.shape[1] - n_free), 0.0]
    if w0 is None and 0 < n_free < x.shape[1]:        # start from the free columns' own fit: the penalised block is stiff
        free, _ = logistic_ridge(np.ascontiguousarray(x[:, :n_free]), y, 0.0, n_free, max_iter)
        w0 = np.r_[free[:-1], np.zeros(x.shape[1] - n_free), free[-1]]

    def f(w):
        z = s * (x @ w[:-1].astype(np.float32) + w[-1])
        g_eta = -s * expit(-z)                               # d log(1 + exp(-z)) / d eta
        grad = np.r_[x.T @ g_eta.astype(np.float32), g_eta.sum()] + pen * w
        return np.logaddexp(0.0, -z).sum() + 0.5 * (pen * w * w).sum(), grad

    res = minimize(f, np.zeros(x.shape[1] + 1) if w0 is None else w0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "maxfun": 2 * max_iter})
    return res.x, int(res.nit)


def predict(x, w):
    return (x @ w[:-1].astype(np.float32) + w[-1]).astype(np.float64)


def features(rung, split, cov, scale, chunk=4096):
    """[standardised covariates, channels / INT8_SCALE] float32 [n, n_cov + blocks * channels]."""
    ch = np.load(os.path.join(rung.blocks, f"channels_{split}.npy"), mmap_mode="r")
    x = np.empty((len(ch), cov.shape[1] + ch.shape[1] * ch.shape[2]), dtype=np.float32)
    x[:, :cov.shape[1]] = cov
    for i in range(0, len(ch), chunk):
        x[i:i + chunk, cov.shape[1]:] = ch[i:i + chunk].reshape(min(chunk, len(ch) - i), -1).astype(np.float32) / scale
    return x


def write_scores(rung, method, iid, logit, y):
    d = os.path.join(rung.dir, "scores", "blocks_linear", method)
    os.makedirs(d, mode=0o700, exist_ok=True)
    save_npz(os.path.join(d, "val_best.npz"), iid=iid, labels=np.array(rung.labels),
             logit_population=logit.astype(np.float32), y=y.astype(np.float32))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--lam_grid", type=float, nargs="+", default=[1e2, 1e3, 1e4, 1e5, 1e6])
    ap.add_argument("--max_iter", type=int, default=500)
    ap.add_argument("--prs_only", action="store_true", help="the ridge PRS with the covariates only, without the logistic ridge on the channels")
    a = ap.parse_args(argv)
    t0 = time.time()
    rung = Rung(a.dataset_dir)
    tr, va = rung.table("train"), rung.table("val")
    prs = {s: np.load(os.path.join(rung.blocks, f"prs_{s}.npz")) for s in ("train", "val")}
    if (prs["train"]["iid"] != tr["iid"]).any() or (prs["val"]["iid"] != va["iid"]).any():
        raise SystemExit("prs_<split>.npz do not match the tables: rebuild the block ridge")
    cm, cs = tr["cov"].mean(0), tr["cov"].std(0)
    cs[cs == 0] = 1.0
    cov = {"train": (tr["cov"] - cm) / cs, "val": (va["cov"] - cm) / cs}
    if not a.prs_only:
        with open(os.path.join(rung.blocks, "channels_manifest.json")) as f:
            scale = json.load(f)["scale"]
        x = {s: features(rung, s, cov[s], scale) for s in ("train", "val")}
    n_free = len(rung.covariates)
    logit_lr, logit_prs, rows = np.zeros_like(va["y"]), np.zeros_like(va["y"]), []
    part = os.path.join(rung.blocks, "blocks_linear_partial.npz")
    if os.path.exists(part):
        z = np.load(part)
        if np.array_equal(z["iid"], va["iid"]):
            logit_lr, logit_prs, rows = z["logit_lr"], z["logit_prs"], json.loads(str(z["rows"]))
            print(f"resuming after {len(rows)} diseases", flush=True)
    done = {r["label"] for r in rows}
    for j, lab in enumerate(rung.labels):
        if lab in done:
            continue
        w, best = None, None
        for lam in [] if a.prs_only else sorted(a.lam_grid, reverse=True):   # the path from the strongest penalty, warm-started
            w, nit = logistic_ridge(x["train"], tr["y"][:, j], lam, n_free, a.max_iter, w)
            v = predict(x["val"], w)
            au = auroc(v, va["y"][:, j])
            if best is None or au > best["auroc_val_logistic_ridge"]:
                best = dict(lam=lam, auroc_val_logistic_ridge=au, auroc_train_logistic_ridge=auroc(predict(x["train"], w), tr["y"][:, j]),
                            n_iter=nit, logit=v)
        if best is None:
            best = {}
        else:
            logit_lr[:, j] = best.pop("logit")
        pm, ps = prs["train"]["score"][:, j].mean(), prs["train"]["score"][:, j].std() or 1.0    # the raw PRS is tiny: standardise
        xp = {s: np.column_stack([(prs[s]["score"][:, j] - pm) / ps, cov[s]]).astype(np.float32) for s in ("train", "val")}
        w, _ = logistic_ridge(xp["train"], tr["y"][:, j], 0.0, xp["train"].shape[1], a.max_iter)
        logit_prs[:, j] = predict(xp["val"], w)
        rows.append(dict(label=lab, cases_train=int(tr["y"][:, j].sum()), cases_val=int(va["y"][:, j].sum()), **best,
                         lam_at_grid_edge=best.get("lam") in (min(a.lam_grid), max(a.lam_grid)),
                         auroc_val_ridge_prs=auroc(logit_prs[:, j], va["y"][:, j])))
        twin = "" if a.prs_only else f"logistic ridge {best['auroc_val_logistic_ridge']:.4f} (lambda {best['lam']:g}, {best['n_iter']} iterations), "
        print(f"{lab}: {twin}ridge PRS {rows[-1]['auroc_val_ridge_prs']:.4f}; {time.time() - t0:.0f} s", flush=True)
        save_npz(part, iid=va["iid"], logit_lr=logit_lr, logit_prs=logit_prs, rows=np.array(json.dumps(rows)))
    if not a.prs_only:
        write_scores(rung, "logistic_ridge", va["iid"], logit_lr, va["y"])
    write_scores(rung, "ridge_prs", va["iid"], logit_prs, va["y"])
    table = pd.DataFrame(rows)
    res = os.path.join(HERE, "results", os.path.basename(rung.dir))
    os.makedirs(res, exist_ok=True)
    table.to_csv(os.path.join(res, "blocks_linear.csv"), index=False, float_format="%.4f")
    if os.path.exists(part):
        os.remove(part)
    twin = "" if a.prs_only else (f"logistic ridge {table.auroc_val_logistic_ridge.mean():.4f}, "
                                  f"{int(table.lam_at_grid_edge.sum())} of {len(table)} at a grid edge; ")
    print(f"mean tuning AUROC: {twin}ridge PRS {table.auroc_val_ridge_prs.mean():.4f}; {time.time() - t0:.0f} s")
    return table


if __name__ == "__main__":
    main()
