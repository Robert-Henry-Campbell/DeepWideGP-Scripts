#!/usr/bin/env python3
"""The SNP, haplotype and combined ridges of one chromosome, cross-fitted (methods.md sections 5 and 6).

Per Prive block, with build_block_ridge's residualisation and solver: A = standardised dosages, B = standardised
haplotype counts of the haploblocks (haploblocks.py's codes), C = both; penalties lambda = M / h2 with M the
genome-wide feature count of the class (haplotype features scaled by sqrt(M_snp / M_hap) so that one scalar serves
the grid). Training rows are scored by the fold model that excluded them, tuning rows by their pseudo-fold model.
The penalty per disease and parametrisation is chosen on the out-of-fold training scores. Metrics on the tuning set
and on the out-of-fold training rows: the correlation with the residualised label and the AUROC gain of covariates
plus score over covariates (logistic combination fitted on the out-of-fold training rows); B - A and C - A with
paired participant bootstrap intervals, per disease and pooled. A's fold weights are checked against the stored
ridge_weights_fold<k>.npy. Aggregates to results/, participant scores to <dataset_dir>/haplo/ (owner-only).

    OMP_NUM_THREADS=10 python compare_ridge.py <dataset_dir> --chrom 22 --partition data/reference/partitions/prive431
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "widedeepgp-blocks"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_common import K_FOLDS, MISSING, Rung, auroc, inverse_sd, load_partition, prefetch, save_json, save_npz  # noqa: E402
from build_block_ridge import residual_coefficients, solve_block  # noqa: E402
from haploblocks import M_GENOME, RESULTS, cell_name  # noqa: E402

H2 = [0.0167, 0.05, 0.15, 0.45, 1.35]
PARAMS = ["snp", "hap", "both"]


def hap_counts(codes, H):
    """Copies of every common haplotype and of "other" per segment: float64 [n, sum(H + 1)] from codes [n, S, 2]."""
    off = np.r_[0, np.cumsum(H + 1)]
    X = np.zeros((len(codes), off[-1]))
    r = np.arange(len(codes))[:, None]
    for h in range(2):
        X[r, off[:-1][None, :] + codes[:, :, h]] += 1.0
    return X


def standardise_columns(X, n_train):
    """(X - mean) * 1/SD with the moments of the first n_train rows; a constant column becomes 0."""
    mu, sd = X[:n_train].mean(0), X[:n_train].std(0)
    return (X - mu) * inverse_sd(sd)


def snp_features(q, mean, inv):
    """uint8 q [n, m] -> standardised dosages float64 [n, m]; a missing genotype is the mean (0)."""
    z = (q.astype(np.float64) * 0.01 - mean) * inv
    z[q == MISSING] = 0.0
    return z


def block_grams(Z, folds, t):
    """Per-fold grams Z_f'Z_f ([K, p, p]) and Z_f'[Y, 1, C] ([K, p, 43]) in float64 (torch, CPU)."""
    Zt, tt = torch.from_numpy(np.ascontiguousarray(Z)), torch.from_numpy(np.ascontiguousarray(t))
    G = torch.zeros(K_FOLDS, Z.shape[1], Z.shape[1], dtype=torch.float64)
    T = torch.zeros(K_FOLDS, Z.shape[1], t.shape[1], dtype=torch.float64)
    for f in range(K_FOLDS):
        sel = torch.from_numpy(np.flatnonzero(folds == f))
        zf = Zt[sel]
        G[f], T[f] = zf.T @ zf, zf.T @ tt[sel]
    return G, T


def route_scores(Z, folds, W):
    """Every row scored by its fold's weights: [n, W.shape[2]] float64; W is [K (+1), p, k] (torch or numpy)."""
    W = np.asarray(W, dtype=np.float64)
    out = np.zeros((len(Z), W.shape[2]))
    for f in range(K_FOLDS):
        sel = np.flatnonzero(folds == f)
        if len(sel):
            out[sel] = Z[sel] @ W[f]
    return out


def choose(S, y, n_pen):
    """Penalty index per disease with the highest AUROC of the raw score on the given rows."""
    return np.array([int(np.nanargmax([auroc(S[:, n_pen * j + p], y[:, j]) for p in range(n_pen)])) for j in range(y.shape[1])])


def pick(S, choice, n_pen):
    return S[:, n_pen * np.arange(len(choice)) + choice]


def boot_weights(n, n_boot, seed=1):
    """Paired bootstrap counts: int32 [n_boot + 1, n], the first row all ones (the point estimate)."""
    rng = np.random.default_rng(seed)
    W = np.empty((n_boot + 1, n), dtype=np.int32)
    W[0] = 1
    for b in range(1, n_boot + 1):
        W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def boot_auroc(s, y, Wt, chunk=250):
    """AUROC of score s under every weight row of Wt ([R, n] counts): float64 [R]; ties are not averaged."""
    order = np.argsort(s, kind="stable")
    ys = (np.asarray(y)[order] > 0.5)
    out = np.empty(len(Wt))
    for i in range(0, len(Wt), chunk):
        w = Wt[i:i + chunk][:, order].astype(np.int64)
        w0 = w * (~ys)
        before = np.cumsum(w0, axis=1) - w0                      # controls strictly below each position
        n1, n0 = (w * ys).sum(1), w0.sum(1)
        out[i:i + chunk] = (w * ys * before).sum(1) / (n1 * n0)
    return out


def boot_corr(s, r, Wt):
    """Weighted Pearson correlation of s with r under every weight row: float64 [R]."""
    W = Wt.astype(np.float64)
    n = W.sum(1)
    ms, mr = W @ s / n, W @ r / n
    vs, vr = W @ (s * s) / n - ms * ms, W @ (r * r) / n - mr * mr
    return (W @ (s * r) / n - ms * mr) / np.sqrt(np.maximum(vs * vr, 1e-300))


def logistic(x_train, y_train, xs):
    """Logistic regression fitted on standardised x_train (no penalty to speak of); the fitted logit on every xs."""
    mu, sd = x_train.mean(0), x_train.std(0)
    f = lambda x: (x - mu) * inverse_sd(sd)
    lr = LogisticRegression(C=1e6, max_iter=5000, tol=1e-6).fit(f(x_train), y_train > 0.5)
    return [lr.decision_function(f(x)) for x in xs]


def ci(x):
    """Point estimate (row 0) and the 2.5 and 97.5 percentiles of the bootstrap rows."""
    return float(x[0]), float(np.percentile(x[1:], 2.5)), float(np.percentile(x[1:], 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--chrom", default="22")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--cell", default=cell_name(0.01, 0.95))
    ap.add_argument("--h2", type=float, nargs="+", default=H2)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=10)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    rung, t0 = Rung(a.dataset_dir), time.time()
    store, chrom, haplo = rung.store, a.chrom, os.path.join(rung.dir, "haplo")
    tr, va = rung.table("train"), rung.table("val")
    n_tr, n_va = len(tr["iid"]), len(va["iid"])
    folds = np.r_[rung.folds("train", tr["iid"]), rung.folds("val", va["iid"])]
    hb = np.load(os.path.join(haplo, f"chr{chrom}_haploblocks_{a.cell}.npz"), allow_pickle=True)
    if (hb["iid"] != np.r_[tr["iid"], va["iid"]]).any():
        raise SystemExit("the haploblock codes are not in the train-then-val order of the tables")
    blocks = load_partition(a.partition, store)["blocks"][chrom]
    m = store.n_variants(chrom)
    ci_ = list(store.chroms).index(chrom)
    stats_path, mean, std = rung.snp_stats(tr["rows"])
    mean, inv = mean[ci_], inverse_sd(std[ci_])
    rows = np.r_[tr["rows"], va["rows"]]
    q = np.empty((len(rows), m), dtype=np.uint8)
    for idx, qq in prefetch(lambda b: (b, store.read_q(chrom, rows[b])), np.array_split(np.arange(len(rows)), len(rows) // 4096), 8):
        q[idx] = qq
    y, cov = np.r_[tr["y"], va["y"]].astype(np.float64), np.r_[tr["cov"], va["cov"]]
    t = np.column_stack([y[:n_tr], np.ones(n_tr), cov[:n_tr]])
    B = residual_coefficients(y[:n_tr], cov[:n_tr], folds[:n_tr])
    Bt = torch.from_numpy(B)
    n_pen, n_lab = len(a.h2), y.shape[1]
    F = int((hb["H"] + 1).sum())
    s_hap = np.sqrt(m / F)                                   # lambda_hap = M_hap / h2 with M_hap = F * M_GENOME / m
    lams = [M_GENOME / h for h in a.h2]
    print(f"chr{chrom}: {m} SNPs, {len(hb['H'])} haploblocks, {F} haplotype features (scale {s_hap:.3f}); "
          f"{n_tr} training and {n_va} tuning rows loaded in {time.time() - t0:.0f} s", flush=True)

    # 1. the three ridges per Prive block, the scores summed over blocks
    S = {p: np.zeros((len(rows), n_pen * n_lab)) for p in PARAMS}
    checks = []
    offsets = int(sum(store.n_variants(c) for c in store.chroms[:ci_]))
    stored = [np.load(os.path.join(rung.blocks, f"ridge_weights_fold{f}.npy"), mmap_mode="r") for f in range(K_FOLDS)]
    man = json.load(open(os.path.join(rung.blocks, "ridge_manifest.json")))
    pipe_cols = [a.h2.index(h) for h in man["penalties_h2"]] if all(h in a.h2 for h in man["penalties_h2"]) else None
    for b, v0, v1 in blocks:
        t1 = time.time()
        segs = np.flatnonzero(hb["block"] == b)
        Zs = snp_features(q[:, v0:v1], mean[v0:v1], inv[v0:v1])
        Zh = standardise_columns(hap_counts(hb["codes"][:, segs], hb["H"][segs]), n_tr) * s_hap
        Z = np.concatenate([Zs, Zh], axis=1)
        del Zs, Zh
        G, T = block_grams(Z[:n_tr], folds[:n_tr], t)
        ms = v1 - v0
        parts = {"snp": (slice(0, ms), G[:, :ms, :ms], T[:, :ms]), "hap": (slice(ms, None), G[:, ms:, ms:], T[:, ms:]),
                 "both": (slice(0, None), G, T)}
        for p, (sl, Gp, Tp) in parts.items():
            W = solve_block(Gp.clone(), Tp.clone(), Bt, lams)
            S[p] += route_scores(Z[:, sl], folds, W.numpy())
            if p == "snp" and pipe_cols is not None:
                for f in range(K_FOLDS):
                    ref = np.asarray(stored[f][offsets + v0:offsets + v1], dtype=np.float64)
                    mine = W[f].numpy()[:, (n_pen * np.arange(n_lab))[:, None] + np.array(pipe_cols)[None, :]].reshape(ms, -1)
                    checks.append(dict(block=b, fold=f, rel_err=float(np.linalg.norm(mine - ref) / np.linalg.norm(ref))))
        print(f"block {b}: {ms} SNPs, {Z.shape[1] - ms} haplotype features, {time.time() - t1:.0f} s", flush=True)
        del Z, G, T
    os.makedirs(RESULTS, exist_ok=True)
    if checks:
        pd.DataFrame(checks).to_csv(os.path.join(RESULTS, f"check_snp_weights_chr{chrom}.csv"), index=False, float_format="%.3e")
        print(f"SNP fold weights against the stored ridge: max relative error {max(c['rel_err'] for c in checks):.2e}", flush=True)
    save_npz(os.path.join(haplo, f"scores_chr{chrom}_{a.cell}.npz"), iid=hb["iid"], split=hb["split"], fold=folds, h2=np.array(a.h2),
             labels=np.array(rung.labels), **{f"score_{p}": S[p].astype(np.float32) for p in PARAMS})

    # 2. evaluation: penalty chosen on the out-of-fold training rows; metrics there and on the tuning set
    R = y - np.column_stack([np.ones(len(y)), cov]) @ B[K_FOLDS]
    sets = {"oof_train": slice(0, n_tr), "val": slice(n_tr, None)}
    Wt = {k: boot_weights(len(range(*sl.indices(len(y)))), a.n_boot) for k, sl in sets.items()}
    choice = {p: choose(S[p][:n_tr], y[:n_tr], n_pen) for p in PARAMS}
    fixed = {p: np.full(n_lab, a.h2.index(0.15) if 0.15 in a.h2 else n_pen // 2) for p in PARAMS}
    rows_out, pooled = [], []
    for tag, ch in (("chosen", choice), ("h2_0.15", fixed)):
        boots = {}                                             # (metric, set, param, j) -> bootstrap rows
        for j, lab in enumerate(rung.labels):
            cov_logit = logistic(cov[:n_tr], y[:n_tr, j], [cov[:n_tr], cov[n_tr:]])
            logits = {p: logistic(np.column_stack([cov[:n_tr], pick(S[p][:n_tr], ch[p], n_pen)[:, j]]), y[:n_tr, j],
                                  [np.column_stack([cov[sl], pick(S[p][sl], ch[p], n_pen)[:, j]]) for sl in sets.values()]) for p in PARAMS}
            for k, (name, sl) in enumerate(sets.items()):
                boots[("auroc", name, "cov", j)] = boot_auroc(cov_logit[k], y[sl, j], Wt[name])
                for p in PARAMS:
                    boots[("auroc", name, p, j)] = boot_auroc(logits[p][k], y[sl, j], Wt[name])
                    boots[("r", name, p, j)] = boot_corr(pick(S[p][sl], ch[p], n_pen)[:, j], R[sl, j], Wt[name])
            row = dict(label=lab, penalties=tag, cases_train=int(y[:n_tr, j].sum()), cases_val=int(y[n_tr:, j].sum()),
                       **{f"h2_{p}": a.h2[ch[p][j]] for p in PARAMS})
            for name in sets:
                row[f"auroc_cov_{name}"] = boots[("auroc", name, "cov", j)][0]
                for p in PARAMS:
                    row[f"r_{p}_{name}"] = boots[("r", name, p, j)][0]
                    row[f"dauroc_{p}_{name}"] = boots[("auroc", name, p, j)][0] - boots[("auroc", name, "cov", j)][0]
                for p in ("hap", "both"):
                    for metric in ("r", "auroc"):
                        d = boots[(metric, name, p, j)] - boots[(metric, name, "snp", j)]
                        e, lo, hi = ci(d)
                        row[f"d{metric}_{p}_minus_snp_{name}"], row[f"d{metric}_{p}_minus_snp_{name}_lo"], row[f"d{metric}_{p}_minus_snp_{name}_hi"] = e, lo, hi
            rows_out.append(row)
            print(f"{tag} {lab}: val dAUROC snp {row['dauroc_snp_val']:+.4f} hap {row['dauroc_hap_val']:+.4f} both {row['dauroc_both_val']:+.4f}; "
                  f"oof r snp {row['r_snp_oof_train']:.4f} hap {row['r_hap_oof_train']:.4f} both {row['r_both_oof_train']:.4f}", flush=True)
        for name in sets:
            for p in ("hap", "both"):
                for metric in ("r", "auroc"):
                    d = np.mean([boots[(metric, name, p, j)] - boots[(metric, name, "snp", j)] for j in range(n_lab)], axis=0)
                    e, lo, hi = ci(d)
                    point = [boots[(metric, name, p, j)][0] - boots[(metric, name, "snp", j)][0] for j in range(n_lab)]
                    pooled.append(dict(penalties=tag, metric=metric, set=name, comparison=f"{p}_minus_snp", mean_diff=e, lo=lo, hi=hi,
                                       median_diff=float(np.median(point)), n_positive=int(sum(v > 0 for v in point)),
                                       n_negative=int(sum(v < 0 for v in point)), n_boot=a.n_boot))
    pd.DataFrame(rows_out).to_csv(os.path.join(RESULTS, f"compare_chr{chrom}_{a.cell}.csv"), index=False, float_format="%.5f")
    pd.DataFrame(pooled).to_csv(os.path.join(RESULTS, f"compare_chr{chrom}_{a.cell}_pooled.csv"), index=False, float_format="%.5f")
    save_json(os.path.join(haplo, f"compare_chr{chrom}_{a.cell}.json"),
              dict(chrom=chrom, cell=a.cell, snps=m, haploblocks=int(len(hb["H"])), hap_features=F, hap_scale=float(s_hap),
                   h2=a.h2, M_snp=M_GENOME, M_hap=F * M_GENOME / m, n_train=n_tr, n_val=n_va, n_boot=a.n_boot,
                   standardisation=os.path.basename(stats_path), partition=os.path.abspath(a.partition),
                   max_snp_weight_rel_err=max([c["rel_err"] for c in checks], default=None), seconds=round(time.time() - t0, 1)))
    print(pd.DataFrame(pooled).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
