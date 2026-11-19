#!/usr/bin/env python3
"""Clumping and thresholding on SNP dosages, on haplotype columns and on both, cross-fitted (methods.md section 9).

Per fold model and Prive block: the marginal correlation of every standardised column with the label residualised
on the covariates (fold-fitted) and its p-value, from the block's grams; greedy clumping by p inside the block
within +-window kb at each r2 of the grid, r2 from the model's own training rows; at each p threshold the score is
the sum of marginal correlation x standardised column over the kept columns (PLINK --score with BETA weights, up to
scale). A haplotype column is the copy count of one common haplotype (or "other") of one haploblock, positioned at
the haploblock's midpoint. Training rows are scored by the fold that excluded them, tuning rows by their pseudo-fold;
the candidate (r2, p) per disease and parametrisation is chosen on the out-of-fold training AUROC; evaluation as
compare_ridge.evaluate. Aggregates to results/, the chosen scores to <dataset_dir>/haplo/ (owner-only).

    OMP_NUM_THREADS=10 python compare_ct.py <dataset_dir> --chrom 22 --partition data/reference/partitions/prive431
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "widedeepgp-blocks"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_common import K_FOLDS, Rung, inverse_sd, load_partition, prefetch, save_json, save_npz  # noqa: E402
from build_block_ridge import residual_coefficients  # noqa: E402
from compare_ridge import PARAMS, block_grams, evaluate, hap_counts, pick, route_scores, snp_features, standardise_columns  # noqa: E402
from haploblocks import RESULTS, cell_name  # noqa: E402

R2_GRID = [0.1, 0.2, 0.5, 0.8]
P_GRID = np.logspace(np.log10(5e-8), 0, 15)


def cand_label(i, r2_grid):
    r2, p = r2_grid[i // len(P_GRID)], P_GRID[i % len(P_GRID)]
    return f"r2_{r2:g}_p_{p:.1e}"


def marginal(Gd, Tz, B, n, sum_r, sum_r2):
    """Correlation of every column with the residualised label over one fold model's rows and its two-sided p-value.
    Gd: the model's diag(Z'Z) [p]; Tz: Z'[Y, 1, C] [p, n_lab + 1 + n_cov]; B: the model's coefficients of Y on [1, C];
    n: its rows; sum_r, sum_r2: sums of the residual and its square over them [n_lab]. Returns corr, pval [p, n_lab]."""
    n_lab = B.shape[1]
    zr = Tz[:, :n_lab] - Tz[:, n_lab:] @ B
    sz = Tz[:, n_lab]
    vz, vr = Gd - sz * sz / n, sum_r2 - sum_r * sum_r / n
    corr = (zr - np.outer(sz, sum_r) / n) / np.sqrt(np.maximum(np.outer(vz, vr), 1e-300))
    corr = np.clip(np.where(vz[:, None] > 1e-12 * n, corr, 0.0), -1, 1)
    tstat = corr * np.sqrt((n - 2) / np.maximum(1 - corr * corr, 1e-300))
    return corr, 2 * student_t.sf(np.abs(tstat), n - 2)


def correlation(Gm, sz, n):
    """Correlation matrix of the columns over one fold model's rows from their gram Gm [p, p] and column sums sz [p]."""
    C = Gm - np.outer(sz, sz) / n
    d = np.sqrt(np.maximum(np.diag(C), 1e-300))
    return C / np.outer(d, d)


def clump(pval, remove):
    """Greedy clumping: columns in order of p; a kept column removes every column that remove[column] marks (within
    the window and above the r2 threshold). Returns the kept columns in that order."""
    order, alive, kept = np.argsort(pval, kind="stable"), np.ones(len(pval), dtype=bool), []
    for i in order:
        if alive[i]:
            kept.append(i)
            alive &= ~remove[i]
    return np.array(kept, dtype=np.int64)


def ct_weights(corr, pval, C, pos, r2_grid, window):
    """Weights of every candidate for one fold model and block: float64 [p, n_cand * n_lab], column
    n_cand * disease + r2 index * len(P_GRID) + threshold index; a kept column passing the threshold carries its
    marginal correlation."""
    p, n_lab = corr.shape
    n_cand = len(r2_grid) * len(P_GRID)
    W = np.zeros((p, n_cand * n_lab))
    near = np.abs(pos[:, None] - pos[None, :]) <= window
    R2 = C * C
    for ir, r2 in enumerate(r2_grid):
        remove = near & (R2 > r2)
        for j in range(n_lab):
            kept = clump(pval[:, j], remove)
            for ip, thr in enumerate(P_GRID):
                k = kept[pval[kept, j] <= thr]
                W[k, n_cand * j + ir * len(P_GRID) + ip] = corr[k, j]
    return W


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--chrom", default="22")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--cell", default=cell_name(0.01, 0.95))
    ap.add_argument("--r2", type=float, nargs="+", default=R2_GRID)
    ap.add_argument("--window_kb", type=float, default=250.0)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--results", default=RESULTS, help="where the aggregate tables go")
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
    pos = store.positions(chrom).astype(np.float64)
    mid = (pos[hb["start"]] + pos[hb["end"] - 1]) / 2
    rows = np.r_[tr["rows"], va["rows"]]
    q = np.empty((len(rows), m), dtype=np.uint8)
    for idx, qq in prefetch(lambda b: (b, store.read_q(chrom, rows[b])), np.array_split(np.arange(len(rows)), max(1, len(rows) // 4096)), 8):
        q[idx] = qq
    y, cov = np.r_[tr["y"], va["y"]].astype(np.float64), np.r_[tr["cov"], va["cov"]]
    t = np.column_stack([y[:n_tr], np.ones(n_tr), cov[:n_tr]])
    B = residual_coefficients(y[:n_tr], cov[:n_tr], folds[:n_tr])
    A = np.column_stack([np.ones(n_tr), cov[:n_tr]])
    n_f, sum_r, sum_r2 = np.zeros(K_FOLDS), np.zeros((K_FOLDS, y.shape[1])), np.zeros((K_FOLDS, y.shape[1]))
    for f in range(K_FOLDS):                                 # the residual over each fold model's own training rows
        sel = folds[:n_tr] != f
        r = y[:n_tr][sel] - A[sel] @ B[f]
        n_f[f], sum_r[f], sum_r2[f] = sel.sum(), r.sum(0), (r * r).sum(0)
    n_cand, n_lab, window = len(a.r2) * len(P_GRID), y.shape[1], a.window_kb * 1e3
    F = int((hb["H"] + 1).sum())
    print(f"chr{chrom}: {m} SNPs, {len(hb['H'])} haploblocks, {F} haplotype features; {n_cand} candidates; "
          f"{n_tr} training and {n_va} tuning rows loaded in {time.time() - t0:.0f} s", flush=True)

    # 1. marginal effects, clumping and thresholding per fold model and Prive block; the scores summed over blocks
    S = {p: np.zeros((len(rows), n_cand * n_lab), dtype=np.float32) for p in PARAMS}
    kept_counts = []
    for b, v0, v1 in blocks:
        t1 = time.time()
        segs = np.flatnonzero(hb["block"] == b)
        Zs = snp_features(q[:, v0:v1], mean[v0:v1], inv[v0:v1])
        Zh = standardise_columns(hap_counts(hb["codes"][:, segs], hb["H"][segs]), n_tr)
        Z = np.concatenate([Zs, Zh], axis=1)
        posZ = np.r_[pos[v0:v1], np.repeat(mid[segs], hb["H"][segs] + 1)]
        del Zs, Zh
        G, T = block_grams(Z[:n_tr], folds[:n_tr], t)
        G, T = G.numpy(), T.numpy()
        Gt, Tt = G.sum(0), T.sum(0)
        ms = v1 - v0
        for p, sl in (("snp", slice(0, ms)), ("hap", slice(ms, None)), ("both", slice(0, None))):
            W = np.zeros((K_FOLDS, Z[:, sl].shape[1], n_cand * n_lab))
            for f in range(K_FOLDS):
                Gm, Tm = (Gt - G[f])[sl, sl], (Tt - T[f])[sl]
                corr, pval = marginal(np.diag(Gm), Tm, B[f], n_f[f], sum_r[f], sum_r2[f])
                W[f] = ct_weights(corr, pval, correlation(Gm, Tm[:, n_lab], n_f[f]), posZ[sl], a.r2, window)
            S[p] += route_scores(Z[:, sl], folds, W)
            kept_counts.append(dict(block=b, param=p, columns=int(Z[:, sl].shape[1]),
                                    **{f"kept_r2_{r2:g}_p_1": float((W[:, :, n_cand * np.arange(n_lab) + ir * len(P_GRID) + len(P_GRID) - 1] != 0).sum(1).mean())
                                       for ir, r2 in enumerate(a.r2)}))
            del W
        print(f"block {b}: {ms} SNPs, {Z.shape[1] - ms} haplotype features, {time.time() - t1:.0f} s", flush=True)
        del Z, G, T
    os.makedirs(a.results, exist_ok=True)
    pd.DataFrame(kept_counts).to_csv(os.path.join(a.results, f"ct_clumps_chr{chrom}_{a.cell}.csv"), index=False, float_format="%.1f")

    # 2. evaluation: candidate chosen on the out-of-fold training rows; metrics there and on the tuning set
    rows_out, pooled, by_cand = evaluate(S, y, cov, B, n_tr, rung.labels, n_cand, "cand", lambda i: cand_label(i, a.r2),
                                         a.r2.index(0.1) * len(P_GRID) + len(P_GRID) - 1 if 0.1 in a.r2 else None, a.n_boot,
                                         jobs=min(a.threads, n_lab))
    curve = [dict(param=p, label=lab, r2=a.r2[c // len(P_GRID)], p=P_GRID[c % len(P_GRID)], auroc_oof_train=by_cand[p][j, c])
             for p in PARAMS for j, lab in enumerate(rung.labels) for c in range(n_cand)]
    pd.DataFrame(curve).to_csv(os.path.join(a.results, f"ct_curve_chr{chrom}_{a.cell}.csv"), index=False, float_format="%.5g")
    pd.DataFrame(rows_out).to_csv(os.path.join(a.results, f"compare_ct_chr{chrom}_{a.cell}.csv"), index=False, float_format="%.5f")
    pd.DataFrame(pooled).to_csv(os.path.join(a.results, f"compare_ct_chr{chrom}_{a.cell}_pooled.csv"), index=False, float_format="%.5f")
    choice = {p: np.nanargmax(by_cand[p], axis=1) for p in PARAMS}
    save_npz(os.path.join(haplo, f"scores_ct_chr{chrom}_{a.cell}.npz"), iid=hb["iid"], split=hb["split"], fold=folds,
             labels=np.array(rung.labels), candidates=np.array([cand_label(i, a.r2) for i in range(n_cand)]),
             **{f"choice_{p}": choice[p] for p in PARAMS}, **{f"score_{p}": pick(S[p], choice[p], n_cand) for p in PARAMS})
    save_json(os.path.join(haplo, f"compare_ct_chr{chrom}_{a.cell}.json"),
              dict(chrom=chrom, cell=a.cell, snps=m, haploblocks=int(len(hb["H"])), hap_features=F, r2=a.r2,
                   p_grid=[float(x) for x in P_GRID], window_kb=a.window_kb, n_train=n_tr, n_val=n_va, n_boot=a.n_boot,
                   seconds=round(time.time() - t0)))
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
