#!/usr/bin/env python3
"""The closed-form block ridge of one rung (methods.md section 11).

Per fold model f (fitted without fold f; a sixth model on every training row) and per Prive block, the ridge
of the labels residualised on [1, covariates] (least squares on the model's training rows) on standardised
dosages (the rung's cached training-set SNP statistics; a missing genotype is the mean) at the penalties
lambda = M / h2 (the LDpred-inf form; M = the store's SNP count), all diseases at once: per-block grams Z'Z
and Z'[Y, 1, C] are accumulated per fold in float64 on the GPU in one pass over the store, then one
eigendecomposition per block and model gives the weights at every penalty. Every split is then scored,
training rows by the model that excluded them and the others by their pseudo-fold model (folds.npz from
make_folds.py); the penalty per disease is chosen on the tuning set (highest AUROC of the cross-fitted PRS).
Products, owner-only, under <dataset_dir>/blocks/: ridge_weights_fold<f>.npy and ridge_weights_full.npy
(float32 [M, 90], column 3 j + p for disease j and penalty p), ridge_manifest.json, prs_<split>.npz,
channels_<split>.npy (int8 [n, blocks, 90]; not with --no_channels), channels_manifest.json, and blocks.json beside store.json.
Resumable: a chromosome whose weights sit in blocks/partial/ and a split whose files exist are not redone.
The test split needs --unlock_test (logged in blocks/test_access.log).

    CUDA_VISIBLE_DEVICES=2 python build_block_ridge.py <dataset_dir> --partition <partition dir or npz>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_common import (K_FOLDS, Rung, auroc, chmod600, inverse_sd, load_partition, prefetch,  # noqa: E402
                          save_json, save_npy, save_npz, sha256_file, sha256_iids, standardise, unlock)

CLIP_SD, INT8_SCALE = 3.5, 36                 # channels: clipped at +-3.5 SD, stored as round(x * 36)
MODELS = [f"fold{f}" for f in range(K_FOLDS)] + ["full"]


def residual_coefficients(y, cov, folds):
    """Least-squares coefficients of the labels on [1, covariates] over each model's training rows: float64
    [K_FOLDS + 1, 1 + n_cov, n_labels]; model f excludes fold f, the last model excludes nothing."""
    A = np.column_stack([np.ones(len(y)), cov])
    return np.stack([np.linalg.lstsq(A[folds != f], y[folds != f], rcond=None)[0] for f in range(K_FOLDS + 1)])


def accumulate(store, chrom, group, rows, folds, t, mean, inv, a):
    """Per block of ``group`` and per fold, Z'Z (float64 [K_FOLDS, n_b, n_b]) and Z'[Y, 1, C] ([K_FOLDS, n_b, 43])
    over the training rows, read in batches on threads; Z = standardised dosages of the group's variants."""
    v0, v1, dev = group[0][1], group[-1][2], mean.device
    G = [torch.zeros(K_FOLDS, e - s, e - s, dtype=torch.float64, device=dev) for _, s, e in group]
    T = [torch.zeros(K_FOLDS, e - s, t.shape[1], dtype=torch.float64, device=dev) for _, s, e in group]
    batches = np.array_split(np.arange(len(rows)), max(1, len(rows) // a.batch_rows))
    for idx, q in prefetch(lambda b: (b, store.read_q(chrom, rows[b], v0, v1)), batches, a.threads):
        z = standardise(q, mean[v0:v1], inv[v0:v1])
        for f in range(K_FOLDS):
            pos = np.flatnonzero(folds[idx] == f)
            zf, tf = z[torch.as_tensor(pos, device=dev)], t[torch.as_tensor(idx[pos], device=dev)]
            for g, (_, s, e) in enumerate(group):
                zb = zf[:, s - v0:e - v0]
                G[g][f] += zb.T @ zb
                T[g][f] += zb.T @ tf
    return G, T


def solve_block(G, T, B, lams):
    """Weights of one block for the K_FOLDS fold models and the full model: float32 [K_FOLDS + 1, n_b, P n_labels],
    column P j + p for disease j and penalty p. Model f uses the grams of the other folds; eigh once, every
    penalty from it. A SNP without variation has a zero row and column, so its weight is 0."""
    n_lab, Ts, out = B.shape[2], T.sum(0), []
    for f in range(K_FOLDS + 1):
        g, t = G.sum(0), (Ts - T[f] if f < K_FOLDS else Ts)
        if f < K_FOLDS:
            g.sub_(G[f])                                     # in place: one n x n temporary per model
        zr = t[:, :n_lab] - t[:, n_lab:] @ B[f]            # Z'R with R = Y - [1 C] B, the residualised labels
        lam, Q = torch.linalg.eigh(g)
        del g
        p = Q.T @ zr
        out.append(torch.stack([Q @ (p / (lam + l)[:, None]) for l in lams], dim=2).reshape(len(Q), -1).float())
    return torch.stack(out)


def solve_chromosome(rung, part, chrom, rows, folds, t, B, mean, inv, lams, a):
    """Weights [K_FOLDS + 1, L_c, P n_labels] of one chromosome, in groups of blocks whose per-fold grams fit
    a.gram_gb, one pass over the training rows per group."""
    blocks = part["blocks"][chrom]
    W = torch.zeros(K_FOLDS + 1, rung.store.n_variants(chrom), len(lams) * B.shape[2], dtype=torch.float32)
    i = 0
    while i < len(blocks):
        j, size = i, 0.0
        while j < len(blocks) and (j == i or size + 8 * K_FOLDS * (blocks[j][2] - blocks[j][1]) ** 2 <= a.gram_gb * 1e9):
            size += 8 * K_FOLDS * (blocks[j][2] - blocks[j][1]) ** 2
            j += 1
        group = blocks[i:j]
        G, T = accumulate(rung.store, chrom, group, rows, folds, t, mean, inv, a)
        for (_, s, e), g, tt in zip(group, G, T):
            W[:, s:e] = solve_block(g, tt, B, lams).cpu()
        del G, T
        if mean.is_cuda:
            torch.cuda.empty_cache()
        i = j
    return W.numpy()


def score_split(rung, part, split, tab, folds, W, mean, inv, a, keep=True):
    """Scores a split: every row's block channels with its fold model's weights, written raw (float32) to
    blocks/partial/channels_<split>_raw.npy unless not ``keep``; returns that memmap (or None) and the PRS at every
    penalty, float32 [n, 90]."""
    store, rows, dev = tab["store"], tab["rows"], mean[0].device
    chroms = list(store.chroms)
    shape = (len(rows), part["n_blocks"], W[0][0].shape[1])
    raw = None
    if keep:
        raw = np.lib.format.open_memmap(os.path.join(rung.blocks, "partial", f"channels_{split}_raw.npy"), mode="w+",
                                        dtype=np.float32, shape=shape)
        chmod600(raw.filename)
    total = np.zeros((len(rows), shape[2]), dtype=np.float32)
    batches = np.array_split(np.arange(len(rows)), max(1, len(rows) // a.batch_rows))
    for idx, qs in prefetch(lambda b: (b, [store.read_q(c, rows[b]) for c in chroms]), batches, a.threads):
        ch = torch.zeros(len(idx), shape[1], shape[2], device=dev)
        sel = [torch.as_tensor(np.flatnonzero(folds[idx] == f), device=dev) for f in range(K_FOLDS)]
        for ci, (c, q) in enumerate(zip(chroms, qs)):
            z = standardise(q, mean[ci], inv[ci])
            for f in range(K_FOLDS):
                if len(sel[f]):
                    zf = z[sel[f]]
                    for b, s, e in part["blocks"][c]:
                        ch[sel[f], b] = zf[:, s:e] @ W[f][ci][s:e]
        if keep:
            raw[idx] = ch.cpu().numpy()
        total[idx] = ch.sum(1).cpu().numpy()
    if keep:
        raw.flush()
    return raw, total


def channel_scale(raw, chunk=2048):
    """Mean and SD per channel over the rows of ``raw`` (float64 sums); an SD of 0 becomes 1, so a constant channel
    is stored as 0."""
    s0, s1 = 0.0, 0.0
    for i in range(0, len(raw), chunk):
        x = raw[i:i + chunk].astype(np.float64)
        s0, s1 = s0 + x.sum(0), s1 + (x * x).sum(0)
    mean = s0 / len(raw)
    sd = np.sqrt(np.maximum(s1 / len(raw) - mean * mean, 0.0))
    return mean.astype(np.float32), np.where(sd > 0, sd, 1.0).astype(np.float32)


def quantise(raw, scale, path, chunk=2048):
    """int8 channels: standardised with the training rows' scale, clipped at +-CLIP_SD, round(x * INT8_SCALE)."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.int8, shape=raw.shape)
    for i in range(0, len(raw), chunk):
        x = (raw[i:i + chunk] - scale[0]) / scale[1]
        out[i:i + chunk] = np.rint(np.clip(x, -CLIP_SD, CLIP_SD) * INT8_SCALE).astype(np.int8)
    out.flush()
    del out
    os.remove(raw.filename)
    return chmod600(path)


def choose_penalties(total, y, n_pen):
    """The penalty with the highest tuning AUROC per disease (int [n_labels]) and the AUROC table [n_labels, n_pen]."""
    table = np.array([[auroc(total[:, n_pen * j + p], y[:, j]) for p in range(n_pen)] for j in range(y.shape[1])])
    return np.nan_to_num(table, nan=-1.0).argmax(1), table


def products(blocks_dir, split):
    return os.path.join(blocks_dir, f"prs_{split}.npz"), os.path.join(blocks_dir, f"channels_{split}.npy")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--partition", required=True, help="partition directory or partition.npz (the Prive blocks)")
    ap.add_argument("--penalties", type=float, nargs="+", default=[0.05, 0.15, 0.45], help="h2 values: lambda = M / h2")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "wb_related", "asian", "black"])
    ap.add_argument("--unlock_test", action="store_true", help="required for the test split; the access is logged")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_rows", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8, help="store reader threads")
    ap.add_argument("--gram_gb", type=float, default=20.0, help="device memory for one group's per-fold grams")
    ap.add_argument("--no_channels", action="store_true",
                    help="write the PRS only, not the int8 block channels (97 GB for the training split at 5,000 blocks; only the linear twin reads them)")
    a = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    rung = Rung(a.dataset_dir)
    unlock(a, a.splits, rung.blocks, "build_block_ridge")
    part = load_partition(a.partition, rung.store)
    store, chroms = rung.store, list(rung.store.chroms)
    M = sum(store.n_variants(c) for c in chroms)
    lams, n_pen = [M / h for h in a.penalties], len(a.penalties)
    tr = rung.table("train")
    folds = rung.folds("train", tr["iid"])
    stats_path, mean, std = rung.snp_stats(tr["rows"])
    partial = os.path.join(rung.blocks, "partial")
    os.makedirs(partial, mode=0o700, exist_ok=True)
    mpath = os.path.join(rung.blocks, "ridge_manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"files": {}, "seconds": {}}
    manifest.update(penalties_h2=a.penalties, **{"lambda": lams}, M=M, n_folds=K_FOLDS, labels=rung.labels,
                    n_train=len(tr["iid"]), train_iids_sha256=sha256_iids(tr["iid"]),
                    fold_counts=np.bincount(folds, minlength=K_FOLDS).tolist(),
                    standardisation={"file": os.path.basename(stats_path), "sha256": sha256_file(stats_path)},
                    partition={k: part[k] for k in ("path", "sha256", "n_blocks")},
                    folds_sha256=sha256_file(os.path.join(rung.blocks, "folds.npz")), device=a.device,
                    channels_written=not a.no_channels)

    # 1. weights per model, chromosome by chromosome (resumable through blocks/partial/)
    weights = {m: os.path.join(rung.blocks, f"ridge_weights_{m}.npy") for m in MODELS}
    if not all(os.path.exists(p) for p in weights.values()):
        t = torch.as_tensor(np.column_stack([tr["y"], np.ones(len(tr["y"])), tr["cov"]]), dtype=torch.float64, device=a.device)
        B = torch.as_tensor(residual_coefficients(tr["y"], tr["cov"], folds), device=a.device)
        for i, c in enumerate(chroms):
            paths = [os.path.join(partial, f"weights_chr{c}_{m}.npy") for m in MODELS]
            if all(os.path.exists(p) for p in paths):
                continue
            t1 = time.time()
            m64 = torch.as_tensor(mean[i], dtype=torch.float64, device=a.device)
            i64 = torch.as_tensor(inverse_sd(std[i]), dtype=torch.float64, device=a.device)
            w_c = solve_chromosome(rung, part, c, tr["rows"], folds, t, B, m64, i64, lams, a)
            for p, w in zip(paths, w_c):
                save_npy(p, w)
            manifest["seconds"][f"chr{c}"] = round(time.time() - t1, 1)
            print(f"chr{c}: {len(part['blocks'][c])} blocks, {store.n_variants(c)} SNPs, {time.time() - t1:.0f} s", flush=True)
        for m in MODELS:
            parts = [os.path.join(partial, f"weights_chr{c}_{m}.npy") for c in chroms]
            save_npy(weights[m], np.concatenate([np.load(p) for p in parts]))
            for p in parts:
                os.remove(p)
        del t, B
    for m in MODELS:
        if os.path.basename(weights[m]) not in manifest["files"]:
            manifest["files"][os.path.basename(weights[m])] = sha256_file(weights[m])
    save_json(mpath, manifest)

    # 2. scores: the training split first (the channel scale), the tuning split next (the penalty choice)
    offsets = np.cumsum([0] + [store.n_variants(c) for c in chroms])
    W = [[torch.as_tensor(np.array(w[offsets[i]:offsets[i + 1]]), device=a.device) for i in range(len(chroms))]
         for w in (np.load(weights[f"fold{f}"], mmap_mode="r") for f in range(K_FOLDS))]
    m32 = [torch.as_tensor(m, dtype=torch.float32, device=a.device) for m in mean]
    i32 = [torch.as_tensor(inverse_sd(s), dtype=torch.float32, device=a.device) for s in std]
    cpath = os.path.join(rung.blocks, "channels_manifest.json")
    cman = json.load(open(cpath)) if os.path.exists(cpath) else {"clip_sd": CLIP_SD, "scale": INT8_SCALE, "standardised_on": "train", "files": {}}
    scale = (np.array(cman["mean"], np.float32), np.array(cman["sd"], np.float32)) if "mean" in cman else None
    choice = manifest.get("chosen_penalty_index")
    order = [s for s in ("train", "val") if s in a.splits] + [s for s in a.splits if s not in ("train", "val")]
    pending = []
    for split in order:
        if all(os.path.exists(p) for p in products(rung.blocks, split)[:1 if a.no_channels else 2]):
            continue
        tab = rung.table(split)
        fold = rung.folds(split, tab["iid"])
        t1 = time.time()
        raw, total = score_split(rung, part, split, tab, fold, W, m32, i32, a, keep=not a.no_channels)
        manifest["seconds"][f"score_{split}"] = round(time.time() - t1, 1)
        print(f"{split}: {len(fold)} rows scored in {time.time() - t1:.0f} s", flush=True)
        if split == "train" and not a.no_channels:
            scale = channel_scale(raw)
            cman["mean"], cman["sd"] = scale[0].tolist(), scale[1].tolist()
        if split == "val":
            choice, table = choose_penalties(total, tab["y"], n_pen)
            manifest["chosen_penalty_index"] = choice.tolist()
            manifest["chosen_h2"] = {lab: a.penalties[p] for lab, p in zip(rung.labels, choice)}
            manifest["tuning_auroc_by_penalty"] = {lab: [None if np.isnan(v) else round(float(v), 4) for v in row]
                                                  for lab, row in zip(rung.labels, table)}
        pending.append((split, tab, fold, raw, total))
        if (scale is None and not a.no_channels) or choice is None:
            continue
        for sp, tb, fd, rw, tt in pending:
            prs_path, ch_path = products(rung.blocks, sp)
            if not a.no_channels:
                shape = list(rw.shape)
                quantise(rw, scale, ch_path)
                cman["files"][os.path.basename(ch_path)] = {"shape": shape, "sha256": sha256_file(ch_path)}
            save_npz(prs_path, iid=tb["iid"], labels=np.array(rung.labels), fold=fd.astype(np.int8), y=tb["y"],
                     score=tt[:, n_pen * np.arange(len(rung.labels)) + np.asarray(choice)], score_all=tt)
            manifest["files"][os.path.basename(prs_path)] = sha256_file(prs_path)
        pending.clear()
        save_json(mpath, manifest)
        save_json(cpath, cman)
    if pending:
        raise SystemExit("the train and val splits must be scored before any other split is written")
    if not os.listdir(partial):
        os.rmdir(partial)
    # the aggregate summary the E0 read-out (e0_summary.py) and the results tables read: tuning AUROC per disease
    y_val = np.load(products(rung.blocks, "val")[0])["y"]
    table = manifest["tuning_auroc_by_penalty"]
    rows = [dict(label=lab, auroc_val=table[lab][p], chosen_h2=a.penalties[p], cases_val=int(y_val[:, j].sum()),
                 **{f"auroc_val_h2_{h}": v for h, v in zip(a.penalties, table[lab])})
            for j, (lab, p) in enumerate(zip(rung.labels, manifest["chosen_penalty_index"]))]
    spath = os.path.join(rung.blocks, "ridge_summary.csv")
    pd.DataFrame(rows).to_csv(spath, index=False, float_format="%.4f")
    files = {"folds.npz": manifest["folds_sha256"], "ridge_manifest.json": sha256_file(mpath), **manifest["files"],
             **{k: v["sha256"] for k, v in cman["files"].items()}, "channels_manifest.json": sha256_file(cpath),
             "ridge_summary.csv": sha256_file(chmod600(spath))}
    save_json(os.path.join(rung.dir, "blocks.json"),
              {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "partition": manifest["partition"], "penalties_h2": a.penalties,
               "chosen_h2": manifest.get("chosen_h2"), "files": files})


if __name__ == "__main__":
    main()
