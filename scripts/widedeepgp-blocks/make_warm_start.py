#!/usr/bin/env python3
"""Ridge warm start for the per-SNP table (DeepWideGP --init_encoder --folds): one table copy per fold, the fold's
ridge weights [M, 90] (every disease and penalty; standardised-dosage form) in the d_e-dimensional basis of the
full-training ridge's right singular vectors, split per chromosome as tables.<i> [K, L_i, d_e] float32, dosage code,
with the manifest of the contract. A training participant is pooled with the copy of the fold that excluded them and
every other participant with the copy of their pseudo-fold (the comparators' cross-fitting), so no participant's
tokens carry their own label: the full-training ridge does (in-sample AUROC 0.96 against 0.57 out of sample at the
top rung), and the first warm-started runs learnt that shortcut (18 November 2026). --in_sample writes that one copy
(U_k S_k of the full ridge, [L_i, d_e]) for the record. The table is scaled so that
the block tokens the model forms from it (standardised dosages times the rows, summed over a block and divided by
sqrt(SNPs)) have RMS --token_rms (1.0) over a sample of training participants: the raw ridge weights are of order 1e-5
and a token made of them is invisible next to the identity embeddings of order 1 (18 November 2026). --token_rms 0
keeps the raw scale.

    python make_warm_start.py <dataset_dir> --embed_dim 16 [--out <dataset_dir>/blocks/init_ridge_d16.pt] [--token_rms 1.0] [--in_sample]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_common import SEED, Rung, chmod600, inverse_sd, load_partition, standardise  # noqa: E402


def partition_digest(path, chroms):
    """The model's digest of the partition (genostore.datamodule.partition_digest over the per-chromosome compact
    block ids), or None while the model code does not publish it."""
    try:
        from genostore.datamodule import partition_digest as digest
    except ImportError:
        return None
    z = np.load(path)
    raw = [z[f"block_chr{c}"] for c in chroms]
    _, inv = np.unique(np.concatenate(raw), return_inverse=True)
    return digest(np.split(inv.astype(np.int64), np.cumsum([len(r) for r in raw])[:-1]))


def token_rms(store, part, tables, mean, std, rows, fold, chunk=512):
    """RMS over participants, blocks and dimensions of the block tokens the model's first layer forms from the tables
    (tables[k][i]: fold k's copy for chromosome i): standardised dosages times the rows of the participant's copy,
    summed over a block's SNPs and divided by sqrt(SNPs). ``fold``: each row's copy."""
    s2 = n = 0.0
    for i, c in enumerate(store.chroms):
        m, inv = torch.as_tensor(mean[i], dtype=torch.float32), torch.as_tensor(inverse_sd(std[i]), dtype=torch.float32)
        for k in np.unique(fold):
            t, rows_k = torch.from_numpy(np.ascontiguousarray(tables[k][i])), rows[fold == k]
            for r in np.array_split(rows_k, max(1, len(rows_k) // chunk)):
                x = standardise(store.read_q(c, r), m, inv)
                for _, s, e in part["blocks"][c]:
                    v = x[:, s:e] @ t[s:e] / (e - s) ** 0.5
                    s2 += float((v * v).sum())
                    n += v.numel()
    return float(np.sqrt(s2 / n))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--embed_dim", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--token_rms", type=float, default=1.0, help="RMS of the block tokens over training participants (0: the raw ridge scale)")
    ap.add_argument("--scale_rows", type=int, default=2048, help="training participants sampled (seed 1) to measure the token RMS")
    ap.add_argument("--in_sample", action="store_true", help="one copy from the full-training ridge (leaks the training rows' labels; the record)")
    a = ap.parse_args(argv)
    rung = Rung(a.dataset_dir)
    with open(os.path.join(rung.blocks, "ridge_manifest.json")) as f:
        man = json.load(f)
    w = np.load(os.path.join(rung.blocks, "ridge_weights_full.npy"))
    if not 0 < a.embed_dim <= w.shape[1]:
        raise SystemExit(f"--embed_dim must be 1..{w.shape[1]}")
    _, s, vt = np.linalg.svd(w, full_matrices=False)
    basis = vt[:a.embed_dim].T                               # the full ridge's d_e directions in disease-penalty space
    tr = rung.table("train")
    if a.in_sample:
        weights, fold = ["ridge_weights_full.npy"], np.zeros(len(tr["rows"]), dtype=np.int64)
    else:
        fold = rung.folds("train", tr["iid"]).astype(np.int64)
        weights = [f"ridge_weights_fold{k}.npy" for k in range(int(fold.max()) + 1)]
    tables = [(w if name == "ridge_weights_full.npy" else np.load(os.path.join(rung.blocks, name))) @ basis for name in weights]
    tables = [t.astype(np.float32) for t in tables]         # one [M, d_e] per copy, all in the same basis
    chroms = list(rung.store.chroms)
    seq_len = [rung.store.n_variants(c) for c in chroms]
    offsets = np.cumsum([0] + seq_len)
    per_chrom = [[t[offsets[i]:offsets[i + 1]] for i in range(len(chroms))] for t in tables]
    scaling = {}
    if a.token_rms > 0:                                   # unit-RMS tokens, so that the block MLP can read the ridge
        part = load_partition(man["partition"]["path"], rung.store)
        _, mean, std = rung.snp_stats(tr["rows"])
        pick = np.sort(np.random.default_rng(SEED).choice(len(tr["rows"]), size=min(a.scale_rows, len(tr["rows"])), replace=False))
        rms = token_rms(rung.store, part, per_chrom, mean, std, tr["rows"][pick], fold[pick])
        tables = [t * np.float32(a.token_rms / rms) for t in tables]
        per_chrom = [[t[offsets[i]:offsets[i + 1]] for i in range(len(chroms))] for t in tables]
        scaling = {"token_rms_unscaled": rms, "token_rms": a.token_rms, "scale": a.token_rms / rms, "scale_rows": int(len(pick))}
    state = {f"tables.{i}": torch.from_numpy(np.ascontiguousarray(per_chrom[0][i] if a.in_sample else np.stack([c[i] for c in per_chrom])))
             for i in range(len(chroms))}
    digest = partition_digest(man["partition"]["path"], chroms)
    source = {"rung": os.path.basename(rung.dir), "weights": weights, "basis": "right singular vectors of ridge_weights_full.npy",
              "weights_sha256": [man["files"][name] for name in weights], "train_iids_sha256": man["train_iids_sha256"],
              "penalties_h2": man["penalties_h2"], "lambda": man["lambda"], "singular_values": s[:a.embed_dim].tolist(),
              "variance_kept": float((s[:a.embed_dim] ** 2).sum() / (s ** 2).sum()), "date": time.strftime("%Y-%m-%d"), **scaling}
    if digest is None:
        source["TODO"] = "partition_digest: genostore.datamodule.partition_digest was not importable; fill it in before use"
    manifest = {"kind": "ridge", "partition_digest": digest, "snp_code": "dosage", "embed_dim": a.embed_dim, "n_folds": len(tables),
                "fold_rule": "training rows: blocks/folds.npz fold_train; every other participant: pseudo-fold (sha256 of IID and seed modulo k)",
                "n_blocks": man["partition"]["n_blocks"], "seq_len": seq_len, "source": source}
    out = a.out or os.path.join(rung.blocks, f"init_ridge_d{a.embed_dim}.pt")
    torch.save({"manifest": manifest, "state": state}, out)
    chmod600(out)
    print(f"{out}: {len(state)} tables x {len(tables)} fold copies, embed_dim {a.embed_dim}, variance kept {source['variance_kept']:.3f}, digest {digest}"
          + (f", token RMS {scaling['token_rms_unscaled']:.2e} scaled by {scaling['scale']:.3g} to {a.token_rms:g}" if scaling else ", raw scale"))
    return out


if __name__ == "__main__":
    main()
