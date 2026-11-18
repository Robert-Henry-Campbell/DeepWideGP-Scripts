#!/usr/bin/env python3
"""Ridge warm start for the per-SNP table (DeepWideGP --init_encoder): the rank-d_e SVD U_k S_k of a rung's
full-training ridge weights [M, 90] (every disease and penalty; standardised-dosage form), split per
chromosome as tables.<i> [L_i, d_e] float32, dosage code, with the manifest of the contract. The table is scaled so that
the block tokens the model forms from it (standardised dosages times the rows, summed over a block and divided by
sqrt(SNPs)) have RMS --token_rms (1.0) over a sample of training participants: the raw ridge weights are of order 1e-5
and a token made of them is invisible next to the identity embeddings of order 1 (18 November 2026). --token_rms 0
keeps the raw scale.

    python make_warm_start.py <dataset_dir> --embed_dim 16 [--out <dataset_dir>/blocks/init_ridge_d16.pt] [--token_rms 1.0]
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


def token_rms(store, part, tables, mean, std, rows, chunk=512):
    """RMS over participants, blocks and dimensions of the block tokens the model's first layer forms from the tables:
    standardised dosages times the table rows, summed over a block's SNPs and divided by sqrt(SNPs)."""
    s2 = n = 0.0
    for i, c in enumerate(store.chroms):
        m, inv = torch.as_tensor(mean[i], dtype=torch.float32), torch.as_tensor(inverse_sd(std[i]), dtype=torch.float32)
        t = torch.from_numpy(np.ascontiguousarray(tables[i]))
        for r in np.array_split(rows, max(1, len(rows) // chunk)):
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
    a = ap.parse_args(argv)
    rung = Rung(a.dataset_dir)
    with open(os.path.join(rung.blocks, "ridge_manifest.json")) as f:
        man = json.load(f)
    w = np.load(os.path.join(rung.blocks, "ridge_weights_full.npy"))
    if not 0 < a.embed_dim <= w.shape[1]:
        raise SystemExit(f"--embed_dim must be 1..{w.shape[1]}")
    u, s, _ = np.linalg.svd(w, full_matrices=False)
    table = (u[:, :a.embed_dim] * s[:a.embed_dim]).astype(np.float32)
    chroms = list(rung.store.chroms)
    seq_len = [rung.store.n_variants(c) for c in chroms]
    offsets = np.cumsum([0] + seq_len)
    scaling = {}
    if a.token_rms > 0:                                   # unit-RMS tokens, so that the block MLP can read the ridge
        part = load_partition(man["partition"]["path"], rung.store)
        tr = rung.table("train")
        _, mean, std = rung.snp_stats(tr["rows"])
        rows = np.sort(np.random.default_rng(SEED).choice(tr["rows"], size=min(a.scale_rows, len(tr["rows"])), replace=False))
        rms = token_rms(rung.store, part, [table[offsets[i]:offsets[i + 1]] for i in range(len(chroms))], mean, std, rows)
        table *= np.float32(a.token_rms / rms)
        scaling = {"token_rms_unscaled": rms, "token_rms": a.token_rms, "scale": a.token_rms / rms, "scale_rows": int(len(rows))}
    state = {f"tables.{i}": torch.from_numpy(np.ascontiguousarray(table[offsets[i]:offsets[i + 1]])) for i in range(len(chroms))}
    digest = partition_digest(man["partition"]["path"], chroms)
    source = {"rung": os.path.basename(rung.dir), "weights": "ridge_weights_full.npy",
              "weights_sha256": man["files"]["ridge_weights_full.npy"], "train_iids_sha256": man["train_iids_sha256"],
              "penalties_h2": man["penalties_h2"], "lambda": man["lambda"], "singular_values": s[:a.embed_dim].tolist(),
              "variance_kept": float((s[:a.embed_dim] ** 2).sum() / (s ** 2).sum()), "date": time.strftime("%Y-%m-%d"), **scaling}
    if digest is None:
        source["TODO"] = "partition_digest: genostore.datamodule.partition_digest was not importable; fill it in before use"
    manifest = {"kind": "ridge", "partition_digest": digest, "snp_code": "dosage", "embed_dim": a.embed_dim,
                "n_blocks": man["partition"]["n_blocks"], "seq_len": seq_len, "source": source}
    out = a.out or os.path.join(rung.blocks, f"init_ridge_d{a.embed_dim}.pt")
    torch.save({"manifest": manifest, "state": state}, out)
    chmod600(out)
    print(f"{out}: {len(state)} tables, embed_dim {a.embed_dim}, variance kept {source['variance_kept']:.3f}, digest {digest}"
          + (f", token RMS {scaling['token_rms_unscaled']:.2e} scaled by {scaling['scale']:.3g} to {a.token_rms:g}" if scaling else ", raw scale"))
    return out


if __name__ == "__main__":
    main()
