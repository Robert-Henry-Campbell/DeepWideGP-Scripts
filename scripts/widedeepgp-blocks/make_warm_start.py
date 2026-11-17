#!/usr/bin/env python3
"""Ridge warm start for the per-SNP table (DeepWideGP --init_encoder): the rank-d_e SVD U_k S_k of a rung's
full-training ridge weights [M, 90] (every disease and penalty; standardised-dosage form), split per
chromosome as tables.<i> [L_i, d_e] float32, dosage code, with the manifest of the contract.

    python make_warm_start.py <dataset_dir> --embed_dim 16 [--out <dataset_dir>/blocks/init_ridge_d16.pt]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_common import Rung, chmod600  # noqa: E402


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--embed_dim", type=int, required=True)
    ap.add_argument("--out", default=None)
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
    state = {f"tables.{i}": torch.from_numpy(np.ascontiguousarray(table[offsets[i]:offsets[i + 1]])) for i in range(len(chroms))}
    digest = partition_digest(man["partition"]["path"], chroms)
    source = {"rung": os.path.basename(rung.dir), "weights": "ridge_weights_full.npy",
              "weights_sha256": man["files"]["ridge_weights_full.npy"], "train_iids_sha256": man["train_iids_sha256"],
              "penalties_h2": man["penalties_h2"], "lambda": man["lambda"], "singular_values": s[:a.embed_dim].tolist(),
              "variance_kept": float((s[:a.embed_dim] ** 2).sum() / (s ** 2).sum()), "date": time.strftime("%Y-%m-%d")}
    if digest is None:
        source["TODO"] = "partition_digest: genostore.datamodule.partition_digest was not importable; fill it in before use"
    manifest = {"kind": "ridge", "partition_digest": digest, "snp_code": "dosage", "embed_dim": a.embed_dim,
                "n_blocks": man["partition"]["n_blocks"], "seq_len": seq_len, "source": source}
    out = a.out or os.path.join(rung.blocks, f"init_ridge_d{a.embed_dim}.pt")
    torch.save({"manifest": manifest, "state": state}, out)
    chmod600(out)
    print(f"{out}: {len(state)} tables, embed_dim {a.embed_dim}, variance kept {source['variance_kept']:.3f}, digest {digest}")
    return out


if __name__ == "__main__":
    main()
