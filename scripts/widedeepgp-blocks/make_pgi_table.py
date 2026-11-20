#!/usr/bin/env python3
"""Frozen per-SNP table from a PGI method's chosen weights, for DeepWideGP --init_encoder --freeze_encoder (design
document item 22). One column per disease, in the dataset's label order: a SNP's row is its ALT-allele weights times the
training SD the model standardises with, so a block's pooled vector is the block's part of each disease's PGI (centred,
over the square root of the block's SNP count). --drop_zero: SNPs with no weight for any disease leave the model: their
rows are zero and each block is divided by the square root of its kept SNPs instead of all of them. Each channel is
scaled to unit RMS over training participants, so that it is readable beside the identity vectors. One copy, no folds:
when the weights' GWAS shared participants with the dataset's training split, --in_sample must be passed (those
participants' labels are then in their own inputs) and the overlap is recorded. Checks, written to the manifest
(aggregates only): the block sums reproduce the PGI's own tuning scores (correlation per disease), and the PGI's AUROC
on a sample of training participants against the tuning set (the size of any leak).

    PYTHONPATH=<DeepWideGP> python make_pgi_table.py <dataset_dir> --method lassosum2 [--weights_dataset <dir>]
        [--drop_zero] [--in_sample] [--out <file>]
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

ROOT = "/mnt/sdg/robert/deepwidegp"


def chosen_weights(results_dir, cache_dir, weights_dataset, method, labels, n_train):
    """{trait: (DataFrame SNP A1 w, info)}: each trait's chosen weights from the PGI cache entry its tuning row names."""
    tm = pd.read_csv(os.path.join(results_dir, os.path.basename(weights_dataset), "tune_metrics.csv"), dtype={"pgi_key": str})
    tm = tm[tm["method"] == method]
    out = {}
    for t in labels:
        row = tm[tm["trait"] == t]
        if len(row) != 1 or row["status"].iloc[0] != "ok":
            raise SystemExit(f"{method} {t}: no single fitted row in tune_metrics.csv")
        key = row["pgi_key"].iloc[0]
        dirs = glob.glob(os.path.join(cache_dir, key[:2], key + "*"))
        if len(dirs) != 1:
            raise SystemExit(f"{method} {t}: {len(dirs)} cache entries match key {key}")
        man = json.load(open(os.path.join(dirs[0], "manifest.json")))
        if (man["method"], man["trait"], man["status"]) != (method, t, "ok"):
            raise SystemExit(f"{dirs[0]}: not the fitted {method} entry of {t}")
        if int(man["trait_info"]["N"]) != n_train:
            raise SystemExit(f"{method} {t}: GWAS of {man['trait_info']['N']} people, the weights dataset trains {n_train}")
        w = pd.read_csv(os.path.join(dirs[0], "weights.tsv.gz"), sep="\t", dtype={"SNP": str, "A1": str},
                        keep_default_na=False, na_filter=False, float_precision="round_trip")
        out[t] = (w, {"key": man["key"], "candidate": (man.get("choice") or {}).get("candidate"),
                      "n_weights": int(man["n_weights"])})
    return out


def weight_tables(weights, labels, store, regions):
    """float64 [L_i, n_labels] per sequence: each trait's weight per ALT allele, in the model's SNP order (0 where none)."""
    tabs = {c: store.variants(c) for c in {c for seq in regions for c, _, _ in seq}}
    ids = np.concatenate([tabs[c]["id"][v0:v1] for seq in regions for c, v0, v1 in seq]).astype(str)
    alt = np.concatenate([tabs[c]["alt"][v0:v1] for seq in regions for c, v0, v1 in seq]).astype(str)
    index = pd.Index(ids)
    if not index.is_unique:
        raise SystemExit("the store's variant ids are not unique")
    W = np.zeros((len(ids), len(labels)))
    for j, t in enumerate(labels):
        w = weights[t][0]
        pos = index.get_indexer(w["SNP"].to_numpy(str))
        if (pos < 0).any():
            raise SystemExit(f"{t}: {int((pos < 0).sum())} weighted SNPs are not in the store")
        if (alt[pos] != w["A1"].to_numpy(str)).any():
            raise SystemExit(f"{t}: {int((alt[pos] != w['A1'].to_numpy(str)).sum())} weights are not for the store's ALT allele")
        W[pos, j] = w["w"].to_numpy(np.float64)
    return np.split(W, np.cumsum([sum(v1 - v0 for _, v0, v1 in seq) for seq in regions])[:-1])


def block_counts(partition, tables, drop_zero):
    """Per sequence, per block in the model's order: its SNPs, all or (drop_zero) those with any weight."""
    out = []
    for ids, W in zip(partition["ids"], tables):
        keep = W.any(axis=1) if drop_zero else np.ones(len(W), bool)
        _, inv = np.unique(ids, return_inverse=True)
        out.append(np.bincount(inv, weights=keep).astype(np.int64))
    return out


def pooled(dm, enc, split, rows, device):
    """The encoder's pooled vectors [n, n_blocks, d] (float32, no autocast) of some rows of a split."""
    from lightning_fabric.utilities.apply_func import move_data_to_device
    sp = {k: v[rows] for k, v in dm.splits[split].items()}
    parts = []
    with torch.no_grad():
        for b in dm._loader(dm._dataset(sp), False):
            x_all = dm.on_after_batch_transfer(move_data_to_device(b, device), 0)[0]
            parts.append(enc.pool(x_all[:len(dm.regions)]).cpu())
    return torch.cat(parts).numpy()


def auroc(s, y):
    from sklearn.metrics import roc_auc_score
    ok = ~np.isnan(y)
    return float(roc_auc_score(y[ok], s[ok])) if 0 < np.nansum(y) < ok.sum() else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", help="the dataset the network trains on")
    ap.add_argument("--method", required=True, help="lassosum2, ldpred2_auto, ...: a method of PGI/results/<rung>/tune_metrics.csv")
    ap.add_argument("--weights_dataset", default=None, help="the rung whose PGI weights are used (default: dataset_dir)")
    ap.add_argument("--drop_zero", action="store_true", help="SNPs with no weight for any disease leave the model")
    ap.add_argument("--in_sample", action="store_true", help="allow weights whose GWAS included the training participants")
    ap.add_argument("--partition", default=f"{ROOT}/data/reference/partitions/prive431")
    ap.add_argument("--pgi_results", default=f"{ROOT}/PGI/results")
    ap.add_argument("--pgi_cache", default=f"{ROOT}/data/pgi/cache/pgi")
    ap.add_argument("--check_rows", type=int, default=20000, help="training participants (seed 1) for the scale and the leak check")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from args_generator import args_initial
    from widedeepgp.blockencoder import BlockEncoder
    from widedeepgp.train import build_data

    dataset_dir = os.path.normpath(a.dataset_dir)
    weights_dataset = os.path.normpath(a.weights_dataset or dataset_dir)
    cfg = args_initial(["--label", os.path.basename(os.path.dirname(dataset_dir)), "--rd_mode", os.path.basename(dataset_dir),
                        "--dm", "snps_covs", "--input_mode", "blocks", "--partition", a.partition, "--snp_stats", "train",
                        "--num_workers", str(a.num_workers), "--eval_batch_size", "1024", "--wide", "off"])
    cfg.data_dir = os.path.dirname(os.path.dirname(dataset_dir))
    dm = build_data(cfg)
    labels = list(dm.label_names)
    iids = {d: set(pd.read_csv(os.path.join(d, "samples_train.tsv"), sep="\t", dtype={"IID": str}, usecols=["IID"])["IID"])
            for d in {dataset_dir, weights_dataset}}
    overlap = len(iids[dataset_dir] & iids[weights_dataset])
    if overlap and not a.in_sample:
        raise SystemExit(f"the weights' GWAS includes {overlap} of the {len(iids[dataset_dir])} training participants: their "
                         "labels would be in their own inputs; pass --in_sample to build it anyway")

    weights = chosen_weights(a.pgi_results, a.pgi_cache, weights_dataset, a.method, labels, len(iids[weights_dataset]))
    W = weight_tables(weights, labels, dm.store, dm.regions)
    n_all, n_div = block_counts(dm.partition, W, False), block_counts(dm.partition, W, a.drop_zero)
    sd = [np.where(inv > 0, 1.0 / np.where(inv > 0, inv, 1.0), 0.0) for inv in dm.snp_inv_std]
    # the model divides a block by sqrt(all its SNPs); for the kept SNPs only, rows are rescaled by sqrt(all / kept)
    T = [w * s[:, None] * np.sqrt(na / np.maximum(nd, 1))[np.unique(ids, return_inverse=True)[1]][:, None]
         for w, s, na, nd, ids in zip(W, sd, n_all, n_div, dm.partition["ids"])]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = BlockEncoder(dm.partition, len(labels), 8, "dosage", 0.0, 1)
    for t, w in zip(enc.tables, T):
        t.data = torch.from_numpy(w.astype(np.float32))
    enc.to(device)
    n_train = len(dm.splits["train"]["y"])
    pick = np.sort(np.random.default_rng(1).choice(n_train, size=min(a.check_rows, n_train), replace=False))
    p = {"train": pooled(dm, enc, "train", pick, device), "val": pooled(dm, enc, "val", np.arange(len(dm.splits["val"]["y"])), device)}
    rms = np.sqrt((p["train"].astype(np.float64) ** 2).mean(axis=(0, 1)))
    scale = np.where(rms > 0, 1.0 / np.where(rms > 0, rms, 1.0), 0.0)
    div = np.sqrt(np.concatenate(n_div).astype(np.float64))          # a block's pooled vector x sqrt(its divisor) = its PGI part
    pgi = {s: (v.astype(np.float64) * div[None, :, None]).sum(axis=1) for s, v in p.items()}
    files = glob.glob(os.path.join(weights_dataset, "scores", "pgi", "*", a.method, "val.npz"))
    if len(files) != 1:
        raise SystemExit(f"{len(files)} tuning-score files for {a.method} under {weights_dataset}/scores/pgi")
    z = np.load(files[0], allow_pickle=False)
    val_iids = pd.read_csv(os.path.join(dataset_dir, "samples_val.tsv"), sep="\t", dtype={"IID": str}, usecols=["IID"])["IID"]
    if z["iid"].astype(str).tolist() != val_iids.tolist():
        raise SystemExit(f"{files[0]}: not this dataset's tuning set in its order")
    r = np.array([np.corrcoef(pgi["val"][:, j], z["pgi"][:, j])[0, 1] for j in range(len(labels))])
    if not (r > 0.9999).all():
        raise SystemExit(f"block sums do not reproduce the PGI's tuning scores (lowest correlation {np.nanmin(r):.6f})")
    y_tr, y_val = dm.splits["train"]["y"][pick], dm.splits["val"]["y"]
    leak = [{"trait": t, "auroc_pgi_training_sample": auroc(pgi["train"][:, j], y_tr[:, j]),
             "auroc_pgi_tuning": auroc(pgi["val"][:, j], y_val[:, j])} for j, t in enumerate(labels)]

    state = {f"tables.{i}": torch.from_numpy((w * scale[None, :]).astype(np.float32)) for i, w in enumerate(T)}
    kept, n_snps = int(sum(int(n.sum()) for n in n_div)), int(sum(len(w) for w in W))
    source = {"method": a.method, "weights_dataset": weights_dataset, "dataset": dataset_dir, "training_overlap": overlap,
              "n_training": len(iids[dataset_dir]), "labels": labels, "entries": {t: info for t, (_, info) in weights.items()},
              "drop_zero": a.drop_zero, "snps_kept": kept, "snps_all": n_snps,
              "blocks_empty": int(sum(int((n == 0).sum()) for n in n_div)), "channel_rms_unscaled": rms.tolist(),
              "check_rows": int(len(pick)), "tuning_correlation_min": float(r.min()), "leak_check": leak,
              "date": time.strftime("%Y-%m-%d")}
    manifest = {"kind": "pgi", "partition_digest": dm.partition["digest"], "snp_code": "dosage", "embed_dim": len(labels),
                "n_folds": 1, "n_blocks": int(dm.partition["n_blocks"]), "seq_len": [len(w) for w in W], "source": source}
    out = a.out or os.path.join(dataset_dir, "blocks", f"init_pgi_{a.method}_d{len(labels)}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"manifest": manifest, "state": state}, out)
    os.chmod(out, 0o600)
    tr, tu = (np.nanmean([d[k] for d in leak]) for k in ("auroc_pgi_training_sample", "auroc_pgi_tuning"))
    print(f"{out}: {a.method} from {os.path.basename(weights_dataset)}, {len(labels)} channels, {kept:,} of {n_snps:,} SNPs "
          f"({source['blocks_empty']} empty blocks), tuning correlation >= {r.min():.6f}; mean PGI AUROC on {len(pick):,} "
          f"training participants {tr:.4f} against {tu:.4f} on the tuning set; training overlap {overlap:,}", flush=True)
    return out


if __name__ == "__main__":
    main()
