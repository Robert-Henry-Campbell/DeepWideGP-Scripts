#!/usr/bin/env python
"""Check a block partition against the genotype store and write partition.npz and partition.json.

    check_partition.py prive431 --from_reference REF.tsv             Prive's blocks: the reference table's block_id
    check_partition.py NAME --tsv BLOCKS.tsv --parent PARENT [--built JSON]   a TSV (CHROM ID block) nested in PARENT
    check_partition.py --csv OUT.csv NAME [NAME ...]                  the aggregate table of built partitions

A TSV must list the store's variants row by row in store order. Blocks must be contiguous runs (a label never
returns once another has started) and never span two chromosomes; labels are renumbered 0-based in order of
appearance. The npz and json formats are the contract's (methods.md, section "Partitions").
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP"))
from genostore.store import GenoStore  # noqa: E402

STORE = "/mnt/sdg/robert/deepwidegp/data/genostore/hm3plus"
PARTITIONS = "/mnt/sdg/robert/deepwidegp/data/reference/partitions"
QUANTILES = {"min": 0, "q05": 5, "q25": 25, "median": 50, "q75": 75, "q95": 95, "max": 100}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def variant_digest(ids):
    """ids: the per-chromosome id arrays in store order."""
    return hashlib.sha256("\n".join(np.concatenate(ids)).encode()).hexdigest()


def read_labels(path, column, store, ids):
    """The label column of a TSV whose rows are the store's variants in store order; refused otherwise."""
    tab = np.atleast_1d(np.genfromtxt(path, delimiter="\t", names=True, dtype=None, encoding="utf-8",
                                      usecols=("CHROM", "ID", column)))
    want = np.concatenate([np.char.add(c + "\t", i) for c, i in zip(store.chroms, ids)])
    if len(tab) != len(want):
        raise SystemExit(f"{path}: {len(tab)} rows, but the store has {len(want)} variants")
    bad = np.flatnonzero(np.char.add(tab["CHROM"].astype(str), np.char.add("\t", tab["ID"].astype(str))) != want)
    if len(bad):
        raise SystemExit(f"{path}: {len(bad)} rows are not the store's variant in store order (first at row {bad[0] + 1})")
    return tab[column]


def block_ids(labels, store):
    """Global 0-based ids in order of appearance; refuses a label that returns after another (not contiguous) and a
    label shared by the end of one chromosome and the start of the next (a block spanning chromosomes)."""
    _, first, inv = np.unique(labels, return_index=True, return_inverse=True)
    ids = np.argsort(np.argsort(first))[inv].astype(np.int32)
    ends = np.cumsum([store.n_variants(c) for c in store.chroms])
    back = np.flatnonzero(np.diff(ids) < 0)
    if len(back):
        i = back[0] + 1
        raise SystemExit(f"chr{store.chroms[np.searchsorted(ends, i, side='right')]}: not contiguous: block "
                         f"{labels[i]} returns after block {labels[i - 1]} (store row {i})")
    for c0, c1, a in zip(store.chroms[:-1], store.chroms[1:], ends[:-1]):
        if ids[a - 1] == ids[a]:
            raise SystemExit(f"block {labels[a]} spans chr{c0} and chr{c1}")
    return ids


def check_nested(ids, parent_npz, digest):
    p = np.load(parent_npz)
    if str(p["variant_digest"]) != digest:
        raise SystemExit(f"{parent_npz}: built on other variants (digest {p['variant_digest']}, store {digest})")
    parent = np.concatenate([p[f"block_chr{c}"] for c in p["chroms"]])
    straddling = len(np.unique(ids.astype(np.int64) * len(parent) + parent)) - (int(ids.max()) + 1)
    if straddling:
        raise SystemExit(f"not nested in {parent_npz}: {straddling} blocks straddle a parent block boundary")


def size_stats(sizes):
    return {k: int(np.percentile(sizes, q, method="nearest")) for k, q in QUANTILES.items()} | {"mean": round(float(sizes.mean()), 1)}


def write_partition(name, ids, store, var_ids, digest, built, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ends = np.cumsum([len(v) for v in var_ids])
    starts = ends - [len(v) for v in var_ids]
    npz = os.path.join(out_dir, "partition.npz")
    np.savez(npz, chroms=np.array(store.chroms), n_variants=np.array([len(v) for v in var_ids], dtype=np.int64),
             variant_digest=np.array(digest), n_blocks=np.int64(ids.max() + 1),
             **{f"block_chr{c}": ids[a:b] for c, a, b in zip(store.chroms, starts, ends)})
    per_chrom = [{"chrom": c, "n_variants": int(b - a), "n_blocks": int(ids[b - 1] - ids[a] + 1),
                  **size_stats(np.bincount(ids[a:b] - ids[a]))} for c, a, b in zip(store.chroms, starts, ends)]
    info = {"name": name, "n_blocks": int(ids.max() + 1), "n_variants": int(len(ids)), "variant_digest": digest,
            "partition_npz_sha256": sha256(npz), "block_size": size_stats(np.bincount(ids)),
            "per_chromosome": per_chrom, "built": built}
    with open(os.path.join(out_dir, "partition.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def write_csv(names, partitions, path):
    cols = ["n_variants", "n_blocks", "min", "q25", "median", "q75", "max"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["partition", "chrom"] + [c if c in cols[:2] else f"size_{c}" for c in cols])
        for name in names:
            with open(os.path.join(partitions, name, "partition.json")) as g:
                info = json.load(g)
            rows = info["per_chromosome"] + [{"chrom": "all", "n_variants": info["n_variants"],
                                              "n_blocks": info["n_blocks"], **info["block_size"]}]
            for r in rows:
                w.writerow([name, r["chrom"]] + [r[c] for c in cols])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", metavar="NAME")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from_reference", metavar="TSV")
    src.add_argument("--tsv", metavar="TSV")
    src.add_argument("--csv", metavar="OUT")
    ap.add_argument("--parent", metavar="NAME", help="partition this one must nest in")
    ap.add_argument("--built", metavar="JSON", help="the builder's aggregates, embedded in partition.json")
    ap.add_argument("--store", default=STORE)
    ap.add_argument("--partitions", default=PARTITIONS)
    a = ap.parse_args(argv)
    if a.csv:
        return write_csv(a.names, a.partitions, a.csv)
    if len(a.names) != 1:
        ap.error("one NAME per partition")
    store = GenoStore(a.store)
    var_ids = [store.variants(c)["id"].astype(str) for c in store.chroms]
    digest = variant_digest(var_ids)
    path, column = (a.from_reference, "block_id") if a.from_reference else (a.tsv, "block")
    ids = block_ids(read_labels(path, column, store, var_ids), store)
    if a.parent:
        check_nested(ids, os.path.join(a.partitions, a.parent, "partition.npz"), digest)
    built = {"source": path, "source_sha256": sha256(path), "column": column, "parent": a.parent}
    if a.built:
        with open(a.built) as f:
            built.update(json.load(f))
    info = write_partition(a.names[0], ids, store, var_ids, digest, built, os.path.join(a.partitions, a.names[0]))
    print(f"{a.names[0]}: {info['n_blocks']} blocks over {info['n_variants']} variants, sizes {info['block_size']}")


if __name__ == "__main__":
    main()
