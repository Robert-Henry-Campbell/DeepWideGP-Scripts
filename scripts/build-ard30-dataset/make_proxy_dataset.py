#!/usr/bin/env python3
"""Make the small "proxy" version of the 30-disease dataset used for pilots and for the first
stage of the hyper-parameter search.

The proxy keeps the 30 labels and 12 covariates and shrinks the two things training time is
proportional to:

    SNPs          one contiguous stretch of every chromosome: the SNPs from --from-eighth/8 to
                  (--from-eighth + 1)/8 of the way along it, by SNP index. A contiguous stretch
                  keeps the local LD the full model sees. Nothing is copied: store.json points at
                  the same genotype store and names the stretch as a region.
    participants  --n-train drawn at random from the training split; --n-tune from the tuning
                  split.

The proxy's samples_test.tsv is a further random draw from the TRAINING split, disjoint from
the proxy's training participants. A pilot therefore cannot read the real test set, by
construction.

Writes <data_dir>/datasets/ard30/<name>/{store.json, samples_train.tsv, samples_val.tsv,
samples_test.tsv, labels.json, proxy_log.json}. Participant-level; aggregate numbers only are
printed and logged.

    /home/robert/envs/widedeepgp/bin/python scripts/build-ard30-dataset/make_proxy_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261112


def stretch(n_variants, from_eighth):
    """[v0, v1): the SNPs from from_eighth/8 to (from_eighth + 1)/8 of the way along a chromosome."""
    if not 0 <= from_eighth <= 7:
        raise ValueError("from_eighth must be 0..7")
    return from_eighth * n_variants // 8, (from_eighth + 1) * n_variants // 8


def draw(ids, sizes, seed):
    """Disjoint random draws of the given sizes from ``ids`` (sorted first, so the order of the
    input does not matter). Returns one sorted array per size."""
    ids = np.sort(np.asarray(ids, dtype=str))
    if sum(sizes) > len(ids):
        raise ValueError(f"cannot draw {sum(sizes)} from {len(ids)}")
    perm = np.random.default_rng(seed).permutation(len(ids))
    out, start = [], 0
    for n in sizes:
        out.append(np.sort(ids[perm[start:start + n]]))
        start += n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="/mnt/sdg/robert/deepwidegp/data/datasets/ard30/hm3plus")
    ap.add_argument("--name", default="hm3plus_proxy8")
    ap.add_argument("--from-eighth", type=int, default=1, help="0..7: which eighth of each chromosome (default 1, the second)")
    ap.add_argument("--n-train", type=int, default=40_000)
    ap.add_argument("--n-tune", type=int, default=10_000)
    ap.add_argument("--n-test", type=int, default=10_000, help="drawn from the TRAINING split (see the header)")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    t0 = time.time()
    src = Path(a.dataset)
    out = src.parent / a.name
    os.makedirs(out, mode=0o700, exist_ok=True)

    spec = json.load(open(src / "store.json"))
    store = spec["store"] if os.path.isabs(spec["store"]) else str(src / spec["store"])
    labels, covs = spec["labels"], spec["covariates"]
    regions, snps = [], {}
    for c in range(1, 23):
        meta = json.load(open(os.path.join(store, f"chr{c}.json")))
        pos = np.load(os.path.join(store, f"chr{c}.pos.npy"), mmap_mode="r")
        v0, v1 = stretch(len(pos), a.from_eighth)
        regions.append([[str(c), int(v0), int(v1)]])
        snps[str(c)] = {"n_variants": int(v1 - v0), "of": int(len(pos)), "first_index": int(v0),
                        "bp_from": int(pos[v0]), "bp_to": int(pos[v1 - 1])}
        assert int(meta.get("n_variants", len(pos))) == len(pos)

    train = pd.read_csv(src / "samples_train.tsv", sep="\t", dtype={"IID": str})
    tune = pd.read_csv(src / "samples_val.tsv", sep="\t", dtype={"IID": str})
    tr_ids, te_ids = draw(train["IID"], [a.n_train, a.n_test], a.seed)
    (tu_ids,) = draw(tune["IID"], [a.n_tune], a.seed + 1)
    assert not set(tr_ids) & set(te_ids)
    real_test = set(pd.read_csv(src / "samples_test.tsv", sep="\t", usecols=["IID"], dtype={"IID": str})["IID"])
    assert not (set(tr_ids) | set(te_ids) | set(tu_ids)) & real_test, "the proxy must not contain test participants"

    tables = {"train": train[train["IID"].isin(set(tr_ids))], "val": tune[tune["IID"].isin(set(tu_ids))],
              "test": train[train["IID"].isin(set(te_ids))]}
    for name, df in tables.items():
        path = out / f"samples_{name}.tsv"
        df.to_csv(path, sep="\t", index=False)
        os.chmod(path, 0o600)

    proxy_spec = {"store": store, "pos_mode": spec.get("pos_mode", "index"), "regions": regions, "labels": labels,
                  "covariates": covs}
    if spec.get("annotations"):
        proxy_spec["annotations"] = dict(spec["annotations"],
                                         file=os.path.relpath(src / spec["annotations"]["file"], out))
    with open(out / "store.json", "w") as f:
        json.dump(proxy_spec, f, indent=1)

    manifest = json.load(open(src / "labels.json"))
    cases = {g: {lab: int(df[lab].sum()) for lab in labels} for g, df in tables.items()}
    with open(out / "labels.json", "w") as f:
        json.dump([{"label": m["label"], "icd10_codes": m["icd10_codes"], "title": m["title"],
                    "definition": m["definition"], "cases_train": cases["train"][m["label"]],
                    "cases_val": cases["val"][m["label"]], "cases_test": cases["test"][m["label"]],
                    "prevalence_train": cases["train"][m["label"]] / len(tables["train"])} for m in manifest], f, indent=1)
    log = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "source_dataset": str(src), "genotype_store": store,
           "from_eighth": a.from_eighth, "seed": a.seed,
           "n_snps": int(sum(s["n_variants"] for s in snps.values())),
           "n_snps_full": int(sum(s["of"] for s in snps.values())), "snps_by_chromosome": snps,
           "groups": {g: int(len(df)) for g, df in tables.items()},
           "test_table_is": "a draw from the training split, disjoint from the proxy's training participants; "
                            "the real test set is not in this dataset",
           "cases_by_group": cases,
           "fewest_cases": {g: {"label": min(c, key=c.get), "cases": min(c.values())} for g, c in cases.items()}}
    with open(out / "proxy_log.json", "w") as f:
        json.dump(log, f, indent=1)
    print(f"wrote {out} in {time.time() - t0:.1f} s: {log['n_snps']} of {log['n_snps_full']} SNPs, "
          f"train {log['groups']['train']}, tune {log['groups']['val']}, proxy test {log['groups']['test']}; "
          f"fewest cases {log['fewest_cases']}", flush=True)
    return log


if __name__ == "__main__":
    main()
