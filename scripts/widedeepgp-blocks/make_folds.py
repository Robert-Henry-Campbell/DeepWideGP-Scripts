#!/usr/bin/env python3
"""Folds for the closed-form block ridge of one rung (methods.md section 11): writes blocks/folds.npz.

A training participant's fold is their position in the ladder's permutation (default_rng(seed).permutation
of the parent's sorted training IIDs, the rule in ladder.json; a rung's training set is the first n positions)
modulo 5: balanced within every rung and the same for a participant in every rung, so the folds nest. (The
rung tables keep the parent's row order, so "row number mod 5" would not nest.) A participant of any other
split gets a pseudo-fold from a seeded hash of the IID. The test split needs --unlock_test and is logged in
blocks/test_access.log.

    python make_folds.py <dataset_dir> [--splits train val wb_related asian black]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_common import K_FOLDS, SEED, Rung, pseudo_fold, save_npz, unlock  # noqa: E402


def training_folds(rung, iid):
    """Position in the ladder's permutation of the parent's sorted training IIDs, modulo K_FOLDS."""
    parent, seed = rung.dir, SEED
    ladder = os.path.join(rung.dir, "ladder.json")
    if os.path.exists(ladder):
        with open(ladder) as f:
            lad = json.load(f)
        parent, seed = lad["parent"], int(lad["seed"])
    parent_iid = pd.read_csv(os.path.join(parent, "samples_train.tsv"), sep="\t", usecols=["IID"], dtype=str)["IID"]
    order = np.random.default_rng(seed).permutation(np.sort(parent_iid.to_numpy(str)))
    pos = pd.Series(np.arange(len(order)), index=order).reindex(iid)
    if pos.isna().any() or pos.max() >= len(iid):
        raise SystemExit("the training table is not the first n participants of the ladder permutation")
    return (pos.to_numpy(np.int64) % K_FOLDS).astype(np.int8)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "wb_related", "asian", "black"])
    ap.add_argument("--unlock_test", action="store_true", help="required for the test split; the access is logged")
    a = ap.parse_args(argv)
    rung = Rung(a.dataset_dir)
    unlock(a, a.splits, rung.blocks, "make_folds")
    path = os.path.join(rung.blocks, "folds.npz")
    out = dict(np.load(path)) if os.path.exists(path) else {"seed": np.array(SEED), "k": np.array(K_FOLDS)}
    for split in a.splits:
        iid = rung.table(split)["iid"]
        out[f"iid_{split}"] = iid
        out[f"fold_{split}"] = training_folds(rung, iid) if split == "train" else pseudo_fold(iid)
        print(f"{split}: {len(iid)} rows, per fold {np.bincount(out[f'fold_{split}'], minlength=K_FOLDS).tolist()}")
    save_npz(path, **out)


if __name__ == "__main__":
    main()
