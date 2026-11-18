"""A 1:1 case-control training split for one disease (methods.md, "A 1:1 case-control run").

Every participant of the rarer class is kept and as many of the other class are drawn at random without
replacement; participants without the label are left out. The new dataset directory holds that training table
and links to the source's store.json, tuning and test tables, so only the training participants change.

  python make_balanced_split.py --dataset data/datasets/ard30/classic_i10_n215872 --label I10 --seed 1
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd


def balanced_rows(y, rng):
    """Sorted row positions of the 1:1 set: all of the rarer class and the same number of the other class, drawn
    without replacement by ``rng`` (a seed or a numpy Generator). ``y`` holds 0, 1 or NaN per row."""
    y = np.asarray(y, dtype=float)
    cases, controls = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(cases) == 0 or len(controls) == 0:
        raise ValueError(f"a 1:1 set needs both classes; found {len(cases)} cases and {len(controls)} controls")
    keep, pool = (cases, controls) if len(cases) <= len(controls) else (controls, cases)
    drawn = np.random.default_rng(rng).choice(pool, size=len(keep), replace=False)
    return np.sort(np.concatenate([keep, drawn]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="source dataset directory (samples_*.tsv, store.json)")
    ap.add_argument("--label", required=True, help="the label column balanced on")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="new dataset directory (default: <dataset>_cc11_<label>)")
    a = ap.parse_args(argv)
    src = a.dataset.rstrip("/")
    out = a.out or f"{src}_cc11_{a.label}"
    os.makedirs(out, mode=0o700, exist_ok=False)          # never overwrite a dataset
    df = pd.read_csv(os.path.join(src, "samples_train.tsv"), sep="\t", dtype=str, keep_default_na=False)
    y = pd.to_numeric(df[a.label].mask(df[a.label] == "")).to_numpy(dtype=float)
    rows = balanced_rows(y, a.seed)
    path = os.path.join(out, "samples_train.tsv")
    df.iloc[rows].to_csv(path, sep="\t", index=False)     # rows copied as they are
    os.chmod(path, 0o600)
    for name in ("samples_val.tsv", "samples_test.tsv", "store.json"):
        os.symlink(os.path.realpath(os.path.join(src, name)), os.path.join(out, name))
    sub = y[rows]
    log = {"source": src, "label": a.label, "seed": a.seed,
           "train": {"rows": int(len(rows)), "cases": int((sub == 1).sum()), "controls": int((sub == 0).sum())},
           "source_train": {"rows": int(len(y)), "cases": int((y == 1).sum()), "controls": int((y == 0).sum()),
                            "unlabelled": int(np.isnan(y).sum())},
           "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out, "build_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print(json.dumps(log["train"]))


if __name__ == "__main__":
    main()
