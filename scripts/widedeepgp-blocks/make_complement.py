#!/usr/bin/env python3
"""A dataset whose training split is a parent's minus a rung's: the training participants that the rung's GWAS never saw,
for a frozen PGI table without a label leak (design document item 22). The tuning and test tables, the annotations and
the store are the parent's (symlinks, as the ladder's rungs have them); the training table keeps the parent's lines
unchanged and is written owner-only.

    python make_complement.py <parent dir> <rung dir> <out dir>
"""
import hashlib
import json
import os
import shutil
import sys


def iid_column(header):
    return header.rstrip("\n").split("\t").index("IID")


def main(argv=None):
    parent, rung, out = (os.path.normpath(p) for p in (argv or sys.argv[1:]))
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    with open(os.path.join(rung, "samples_train.tsv")) as f:
        k = iid_column(next(f))
        drop = {line.split("\t")[k].rstrip("\n") for line in f}
    with open(os.path.join(parent, "samples_train.tsv")) as f:
        lines = f.readlines()
    k = iid_column(lines[0])
    iids = [line.split("\t")[k].rstrip("\n") for line in lines[1:]]
    if not drop <= set(iids):
        raise SystemExit("the rung's training participants are not all in the parent's")
    keep = [line for line, i in zip(lines[1:], iids) if i not in drop]
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, mode=0o700)
    for name in ("annotations.npz", "samples_val.tsv", "samples_test.tsv", "samples_wb_related.tsv"):
        if os.path.exists(os.path.join(parent, name)):
            os.symlink(os.path.realpath(os.path.join(parent, name)), os.path.join(tmp, name))
    with open(os.path.join(parent, "store.json")) as f:
        spec = json.load(f)
    spec["store"] = os.path.join(parent, spec["store"])     # a relative store is the parent's (absolute stays as it is)
    with open(os.open(os.path.join(tmp, "store.json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        json.dump(spec, f, indent=1)
    if os.path.exists(os.path.join(parent, "labels.json")):
        shutil.copyfile(os.path.join(parent, "labels.json"), os.path.join(tmp, "labels.json"))
        os.chmod(os.path.join(tmp, "labels.json"), 0o600)
    path = os.path.join(tmp, "samples_train.tsv")
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        f.writelines([lines[0]] + keep)
    kept = sorted(i for i in iids if i not in drop)
    record = {"parent": parent, "excluded": rung, "n_train": len(keep), "n_excluded": len(drop),
              "rule": "the parent's training participants not in the excluded rung's training split",
              "train_iids_sha256": hashlib.sha256("\n".join(kept).encode()).hexdigest()}
    with open(os.open(os.path.join(tmp, "complement.json"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as f:
        json.dump(record, f, indent=1)
    os.rename(tmp, out)
    print(f"{out}: {len(keep):,} training participants ({len(drop):,} of {len(iids):,} excluded)")
    return out


if __name__ == "__main__":
    main()
