#!/usr/bin/env python3
"""Check a genotype store rebuilt without withdrawn participants against the store it replaces.

Run after genostore.convert has written the new store and `genostore.convert verify` has passed
on it. Prints counts only, never an identifier or a genotype:
  - the new store holds exactly the old store's participants minus those on the withdrawal
    list, in the same order, and nobody on the list;
  - every participant of the given sample tables is in the new store;
  - every chromosome has the same variants in the same order;
  - for a random draw of participants, every genotype on the listed chromosomes is the same in
    the two stores.
Exit status 0 when every check passes.

    python check_rebuilt_store.py --old <data>/genostore/hm3plus --new <data>/genostore/hm3plus.rebuild \
        --withdrawn <withdrawal list> --tables <data>/datasets/ard30/hm3plus/samples_*.tsv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

DEEPGP_REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
SEED = 1                                             # the project's single seed


def table_ids(path):
    """The IID column of a sample table (tab-separated, with a header)."""
    with open(path) as f:
        col = f.readline().rstrip("\n").split("\t").index("IID")
        return [line.rstrip("\n").split("\t")[col] for line in f if line.strip()]


def withdrawn_ids(path):
    """A withdrawal list: one participant ID per line."""
    with open(path) as f:
        return {line.strip() for line in f if line.strip().isdigit()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--withdrawn", required=True)
    ap.add_argument("--tables", nargs="*", default=[])
    ap.add_argument("--n-check", type=int, default=2000, help="participants whose genotypes are compared")
    ap.add_argument("--chroms", nargs="*", default=["1", "6", "22"], help="chromosomes compared in full")
    a = ap.parse_args(argv)
    sys.path.insert(0, DEEPGP_REPO)
    from genostore.store import GenoStore

    old, new = GenoStore(a.old), GenoStore(a.new)
    ids_old, ids_new = [str(s) for s in old.samples], [str(s) for s in new.samples]
    gone = withdrawn_ids(a.withdrawn)
    failed = []

    def check(name, passed, detail):
        print(f"{'ok  ' if passed else 'FAIL'} {name}: {detail}", flush=True)
        if not passed:
            failed.append(name)

    expected = [s for s in ids_old if s not in gone]
    check("participants", ids_new == expected,
          f"{len(ids_old)} before, {len(ids_old) - len(expected)} on the withdrawal list, {len(ids_new)} after")
    left = len(set(ids_new) & gone)
    check("withdrawn participants left", left == 0, f"{left}")
    present = set(ids_new)
    for t in a.tables:
        ids = table_ids(t)
        missing = sum(s not in present for s in ids)
        check(f"table {os.path.basename(os.path.dirname(t))}/{os.path.basename(t)}", missing == 0,
              f"{len(ids)} participants, {missing} not in the new store")
    check("chromosomes", [str(c) for c in old.chroms] == [str(c) for c in new.chroms], f"{len(new.chroms)}")
    differ = [c for c in old.chroms if old.n_variants(c) != new.n_variants(c)
              or not np.array_equal(old.variants(c)["id"], new.variants(c)["id"])]
    check("variants", not differ, f"chromosomes whose variant list differs: {len(differ)}")

    pick = np.sort(np.random.default_rng(SEED).choice(len(ids_new), size=min(a.n_check, len(ids_new)),
                                                      replace=False))
    row_old = {s: i for i, s in enumerate(ids_old)}
    rows_new = pick.astype(np.int64)
    rows_old = np.array([row_old[ids_new[i]] for i in pick], dtype=np.int64)
    for c in a.chroms:
        m, bad = new.n_variants(c), 0
        for v0 in range(0, m, 1 << 16):
            v1 = min(m, v0 + (1 << 16))
            bad += int((old.read_q(c, rows_old, v0, v1) != new.read_q(c, rows_new, v0, v1)).sum())
        check(f"chr{c} genotypes of {len(pick)} participants", bad == 0, f"{bad} of {len(pick) * m} differ")

    print("ALL CHECKS PASSED" if not failed else f"FAILED: {', '.join(failed)}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
