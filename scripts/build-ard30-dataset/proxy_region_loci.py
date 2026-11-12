#!/usr/bin/env python3
"""Write results/proxy_region_loci.csv: the 20 loci behind the choice of the proxy's region.

The list and its approximate GRCh37 positions are as typed from recall on 11 Nov 2026 (session
transcript, 22:48:38Z), before the proxy was built; they were not looked up in a catalogue. Each
locus's eighth is that of the first SNP of the genotype store at or after its position, with the
eighths defined exactly as make_proxy_dataset.stretch() defines them (numbered 1 to 8 here; the
proxy uses --from-eighth 1, the second). Reads SNP positions only.

    python scripts/build-ard30-dataset/proxy_region_loci.py [--store <data>/genostore/hm3plus]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = str(Path(__file__).resolve().parent)
STORE = "/mnt/sdg/robert/deepwidegp/data/genostore/hm3plus"
sys.path.insert(0, HERE)
import make_proxy_dataset as mp  # noqa: E402

LOCI = [  # locus, chromosome, approximate GRCh37 position (Mb), disease area
    ("9p21 (CDKN2B-AS1)", 9, 22.1, "coronary disease"),
    ("HLA", 6, 31.0, "immune-mediated (e.g. hypothyroidism)"),
    ("LDLR", 19, 11.2, "lipids"),
    ("PCSK9", 1, 55.5, "lipids"),
    ("UMOD", 16, 20.36, "kidney function"),
    ("TCF7L2", 10, 114.76, "type 2 diabetes"),
    ("APOE", 19, 45.41, "lipids"),
    ("LPA", 6, 161.0, "coronary disease"),
    ("FTO", 16, 53.8, "obesity"),
    ("SORT1", 1, 109.8, "lipids"),
    ("PITX2 (4q25)", 4, 111.7, "atrial fibrillation"),
    ("ZFHX3", 16, 73.05, "atrial fibrillation"),
    ("ABO", 9, 136.15, "cardiovascular"),
    ("APOB", 2, 21.26, "lipids"),
    ("GCKR", 2, 27.73, "lipids and glucose"),
    ("MC4R", 18, 57.85, "obesity"),
    ("GDF5", 20, 34.0, "osteoarthritis"),
    ("HNF1B", 17, 36.1, "type 2 diabetes"),
    ("KCNQ1", 11, 2.85, "type 2 diabetes"),
    ("SLC2A9", 4, 9.9, "urate"),
]


def eighth_of(pos, bp):
    """1..8: the eighth (by SNP index, as make_proxy_dataset.stretch defines it) holding the first
    SNP at or after bp."""
    i = int(np.searchsorted(pos, bp))
    n = len(pos)
    hits = [k for k in range(8) if mp.stretch(n, k)[0] <= i < mp.stretch(n, k)[1]]
    assert len(hits) == 1, (bp, i, n)
    return hits[0] + 1, i / n


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--store", default=STORE, help="genotype store whose SNP positions define the eighths")
STORE = ap.parse_args().store

rows, old = [], [0] * 8
for locus, c, mb, area in LOCI:
    pos = np.load(os.path.join(STORE, f"chr{c}.pos.npy"), mmap_mode="r")
    assert np.all(np.diff(pos) >= 0)
    e, frac = eighth_of(pos, mb * 1e6)
    old[min(7, 8 * int(np.searchsorted(pos, mb * 1e6)) // len(pos))] += 1   # the 11 Nov rule
    v0, v1 = mp.stretch(len(pos), 1)
    print(f"{locus:20} chr{c:<2} {mb:>7} Mb  index fraction {frac:.3f}  eighth {e}"
          + (f"  (proxy stretch {pos[v0] / 1e6:.2f}-{pos[v1 - 1] / 1e6:.2f} Mb)" if e == 2 else ""))
    rows.append({"locus": locus, "chromosome": c, "position_mb_grch37_approx": mb,
                 "eighth_by_snp_index": e, "disease_area": area})
df = pd.DataFrame(rows)
tally = [int((df["eighth_by_snp_index"] == k).sum()) for k in range(1, 9)]
print("loci per eighth:", tally, "| 11 Nov rule:", old)
assert tally == [3, 5, 1, 3, 2, 3, 1, 2] and tally == old
assert sorted(df.loc[df["eighth_by_snp_index"] == 2, "locus"]) == sorted(
    ["9p21 (CDKN2B-AS1)", "HLA", "LDLR", "PCSK9", "GCKR"])
out = os.path.join(HERE, "results", "proxy_region_loci.csv")
df.to_csv(out, index=False)
print("wrote", out)
