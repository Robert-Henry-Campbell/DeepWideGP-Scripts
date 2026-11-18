#!/usr/bin/env python3
"""Hard calls of one chromosome for the haploblock pre-test (methods.md sections 2 and 3).

Reads the store's dosages of the training and tuning rows of a rung (never the test split), calls 0, 1 or 2
when the dosage is within --threshold of it (else missing), and writes under <dataset_dir>/haplo/, owner-only:
chr<c>_hardcalls.{bed,bim,fam} (PLINK 1 binary, SNP-major, A1 = ALT so that the BCF plink2 writes has
genotype 1 = the store's ALT allele), rows.npz (sample order: IID, store row, split), chr<c>.gmap.gz (the OMNI
genetic map in SHAPEIT's pos/chr/cM format) and export.json (aggregates). `--check <bcf>` compares the BCF's
allele frequencies with the hard calls' (needs bcftools on PATH).

    python export_chr22.py <dataset_dir> --map data/pgi/ld/genetic_maps/omni/chr22.OMNI.interpolated_genetic_map.gz
    python export_chr22.py <dataset_dir> --check <dataset_dir>/haplo/chr22_unphased.bcf
"""
import argparse
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "widedeepgp-blocks"))
from ridge_common import MISSING, Rung, chmod600, prefetch, save_json, save_npz  # noqa: E402

BED_CODE = np.array([3, 2, 0, 1], dtype=np.uint8)   # call 0 -> 11 (hom A2 = REF), 1 -> 10, 2 -> 00 (hom A1 = ALT), -1 -> 01


def hard_calls(q, threshold=0.1):
    """uint8 q (dosage x 100, 255 missing) -> int8 calls 0/1/2, -1 where no integer lies within ``threshold``."""
    d = q.astype(np.int16)
    call = np.clip(np.rint(d / 100.0), 0, 2).astype(np.int16)
    ok = (q != MISSING) & (np.abs(d - call * 100) <= int(round(threshold * 100)))
    return np.where(ok, call, -1).astype(np.int8)


def pack_bed(calls):
    """int8 calls [n_samples, m] -> SNP-major PLINK 1 .bed bytes (with the 3 magic bytes)."""
    n, m = calls.shape
    c = BED_CODE[np.ascontiguousarray(calls.T) & 3]                      # [m, n] two-bit codes
    pad = (-n) % 4
    if pad:
        c = np.concatenate([c, np.ones((m, pad), np.uint8)], axis=1)     # padding = missing, as plink writes it
    c = c.reshape(m, -1, 4)
    b = c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)
    return b"\x6c\x1b\x01" + b.astype(np.uint8).tobytes()


def unpack_bed(data, n_samples, m):
    """The numpy twin of pack_bed for the tests: .bed bytes -> int8 calls [n_samples, m]."""
    b = np.frombuffer(data[3:], dtype=np.uint8).reshape(m, -1)
    c = np.stack([(b >> (2 * j)) & 3 for j in range(4)], axis=2).reshape(m, -1)[:, :n_samples]
    inv = np.full(4, -1, np.int8)
    inv[BED_CODE] = [0, 1, 2, -1]
    return inv[c].T


def write_gmap(omni_gz, chrom, out_gz):
    """OMNI interpolated map (rsid pos cM) -> SHAPEIT map (pos chr cM, one row per position, ascending)."""
    pos, cm = [], []
    with gzip.open(omni_gz, "rt") as f:
        for line in f:
            p = line.split()
            if len(p) < 3 or not p[1].isdigit():
                continue
            pos.append(int(p[1])); cm.append(float(p[2]))
    pos, cm = np.array(pos), np.array(cm)
    _, first = np.unique(pos, return_index=True)
    pos, cm = pos[first], cm[first]
    if (np.diff(cm) < 0).any():
        raise SystemExit(f"{omni_gz}: the map is not monotone")
    with gzip.open(out_gz, "wt") as f:
        f.write("pos\tchr\tcM\n")
        for p, c in zip(pos, cm):
            f.write(f"{p}\t{chrom}\t{c:.6f}\n")
    chmod600(out_gz)
    return len(pos)


def read_calls(store, chrom, rows, threshold, threads, batch=4096):
    """Hard calls [len(rows), m] of ``rows`` (in order), read in batches on threads."""
    batches = np.array_split(np.arange(len(rows)), max(1, len(rows) // batch))
    out = np.empty((len(rows), store.n_variants(chrom)), dtype=np.int8)
    for idx, q in prefetch(lambda b: (b, store.read_q(chrom, rows[b])), batches, threads):
        out[idx] = hard_calls(q, threshold)
    return out


def export(a):
    rung = Rung(a.dataset_dir)
    if "test" in a.splits:
        raise SystemExit("the test split is locked and is not phased")
    out = os.path.join(rung.dir, "haplo")
    os.makedirs(out, mode=0o700, exist_ok=True)
    tabs = [rung.table(s) for s in a.splits]
    iid = np.concatenate([t["iid"] for t in tabs])
    rows = np.concatenate([t["rows"] for t in tabs])
    split = np.concatenate([np.full(len(t["iid"]), s) for s, t in zip(a.splits, tabs)])
    if len(np.unique(iid)) != len(iid):
        raise SystemExit("a participant appears in two splits")
    var = rung.store.variants(a.chrom)
    t0 = time.time()
    calls = read_calls(rung.store, a.chrom, rows, a.threshold, a.threads)
    missing = calls < 0
    prefix = os.path.join(out, f"chr{a.chrom}_hardcalls")
    with open(prefix + ".bed", "wb") as f:
        f.write(pack_bed(calls))
    chmod600(prefix + ".bed")
    with open(prefix + ".bim", "w") as f:
        for p, i, r, al in zip(var["pos"], var["id"], var["ref"], var["alt"]):
            f.write(f"{a.chrom}\t{i}\t0\t{p}\t{al}\t{r}\n")                 # A1 = ALT, A2 = REF
    with open(prefix + ".fam", "w") as f:
        for i in iid:
            f.write(f"0\t{i}\t0\t0\t0\t-9\n")
    for ext in (".bim", ".fam"):
        chmod600(prefix + ext)
    save_npz(os.path.join(out, "rows.npz"), iid=iid, rows=rows, split=split, chrom=np.array(a.chrom))
    ok = ~missing
    af = np.where(ok.sum(0) > 0, (np.where(ok, calls, 0).sum(0) / np.maximum(ok.sum(0), 1)) / 2.0, np.nan)
    np.save(os.path.join(out, f"chr{a.chrom}_hardcall_af.npy"), af.astype(np.float64))
    chmod600(os.path.join(out, f"chr{a.chrom}_hardcall_af.npy"))
    n_map = write_gmap(a.map, a.chrom, os.path.join(out, f"chr{a.chrom}.gmap.gz")) if a.map else 0
    rec = {"chrom": a.chrom, "splits": a.splits, "n_rows": {s: int(len(t["iid"])) for s, t in zip(a.splits, tabs)},
           "n_variants": int(calls.shape[1]), "threshold": a.threshold, "missing_rate": float(missing.mean()),
           "variants_missing_over_5pct": int((missing.mean(0) > 0.05).sum()),
           "rows_missing_over_5pct": int((missing.mean(1) > 0.05).sum()),
           "map_positions": n_map, "seconds": round(time.time() - t0, 1), "built": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_json(os.path.join(out, "export.json"), rec)
    print(json.dumps(rec, indent=1))


def check(a):
    """The BCF's INFO/AF per variant (bcftools +fill-tags) against the hard calls'; REF/ALT against the store's."""
    rung = Rung(a.dataset_dir)
    out = os.path.join(rung.dir, "haplo")
    af0 = np.load(os.path.join(out, f"chr{a.chrom}_hardcall_af.npy"))
    var = rung.store.variants(a.chrom)
    cmd = ["bcftools", "query", "-f", "%POS\\t%REF\\t%ALT\\t%INFO/AF\\n", a.check]
    txt = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.splitlines()
    if len(txt) != len(var["pos"]):
        raise SystemExit(f"{a.check}: {len(txt)} variants against the store's {len(var['pos'])}")
    pos, ref, alt, af = zip(*(l.split("\t") for l in txt))
    if (np.array(pos, np.int64) != var["pos"]).any() or (np.array(ref) != var["ref"]).any() or (np.array(alt) != var["alt"]).any():
        raise SystemExit(f"{a.check}: positions or alleles differ from the store's")
    diff = np.abs(np.array(af, np.float64) - af0)
    print(f"{a.check}: {len(txt)} variants, max |AF(bcf) - AF(hard calls)| = {np.nanmax(diff):.2e}")
    if np.nanmax(diff) > 1e-5:
        raise SystemExit("allele frequencies differ: the A1/A2 convention or the sample order is wrong")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--chrom", default="22")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--map", help="OMNI interpolated genetic map (rsid pos cM), gzipped")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--check", help="an exported BCF to check against the hard calls")
    a = ap.parse_args(argv)
    check(a) if a.check else export(a)


if __name__ == "__main__":
    main()
