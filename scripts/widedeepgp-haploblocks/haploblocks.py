#!/usr/bin/env python3
"""Haploblocks of one phased chromosome (methods.md section 4).

Reads the phased BCF (bcftools on PATH) of the participants export_chr22.py wrote, checks their order and the
variants against the store, and cuts every Prive block into segments from left to right: a segment grows while
the haplotypes with frequency >= --freq among the training chromosomes cover >= --cover of them; a single SNP is
always a segment. Common haplotypes are coded 0..H-1 by decreasing training frequency, every other haplotype H.
Writes <dataset_dir>/haplo/chr<c>_haploblocks_f<freq>_c<cover>.npz (segments and every participant's two codes
per segment; owner-only) and the aggregate tables results/haploblocks_grid.csv (one row per cell) and
results/haploblocks_<cell>_segments.csv (one row per segment of the primary cell).

    python haploblocks.py <dataset_dir> --chrom 22 --partition data/reference/partitions/prive431
    python haploblocks.py <dataset_dir> --chrom 22 --partition ... --grid        # the nine sensitivity cells
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "widedeepgp-blocks"))
from ridge_common import Rung, load_partition, save_npz  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
M_GENOME = 1258200                      # the model's SNPs: the projection factor is M_GENOME / (SNPs on this chromosome)


def cell_name(freq, cover):
    return f"f{freq:g}_c{cover:g}"


def read_haplotypes(bcf, iid, positions):
    """Alleles uint8 [m, 2 n] of the phased BCF (chromosome 2 i and 2 i + 1 belong to sample i), after checking the
    sample order against ``iid`` and the positions against the store's."""
    names = subprocess.run(["bcftools", "query", "-l", bcf], check=True, capture_output=True, text=True).stdout.split()
    if len(names) != len(iid) or (np.array(names) != np.asarray(iid)).any():
        raise SystemExit(f"{bcf}: the samples are not the exported rows in their order")
    H = np.empty((len(positions), 2 * len(iid)), dtype=np.uint8)
    proc = subprocess.Popen(["bcftools", "query", "-f", "%POS\\t[%GT]\\n", bcf], stdout=subprocess.PIPE, bufsize=1 << 24)
    j = 0
    for line in proc.stdout:
        pos, gts = line.rstrip(b"\n").split(b"\t", 1)
        if j >= len(positions) or int(pos) != positions[j]:
            raise SystemExit(f"{bcf}: variant {j} at {pos.decode()} is not the store's")
        g = np.frombuffer(gts, dtype=np.uint8)
        if len(g) != 3 * len(iid) or (g[1::3] != ord("|")).any():
            raise SystemExit(f"{bcf}: variant {j} is not phased for every sample")
        H[j, 0::2], H[j, 1::2] = g[0::3] - ord("0"), g[2::3] - ord("0")
        j += 1
    if proc.wait() != 0 or j != len(positions):
        raise SystemExit(f"{bcf}: read {j} of {len(positions)} variants")
    return H


def _close(ids, n_ids, is_train, freq):
    """Codes [2 n] of a finished segment from its dense haplotype ids: common ones 0..H-1 by decreasing training
    count, the rest H; returns (codes int16, H, training coverage)."""
    cnt = np.bincount(ids[is_train], minlength=n_ids)
    common = np.flatnonzero(cnt >= freq * is_train.sum())
    common = common[np.argsort(-cnt[common], kind="stable")]
    code = np.full(n_ids, len(common), dtype=np.int16)
    code[common] = np.arange(len(common), dtype=np.int16)
    return code[ids], len(common), float(cnt[common].sum() / is_train.sum())


def segment_block(H, v0, v1, is_train, freq, cover):
    """Greedy segments of the Prive block [v0, v1): list of (start, end, codes [2 n] int16, H, coverage)."""
    n_tr, out = is_train.sum(), []
    start, ids, n_ids = v0, np.zeros(H.shape[1], dtype=np.int64), 1
    for j in range(v0, v1):
        key = ids * 2 + H[j]
        occupied = np.bincount(key, minlength=2 * n_ids) > 0
        new = np.cumsum(occupied) - 1
        ids_new, n_new = new[key], int(occupied.sum())
        cnt = np.bincount(ids_new[is_train], minlength=n_new)
        coverage = cnt[cnt >= freq * n_tr].sum() / n_tr
        if j > start and coverage < cover:
            out.append((start, j) + _close(ids, n_ids, is_train, freq))
            start, ids, n_ids = j, H[j].astype(np.int64), 2
        else:
            ids, n_ids = ids_new, n_new
    out.append((start, v1) + _close(ids, n_ids, is_train, freq))
    return out


def segment_chromosome(H, blocks, is_train, freq, cover):
    """Every Prive block's segments: dict of arrays (start, end, block, H, cover) and codes int16 [n, n_seg, 2]."""
    segs = [(b,) + s for b, v0, v1 in blocks for s in segment_block(H, v0, v1, is_train, freq, cover)]
    codes = np.stack([s[3] for s in segs], axis=1).reshape(-1, 2, len(segs)).transpose(0, 2, 1)
    return dict(start=np.array([s[1] for s in segs]), end=np.array([s[2] for s in segs]),
                block=np.array([s[0] for s in segs]), H=np.array([s[4] for s in segs]),
                cover=np.array([s[5] for s in segs])), np.ascontiguousarray(codes)


def summarise(seg, pos, m, cell, freq, cover):
    """One aggregate row of the grid table."""
    n_snps = seg["end"] - seg["start"]
    kb = (pos[seg["end"] - 1] - pos[seg["start"]]) / 1e3
    feats = int((seg["H"] + 1).sum())
    q = lambda x, p: float(np.quantile(x, p))
    return dict(cell=cell, freq=freq, cover=cover, n_segments=len(n_snps), snps=m,
                snps_per_segment_mean=float(n_snps.mean()), snps_per_segment_median=q(n_snps, 0.5),
                snps_per_segment_q10=q(n_snps, 0.1), snps_per_segment_q90=q(n_snps, 0.9), snps_per_segment_max=int(n_snps.max()),
                kb_per_segment_median=q(kb, 0.5), kb_per_segment_q90=q(kb, 0.9),
                H_mean=float(seg["H"].mean()), H_median=q(seg["H"], 0.5), H_q10=q(seg["H"], 0.1), H_q90=q(seg["H"], 0.9),
                H_max=int(seg["H"].max()), share_H_2_to_15=float(((seg["H"] >= 2) & (seg["H"] <= 15)).mean()),
                coverage_mean=float(seg["cover"].mean()), coverage_min=float(seg["cover"].min()),
                features=feats, features_per_snp=feats / m, features_genome_projected=int(round(feats * M_GENOME / m)),
                segments_genome_projected=int(round(len(n_snps) * M_GENOME / m)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--chrom", default="22")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--freq", type=float, default=0.01)
    ap.add_argument("--cover", type=float, default=0.95)
    ap.add_argument("--grid", action="store_true", help="also the cells freq in {0.005, 0.01, 0.02} x cover in {0.9, 0.95, 0.99}")
    a = ap.parse_args(argv)
    rung = Rung(a.dataset_dir)
    haplo = os.path.join(rung.dir, "haplo")
    rows = np.load(os.path.join(haplo, "rows.npz"), allow_pickle=True)
    if str(rows["chrom"]) != a.chrom:
        raise SystemExit("rows.npz was exported for another chromosome")
    pos = rung.store.positions(a.chrom)
    blocks = load_partition(a.partition, rung.store)["blocks"][a.chrom]
    t0 = time.time()
    H = read_haplotypes(os.path.join(haplo, f"chr{a.chrom}_phased.bcf"), rows["iid"], pos)
    is_train = np.repeat(rows["split"] == "train", 2)
    print(f"read {H.shape[0]} variants x {H.shape[1]} chromosomes ({is_train.sum()} training) in {time.time() - t0:.0f} s", flush=True)
    cells = [(a.freq, a.cover)]
    if a.grid:
        cells += [(f, c) for f in (0.005, 0.01, 0.02) for c in (0.9, 0.95, 0.99) if (f, c) != (a.freq, a.cover)]
    os.makedirs(RESULTS, exist_ok=True)
    gpath = os.path.join(RESULTS, "haploblocks_grid.csv")
    grid = pd.read_csv(gpath) if os.path.exists(gpath) else pd.DataFrame()
    for freq, cover in cells:
        t1, cell = time.time(), cell_name(freq, cover)
        seg, codes = segment_chromosome(H, blocks, is_train, freq, cover)
        save_npz(os.path.join(haplo, f"chr{a.chrom}_haploblocks_{cell}.npz"), iid=rows["iid"], split=rows["split"],
                 codes=codes, freq=np.array(freq), cover=np.array(cover), chrom=np.array(a.chrom), **seg)
        row = dict(chrom=a.chrom, **summarise(seg, pos, H.shape[0], cell, freq, cover), seconds=round(time.time() - t1, 1))
        grid = pd.concat([grid[(grid.get("cell") != cell) | (grid.get("chrom").astype(str) != a.chrom)] if len(grid) else grid,
                          pd.DataFrame([row])], ignore_index=True)
        grid.to_csv(gpath, index=False, float_format="%.4f")
        if (freq, cover) == (a.freq, a.cover):
            pd.DataFrame(dict(block=seg["block"], start_pos=pos[seg["start"]], end_pos=pos[seg["end"] - 1],
                              n_snps=seg["end"] - seg["start"], H=seg["H"], coverage=np.round(seg["cover"], 4))
                         ).to_csv(os.path.join(RESULTS, f"haploblocks_chr{a.chrom}_{cell}_segments.csv"), index=False)
        print(f"{cell}: {row['n_segments']} segments, H median {row['H_median']:.0f} (q90 {row['H_q90']:.0f}), "
              f"{row['features']} features ({row['features_per_snp']:.2f} per SNP), {row['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
