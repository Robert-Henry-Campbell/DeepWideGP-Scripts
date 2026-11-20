#!/usr/bin/env python3
"""Datasets for the classic DeepGP comparator: hypertension (I10), at the five training sizes.

For each training size of the PGI arm's ladder, the SNPs that the published DeepGP pipeline
gives the model:
- source: the PGI arm's REGENIE GWAS of I10 on that size's training participants;
- the lead SNPs with p <= 0.05;
- leads as plink2 keeps them when clumping at r^2 0.1 within 250 kb on the PGI arm's LD panel
  (20,000 training participants; the panel its C+T clumps on).

The genotypes of those SNPs, for that size's training participants and the tuning and test
participants, are written as a small genotype store of their own. All five sizes are built in
one pass over the main store, chromosome by chromosome. A dataset folder
`data/datasets/ard30/classic_i10_n<size>/` names each store and links its sample tables.
Afterwards, the genotypes of 200 participants on every selected SNP of chromosomes 1, 6 and 22
are compared with the main store.

With several diseases (Robert, 18 November 2026) the SNPs are those with p <= 0.05 in any of their
GWAS, clumped once with each SNP's smallest p over the diseases choosing the index SNPs: the classic
selection extended to the 30-disease model (`--diseases`, `--name`; `--rungs` restricts the sizes).
One disease gives the classic selection exactly.

Two options depart from the classic rule (Robert, 19 November 2026, for the selected-SNP arms of the LD-block
model): `--p_max` sets the threshold (0.05 by default) and `--no_clump` keeps every SNP at or below it without
clumping (PRSformer's input rule); `--source` reads the genotypes from another store than the main one, such as
a store this script built before, when the selection is a subset of it (a fraction of the time).

Prints and logs counts only, never an identifier or a genotype. Participant-level files are
written only under /mnt/sdg/robert/deepwidegp/data, owner-only.

    python build_datasets.py --threads 8
    python build_datasets.py --threads 8 --name classic30 --rungs hm3plus --diseases I10 E78 ... I21
    python build_datasets.py --threads 8 --name p01any --rungs hm3plus --diseases I10 E78 ... I21 --p_max 0.01 --no_clump
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

DATA = "/mnt/sdg/robert/deepwidegp/data"
REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
PLINK2 = "/mnt/sdg/robert/miniforge3/envs/ard-gwas/bin/plink2"      # v2.0.0-a.7.13, as the PGI arm
PANEL = f"{DATA}/pgi/ld_panel/panel"
MAIN = f"{DATA}/genostore/hm3plus"
DS = f"{DATA}/datasets/ard30"
DISEASE = "I10"
P_MAX, R2, KB = 0.05, 0.1, 250
MIN_PER_CHROM = 3                    # the token embedding convolves over three SNPs
SIZES = {"hm3plus_n13199": 13199, "hm3plus_n26287": 26287, "hm3plus_n53498": 53498,
         "hm3plus_n107743": 107743, "hm3plus": 215872}
ROW_CHUNK = 8192
SEED = 1                             # the project's single seed (the participants checked)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def table_ids(path):
    """The IID column of a sample table (tab-separated, with a header)."""
    with open(path) as f:
        col = f.readline().rstrip("\n").split("\t").index("IID")
        return [line.rstrip("\n").split("\t")[col] for line in f if line.strip()]


def private(path):
    """Owner-only, as the other stores: directories 0700, files 0600."""
    for root, dirs, files in os.walk(path):
        os.chmod(root, 0o700)
        for f in files:
            os.chmod(os.path.join(root, f), 0o600)


def min_p(tables):
    """One [SNP, P] table from several (the same SNPs, in any order): each SNP's smallest finite p over
    them, NaN where none is finite."""
    first = tables[0].set_index("SNP")["P"]
    cols = [first]
    for t in tables[1:]:
        other = t.set_index("SNP")["P"]
        if len(other) != len(first) or not other.index.isin(first.index).all():
            raise ValueError("the summary statistics do not hold the same SNPs")
        cols.append(other.reindex(first.index))
    p = np.vstack([c.to_numpy(dtype=float) for c in cols])
    best = np.where(np.isfinite(p), p, np.inf).min(axis=0)
    return pd.DataFrame({"SNP": first.index.to_numpy(), "P": np.where(np.isfinite(best), best, np.nan)})


def threshold_ids(ss, p_max):
    """The SNPs of a [SNP, P] table with a finite p at or below p_max: the selection without clumping."""
    p = ss["P"].to_numpy(dtype=float)
    return set(ss["SNP"].to_numpy()[np.isfinite(p) & (p <= p_max)])


def selection_rule(p_max, clump, r2=R2, kb=KB):
    return f"p <= {p_max:g} in any disease's GWAS, " + (f"clumped once on the smallest p (r2 {r2:g}, {kb} kb)" if clump else "not clumped")


def leads(rung, diseases, name, workdir, threads, p_max=P_MAX, clump=True, r2=R2, kb=KB):
    """The selected SNPs of one training size, from each SNP's smallest p over the diseases' GWAS (one disease:
    that GWAS). Clumped: plink2's leads on the LD panel; with --clump-p1 p_max and --clump-p2 1 they are the leads
    that the greedy clumping of all SNPs gives (as the PGI arm's C+T), kept where p <= p_max. Not clumped: every
    SNP with p <= p_max."""
    tables = []
    for d in diseases:
        t = pd.read_csv(f"{DATA}/pgi/gwas_sumstats/{rung}/{d}.sumstats.tsv.gz", sep="\t",
                        usecols=["CHR", "SNP", "P"], dtype={"SNP": str, "CHR": str})
        if t["SNP"].duplicated().any():
            raise ValueError(f"{rung} {d}: repeated SNP ids in the summary statistics")
        tables.append(t)
    n_tested = len(tables[0])
    ss = min_p([t[["SNP", "P"]] for t in tables])
    ss = ss[np.isfinite(ss["P"].to_numpy(dtype=float))]
    inp = os.path.join(workdir, f"{rung}_{name}_p.tsv")
    ss[["SNP", "P"]].to_csv(inp, sep="\t", index=False, float_format="%.17g")
    p_by_id = dict(zip(ss["SNP"], ss["P"]))
    counts = {"snps_tested": int(n_tested), "snps_p_at_most_threshold": int((ss["P"] <= p_max).sum())}
    if not clump:
        ids = threshold_ids(ss, p_max)
        return ids, {**counts, "lead_snps": int(len(ids))}, p_by_id
    out = os.path.join(workdir, f"{rung}_{name}")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    cmd = [PLINK2, "--bfile", PANEL, "--clump", inp, "--clump-id-field", "SNP", "--clump-p-field", "P",
           "--clump-p1", f"{p_max:g}", "--clump-p2", "1", "--clump-r2", f"{r2:g}", "--clump-kb", str(kb),
           "--threads", str(threads), "--memory", "8000", "--out", out]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{rung}: plink2 --clump failed (exit {r.returncode}); see {out}.log")
    if os.path.exists(out + ".clumps.missing_id"):
        raise RuntimeError(f"{rung}: plink2 could not find some SNPs in the panel ({out}.clumps.missing_id)")
    ids = pd.read_csv(out + ".clumps", sep="\t", usecols=["ID"], dtype={"ID": str})["ID"].to_numpy()
    if len(set(ids)) != len(ids) or any(p_by_id[i] > p_max for i in ids):
        raise RuntimeError(f"{rung}: plink2's leads are not distinct SNPs with p <= {p_max:g}")
    return set(ids), {**counts, "lead_snps": int(len(ids))}, p_by_id


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--work", default=f"{DATA}/classic_deepgp_work")
    ap.add_argument("--diseases", nargs="+", default=[DISEASE],
                    help="the GWAS whose p <= --p_max SNPs are kept (several: in any of them, clumped once on the smallest p)")
    ap.add_argument("--rungs", nargs="+", default=list(SIZES), choices=list(SIZES), help="the training sizes to build")
    ap.add_argument("--name", default="classic_i10", help="dataset and store prefix: <name>_n<size>")
    ap.add_argument("--p_max", type=float, default=P_MAX, help="the p-value threshold (the classic rule: 0.05)")
    ap.add_argument("--no_clump", action="store_true", help="keep every SNP at or below --p_max, without clumping")
    ap.add_argument("--r2", type=float, default=R2, help="clumping r2 (the classic rule 0.1; the published paper 0.7)")
    ap.add_argument("--kb", type=int, default=KB, help="clumping window in kb (the classic rule 250; the published paper 500)")
    ap.add_argument("--source", default=MAIN,
                    help="the store the genotypes are read from (one built here before, when the selection is a subset of it)")
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    clump = not a.no_clump
    rule = selection_rule(a.p_max, clump, a.r2, a.kb)
    sizes = {r: SIZES[r] for r in a.rungs}
    sys.path.insert(0, REPO)
    from genostore import format as gf
    from genostore.store import GenoStore

    os.makedirs(a.work, mode=0o700, exist_ok=True)
    main_store = GenoStore(a.source)
    chroms = [str(c) for c in main_store.chroms]
    variants = {c: main_store.variants(c) for c in chroms}
    row_of = {str(s): i for i, s in enumerate(main_store.samples)}
    val_ids, test_ids = table_ids(f"{DS}/hm3plus/samples_val.tsv"), table_ids(f"{DS}/hm3plus/samples_test.tsv")

    # ---- the SNPs and participants of each size
    plan, logs = {}, {}
    for rung, size in sizes.items():
        lead_ids, counts, p_by_id = leads(rung, a.diseases, a.name, a.work, a.threads, a.p_max, clump, a.r2, a.kb)
        sel, topped = {}, {}
        for c in chroms:
            ids_c = variants[c]["id"]
            # set lookups: np.isin on string (object) arrays compares every pair, hours per chromosome
            idx = np.flatnonzero(np.fromiter((i in lead_ids for i in ids_c), dtype=bool, count=len(ids_c)))
            if len(idx) < MIN_PER_CHROM:              # a sequence needs a few SNPs: the next strongest
                p_c = np.array([p_by_id.get(i, np.inf) for i in ids_c])
                extra = [j for j in np.argsort(p_c, kind="stable") if j not in set(idx)][:MIN_PER_CHROM - len(idx)]
                topped[c] = len(extra)
                idx = np.sort(np.concatenate([idx, np.asarray(extra, dtype=np.int64)]))
            sel[c] = idx.astype(np.int64)
        found = sum(sum(1 for i in variants[c]["id"] if i in lead_ids) for c in chroms)
        if found != len(lead_ids):
            raise RuntimeError(f"{rung}: {len(lead_ids) - found} lead SNPs are not in the genotype store")
        train_ids = table_ids(f"{DS}/{rung}/samples_train.tsv")
        if len(train_ids) != size:
            raise RuntimeError(f"{rung}: {len(train_ids)} training participants, expected {size}")
        ids = train_ids + val_ids + test_ids
        rows = np.sort(np.array([row_of[s] for s in ids], dtype=np.int64))
        if len(np.unique(rows)) != len(rows):
            raise RuntimeError(f"{rung}: a participant is in two of the sample tables")
        plan[rung] = {"sel": sel, "rows": rows}
        logs[rung] = {**counts, "snps": int(sum(len(v) for v in sel.values())), "chromosomes_topped_up": topped,
                      "diseases": a.diseases, "selection": rule,
                      "participants": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}}
        log(f"{rung}: {counts['lead_snps']} SNPs selected ({rule}; {counts['snps_p_at_most_threshold']} SNPs with "
            f"p <= {a.p_max:g} of {counts['snps_tested']}); {len(rows)} participants")

    # ---- one pass over the main store, chromosome by chromosome
    union = np.unique(np.concatenate([p["rows"] for p in plan.values()]))
    out_store = {r: f"{DATA}/genostore/{a.name}_n{sizes[r]}" for r in sizes}
    for r, path in out_store.items():
        if os.path.exists(path):
            raise RuntimeError(f"{path} exists: move it aside before building again")
        ids_sorted = [str(main_store.samples[i]) for i in plan[r]["rows"]]
        gf.create_store(path, ids_sorted, gf.BLOCK_VARIANTS,
                        {"selected_from": a.source, "diseases": a.diseases, "gwas": f"{DATA}/pgi/gwas_sumstats/{r}",
                         "selection": rule, "p_max": a.p_max, "clump_r2": a.r2 if clump else None,
                         "clump_kb": a.kb if clump else None, "ld_panel": PANEL if clump else None})
    w = main_store.w
    for c in chroms:
        t0 = time.time()
        need = np.unique(np.concatenate([plan[r]["sel"][c] for r in sizes]))
        arrays = {r: np.empty((len(plan[r]["rows"]), len(plan[r]["sel"][c])), dtype=np.uint8) for r in sizes}
        for b0 in range(0, main_store.n_variants(c), w):
            b1 = min(main_store.n_variants(c), b0 + w)
            cols = need[(need >= b0) & (need < b1)]
            if not len(cols):
                continue
            for i0 in range(0, len(union), ROW_CHUNK):
                urows = union[i0:i0 + ROW_CHUNK]
                q = main_store.read_q(c, urows, b0, b1)
                for r in sizes:
                    rows_r, sel_r = plan[r]["rows"], plan[r]["sel"][c]
                    in_r = np.isin(urows, rows_r)
                    want = (sel_r >= b0) & (sel_r < b1)
                    if not in_r.any() or not want.any():
                        continue
                    ri = np.searchsorted(rows_r, urows[in_r])
                    arrays[r][np.ix_(ri, np.flatnonzero(want))] = q[np.ix_(np.flatnonzero(in_r), sel_r[want] - b0)]
        for r in sizes:
            sel_r = plan[r]["sel"][c]
            v = variants[c]
            lines = (f"{c}\t{int(v['pos'][j])}\t{v['id'][j]}\t{v['ref'][j]}\t{v['alt'][j]}" for j in sel_r)
            gf.write_chrom_from_array(out_store[r], c, arrays[r], lines, gf.BLOCK_VARIANTS, pos=np.arange(len(sel_r)))
        log(f"chr{c}: {len(need)} SNPs over all sizes, {time.time() - t0:.0f} s")
        del arrays

    # ---- check, dataset folders, logs
    rng = np.random.default_rng(SEED)
    for r, path in out_store.items():
        private(path)
        new = GenoStore(path)
        rows_r = plan[r]["rows"]
        pick = np.sort(rng.choice(len(rows_r), size=min(200, len(rows_r)), replace=False))
        bad = 0
        for c in ("1", "6", "22"):
            sel_c = plan[r]["sel"][c]
            got = new.read_q(c, pick)
            ref = main_store.read_q(c, rows_r[pick])[:, sel_c]
            bad += int((got != ref).sum()) + int(got.shape != ref.shape)
        if bad:
            raise RuntimeError(f"{path}: {bad} genotypes differ from the main store")
        spec = json.load(open(f"{DS}/{r}/store.json"))
        out = f"{DS}/{a.name}_n{sizes[r]}"
        os.makedirs(out, mode=0o700, exist_ok=True)
        with open(f"{out}/store.json", "w") as f:
            json.dump({"store": path, "pos_mode": "index", "labels": spec["labels"], "covariates": spec["covariates"]},
                      f, indent=1)
        links = {"samples_train.tsv": f"{DS}/{r}/samples_train.tsv", "samples_val.tsv": f"{DS}/hm3plus/samples_val.tsv",
                 "samples_test.tsv": f"{DS}/hm3plus/samples_test.tsv"}
        for name, target in links.items():
            dst = f"{out}/{name}"
            if os.path.islink(dst) or os.path.exists(dst):
                os.unlink(dst)
            os.symlink(os.path.realpath(target), dst)
        logs[r].update(built=time.strftime("%Y-%m-%d %H:%M:%S"), store=path, dataset=out,
                       check="0 differing genotypes, 200 participants, every selected SNP of chromosomes 1, 6 and 22")
        with open(f"{out}/build_log.json", "w") as f:
            json.dump(logs[r], f, indent=1)
        log(f"{r}: dataset {out} ({logs[r]['snps']} SNPs), check passed")
    print(json.dumps({r: {k: v for k, v in logs[r].items() if k in ("lead_snps", "snps", "chromosomes_topped_up")}
                      for r in sizes}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
