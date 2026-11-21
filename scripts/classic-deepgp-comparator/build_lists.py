"""SNP lists for the classic DeepGP ladder (methods.md, section "Ladder runs"): one dataset folder per (disease,
rung) that reads the main HapMap3+ store, with no genotypes copied. For each cell, the rung's own GWAS of that
disease (the PGI arm's), p <= 0.05, clumped at r2 0.7 within 500 kb on the LD panel (build_datasets.leads), gives
the SNPs; each SNP's sign turns its dosage into the risk allele's (log odds ratio > 0). Writes
<out>/<disease>/<rung>/{store.json, snps.npz, build_log.json} and links the rung's sample tables; the training SNP
statistics are shared per rung in <out>/stats/<rung>. Prints aggregates only. Resumes: finished cells are skipped.
Usage: python build_lists.py --out <dir> [--rungs ...] [--diseases ...] [--threads 8]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_datasets as bd                                         # leads(), the comparator's clumping

P_MAX, R2, KB = 0.05, 0.7, 500                                       # the paper's rule (Zhu et al. 2026)


def risk_list(lead_ids, ss, variants):
    """Store indices (sorted, per chromosome) and signs of the lead SNPs. ss: sumstats [SNP, A1, A2, BETA];
    variants: {chrom: store.variants(chrom)}. Sign +1 when the store's ALT allele raises risk, -1 when REF does."""
    ss = ss.set_index("SNP")
    out, n_neg = {}, 0
    for c, v in variants.items():
        ids = np.asarray(v["id"]).astype(str)
        keep = np.flatnonzero(np.isin(ids, list(lead_ids)))
        if len(keep) < bd.MIN_PER_CHROM:
            raise ValueError(f"chr{c}: {len(keep)} lead SNPs, fewer than {bd.MIN_PER_CHROM}")
        s = ss.reindex(ids[keep])
        alt, ref = np.asarray(v["alt"])[keep].astype(str), np.asarray(v["ref"])[keep].astype(str)
        a1 = s["A1"].to_numpy(str)
        if not np.all((a1 == alt) | (a1 == ref)) or s["BETA"].isna().any():
            raise ValueError(f"chr{c}: summary-statistic alleles do not match the store's")
        beta_alt = np.where(a1 == alt, s["BETA"].to_numpy(float), -s["BETA"].to_numpy(float))
        sign = np.where(beta_alt < 0, -1.0, 1.0).astype(np.float32)
        out[f"idx_{c}"], out[f"sign_{c}"] = keep.astype(np.int64), sign
        n_neg += int((sign < 0).sum())
    found = sum(len(v) for k, v in out.items() if k.startswith("idx_"))
    if found != len(lead_ids):
        raise ValueError(f"{len(lead_ids) - found} lead SNPs are not in the store")
    return out, n_neg


def build_cell(out, disease, rung, store, variants, threads, tables=None):
    d = os.path.join(out, disease, rung)
    if os.path.exists(os.path.join(d, "build_log.json")):
        return None
    os.makedirs(d, mode=0o700, exist_ok=True)
    work = os.path.join(out, "clump")
    os.makedirs(work, mode=0o700, exist_ok=True)
    t0 = time.time()
    ids, counts, _ = bd.leads(rung, [disease], f"ladder_{disease}", work, threads, P_MAX, True, R2, KB)
    ss = pd.read_csv(f"{bd.DATA}/pgi/gwas_sumstats/{rung}/{disease}.sumstats.tsv.gz", sep="\t",
                     usecols=["SNP", "A1", "A2", "BETA"], dtype={"SNP": str})
    lst, n_neg = risk_list(ids, ss, variants)
    np.savez(os.path.join(d, "snps.npz"), **lst)
    tables = tables or os.path.join(bd.DS, rung)
    for split in ("train", "val", "test"):
        link = os.path.join(d, f"samples_{split}.tsv")
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(os.path.join(tables, f"samples_{split}.tsv")), link)
    spec = json.load(open(os.path.join(tables, "store.json")))
    stats = os.path.join(out, "stats", rung)
    os.makedirs(stats, mode=0o700, exist_ok=True)
    json.dump({"store": store, "pos_mode": "pos", "labels": spec["labels"], "covariates": spec["covariates"],
               "snps": "snps.npz", "stats_dir": stats}, open(os.path.join(d, "store.json"), "w"), indent=1)
    log = {**counts, "snps_turned_to_ref": n_neg, "selection": f"p <= {P_MAX:g}, clumped at r2 {R2:g} within {KB} kb",
           "per_chrom": {k[4:]: int(len(v)) for k, v in lst.items() if k.startswith("idx_")},
           "seconds": round(time.time() - t0, 1)}
    json.dump(log, open(os.path.join(d, "build_log.json"), "w"), indent=1)
    for f in os.listdir(d):
        if not os.path.islink(os.path.join(d, f)):
            os.chmod(os.path.join(d, f), 0o600)
    return log


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rungs", nargs="+", default=list(bd.SIZES))
    ap.add_argument("--diseases", nargs="+")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tables", help="the folder whose store.json and sample tables the cells use (default: data/datasets/ard30/<rung>)")
    a = ap.parse_args(argv)
    sys.path.insert(0, bd.REPO)
    from genostore.store import GenoStore
    store = bd.MAIN
    gs = GenoStore(store)
    variants = {c: gs.variants(c) for c in map(str, gs.chroms)}
    for rung in a.rungs:
        tables = a.tables or os.path.join(bd.DS, rung)
        diseases = a.diseases or json.load(open(os.path.join(tables, "store.json")))["labels"]
        for disease in diseases:
            log = build_cell(a.out, disease, rung, store, variants, a.threads, tables)
            if log:
                print(json.dumps({"disease": disease, "rung": rung, **{k: log[k] for k in ("snps_p_at_most_threshold", "lead_snps", "snps_turned_to_ref", "seconds")}}), flush=True)


if __name__ == "__main__":
    main()
