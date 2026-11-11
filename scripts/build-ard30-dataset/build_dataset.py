#!/usr/bin/env python3
"""Build the multi-disease dataset for DeepWideGP: labels, covariates, splits and SNP annotations.

Labels are built with the case-definition code of ``find-ukb-ard-phenos`` for the phenotypes
listed in its ``results/ukb_ard_selected_phenos_overview.csv``, and the case counts of the
result must equal the counts in that file exactly, or the build stops.

Written under <data_dir> (participant-level; nothing is written into a git repository):

    datasets/ard30/hm3plus/
        store.json                  points at the genotype store; names labels, covariates, annotations
        samples_train.tsv           215,872 unrelated White-British participants
        samples_val.tsv             20,000 (the tune set)
        samples_test.tsv            40,000
        samples_wb_related.tsv      related White-British participants: held out, used for nothing yet
        labels.json                 per phenotype: codes, title, definition, counts by split
        annotations.npz             per-chromosome [SNPs, annotations] matrices, in store order
        build_log.json              participant flow, checks, checksums (aggregate only)
    datasets/ard30/hm3plus_asian_black/
        samples_asian.tsv, samples_black.tsv   the second genotype set: held out, used for nothing yet

Only aggregate numbers are printed. The scientific description is in methods.md next to this script.

    /home/robert/envs/widedeepgp/bin/python scripts/build-ard30-dataset/build_dataset.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "find-ukb-ard-phenos"))
import find_ukb_ard_phenos as fp  # noqa: E402  (the case definition lives there)

DEFAULTS = {
    "data_dir": "/mnt/sdg/robert/deepwidegp/data",
    "pheno": fp.DEFAULTS["pheno"],
    "withdrawals": fp.DEFAULTS["withdrawals"],
    "psam": fp.DEFAULTS["psam"],
    "psam_other": "/mnt/sdh/upamanyu/UKB/Genomics_PGEN/Asian_Black/UKB_chr1.psam",
    "asian_ids": "/mnt/sdh/upamanyu/UKB/Genomics_PGEN/Asian_Black/UKB_Asian.grm.id",
    "black_ids": "/mnt/sdh/upamanyu/UKB/Genomics_PGEN/Asian_Black/UKB_Black.grm.id",
    "selected": str(HERE.parent / "find-ukb-ard-phenos" / "results" / "ukb_ard_selected_phenos_overview.csv"),
    "annot": "/mnt/sdg/robert/ardmr/GSEM/reference/postgwas/sbayesrc/annot_baseline2.2.txt",
    "snps": None,       # default: <data_dir>/reference/hm3plus_in_ukb_pgen.tsv
    "store": None,      # default: <data_dir>/genostore/hm3plus
}
N_TUNE, N_TEST, SPLIT_SEED = 20_000, 40_000, 20261111
PCS = [f"22009.0.{i}" for i in range(1, 11)]
COVARIATES = ["birth_year", "sex"] + [f"pc{i}" for i in range(1, 11)]


def md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def read_ids(path, col=1):
    """Second column (IID) of a whitespace-delimited ID file; a '#...' header line is skipped."""
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if p and not line.startswith("#"):
                out.append(p[col])
    return out


def read_selected(path):
    """[(label, (codes...), title)] from the selection table; a label may unite several categories."""
    t = pd.read_csv(path, dtype=str)
    need = {"phenotype", "icd10_codes", "icd10_title", "n_cases__geno_wb_unrel"}
    if not need <= set(t.columns):
        raise ValueError(f"{path} lacks columns {sorted(need - set(t.columns))}")
    phenos = [(r.phenotype, tuple(c for c in r.icd10_codes.split(";") if c), r.icd10_title) for r in t.itertuples()]
    codes = [c for _, cs, _ in phenos for c in cs]
    if len(set(codes)) != len(codes):
        raise ValueError("an ICD-10 category is listed under more than one phenotype")
    return phenos, t.set_index("phenotype")


def label_matrix(records, eids, phenos):
    """uint8 [len(eids), len(phenos)]: 1 if the participant has a record of any of the phenotype's
    categories in any of the three linked sources (the primary definition of find-ukb-ard-phenos)."""
    code_to_label = {c: j for j, (_, cs, _) in enumerate(phenos) for c in cs}
    row = {e: i for i, e in enumerate(eids)}
    y = np.zeros((len(eids), len(phenos)), dtype=np.uint8)
    for rec in records.values():
        r = rec.loc[rec["cat"].isin(code_to_label), ["eid", "cat"]]
        i = r["eid"].map(row)
        ok = i.notna().to_numpy()
        y[i[ok].astype(np.int64).to_numpy(), r["cat"].map(code_to_label)[ok].to_numpy()] = 1
    return y


def decimal_year(dates):
    d = pd.to_datetime(dates, errors="coerce")
    return d.dt.year + (d.dt.dayofyear - 1) / 365.25


def birth_year(assessment_date, age_at_recruitment):
    """Year of birth reconstructed from the date of the first assessment and age in completed
    years at that date. The true date of birth lies in the year ending at (assessment - age);
    the midpoint of that year is returned, so the error is at most half a year."""
    return decimal_year(assessment_date) - pd.to_numeric(age_at_recruitment, errors="coerce") - 0.5


def split_ids(eids, n_tune=N_TUNE, n_test=N_TEST, seed=SPLIT_SEED):
    """Disjoint test / tune / train sets from a seeded permutation of the sorted IDs."""
    ids = np.array(sorted(eids, key=int))
    perm = np.random.default_rng(seed).permutation(len(ids))
    return {"test": set(ids[perm[:n_test]]), "val": set(ids[perm[n_test:n_test + n_tune]]),
            "train": set(ids[perm[n_test + n_tune:]])}


def build_annotations(annot_path, snps, out_path):
    """Per-chromosome annotation matrices in store order. SNPs absent from the table get zeros
    and a 1 in the last channel ('no_annotation')."""
    want = set(snps["ID"])
    parts, names = [], None
    for chunk in pd.read_csv(annot_path, sep="\t", chunksize=400_000):
        if names is None:
            names = [c for c in chunk.columns if c not in ("SNP", "Intercept")]
        parts.append(chunk.loc[chunk["SNP"].isin(want), ["SNP"] + names])
    a = pd.concat(parts, ignore_index=True).drop_duplicates("SNP").set_index("SNP")
    out, covered = {}, 0
    for chrom, g in snps.groupby("CHROM", sort=False):
        m = a.reindex(g["ID"].to_numpy())
        miss = m.iloc[:, 0].isna().to_numpy()
        arr = np.concatenate([m.fillna(0.0).to_numpy(dtype=np.float32), miss[:, None].astype(np.float32)], axis=1)
        out[f"chr{chrom}"] = arr
        covered += int((~miss).sum())
    np.savez_compressed(out_path, **out)
    return names + ["no_annotation"], covered


def write_table(path, eids, y, cov, labels):
    t = pd.DataFrame(y, columns=labels)
    t.insert(0, "IID", list(eids))
    for k, name in enumerate(COVARIATES):
        t[name] = cov[:, k]
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        t.to_csv(f, sep="\t", index=False, float_format="%.8g")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v)
    ap.add_argument("--skip-annotations", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    data_dir = Path(a.data_dir)
    snps_path = a.snps or str(data_dir / "reference" / "hm3plus_in_ukb_pgen.tsv")
    store = a.store or str(data_dir / "genostore" / "hm3plus")
    out = data_dir / "datasets" / "ard30" / "hm3plus"
    out_other = data_dir / "datasets" / "ard30" / "hm3plus_asian_black"
    for d in (out, out_other):
        os.makedirs(d, mode=0o700, exist_ok=True)

    phenos, selected = read_selected(a.selected)
    labels = [p[0] for p in phenos]
    print(f"{len(labels)} phenotypes from {a.selected}", flush=True)

    # ---- cohort and labels: the code and definitions of find-ukb-ard-phenos ----------------------
    df, death_cols, _ = fp.load_pheno(a.pheno)
    withdrawn = set(pd.read_csv(a.withdrawals, header=None, dtype=str)[0].str.strip())
    main_ids, other_ids = read_ids(a.psam), read_ids(a.psam_other)
    flags, flow = fp.define_cohorts(df, set(main_ids), withdrawn)
    records, _ = fp.extract_records(df, death_cols)
    del df
    eids = flags["eid"].to_numpy()
    y = label_matrix(records, eids, phenos)
    del records

    checks = {}
    for cohort in ("geno_wb_unrel", "geno_wb", "ukb_all"):
        got = y[flags[cohort].to_numpy()].sum(axis=0)
        want = selected.loc[labels, f"n_cases__{cohort}"].astype(int).to_numpy()
        checks[f"case_counts_equal_selection_table__{cohort}"] = bool((got == want).all())
        if not (got == want).all():
            bad = [f"{l}: built {g}, table {w}" for l, g, w in zip(labels, got, want) if g != w]
            raise SystemExit(f"case counts differ from the selection table in {cohort}: {bad[:5]}")
    print("case counts equal the selection table in all three populations", flush=True)

    # ---- covariates ----------------------------------------------------------------------------
    cols = ["eid", "21022.0.0", "53.0.0", "31.0.0", "21000.0.0", *PCS]
    extra = pd.read_csv(a.pheno, sep="\t", usecols=cols, dtype=str)
    extra["eid"] = extra["eid"].str.strip()
    extra = extra.set_index("eid").reindex(eids)
    cov = np.column_stack([birth_year(extra["53.0.0"], extra["21022.0.0"]).to_numpy(),
                           pd.to_numeric(extra["31.0.0"], errors="coerce").to_numpy()]
                          + [pd.to_numeric(extra[c], errors="coerce").to_numpy() for c in PCS]).astype(np.float64)
    complete = ~np.isnan(cov).any(axis=1)

    # ---- who goes where ------------------------------------------------------------------------
    eid_set_other = set(other_ids) - withdrawn
    unrel = flags["geno_wb_unrel"].to_numpy()
    wb = flags["geno_wb"].to_numpy()
    n_unrel_incomplete = int((unrel & ~complete).sum())
    if n_unrel_incomplete:
        raise SystemExit(f"{n_unrel_incomplete} unrelated White-British participants lack a covariate")
    sp = split_ids(eids[unrel])
    assert len(sp["train"]) + len(sp["val"]) + len(sp["test"]) == int(unrel.sum())
    assert not (sp["train"] & sp["val"]) and not (sp["train"] & sp["test"]) and not (sp["val"] & sp["test"])
    asian, black = set(read_ids(a.asian_ids)) & eid_set_other, set(read_ids(a.black_ids)) & eid_set_other
    member = lambda ids: np.asarray(pd.Index(eids).isin(list(ids)))    # hashed; numpy's isin is quadratic on text arrays
    groups = {"train": member(sp["train"]), "val": member(sp["val"]), "test": member(sp["test"]),
              "wb_related": wb & ~unrel & complete,
              "asian": member(asian) & complete, "black": member(black) & complete}

    store_ids = None
    if os.path.exists(os.path.join(store, "samples.txt")):
        with open(os.path.join(store, "samples.txt")) as f:
            store_ids = {line.strip() for line in f}
        for g in ("train", "val", "test", "wb_related"):
            assert set(eids[groups[g]]) <= store_ids, f"{g}: participants missing from the genotype store"
        checks["all_main_tables_in_genotype_store"] = True

    for g, where in (("train", out), ("val", out), ("test", out), ("wb_related", out), ("asian", out_other),
                     ("black", out_other)):
        m = groups[g]
        write_table(where / f"samples_{g}.tsv", eids[m], y[m], cov[m], labels)

    # ---- manifest, store.json, annotations -----------------------------------------------------
    counts = {g: {"n": int(m.sum()), "cases": [int(v) for v in y[m].sum(axis=0)]} for g, m in groups.items()}
    manifest = []
    for j, (label, codes, title) in enumerate(phenos):
        manifest.append({
            "label": label, "icd10_codes": list(codes), "title": title,
            "definition": "a record of any listed ICD-10 category in hospital inpatient diagnoses (field 41270), "
                          "the death registry (40001, 40002) or the cancer registry (40006), at any date",
            **{f"cases_{g}": counts[g]["cases"][j] for g in counts},
            "prevalence_train": counts["train"]["cases"][j] / counts["train"]["n"]})
    with open(out / "labels.json", "w") as f:
        json.dump(manifest, f, indent=1)

    spec = {"store": store, "pos_mode": "index", "labels": labels, "covariates": COVARIATES}
    annot_info = None
    if not a.skip_annotations:
        snps = pd.read_csv(snps_path, sep="\t", usecols=["CHROM", "ID"], dtype=str)
        names, covered = build_annotations(a.annot, snps, out / "annotations.npz")
        spec["annotations"] = {"file": "annotations.npz", "names": names}
        annot_info = {"source": a.annot, "channels": len(names), "snps": int(len(snps)), "snps_covered": covered}
        print(f"annotations: {len(names)} channels, {covered} of {len(snps)} SNPs covered", flush=True)
    with open(out / "store.json", "w") as f:
        json.dump(spec, f, indent=1)
    # the second genotype set has a store of its own (same SNPs, same order); its dataset
    # directory names it, and reuses the labels, covariates and annotation matrices of the first
    spec_other = {"store": str(Path(store).parent / (Path(store).name + "_asian_black")),
                  "pos_mode": spec["pos_mode"], "labels": spec["labels"], "covariates": spec["covariates"]}
    if spec.get("annotations"):
        spec_other["annotations"] = dict(spec["annotations"],
                                         file=os.path.relpath(out / spec["annotations"]["file"], out_other))
    with open(out_other / "store.json", "w") as f:
        json.dump(spec_other, f, indent=1)
    with open(out_other / "README.txt", "w") as f:
        f.write("Labels and covariates for the participants of the second genotype set (Asian and Black),\n"
                "built with the same definitions as ../hm3plus. Held out: not used in training, tuning or testing.\n"
                "To score them with a trained model, standardise SNPs and covariates with the training-split\n"
                "statistics of that model's dataset (a run's config.json records the covariate scaling), not\n"
                "with statistics of these participants.\n")

    tr = groups["train"]
    ytr = y[tr].astype(np.float64)
    log = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"), "runtime_s": round(time.time() - t0, 1),
        "inputs": {k: {"path": p, "md5": md5(p)} for k, p in (("pheno", a.pheno), ("withdrawals", a.withdrawals),
                                                              ("selected", a.selected), ("snps", snps_path))},
        "participant_flow": flow, "groups": {g: counts[g]["n"] for g in counts},
        "second_genotype_set": {"samples": len(other_ids), "asian": len(asian), "black": len(black),
                                "dropped_incomplete_covariates": int((member(asian | black) & ~complete).sum())},
        "wb_related_dropped_incomplete_covariates": int((wb & ~unrel & ~complete).sum()),
        "split": {"seed": SPLIT_SEED, "n_tune": N_TUNE, "n_test": N_TEST, "order": "seeded permutation of sorted IDs"},
        "labels": labels, "covariates": COVARIATES,
        "covariate_summary_train": {n: {"mean": float(cov[tr][:, k].mean()), "sd": float(cov[tr][:, k].std()),
                                        "min": float(cov[tr][:, k].min()), "max": float(cov[tr][:, k].max())}
                                    for k, n in enumerate(COVARIATES)},
        "cases_by_group": {g: dict(zip(labels, counts[g]["cases"])) for g in counts},
        "diseases_per_person_train": {"mean": float(ytr.sum(1).mean()), "none": float((ytr.sum(1) == 0).mean()),
                                      "five_or_more": float((ytr.sum(1) >= 5).mean())},
        "co_occurrence_train": (ytr.T @ ytr).astype(int).tolist(),
        "annotations": annot_info, "checks": checks, "genotype_store": store,
        "store_checked": store_ids is not None,
    }
    with open(out / "build_log.json", "w") as f:
        json.dump(log, f, indent=1)
    print(f"wrote {out} in {log['runtime_s']} s: train {counts['train']['n']}, tune {counts['val']['n']}, "
          f"test {counts['test']['n']}, related held out {counts['wb_related']['n']}, "
          f"asian {counts['asian']['n']}, black {counts['black']['n']}", flush=True)
    return log


if __name__ == "__main__":
    main()
