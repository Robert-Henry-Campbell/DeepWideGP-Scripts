#!/usr/bin/env python3
"""Association statistics of the 30 diseases as per-SNP annotation vectors for the proxy dataset.

The simulated-trait control showed that the network learns from the SNPs when each SNP comes
with a signed association statistic, and not otherwise. This script makes the same thing for
the real labels, without letting any participant inform their own annotation:

    GWAS sample   the participants of the full training split that are in none of the proxy's
                  tables (so neither the proxy's training nor its tuning participants, and no
                  participant of the real test set)
    statistic     for every proxy SNP and disease, the z-score of the SNP in a linear regression
                  of case status on the standardised dosage, after regressing year of birth, sex
                  and ten principal components out of case status
    annotation    30 channels per SNP, one z-score per disease

It also measures what a linear method draws from the same statistics, as a yardstick for the
network: a clumped polygenic score per disease (lead SNPs with p < 1e-3, r2 > 0.1 within 500
SNPs removed, LD from the proxy's training participants), added to the covariates in a logistic
model fitted on the proxy's training participants and scored on its tuning participants.

Writes <data>/datasets/ard30/hm3plus_proxy8_gwas/ (store.json, annotations_gwas.npz, links to
the proxy's tables, gwas_log.json), a second dataset hm3plus_proxy8_gwasclump/ whose channels
keep the z-score only at each disease's clumped lead SNPs (all of them, p < 1e-3: the candidate
weights of the clumped score, whose threshold keeps fewer for some diseases), and
results/gwas_annotation_reference.csv. Aggregate numbers only are printed or written into the
repository. Needs one GPU for a few minutes. It stops unless the proxy's third table is
recorded (proxy_log.json) as a draw from the training split, because it chooses the score's
threshold on that table without the test lock.

    CUDA_VISIBLE_DEVICES=<gpu> python scripts/widedeepgp-pilot/gwas_annotation.py
    python scripts/widedeepgp-pilot/run_pilot.py --arms gwas_annot --rd-mode hm3plus_proxy8_gwas
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE))
from simulated_control import require_training_draw  # noqa: E402
DEEPGP_REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
DATA = "/mnt/sdg/robert/deepwidegp/data/datasets"
CLUMP_Z, CLUMP_R2, CLUMP_WINDOW = 3.29, 0.1, 500
Z_THRESHOLDS = (3.29, 4.42, 5.45)                      # p < 1e-3, 1e-5, 5e-8


def residualise(y, cov):
    """y [n, k] with the columns of cov (and an intercept) regressed out, then scaled to unit variance."""
    c = np.column_stack([np.ones(len(cov)), (cov - cov.mean(0)) / cov.std(0)])
    r = y - c @ np.linalg.lstsq(c, y, rcond=None)[0]
    return r / r.std(axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--proxy", default="hm3plus_proxy8")
    ap.add_argument("--full", default="hm3plus")
    ap.add_argument("--name", default="hm3plus_proxy8_gwas")
    a = ap.parse_args(argv)
    require_training_draw(Path(a.data) / "ard30" / a.proxy)
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, DEEPGP_REPO)
    os.chdir(DEEPGP_REPO)
    from args_generator import args_initial
    from genostore.datamodule import SNPStoreDataModule
    t0, dev = time.time(), "cuda"
    cfg = args_initial(["--label", "ard30", "--rd_mode", a.proxy, "--dm", "snps_covs", "--snp_embed", "cov",
                        "--final_pool", "atten", "--snp_stats", "train", "--batch_size", "16"])
    cfg.data_dir = a.data
    dm = SNPStoreDataModule(cfg)
    src = Path(dm.dir)
    spec = json.load(open(src / "store.json"))
    labels, covs, m = spec["labels"], spec["covariates"], int(sum(dm.seq_len))

    def genotypes(rows):
        """uint8 dosages x 100 of the proxy SNPs for these store rows, on the GPU."""
        ds = dm._dataset({"rows": np.asarray(rows), "cov": np.zeros((len(rows), 0)), "y": np.zeros(len(rows))})
        q = torch.empty((len(rows), m), dtype=torch.uint8, device=dev)
        for i in range(0, len(rows), 512):
            part = torch.cat(dm.decode_q(ds[np.arange(i, min(len(rows), i + 512))]["geno"], dev), dim=1)
            q[i:i + len(part)] = part
        return q

    # ---- the GWAS sample: training participants that are in none of the proxy's tables -----------
    full = pd.read_csv(Path(a.data) / "ard30" / a.full / "samples_train.tsv", sep="\t", dtype={"IID": str})
    used = set()
    for s in ("train", "val", "test"):
        used |= set(pd.read_csv(src / f"samples_{s}.tsv", sep="\t", usecols=["IID"], dtype={"IID": str})["IID"])
    real_test = set(pd.read_csv(Path(a.data) / "ard30" / a.full / "samples_test.tsv", sep="\t", usecols=["IID"],
                                dtype={"IID": str})["IID"])
    gw = full[~full["IID"].isin(used)].reset_index(drop=True)
    assert not set(gw["IID"]) & (used | real_test)
    n = len(gw)
    yr = torch.as_tensor(residualise(gw[labels].to_numpy(np.float64), gw[covs].to_numpy(np.float64)), dtype=torch.float32,
                         device=dev)
    q = genotypes([dm._row_of[i] for i in gw["IID"]])
    z = np.zeros((m, len(labels)), dtype=np.float32)
    for c in range(0, m, 4096):
        x = q[:, c:c + 4096].float() * 0.01
        x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
        z[c:c + 4096] = ((x.T @ yr) / np.sqrt(n)).cpu().numpy()
        del x
    del q, yr
    torch.cuda.empty_cache()
    gwas_summary = {lab: {"max_abs_z": float(np.abs(z[:, j]).max()), "snps_p_below_5e-8": int((np.abs(z[:, j]) >= 5.45).sum()),
                          "lambda_gc": float(np.median(z[:, j] ** 2) / 0.4549)} for j, lab in enumerate(labels)}

    # ---- yardstick: a clumped polygenic score from the same statistics ---------------------------
    sp = dm.splits
    n_tr, n_va, n_te = (len(sp[s]["rows"]) for s in ("train", "val", "test"))
    qp = genotypes(np.concatenate([sp["train"]["rows"], sp["val"]["rows"], sp["test"]["rows"]]))
    mean = torch.as_tensor(np.concatenate(dm.snp_mean), device=dev)
    inv_sd = torch.as_tensor(np.concatenate(dm.snp_inv_std), device=dev)

    def clump(j):
        order, removed, keep = np.argsort(-np.abs(z[:, j])), np.zeros(m, dtype=bool), []
        for s_ in order:
            if abs(z[s_, j]) < CLUMP_Z:
                break
            if removed[s_]:
                continue
            keep.append(s_)
            lo, hi = max(0, s_ - CLUMP_WINDOW), min(m, s_ + CLUMP_WINDOW + 1)
            xw = (qp[:n_tr, lo:hi].float() * 0.01 - mean[lo:hi]) * inv_sd[lo:hi]
            r = (xw.T @ xw[:, s_ - lo]) / n_tr
            removed[lo:hi] |= (r * r > CLUMP_R2).cpu().numpy()
        return np.array(keep, dtype=np.int64)

    cov = {s: np.asarray(sp[s]["cov"], dtype=np.float64) for s in ("train", "val", "test")}        # standardised by the loader
    y = {s: np.asarray(sp[s]["y"], dtype=np.float64) for s in ("train", "val", "test")}
    cut = {"train": slice(0, n_tr), "val": slice(n_tr, n_tr + n_va), "test": slice(n_tr + n_va, n_tr + n_va + n_te)}
    rows, z_clumped = [], np.zeros_like(z)
    for j, lab in enumerate(labels):
        lead = clump(j)
        z_clumped[lead, j] = z[lead, j]                  # every clumped lead: the candidate weights of the score below
        base = LogisticRegression(max_iter=2000).fit(cov["train"], y["train"][:, j])
        auc0 = roc_auc_score(y["val"][:, j], base.decision_function(cov["val"]))
        best = None
        for t in Z_THRESHOLDS:
            use = lead[np.abs(z[lead, j]) >= t]
            if len(use) == 0:
                continue
            w = torch.as_tensor(z[use, j], device=dev)
            prs = ((qp[:, use].float() * 0.01 - mean[use]) * inv_sd[use]) @ w
            prs = ((prs - prs[:n_tr].mean()) / prs[:n_tr].std()).cpu().numpy()
            fit = LogisticRegression(max_iter=2000).fit(np.column_stack([cov["train"], prs[cut["train"]]]), y["train"][:, j])
            score = lambda s: fit.decision_function(np.column_stack([cov[s], prs[cut[s]]]))
            pick = roc_auc_score(y["test"][:, j], score("test"))                # threshold chosen off the tuning set
            if best is None or pick > best[0]:
                best = (pick, t, len(use), roc_auc_score(y["val"][:, j], score("val")), roc_auc_score(y["val"][:, j], prs[cut["val"]]))
        rows.append(dict(label=lab, gwas_n=n, snps_p_below_5e_8=gwas_summary[lab]["snps_p_below_5e-8"],
                         max_abs_z=gwas_summary[lab]["max_abs_z"], lambda_gc=gwas_summary[lab]["lambda_gc"],
                         lead_snps_p_below_1e_3=int(len(lead)), covariates_auroc_tune=auc0,
                         prs_z_threshold=best[1] if best else np.nan, prs_snps=best[2] if best else 0,
                         prs_alone_auroc_tune=best[4] if best else np.nan,
                         covariates_plus_prs_auroc_tune=best[3] if best else auc0,
                         gain_from_prs=(best[3] - auc0) if best else 0.0))
    ref = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    ref.to_csv(RES / "gwas_annotation_reference.csv", index=False, float_format="%.6g")

    # ---- the datasets: the proxy with the z-scores as annotation vectors -------------------------
    def write(folder, file_name, channels, prefix):
        out_ = Path(dm.dir).parent / folder
        os.makedirs(out_, mode=0o700, exist_ok=True)
        arrays, at = {}, 0
        for seq in spec["regions"]:
            c, v0, v1 = seq[0]
            full_chrom = np.zeros((int(json.load(open(os.path.join(spec["store"], f"chr{c}.json")))["n_variants"]),
                                   len(labels)), dtype=np.float32)
            full_chrom[v0:v1] = channels[at:at + (v1 - v0)]
            arrays[f"chr{c}"] = full_chrom
            at += v1 - v0
        np.savez_compressed(out_ / file_name, **arrays)
        with open(out_ / "store.json", "w") as f:
            json.dump(dict(spec, annotations={"file": file_name, "names": [f"{prefix}_{lab}" for lab in labels]}), f)
        for name in [f"samples_{s}.tsv" for s in ("train", "val", "test")] + [p.name for p in src.glob("snp_stats_train_*.npz")]:
            dst = out_ / name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src / name, dst)
        with open(out_ / "gwas_log.json", "w") as f:
            json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "gwas_participants": n, "n_snps": m,
                       "excluded_proxy_participants": len(used), "by_disease": gwas_summary,
                       "non_zero_entries": int((channels != 0).sum()),
                       "mean_gain_from_clumped_prs": float(ref.gain_from_prs.mean())}, f, indent=1)
        return out_

    out = write(a.name, "annotations_gwas.npz", z, "z")
    write(a.name + "clump", "annotations_gwasclump.npz", z_clumped, "clumped_z")     # zero except at clumped lead SNPs
    pd.set_option("display.width", 250)
    print(ref.round(4).to_string(index=False))
    print(f"GWAS of {n} participants on {m} SNPs; mean tuning AUROC: covariates {ref.covariates_auroc_tune.mean():.4f}, "
          f"covariates + clumped score {ref.covariates_plus_prs_auroc_tune.mean():.4f} "
          f"(gain {ref.gain_from_prs.mean():+.4f}); wrote {out} in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
