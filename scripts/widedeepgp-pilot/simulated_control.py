#!/usr/bin/env python3
"""Positive control for the pilot: traits simulated on the real genotypes of the proxy dataset.

On the real labels the pilot models learn the covariates and nothing detectable from the SNPs
(genetic_signal.py). That could mean that the proxy holds too little genetic signal to find, or
that the network cannot find such signal. This script separates the two: it simulates traits
with known causal SNPs and heritability on the same genotypes, so that the signal is there by
construction and its size is known, and it measures what linear methods recover from the same
data.

    build      writes the dataset <data>/datasets/ard30/hm3plus_proxy8_sim (same participants,
               SNPs and covariates as the proxy; the labels are the simulated traits), and the
               reference AUROCs: the true genetic value (the ceiling), polygenic scores from
               marginal effects without and with LD clumping, and GBLUP. Needs one GPU for a
               few minutes.
    select     writes <...>/hm3plus_proxy8_sim_selected: the same traits and participants on
               GWAS-selected SNPs only (the clumped lead SNPs of all traits), as a second
               control of whether the unselected input is what stops learning.
    oracle     writes <...>/hm3plus_proxy8_sim_oracle: the same traits with an annotation
               channel per trait that marks its causal SNPs (train it with --annot).
    gwas       writes <...>/hm3plus_proxy8_sim_gwas: the same traits with an annotation channel
               per trait holding each SNP's signed association z-score in the training
               participants (train it with --annot).
    summarise  after the runs: every run's tuning AUROC per trait beside those references.

Traits (liability-threshold model, prevalence 20%, no covariate effect, so the AUROC is all
genetic):

    sim20      20 causal SNPs, heritability 0.3
    sim200     200 causal SNPs, heritability 0.5
    sim2000    2,000 causal SNPs, heritability 0.5
    sim20000   20,000 causal SNPs, heritability 0.5
    sim200a    200 causal SNPs, heritability 0.5, drawn 20 times more often from coding or
               conserved SNPs (so functional annotations carry information about where they are)

Causal SNPs are drawn from the proxy's 157,279 SNPs; effects on the standardised genotype are
normal; the genetic value is scaled to the stated heritability. Aggregate numbers only are
printed or written into the repository; the tables with simulated labels are participant-level
(real covariates and identifiers) and stay under the data directory.

    CUDA_VISIBLE_DEVICES=<gpu> python scripts/widedeepgp-pilot/simulated_control.py build
    python scripts/widedeepgp-pilot/run_pilot.py --experiment pilot_sim --rd-mode hm3plus_proxy8_sim \
        --arms default annot mamba3 --seeds 1 --gpus <gpu>
    python scripts/widedeepgp-pilot/simulated_control.py summarise
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
DEEPGP_REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
DATA = "/mnt/sdg/robert/deepwidegp/data/datasets"
RUNS = "/mnt/sdg/robert/deepwidegp/runs"
SEED = 20261113
PREVALENCE = 0.20
# name, causal SNPs, heritability of liability, fold enrichment of causal SNPs in coding/conserved SNPs
TRAITS = [("sim20", 20, 0.3, 1.0), ("sim200", 200, 0.5, 1.0), ("sim2000", 2000, 0.5, 1.0),
          ("sim20000", 20000, 0.5, 1.0), ("sim200a", 200, 0.5, 20.0)]
ENRICHED_IN = ("Coding_UCSC", "Conserved_LindbladToh")
Z_THRESHOLDS = (0.0, 1.96, 3.29, 4.42, 5.45)           # p < 1, 0.05, 1e-3, 1e-5, 5e-8
CLUMP_Z, CLUMP_R2, CLUMP_WINDOW = 3.29, 0.1, 500       # clumping: leads with p < 1e-3; r2 > 0.1 within 500 SNPs removed


def draw_effects(n_snps, traits, enriched, rng):
    """Effect of every SNP on every trait: [n_snps, n_traits], zero except at the causal SNPs."""
    beta = np.zeros((n_snps, len(traits)), dtype=np.float32)
    for j, (_, m, _, fold) in enumerate(traits):
        p = np.where(enriched, fold, 1.0).astype(np.float64)
        causal = rng.choice(n_snps, size=m, replace=False, p=p / p.sum())
        beta[causal, j] = rng.standard_normal(m)
    return beta


def liability_to_labels(g, h2, prevalence, rng):
    """Scale genetic values to variance h2, add noise to variance 1, call the top ``prevalence`` cases.
    Returns (labels, scaled genetic values)."""
    g = (g - g.mean(axis=0)) / g.std(axis=0) * np.sqrt(h2)
    liab = g + rng.standard_normal(g.shape) * np.sqrt(1.0 - np.asarray(h2))
    return (liab > np.quantile(liab, 1.0 - prevalence, axis=0)).astype(np.uint8), g


def auroc(score, y):
    from scipy.stats import rankdata
    pos = y > 0.5
    n1, n0 = pos.sum(), (~pos).sum()
    return float((rankdata(score)[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def build(a):
    import torch
    sys.path.insert(0, DEEPGP_REPO)
    os.chdir(DEEPGP_REPO)
    from args_generator import args_initial
    from genostore.datamodule import SNPStoreDataModule
    t0, dev = time.time(), "cuda"
    cfg = args_initial(["--label", "ard30", "--rd_mode", a.source, "--dm", "snps_covs", "--snp_embed", "cov",
                        "--final_pool", "atten", "--snp_stats", "train", "--batch_size", "16"])
    cfg.data_dir = a.data
    dm = SNPStoreDataModule(cfg)
    src, out = Path(dm.dir), Path(dm.dir).parent / a.name
    os.makedirs(out, mode=0o700, exist_ok=True)
    splits = ("train", "val", "test")
    n = {s: len(dm.splits[s]["rows"]) for s in splits}
    m = int(sum(dm.seq_len))

    # every genotype of the proxy as uint8 (dosage x 100) on the GPU: participants x SNPs
    q = torch.empty((sum(n.values()), m), dtype=torch.uint8, device=dev)
    at = 0
    for s in splits:
        ds = dm._dataset(dm.splits[s])
        for i in range(0, n[s], 512):
            geno = ds[np.arange(i, min(n[s], i + 512))]["geno"]
            part = torch.cat(dm.decode_q(geno, dev), dim=1)
            q[at:at + len(part)] = part
            at += len(part)
    mean = torch.as_tensor(np.concatenate(dm.snp_mean), device=dev)
    inv_sd = torch.as_tensor(np.concatenate(dm.snp_inv_std), device=dev)
    x_of = lambda rows: (q[rows].float() * 0.01 - mean) * inv_sd          # standardised with training-split statistics
    sl = {"train": slice(0, n["train"]), "val": slice(n["train"], n["train"] + n["val"]),
          "test": slice(n["train"] + n["val"], sum(n.values()))}

    annot, names = dm.annotations()
    enriched = np.zeros(m, dtype=bool)
    if annot is not None:
        a_all = np.concatenate(annot, axis=0)
        for ch in ENRICHED_IN:
            enriched |= a_all[:, names.index(ch)] > 0
    rng = np.random.default_rng(a.seed)
    beta = draw_effects(m, TRAITS, enriched, rng)
    b = torch.as_tensor(beta, device=dev)
    g = torch.cat([x_of(slice(i, i + 4096)) @ b for i in range(0, len(q), 4096)]).cpu().numpy().astype(np.float64)
    y, g = liability_to_labels(g, np.array([t[2] for t in TRAITS]), PREVALENCE, rng)

    # ---- what linear methods recover from the training participants ----------------------------
    ytr = torch.as_tensor(y[sl["train"]], dtype=torch.float32, device=dev)
    yc = ytr - ytr.mean(dim=0)
    xty = torch.zeros((m, len(TRAITS)), device=dev)
    for i in range(0, n["train"], 4096):
        xty += x_of(slice(i, min(n["train"], i + 4096))).T @ yc[i:i + 4096]
    z = (xty / (n["train"] * ytr.var(dim=0)).sqrt()).cpu().numpy()        # marginal association z-scores
    w = torch.as_tensor(z / np.sqrt(n["train"]), dtype=torch.float32, device=dev)
    prs = {}
    for t in Z_THRESHOLDS:
        wt = torch.where(torch.as_tensor(np.abs(z) >= t, device=dev), w, torch.zeros_like(w))
        prs[t] = torch.cat([x_of(slice(i, i + 4096)) @ wt for i in range(0, len(q), 4096)]).cpu().numpy()
    # clumping and thresholding: the strongest SNP of each LD cluster only, which is what suits few causal SNPs
    def clump(j):
        order, removed, keep = np.argsort(-np.abs(z[:, j])), np.zeros(m, dtype=bool), []
        for s_ in order:
            if abs(z[s_, j]) < CLUMP_Z:
                break
            if removed[s_]:
                continue
            keep.append(s_)
            lo, hi = max(0, s_ - CLUMP_WINDOW), min(m, s_ + CLUMP_WINDOW + 1)
            xw = (q[sl["train"], lo:hi].float() * 0.01 - mean[lo:hi]) * inv_sd[lo:hi]
            r = (xw.T @ xw[:, s_ - lo]) / n["train"]
            removed[lo:hi] |= (r * r > CLUMP_R2).cpu().numpy()
        return np.array(keep, dtype=np.int64)

    ct, leads = {}, []
    for j in range(len(TRAITS)):
        lead = clump(j)
        leads.append(lead)
        for t in Z_THRESHOLDS[2:]:
            wt = torch.zeros((m, 1), device=dev)
            use = lead[np.abs(z[lead, j]) >= t]
            wt[use, 0] = w[use, j]
            ct[(j, t)] = (torch.cat([x_of(slice(i, i + 4096)) @ wt for i in range(0, len(q), 4096)]).squeeze(1).cpu().numpy(),
                          len(use))
    # GBLUP: best linear prediction when every SNP has a small effect; ridge penalty from the true heritability
    k = torch.zeros((n["train"], n["train"]), device=dev)
    kx = torch.zeros((len(q) - n["train"], n["train"]), device=dev)
    for c in range(0, m, 8192):
        xc = (q[:, c:c + 8192].float() * 0.01 - mean[c:c + 8192]) * inv_sd[c:c + 8192]
        k += xc[sl["train"]] @ xc[sl["train"]].T
        kx += xc[n["train"]:] @ xc[sl["train"]].T
        del xc
    k /= m
    kx /= m
    gblup = np.zeros((len(q) - n["train"], len(TRAITS)))
    for j, (_, _, h2, _) in enumerate(TRAITS):
        lam = (1.0 - h2) / h2
        k.diagonal().add_(lam)                           # in place: the matrix is 6 GB
        alpha = torch.linalg.solve(k, yc[:, j:j + 1])
        k.diagonal().sub_(lam)
        gblup[:, j] = (kx @ alpha).squeeze(1).cpu().numpy()
    del k, kx

    rows = []
    va, te = sl["val"], sl["test"]
    off = n["train"]
    for j, (name, mc, h2, fold) in enumerate(TRAITS):
        best_t = max(Z_THRESHOLDS, key=lambda t: auroc(prs[t][te, j], y[te, j]))       # threshold chosen off the tuning set
        best_c = max(Z_THRESHOLDS[2:], key=lambda t: auroc(ct[(j, t)][0][te], y[te, j]) if ct[(j, t)][1] else 0.0)
        rows.append(dict(trait=name, causal_snps=mc, heritability=h2, enrichment=fold, prevalence=float(y[:, j].mean()),
                         causal_in_enriched_snps=int((beta[enriched, j] != 0).sum()),
                         cases_train=int(y[sl["train"], j].sum()), cases_tune=int(y[va, j].sum()),
                         ceiling_auroc_tune=auroc(g[va, j], y[va, j]), prs_z_threshold=best_t,
                         prs_snps=int((np.abs(z[:, j]) >= best_t).sum()), prs_auroc_tune=auroc(prs[best_t][va, j], y[va, j]),
                         clumped_z_threshold=best_c, clumped_snps=ct[(j, best_c)][1],
                         clumped_auroc_tune=auroc(ct[(j, best_c)][0][va], y[va, j]) if ct[(j, best_c)][1] else float("nan"),
                         gblup_auroc_tune=auroc(gblup[va.start - off:va.stop - off, j], y[va, j]),
                         genome_wide_significant_snps=int((np.abs(z[:, j]) >= 5.45).sum())))
    ref = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    ref.to_csv(RES / "simulated_control_references.csv", index=False, float_format="%.6g")

    # ---- the dataset ----------------------------------------------------------------------------
    spec = json.load(open(src / "store.json"))
    labels = [t[0] for t in TRAITS]
    for s in splits:
        df = pd.read_csv(src / f"samples_{s}.tsv", sep="\t", dtype={"IID": str})
        assert len(df) == n[s]
        tab = pd.concat([df[["IID"]], pd.DataFrame(y[sl[s]], columns=labels), df[spec["covariates"]]], axis=1)
        path = out / f"samples_{s}.tsv"
        tab.to_csv(path, sep="\t", index=False)
        os.chmod(path, 0o600)
    new = dict(spec, labels=labels)
    if spec.get("annotations"):
        new["annotations"] = dict(spec["annotations"], file=os.path.relpath((src / spec["annotations"]["file"]).resolve(), out))
    with open(out / "store.json", "w") as f:
        json.dump(new, f, indent=1)
    np.savez_compressed(out / "sim_truth.npz", beta=beta, traits=np.array(labels), enriched=enriched)
    np.savez_compressed(out / "sim_gwas.npz", z=z.astype(np.float32), traits=np.array(labels),
                        **{f"leads_{name}": lead for name, lead in zip(labels, leads)})
    with open(out / "sim_log.json", "w") as f:
        json.dump({"built": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(src), "seed": a.seed, "n": n, "n_snps": m,
                   "enriched_snps": int(enriched.sum()), "enriched_in": list(ENRICHED_IN), "references": rows}, f, indent=1)
    pd.set_option("display.width", 250)
    print(ref.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out} in {time.time() - t0:.0f} s; {int(enriched.sum())} of {m} SNPs are coding or conserved", flush=True)


def select(a):
    """The simulated traits again, on GWAS-selected SNPs only: the union over traits of the
    clumped lead SNPs (p < 1e-3 in the training participants), which is the kind of input
    published DeepGP is given. Same participants and labels. The selected genotypes are written
    as a small store of their own (<data>/../genostore/<name>_selected), because thousands of
    one-SNP regions of the big store are slow to read."""
    import torch
    sys.path.insert(0, DEEPGP_REPO)
    os.chdir(DEEPGP_REPO)
    from args_generator import args_initial
    from genostore import format as gf
    from genostore.datamodule import SNPStoreDataModule
    src = Path(a.data) / "ard30" / a.name
    out = Path(a.data) / "ard30" / (a.name + "_selected")
    store = Path(a.data).parent / "genostore" / (a.name + "_selected")
    os.makedirs(out, mode=0o700, exist_ok=True)
    spec = json.load(open(src / "store.json"))
    gw = np.load(src / "sim_gwas.npz")
    traits = [str(t) for t in gw["traits"]]
    keep = np.unique(np.concatenate([gw[f"leads_{t}"] for t in traits]))
    bounds = np.cumsum([0] + [seq[0][2] - seq[0][1] for seq in spec["regions"]])      # one stretch per chromosome
    strongest = np.argsort(-np.abs(gw["z"]).max(axis=1))
    chosen = []
    for i in range(len(spec["regions"])):
        mine = keep[(keep >= bounds[i]) & (keep < bounds[i + 1])]
        if len(mine) < a.min_per_chromosome:          # a sequence needs a few SNPs: top it up with the next strongest
            extra = [g for g in strongest if bounds[i] <= g < bounds[i + 1] and g not in set(mine)]
            mine = np.sort(np.concatenate([mine, extra[:a.min_per_chromosome - len(mine)]]).astype(np.int64))
        chosen.append(mine)

    # the selected genotypes of the proxy's participants, in the order train, tune, third table
    cfg = args_initial(["--label", "ard30", "--rd_mode", a.name, "--dm", "snps_covs", "--snp_embed", "cov",
                        "--final_pool", "atten", "--snp_stats", "store", "--batch_size", "16"])
    cfg.data_dir = a.data
    dm = SNPStoreDataModule(cfg)
    cols = torch.as_tensor(np.concatenate(chosen))
    ids, parts = [], []
    for s_ in ("train", "val", "test"):
        sp = dm.splits[s_]
        ids += [dm.store.samples[r] for r in sp["rows"]]
        ds = dm._dataset(sp)
        for i in range(0, len(sp["rows"]), 512):
            geno = ds[np.arange(i, min(len(sp["rows"]), i + 512))]["geno"]
            parts.append(torch.cat(dm.decode_q(geno, "cpu"), dim=1)[:, cols].numpy())
    q = np.concatenate(parts, axis=0)
    if store.exists():
        import shutil
        shutil.rmtree(store)
    gf.create_store(str(store), [str(i) for i in ids], gf.BLOCK_VARIANTS, {"selected_from": str(src), "snps": int(q.shape[1])})
    at, per_chrom = 0, {}
    for i, seq in enumerate(spec["regions"]):
        c, v0, _ = seq[0]
        k = len(chosen[i])
        lines = (f"{c}\t{int(v0 + g - bounds[i])}\tproxy_snp_{int(g)}\tN\tN" for g in chosen[i])
        gf.write_chrom_from_array(str(store), str(c), np.ascontiguousarray(q[:, at:at + k]), lines, gf.BLOCK_VARIANTS,
                                  pos=np.arange(k))
        at += k
        per_chrom[str(c)] = int(k)

    new = {"store": str(store), "pos_mode": "index", "labels": spec["labels"], "covariates": spec["covariates"]}
    with open(out / "store.json", "w") as f:
        json.dump(new, f, indent=1)
    for s_ in ("train", "val", "test"):
        dst = out / f"samples_{s_}.tsv"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src / f"samples_{s_}.tsv", dst)
    for stale in out.glob("snp_stats_train_*.npz"):
        stale.unlink()
    truth = np.load(src / "sim_truth.npz")
    causal_kept = {t: int(np.isin(np.flatnonzero(truth["beta"][:, j]), keep).sum()) for j, t in enumerate(traits)}
    log = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(src), "store": str(store), "n_snps": int(q.shape[1]),
           "n_participants": int(q.shape[0]), "snps_by_chromosome": per_chrom,
           "lead_snps_by_trait": {t: int(len(gw[f"leads_{t}"])) for t in traits},
           "causal_snps_among_the_selected": causal_kept}
    with open(out / "select_log.json", "w") as f:
        json.dump(log, f, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "snps_by_chromosome"}, indent=1))


def _with_annotation(a, suffix, file_name, channels, names):
    """A copy of the simulated dataset (same participants, labels and SNPs) whose annotation
    vectors are ``channels`` [proxy SNPs, k]."""
    src = Path(a.data) / "ard30" / a.name
    out = Path(a.data) / "ard30" / (a.name + suffix)
    os.makedirs(out, mode=0o700, exist_ok=True)
    spec = json.load(open(src / "store.json"))
    arrays, at = {}, 0
    for seq in spec["regions"]:
        c, v0, v1 = seq[0]
        full = np.zeros((int(json.load(open(os.path.join(spec["store"], f"chr{c}.json")))["n_variants"]), channels.shape[1]),
                        dtype=np.float32)
        full[v0:v1] = channels[at:at + (v1 - v0)]
        arrays[f"chr{c}"] = full
        at += v1 - v0
    np.savez_compressed(out / file_name, **arrays)
    with open(out / "store.json", "w") as f:
        json.dump(dict(spec, annotations={"file": file_name, "names": names}), f)
    for s_ in ("train", "val", "test"):
        dst = out / f"samples_{s_}.tsv"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src / f"samples_{s_}.tsv", dst)
    for f_ in src.glob("snp_stats_train_*.npz"):                               # same participants and SNPs
        if not (out / f_.name).exists():
            os.symlink(f_, out / f_.name)
    return out


def oracle(a):
    """The simulated traits with an annotation that says where the causal SNPs are: one channel
    per trait, 1 at its causal SNPs. It does not say in which direction each acts."""
    truth = np.load(Path(a.data) / "ard30" / a.name / "sim_truth.npz")
    traits = [str(t) for t in truth["traits"]]
    causal = (truth["beta"] != 0).astype(np.float32)                           # [proxy SNPs, traits]
    out = _with_annotation(a, "_oracle", "annotations_oracle.npz", causal, [f"causal_{t}" for t in traits])
    print(f"wrote {out}: {int(causal.sum())} causal marks over {len(traits)} channels")


def gwas(a):
    """The simulated traits with signed association statistics as annotation: one channel per
    trait holding every SNP's marginal z-score in the training participants. Unlike a mark on
    the causal SNPs, a z-score says which way a SNP's allele points, so the annotation gain can
    weight every SNP's dosage by size and sign. The statistics come from the training
    participants only."""
    gw = np.load(Path(a.data) / "ard30" / a.name / "sim_gwas.npz")
    traits = [str(t) for t in gw["traits"]]
    out = _with_annotation(a, "_gwas", "annotations_gwas.npz", gw["z"].astype(np.float32), [f"z_{t}" for t in traits])
    print(f"wrote {out}: z-scores of {gw['z'].shape[0]} SNPs for {len(traits)} traits; |z| up to {np.abs(gw['z']).max():.1f}")


def summarise(a):
    ref = pd.read_csv(RES / "simulated_control_references.csv").set_index("trait")
    rows = []
    for d in sorted((Path(a.runs) / a.experiment).iterdir()):
        if not (d / "results.json").exists():
            continue
        r = json.load(open(d / "results.json"))
        arm, seed = d.name.rsplit("_s", 1)
        m = pd.read_csv(d / "metrics.csv")
        for trait, v in r["val"].items():
            c = m[(m.label == trait) & (m.metric == "auroc")]
            rows.append(dict(arm=arm, seed=int(seed), trait=trait, auroc_at_best_checkpoint=v["auroc"],
                             auroc_best_of_run=float(c.value.max()), auroc_last=float(c.value.iloc[-1]),
                             ceiling=ref.loc[trait, "ceiling_auroc_tune"], prs=ref.loc[trait, "prs_auroc_tune"],
                             clumped_prs=ref.loc[trait, "clumped_auroc_tune"],
                             gblup=ref.loc[trait, "gblup_auroc_tune"], epoch_equivalents=r["epoch_equivalents"],
                             train_minutes=r["train_seconds"] / 60.0))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "simulated_control_runs.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    if len(out):
        print(out.pivot_table(index="trait", columns=["arm", "seed"], values="auroc_best_of_run").round(4).to_string())
    print(ref[["causal_snps", "heritability", "ceiling_auroc_tune", "prs_auroc_tune", "clumped_auroc_tune",
               "gblup_auroc_tune"]].round(4).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "select", "oracle", "gwas", "summarise"])
    ap.add_argument("--min-per-chromosome", type=int, default=16)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--source", default="hm3plus_proxy8")
    ap.add_argument("--name", default="hm3plus_proxy8_sim")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--experiment", default="pilot_sim")
    a = ap.parse_args(argv)
    {"build": build, "select": select, "oracle": oracle, "gwas": gwas, "summarise": summarise}[a.command](a)


if __name__ == "__main__":
    main()
