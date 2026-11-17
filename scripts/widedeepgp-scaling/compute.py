#!/usr/bin/env python3
"""Compute accounting of a WideDeepGP blocks-mode run (plan section 1; the formulas of the shared
contract, "What a deep run records").

FLOPs per participant, one forward pass, from config.json:
  table              2 M codes d_e              M SNPs; codes 1 (dosage) or 3 (genotype); d_e = embed_dim
  block MLP          2 K (4 d_e^2 + 4 d_e d)    K blocks; d = d_model
  mixer              2 N_mixer T                N_mixer = the mixer's parameters; T = n_tokens
  attention or scan  4 L T^2 d (transformer)    or 2 directions x T x 2d x 16 x 18 per layer (bimamba1)
Training costs 3 forward passes; tokens processed = participants seen, so the cumulative training
compute of a snapshot or validation pass is C = 3 x forward FLOPs per participant x samples_seen.
The head and covariate path are left out (a few thousand FLOPs per participant); pretraining has
its own run records and is reported separately.

    python compute.py <run dir> [<run dir> ...] --out <dir>      # <out>/<run name>_compute.csv per run
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

MAMBA_STATE, MAMBA_SCAN_OPS = 16, 18          # the bimamba1 scan term of the contract: 2 x T x 2d x 16 x 18 per layer


def flops_per_participant(config):
    """Forward FLOPs per participant by term, plus "forward" and "train" (3 x forward)."""
    b, m = config["blocks"], config["model"]
    codes = 3 if b.get("snp_code") == "genotype" else 1
    M, K, T, de, d, L = config["n_snps"], b["n_blocks"], b["n_tokens"], b["embed_dim"], m["d_model"], m["e_layers"]
    out = {"table": 2 * M * codes * de, "block_mlp": 2 * K * (4 * de * de + 4 * de * d),
           "mixer": 2 * b["n_parameters_by_module"]["mixer"] * T}
    if m["mixer"] == "transformer":
        out["attention_or_scan"] = 4 * L * T * T * d
    elif m["mixer"] == "bimamba1":
        out["attention_or_scan"] = 2 * T * 2 * d * MAMBA_STATE * MAMBA_SCAN_OPS * L
    else:
        raise ValueError(f"no FLOPs formula for mixer {m['mixer']!r}")
    out["forward"] = sum(out.values())
    out["train"] = 3 * out["forward"]
    return out


def compute_table(run_dir):
    """One row per snapshot (snapshots.csv) and per validation pass (metrics.csv, with the mean
    tuning AUROC): the cumulative training compute, GPU-hours and the model's counts."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    f = flops_per_participant(config)
    rows = []
    path = os.path.join(run_dir, "snapshots.csv")
    if os.path.exists(path):
        for r in pd.read_csv(path).itertuples():
            rows.append(dict(kind="snapshot", step=r.step, samples_seen=r.samples_seen, epoch_equiv=r.epoch_equiv,
                             train_seconds=r.train_seconds, auroc_mean=np.nan))
    path = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(path):
        m = pd.read_csv(path)
        for r in m[(m.label == "MEAN") & (m.metric == "auroc")].itertuples():
            rows.append(dict(kind="validation", step=r.step, samples_seen=r.samples_seen, epoch_equiv=r.epoch_equiv,
                             train_seconds=np.nan, auroc_mean=r.value))
    t = pd.DataFrame(rows, columns=["kind", "step", "samples_seen", "epoch_equiv", "train_seconds", "auroc_mean"])
    t["flops"] = f["train"] * t.samples_seen.astype(float)
    t["gpu_hours"] = t.train_seconds / 3600.0
    for k, v in f.items():
        t[f"flops_per_participant_{k}"] = v
    b, m = config["blocks"], config["model"]
    if not len(t):                                 # a run without records still describes its model
        t = pd.DataFrame([{c: np.nan for c in t.columns}])
    rung, cell, seed = os.path.basename(os.path.normpath(config["dataset_dir"])), config["args"]["exp_name"], config["args"]["seed"]
    return t.assign(run=f"{rung}__{cell}__seed{seed}", run_dir=run_dir, rung=rung, cell=cell, seed=seed,
                    n_train=config.get("n_train"), mixer=m["mixer"], d_model=m["d_model"], e_layers=m["e_layers"],
                    embed_dim=b["embed_dim"], n_tokens=b["n_tokens"], n_blocks=b["n_blocks"],
                    n_mixer_params=b["n_parameters_by_module"]["mixer"],
                    n_non_embedding_params=b["n_non_embedding_parameters"], n_params=config["n_parameters"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (each with config.json)")
    ap.add_argument("--out", required=True, help="where <rung>__<cell>__seed<s>_compute.csv files go")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for run in a.runs:
        t = compute_table(run)
        path = os.path.join(a.out, f"{t.run.iloc[0]}_compute.csv")
        t.to_csv(path, index=False)
        print(f"{run}: {len(t)} rows, {t.flops.max() if len(t) else 0:.3g} training FLOPs at the last record -> {path}")


if __name__ == "__main__":
    main()
