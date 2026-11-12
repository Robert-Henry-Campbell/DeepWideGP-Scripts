#!/usr/bin/env python3
"""Check of the positive control's data path: the simulated traits' true genetic value,
recomputed from the genotypes exactly as the model's loader delivers them.

simulated_control.py build drew the traits from the proxy's genotypes, decoded and standardised
by its own code, and scored the true genetic value on the tuning participants: the ceilings of
results/simulated_control_references.csv. The networks of the control see the genotypes through
another path: the data module of the simulated dataset (SNPStoreDataModule, built from the
configuration of a control run, so with SNP means and SDs from the training split), its
validation loader and decode_batch, which hands the model its standardised dosages. This script
sends the 10,000 tuning participants through that path, multiplies the dosages by the true
effects (sim_truth.npz) and scores the sum against the labels the same loader delivers. Had the
loader paired participants with other genotype rows than the simulation used, or ordered or
scaled the SNPs differently, the AUROCs would fall short of the ceilings. A participant-to-row
error shared by both paths would not show here; the GWAS of gwas_annotation.py, whose top SNPs
sit at the known loci of these diseases, checks that map for the real labels.

Writes results/control_data_path_check.csv: one row per trait, aggregate numbers only; the
per-person values stay in memory. CPU only.

    DEEPGP_REPO=/mnt/sdg/robert/deepwidegp/runs/pilot_code_8f03ca6 \
        python scripts/widedeepgp-pilot/check_control_data_path.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DEEPGP_REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
RUN = "/mnt/sdg/robert/deepwidegp/runs/pilot_sim/default_s1"
sys.path.insert(0, str(HERE))
from simulated_control import auroc  # noqa: E402  (the AUROC that gave the ceilings)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=RUN, help="a run of the control; its configuration builds the loader")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                                # CPU only, set before torch is imported
    import torch
    torch.set_num_threads(a.threads)
    sys.path.insert(0, DEEPGP_REPO)
    os.chdir(DEEPGP_REPO)
    from genostore.datamodule import SNPStoreDataModule
    t0 = time.time()
    cfg = SimpleNamespace(**json.load(open(Path(a.run) / "config.json"))["args"])
    dm = SNPStoreDataModule(cfg)                    # what widedeepgp.train.build_data builds
    truth = np.load(Path(dm.dir) / "sim_truth.npz")
    traits = [str(t) for t in truth["traits"]]
    assert traits == dm.label_names and len(truth["beta"]) == sum(dm.seq_len)
    beta = torch.as_tensor(truth["beta"], dtype=torch.float64)
    at = np.cumsum([0] + list(dm.seq_len))
    dm.setup("fit")
    g, y = [], []
    for batch in dm.val_dataloader():
        x_all, yb = dm.on_after_batch_transfer(batch, 0)                   # what the model receives
        g.append(sum(x[:, :, 0].double() @ beta[at[k]:at[k + 1]] for k, x in enumerate(x_all[:len(dm.seq_len)])).numpy())
        y.append(yb.numpy())
    g, y = np.concatenate(g), np.concatenate(y)
    ref = pd.read_csv(RES / "simulated_control_references.csv").set_index("trait")
    rows = []
    for j, t in enumerate(traits):
        au = auroc(g[:, j], y[:, j])
        rows.append(dict(trait=t, tuning_participants=len(y), cases=int(y[:, j].sum()),
                         cases_in_references=int(ref.loc[t, "cases_tune"]), auroc_through_loader=au,
                         ceiling_auroc_tune=float(ref.loc[t, "ceiling_auroc_tune"]),
                         difference=au - float(ref.loc[t, "ceiling_auroc_tune"])))
    out = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    out.to_csv(RES / "control_data_path_check.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"{len(y)} tuning participants through the loader of {dm.dir} in {time.time() - t0:.0f} s; "
          f"largest |difference| {out.difference.abs().max():.2g}", flush=True)


if __name__ == "__main__":
    main()
