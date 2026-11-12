#!/usr/bin/env python3
"""How much of a pilot model's tuning AUROC comes from the genotypes?

The covariates (year of birth, sex) predict these diseases well on their own, so the AUROC of
a model says little about what it learned from the SNPs. For each finished run this script
scores the best checkpoint on the proxy's tuning participants twice:

    as they are                      genotypes, covariates and labels of the same person
    with genotypes permuted          each person keeps their covariates and labels and is given
                                     another tuning participant's genotypes (several permutations)

The drop in AUROC when the genotypes are permuted is the part of the AUROC that depends on a
person's own genotypes. A model that ignores the SNPs, or draws only noise from them, shows no
drop. The interval is from a bootstrap over tuning participants of the mean drop over diseases.

Writes results/genetic_signal.csv (one row per run) and results/genetic_signal_by_disease.csv.
Aggregate numbers only; per-person scores stay in memory.

    CUDA_VISIBLE_DEVICES=<gpu> python scripts/widedeepgp-pilot/genetic_signal.py                # runs of the fork environment
    CUDA_VISIBLE_DEVICES=<gpu> <m3 python> scripts/widedeepgp-pilot/genetic_signal.py --arms mamba3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DEEPGP_REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
RUNS = "/mnt/sdg/robert/deepwidegp/runs"
FORK_ARMS = ["default", "target_only", "shuffled", "no_correction", "gamma1", "bf16", "annot"]


def aurocs(logits, y):
    """AUROC of every column, from ranks."""
    from scipy.stats import rankdata
    pos = y > 0.5
    n1 = pos.sum(axis=0)
    n0 = len(y) - n1
    r = rankdata(logits, axis=0)
    return ((r * pos).sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def drop_with_interval(base, perms, y, n_boot=300, seed=0):
    """Per-disease AUROC as scored, mean AUROC over the permutations, and a bootstrap interval
    (over participants) for the mean over diseases of the difference."""
    a = aurocs(base, y)
    p = np.mean([aurocs(q, y) for q in perms], axis=0)
    rng, n, boots = np.random.default_rng(seed), len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if (y[i].sum(axis=0) == 0).any():
            continue
        boots.append(float(np.mean(aurocs(base[i], y[i]) - np.mean([aurocs(q[i], y[i]) for q in perms], axis=0))))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return a, p, float(lo), float(hi), float(np.std(boots))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="pilot")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--arms", nargs="+", default=FORK_ARMS)
    ap.add_argument("--permutations", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=128)
    a = ap.parse_args(argv)

    sys.path.insert(0, DEEPGP_REPO)
    os.chdir(DEEPGP_REPO)
    import torch
    _load = torch.load
    torch.load = lambda *x, **k: _load(*x, **{**k, "weights_only": False})
    from lightning_fabric.utilities.apply_func import move_data_to_device
    from widedeepgp.train import build_data, load_model

    @torch.no_grad()
    def predict(model, dm, sp):
        out = []
        for batch in dm._loader(dm._dataset(sp), False):
            x = dm.on_after_batch_transfer(move_data_to_device(batch, "cuda"), 0)
            out.append(model(x[0]).float().cpu())
        return torch.cat(out).numpy().astype(np.float64)

    RES.mkdir(exist_ok=True)
    run_rows, disease_rows, dm = [], [], None
    for d in sorted((Path(a.runs) / a.experiment).iterdir()):
        if not (d / "results.json").exists() or "_s" not in d.name or d.name.rsplit("_s", 1)[0] not in a.arms:
            continue
        arm, seed = d.name.rsplit("_s", 1)
        cfg = SimpleNamespace(**json.load(open(d / "config.json"))["args"])
        cfg.eval_batch_size = a.batch_size
        if dm is None:                               # every run of the pilot reads the same dataset
            dm = build_data(cfg)
        model, _ = load_model(str(d / "checkpoints" / "best.ckpt"), dm, "cuda")
        sp = dm.splits["val"]
        y = np.asarray(sp["y"], dtype=np.float64)
        base = predict(model, dm, sp)
        rng = np.random.default_rng(1000 + int(seed))
        perms = [predict(model, dm, dict(sp, rows=sp["rows"][rng.permutation(len(y))])) for _ in range(a.permutations)]
        au, pu, lo, hi, se = drop_with_interval(base, perms, y)
        names = model.label_names
        run_rows.append(dict(arm=arm, seed=int(seed), auroc_mean=float(au.mean()), auroc_mean_genotypes_permuted=float(pu.mean()),
                             genetic_part=float((au - pu).mean()), ci_low=lo, ci_high=hi, bootstrap_se=se,
                             diseases_with_a_drop=int(((au - pu) > 0).sum()), n_diseases=len(names),
                             largest_drop=names[int(np.argmax(au - pu))], largest_drop_value=float((au - pu).max())))
        disease_rows += [dict(arm=arm, seed=int(seed), label=n, auroc=float(x), auroc_genotypes_permuted=float(q),
                              genetic_part=float(x - q)) for n, x, q in zip(names, au, pu)]
        print(f"{d.name}: AUROC {au.mean():.4f}, with genotypes permuted {pu.mean():.4f}, genetic part "
              f"{(au - pu).mean():+.4f} ({lo:+.4f} to {hi:+.4f})", flush=True)
        del model
        torch.cuda.empty_cache()

    for name, rows, keys in (("genetic_signal.csv", run_rows, ["arm", "seed"]),
                             ("genetic_signal_by_disease.csv", disease_rows, ["arm", "seed", "label"])):
        new = pd.DataFrame(rows)
        path = RES / name
        if path.exists() and len(new):                # the two environments each add their runs
            old = pd.read_csv(path)
            old = old[~old.arm.isin(set(new.arm))]
            new = pd.concat([old, new], ignore_index=True)
        if len(new):
            new.to_csv(path, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
