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

Writes results/genetic_signal.csv (one row per run) and results/genetic_signal_by_disease.csv;
the runs of another --experiment go to genetic_signal_<experiment>.csv and
genetic_signal_<experiment>_by_disease.csv. A run's rows replace only the earlier rows of the
same arm and seed. Aggregate numbers only; per-person scores stay in memory.

    CUDA_VISIBLE_DEVICES=<gpu> python scripts/widedeepgp-pilot/genetic_signal.py                # runs of the fork environment
    CUDA_VISIBLE_DEVICES=<gpu> <m3 python> scripts/widedeepgp-pilot/genetic_signal.py --arms mamba3

Each run is scored on its own dataset: the data are loaded again whenever a run's dataset
differs from the previous run's (gwas_annot reads a variant of its own), and the checkpoint is
the one results.json names. A label that does not apply to a person (empty) leaves that person
out of the label's AUROC, as in the model's own metrics.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DEEPGP_REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
RUNS = "/mnt/sdg/robert/deepwidegp/runs"
FORK_ARMS = ["default", "target_only", "shuffled", "no_correction", "gamma1", "bf16", "annot"]
# the settings that decide which people, labels and inputs build_data gives a run
DATASET_KEYS = ("data_dir", "label", "rd_mode", "snp_stats", "labels", "pos_mode", "dm", "scaling", "snp_embed")


def aurocs(logits, y):
    """AUROC of every column, from ranks, over the people its label applies to (y not NaN);
    NaN for a label without a case or without a non-case among them."""
    from scipy.stats import rankdata
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        ok = ~np.isnan(y[:, j])
        pos = y[ok, j] > 0.5
        n1, n0 = int(pos.sum()), int((~pos).sum())
        if n1 and n0:
            out[j] = (rankdata(logits[ok, j])[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return out


def drop_with_interval(base, perms, y, n_boot=300, seed=0):
    """Per-disease AUROC as scored, mean AUROC over the permutations, and a bootstrap interval
    (over participants) for the mean over diseases of the difference. Means are over the labels
    that have an AUROC, as in the model's own metrics; a label with no case or no non-case among
    the people it applies to is left out, of the bootstrap's check as well."""
    a = aurocs(base, y)
    p = np.mean([aurocs(q, y) for q in perms], axis=0)
    usable = ~np.isnan(a)
    if not usable.any():
        raise ValueError("no label has both cases and non-cases among the people scored")
    rng, n, boots = np.random.default_rng(seed), len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yi = y[i][:, usable]
        if (np.nansum(yi, axis=0) == 0).any() or (np.nansum(1.0 - yi, axis=0) == 0).any():
            continue
        boots.append(float(np.nanmean(aurocs(base[i], y[i]) - np.mean([aurocs(q[i], y[i]) for q in perms], axis=0))))
    if not boots:
        raise ValueError("every bootstrap resample lacked a case or a non-case of some label")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return a, p, float(lo), float(hi), float(np.std(boots))


def data_for(cfg, cache, build):
    """The data module for a run: the cached one if the run's dataset (DATASET_KEYS) is the
    previous run's, else ``build(cfg)``."""
    key = tuple(getattr(cfg, k, None) for k in DATASET_KEYS)
    if cache.get("key") != key:
        cache.update(key=key, dm=build(cfg))
    return cache["dm"]


def best_checkpoint(run):
    """The checkpoint results.json names, in the run's checkpoints/ (a run copied and resumed
    keeps a stale best.ckpt beside the best-v1.ckpt it wrote)."""
    with open(run / "results.json") as f:
        return run / "checkpoints" / Path(json.load(f)["best_checkpoint"]).name


def table_paths(experiment, res=RES):
    """(per-run table, per-disease table): the pilot's are genetic_signal*.csv, any other
    experiment's carry its name, so experiments that share arm names cannot overwrite each other."""
    stem = "genetic_signal" if experiment == "pilot" else f"genetic_signal_{experiment}"
    return Path(res) / f"{stem}.csv", Path(res) / f"{stem}_by_disease.csv"


def merge_rows(path, rows, keys):
    """Write ``rows`` into the table at ``path``: earlier rows with the same ``keys`` are
    replaced, all others are kept."""
    new = pd.DataFrame(rows)
    if not len(new):
        return
    if path.exists():
        old = pd.read_csv(path)
        replaced = old.set_index(keys).index.isin(new.set_index(keys).index)
        new = pd.concat([old[~replaced], new], ignore_index=True)
    new.to_csv(path, index=False, float_format="%.6g")


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
    run_rows, disease_rows, data = [], [], {}
    for d in sorted((Path(a.runs) / a.experiment).iterdir()):
        if not (d / "results.json").exists() or "_s" not in d.name or d.name.rsplit("_s", 1)[0] not in a.arms:
            continue
        arm, seed = d.name.rsplit("_s", 1)
        cfg = SimpleNamespace(**json.load(open(d / "config.json"))["args"])
        cfg.eval_batch_size = a.batch_size
        dm = data_for(cfg, data, build_data)
        model, _ = load_model(str(best_checkpoint(d)), dm, "cuda")
        sp = dm.splits["val"]
        y = np.asarray(sp["y"], dtype=np.float64)
        base = predict(model, dm, sp)
        rng = np.random.default_rng(1000 + int(seed))
        perms = [predict(model, dm, dict(sp, rows=sp["rows"][rng.permutation(len(y))])) for _ in range(a.permutations)]
        au, pu, lo, hi, se = drop_with_interval(base, perms, y)
        names = model.label_names
        gp = au - pu                                       # NaN for a label that cannot be scored
        run_rows.append(dict(arm=arm, seed=int(seed), auroc_mean=float(np.nanmean(au)),
                             auroc_mean_genotypes_permuted=float(np.nanmean(pu)),
                             genetic_part=float(np.nanmean(gp)), ci_low=lo, ci_high=hi, bootstrap_se=se,
                             diseases_with_a_drop=int((gp > 0).sum()), n_diseases=int((~np.isnan(gp)).sum()),
                             largest_drop=names[int(np.nanargmax(gp))], largest_drop_value=float(np.nanmax(gp))))
        disease_rows += [dict(arm=arm, seed=int(seed), label=n, auroc=float(x), auroc_genotypes_permuted=float(q),
                              genetic_part=float(x - q)) for n, x, q in zip(names, au, pu)]
        print(f"{d.name}: AUROC {np.nanmean(au):.4f}, with genotypes permuted {np.nanmean(pu):.4f}, genetic part "
              f"{np.nanmean(gp):+.4f} ({lo:+.4f} to {hi:+.4f})", flush=True)
        del model
        torch.cuda.empty_cache()

    # several copies may run at once (one per GPU, or one per environment): each replaces the rows
    # of the runs it scored, and the read-merge-write is done under a lock so that none is lost
    with open(os.path.join(tempfile.gettempdir(), "widedeepgp_genetic_signal.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path, rows, keys in zip(table_paths(a.experiment), (run_rows, disease_rows),
                                    (["arm", "seed"], ["arm", "seed", "label"])):
            merge_rows(path, rows, keys)
        fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
