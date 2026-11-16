#!/usr/bin/env python3
"""Tuning-set summary of one finished rung of the WideDeepGP ladder (aggregate numbers only).

For every disease:
- the tuning AUROC of the run's model (the network on top of the ridge);
- the ridge alone (the run's `linear_only` ablation, each disease at its chosen prior variance);
- a logistic regression on the twelve covariates fitted on the same training participants, as
  `score_snapshots.py` computes it.

Also the numbers of tuning cases. All of these are tuning-set figures, chosen on the tuning set.
The comparison with the PGI arm uses the shared protocol and the test set, after the joint unlock.

    python summarise_rung.py hm3plus_n13199     # writes results/rung_<size>.csv, prints a summary
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

RUNS = "/mnt/sdg/robert/deepwidegp/runs/ladder"
REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main(rung):
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from score_snapshots import covariates_only
    from widedeepgp.train import build_data
    root = os.path.join(RUNS, rung)
    r = json.load(open(os.path.join(root, "results.json")))
    lin_dir = os.path.join(root, r["linear_only"]["best_trial_dir"])
    lin = json.load(open(os.path.join(lin_dir, "results.json")))
    chosen = pd.read_csv(os.path.join(lin_dir, "per_disease.csv")).set_index("label")
    cfg = SimpleNamespace(**json.load(open(os.path.join(root, r["best_trial_dir"], "config.json")))["args"])
    dm = build_data(cfg)
    names = list(r["val"].keys())
    cov = covariates_only(dm, "val", names)
    y = np.asarray(dm.splits["val"]["y"], dtype=float)
    t = pd.DataFrame({
        "disease": names,
        "tuning_cases": [int(np.nansum(y[:, j])) for j in range(len(names))],
        "covariates_only": [float(cov["auroc"][j]) for j in range(len(names))],
        "ridge_alone": [lin["val"][n]["auroc"] for n in names],
        "ridge_prior_variance": [float(chosen.loc[n, "prs_var"]) for n in names],
        "widedeepgp": [r["val"][n]["auroc"] for n in names],
    })
    t["network_minus_ridge"] = t.widedeepgp - t.ridge_alone
    t["widedeepgp_minus_covariates"] = t.widedeepgp - t.covariates_only
    t["ridge_minus_covariates"] = t.ridge_alone - t.covariates_only
    os.makedirs(OUT, exist_ok=True)
    size = int(json.load(open(os.path.join(lin_dir, "config.json")))["n_train"])
    t.round(6).to_csv(os.path.join(OUT, f"rung_{size}.csv"), index=False)
    pd.set_option("display.width", 200)
    print(t.sort_values("tuning_cases", ascending=False).round(4).to_string(index=False))
    print(f"\n{rung} ({size} training participants): mean tuning AUROC, covariates only {t.covariates_only.mean():.4f}; "
          f"ridge alone {t.ridge_alone.mean():.4f}; WideDeepGP {t.widedeepgp.mean():.4f}. WideDeepGP minus covariates "
          f"{t.widedeepgp_minus_covariates.mean():+.4f} ({int((t.widedeepgp_minus_covariates > 0).sum())} of {len(t)} "
          f"diseases above); network minus ridge {t.network_minus_ridge.mean():+.4f}.")
    jt = r["stages"]["joint"]
    print(f"Network trials: {jt['n_trials']} ({jt['n_complete']} complete, {jt['n_pruned']} pruned); chosen trial "
          f"{jt['best_trial']}, settings {jt['best_params']}; training time of all trials "
          f"{r['train_seconds_all_trials'] / 3600:.1f} h.")


if __name__ == "__main__":
    main(sys.argv[1])
