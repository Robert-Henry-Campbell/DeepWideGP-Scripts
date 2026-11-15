"""Gate G3, first half: the linear stage on the real labels of the pilot's proxy, against a
logistic model on the covariates alone fitted on the same training participants (tuning AUROC).

Per-disease selection (a prior variance and a validation pass per disease, chosen on the tuning
set) makes the tuning AUROC of the per-disease model optimistic; so this also reports the
model at the end of training with one prior variance for all diseases (the one best on average:
a choice among 7), which carries almost no selection. Aggregate numbers only.
"""
import json  # noqa
import os
import sys

import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/sdg/robert/deepwidegp/runs/wide_gate/real_linear_grid"
sys.path.insert(0, "/mnt/sdg/robert/deepwidegp/DeepWideGP")
os.chdir("/mnt/sdg/robert/deepwidegp/DeepWideGP")
import score_snapshots  # noqa: E402

lin = json.load(open(f"{ROOT}/linear/results.json"))
rows = pd.DataFrame(lin["per_disease"]).set_index("label")
m = pd.read_csv(f"{ROOT}/linear/metrics.csv")
a = m[(m.metric == "auroc") & (m.label != "MEAN")].copy()
a["disease"] = a.label.str.rsplit("@", n=1).str[0]
a["tau"] = a.label.str.rsplit("@", n=1).str[1].astype(float)
last = a[a.validation == a.validation.max()].pivot_table(index="disease", columns="tau", values="value")
tau_mean = last.mean().idxmax()

# the covariates-only model and the per-disease model, scored on the tuning set by the scoring script
score_snapshots.main(["--run", ROOT, "--split", "val", "--snapshots", "best"])
summ = pd.read_csv(f"{ROOT}/linear/per_disease/scores_val_summary.csv").set_index("label")

out = pd.DataFrame({
    "n_cases_tune": summ["n_cases"],
    "auroc_covariates_only": summ["auroc_covariates_only"],
    "auroc_per_disease_choice": summ["auroc"],
    "prs_var_chosen": rows["prs_var"],
    "auroc_end_one_tau": last[tau_mean],
})
out["gain_per_disease_choice"] = out.auroc_per_disease_choice - out.auroc_covariates_only
out["gain_end_one_tau"] = out.auroc_end_one_tau - out.auroc_covariates_only
pd.set_option("display.width", 200)
print(out.round(4).sort_values("n_cases_tune", ascending=False).to_string())
print()
print(f"mean AUROC: covariates only {out.auroc_covariates_only.mean():.4f}; per-disease choice "
      f"{out.auroc_per_disease_choice.mean():.4f} (gain {out.gain_per_disease_choice.mean():+.4f}, optimistic); "
      f"end of training, tau {tau_mean:g} for all {out.auroc_end_one_tau.mean():.4f} "
      f"(gain {out.gain_end_one_tau.mean():+.4f}); diseases with a positive gain at one tau: "
      f"{int((out.gain_end_one_tau > 0).sum())} of {len(out)}")
print("mean AUROC at the end of training, by tau:", last.mean().round(4).to_dict())
out.round(6).to_csv(f"{ROOT}/linear/per_disease/gate_g3_linear_vs_covariates.csv")
print("wrote", f"{ROOT}/linear/per_disease/gate_g3_linear_vs_covariates.csv")
