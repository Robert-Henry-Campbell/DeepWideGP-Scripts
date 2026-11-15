"""Collect the gate results of 12-15 Nov 2026 into aggregate tables (results/). Reads the run
folders under /mnt/sdg/robert/deepwidegp/runs (wide_gate, pilot_sim) and the pilot's control
references; writes no participant-level value.

    python make_tables.py
"""
import json
import os

import pandas as pd

RUNS = "/mnt/sdg/robert/deepwidegp/runs"
PILOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "widedeepgp-pilot", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
G = os.path.join(RUNS, "wide_gate")

# ---- simulated traits: one row per trait
ref = pd.read_csv(os.path.join(PILOT, "simulated_control_references.csv")).set_index("trait")
pilot = pd.read_csv(os.path.join(PILOT, "simulated_control_runs.csv"))
tau1 = json.load(open(os.path.join(G, "sim_linear_tau1_k0.1", "results.json")))["val"]
grid = json.load(open(os.path.join(G, "sim_linear_grid", "linear", "results.json")))
pd_rows = {r["label"]: r for r in grid["per_disease"]}
joint = json.load(open(os.path.join(G, "sim_joint", "results.json")))
jbest = joint["stages"]["joint"]
jtrial = json.load(open(os.path.join(jbest["best_trial_dir"], "results.json")))["val"]
m = pd.read_csv(os.path.join(G, "sim_linear_grid", "linear", "metrics.csv"))
a = m[(m.metric == "auroc") & (m.label != "MEAN")].copy()
a["trait"], a["tau"] = a.label.str.rsplit("@", n=1).str[0], a.label.str.rsplit("@", n=1).str[1].astype(float)
last = a[a.validation == a.validation.max()].pivot_table(index="trait", columns="tau", values="value")
tau_all = last.mean().idxmax()
rows = []
for t in ref.index:
    rows.append({
        "trait": t, "causal_snps": int(ref.loc[t, "causal_snps"]), "heritability": ref.loc[t, "heritability"],
        "ceiling": ref.loc[t, "ceiling_auroc_tune"], "gblup_true_h2": ref.loc[t, "gblup_auroc_tune"],
        "clumped_prs": ref.loc[t, "clumped_auroc_tune"],
        # the pilot's network alone (all SNPs, no PRS path; 6 epoch-equivalents, best pass)
        "pilot_network_all_snps": pilot[(pilot.trait == t) & (pilot.arm == "default")]["auroc_at_best_pass"].iloc[0],
        "linear_tau1_fixed": tau1[t]["auroc"],
        f"linear_grid_end_tau{tau_all:g}_all": last.loc[t, tau_all],
        "linear_grid_per_trait": pd_rows[t]["val_auroc"], "linear_grid_tau_chosen": pd_rows[t]["prs_var"],
        "joint_best_trial": jtrial[t]["auroc"],
    })
sim = pd.DataFrame(rows)
sim.round(6).to_csv(os.path.join(OUT, "simulated_traits.csv"), index=False)

# ---- real labels: one row per disease (the seven-value grid run; the nine-value one was corrupted)
real = pd.read_csv(os.path.join(G, "real_linear_grid", "linear", "per_disease", "gate_g3_linear_vs_covariates.csv"))
real.round(6).to_csv(os.path.join(OUT, "real_diseases_linear.csv"), index=False)

# ---- the network on top (simulated traits): every trial of the joint stage
trials = pd.DataFrame([{"trial": t["trial"], "state": t["state"], "mean_tuning_auroc": t["value"], **t["params"],
                        "epoch_equivalents": t["epoch_equivalents"]} for t in jbest["trials"]])
trials.round(6).to_csv(os.path.join(OUT, "simulated_joint_trials.csv"), index=False)

# ---- one line per gate
summ = [
    ("G1", "simulated", "linear, tau 1 fixed, 40 epoch-equivalents", sim.linear_tau1_fixed.mean(), sim.gblup_true_h2.mean()),
    ("G1", "simulated", "linear, grid, tau and pass per trait", sim.linear_grid_per_trait.mean(), sim.gblup_true_h2.mean()),
    ("G1", "simulated", f"linear, grid, tau {tau_all:g} for all, end of training", sim[f"linear_grid_end_tau{tau_all:g}_all"].mean(), sim.gblup_true_h2.mean()),
    ("G2", "simulated", "network on top of the linear model (best of 12 trials)", jbest["best_value"], joint["stages"]["linear"]["best_value"]),
    ("G3", "real", "linear, seven-value grid, tau 0.1 for all, end of training", real.auroc_end_one_tau.mean(), real.auroc_covariates_only.mean()),
    ("G3", "real", "linear, seven-value grid, tau and pass per disease", real.auroc_per_disease_choice.mean(), real.auroc_covariates_only.mean()),
]
pd.DataFrame(summ, columns=["gate", "labels", "model", "mean_tuning_auroc", "reference_mean_tuning_auroc"]).round(6).to_csv(
    os.path.join(OUT, "gate_summary.csv"), index=False)
print(sim.round(4).to_string(index=False))
print(pd.read_csv(os.path.join(OUT, "gate_summary.csv")).to_string(index=False))
