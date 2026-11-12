#!/usr/bin/env python3
"""Tables and figures of the WideDeepGP pilot (aggregate numbers only).

Reads every run directory <runs>/<experiment>/<arm>_s<seed>/ that has a results.json and
writes into results/:

    pilot_runs.csv            one row per run
    pilot_arms.csv            one row per arm: mean over seeds, seed range, difference from the default
    pilot_equal_time.csv      arms compared with the default at equal GPU time
    pilot_curves.csv          mean tuning AUROC at every validation pass of every run
    pilot_per_disease.csv     tuning AUROC of every disease: covariates only, and each arm
    fig1_curves, fig2_arms, fig3_diseases, fig4_genetic (.pdf and .png), each with _data.csv and _legend.md

genetic_signal.py (run first, on a GPU) supplies results/genetic_signal.csv for figure 4.

    python scripts/widedeepgp-pilot/analyze_pilot.py [--experiment pilot]

The covariates-only baseline is a logistic model per disease on the 12 covariates, fitted on
the proxy's training participants and scored on its tuning participants (the model repository's
own function).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DEEPGP_REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
RUNS = "/mnt/sdg/robert/deepwidegp/runs"
MM = 1 / 25.4
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

ARMS = {  # arm -> (short label, what differs from the default)
    "default": ("Default", "rotation, all outputs learn, sampling-rate correction, square-root class weights, 32-bit"),
    "target_only": ("Only the target learns", "only the output of the disease a batch was balanced on learns from it"),
    "shuffled": ("Shuffled batches", "participants drawn uniformly; same loss"),
    "no_correction": ("No sampling-rate correction", "the loss is not divided by each participant's sampling rate"),
    "gamma1": ("Full class balance", "class-weight exponent 1 instead of 0.5"),
    "bf16": ("bf16 encoders", "encoders in bf16; pooling, head and loss in 32-bit"),
    "annot": ("Annotation vectors", "96 functional annotations per SNP as a gain and offset on its token"),
    "mamba3": ("Mamba-3 block", "Mamba-3 in a bidirectional wrapper instead of the bidirectional Mamba block"),
    "gwas_annot": ("GWAS statistics as annotation", "30 annotation channels: each SNP's association z-score for each disease, "
                                                     "from training participants outside the proxy"),
    "gwasclump_annot": ("Clumped GWAS statistics as annotation", "the same z-scores, kept only at each disease's clumped lead SNPs"),
}
SHORT = {"I10": "Hypertension", "E78": "Lipid disorders", "K57": "Diverticular disease", "K21": "Reflux disease",
         "K29": "Gastritis and duodenitis", "K44": "Diaphragmatic hernia", "I25": "Chronic ischaemic heart disease",
         "M19": "Other arthrosis", "I84+K64": "Haemorrhoids", "I48": "Atrial fibrillation", "M17": "Knee arthrosis",
         "C44": "Other skin cancer", "E11": "Type 2 diabetes", "E66": "Obesity", "M54": "Back pain",
         "D12": "Benign colorectal neoplasm", "I20": "Angina", "E03": "Hypothyroidism", "K40": "Inguinal hernia",
         "M16": "Hip arthrosis", "K80": "Gallstones", "N17": "Acute renal failure", "N18": "Chronic renal failure",
         "J44": "COPD", "M23": "Internal derangement of knee", "M47": "Spondylosis", "I95": "Hypotension",
         "I50": "Heart failure", "M81": "Osteoporosis", "I21": "Myocardial infarction"}


# ---- reading runs -----------------------------------------------------------------------------
def load_runs(exp_dir):
    runs = []
    for d in sorted(Path(exp_dir).iterdir()):
        if not (d / "results.json").exists() or "_s" not in d.name:
            continue
        arm, seed = d.name.rsplit("_s", 1)
        if arm not in ARMS:
            continue
        r, cfg = json.load(open(d / "results.json")), json.load(open(d / "config.json"))
        m = pd.read_csv(d / "metrics.csv")
        runs.append(dict(arm=arm, seed=int(seed), dir=str(d), results=r, config=cfg, metrics=m))
    return runs


def curves(runs):
    rows = []
    for run in runs:
        m, r, sched = run["metrics"], run["results"], run["config"]["schedule"]
        mean = m[m.label == "MEAN"].pivot_table(index=["validation", "step", "samples_seen", "epoch_equiv"],
                                                columns="metric", values="value").reset_index()
        best = -np.inf
        for row in mean.itertuples():
            best = max(best, row.auroc)
            rows.append(dict(arm=run["arm"], seed=run["seed"], validation=int(row.validation), step=int(row.step),
                             epoch_equiv=row.epoch_equiv,
                             gpu_minutes=row.step * sched["accumulate"] * r["seconds_per_batch"] / 60.0,
                             auroc_mean=row.auroc, auroc_mean_best_so_far=best, bce_mean=row.bce, ap_mean=row.ap))
    return pd.DataFrame(rows)


def run_table(runs, cur):
    rows = []
    for run in runs:
        r, c = run["results"], cur[(cur.arm == run["arm"]) & (cur.seed == run["seed"])]
        rows.append(dict(arm=run["arm"], seed=run["seed"], best_auroc_mean=r["best_val_auroc_mean"],
                         last_auroc_mean=float(c.auroc_mean.iloc[-1]), best_at_epoch_equiv=float(c.epoch_equiv[c.auroc_mean.idxmax()]),
                         best_bce_mean=r["val_mean"]["bce"], best_ap_mean=r["val_mean"]["ap"],
                         epoch_equivalents=r["epoch_equivalents"], train_minutes=r["train_seconds"] / 60.0,
                         train_minutes_per_epoch_equiv=r["train_seconds"] / 60.0 / r["epoch_equivalents"],
                         wall_minutes=r["wall_seconds"] / 60.0, gpu_peak_train_gib=r.get("gpu_peak_train_gib"),
                         n_parameters=r["n_parameters"]))
    return pd.DataFrame(rows)


def larger(a, b):
    """The larger of two seed ranges; NaN when neither arm has two seeds."""
    v = [x for x in (a, b) if x is not None and np.isfinite(x)]
    return max(v) if v else np.nan


def arm_table(rt):
    out, base = [], rt[rt.arm == "default"].best_auroc_mean
    for arm in [a for a in ARMS if a in set(rt.arm)]:
        v = rt[rt.arm == arm]
        x = v.best_auroc_mean
        spread = float(x.max() - x.min()) if len(x) > 1 else np.nan
        row = dict(arm=arm, label=ARMS[arm][0], differs=ARMS[arm][1], n_seeds=len(x), best_auroc_mean=float(x.mean()),
                   seeds="; ".join(f"{s}: {a:.4f}" for s, a in zip(v.seed, x)), seed_range=spread,
                   best_bce_mean=float(v.best_bce_mean.mean()),
                   train_minutes_per_epoch_equiv=float(v.train_minutes_per_epoch_equiv.mean()),
                   gpu_peak_train_gib=float(v.gpu_peak_train_gib.mean()), n_parameters=int(v.n_parameters.iloc[0]))
        if len(base):
            bar = larger(spread, float(base.max() - base.min()) if len(base) > 1 else np.nan)
            row.update(difference_from_default=float(x.mean() - base.mean()), bar_seed_range=bar,
                       clears_bar=bool(abs(x.mean() - base.mean()) > bar) if arm != "default" and np.isfinite(bar) else None)
        out.append(row)
    return pd.DataFrame(out)


def equal_time(cur, rt):
    """Each arm against the default at the same GPU time: the best mean AUROC reached so far."""
    rows = []
    base = cur[cur.arm == "default"]
    if base.empty:
        return pd.DataFrame(rows)
    for arm in [a for a in ARMS if a in set(cur.arm) and a != "default"]:
        c = cur[cur.arm == arm]
        t = min(c.groupby("seed").gpu_minutes.max().min(), base.groupby("seed").gpu_minutes.max().min())
        at = lambda d: d[d.gpu_minutes <= t + 1e-9].groupby("seed").auroc_mean_best_so_far.max()
        a_, b_ = at(c), at(base)
        if a_.empty or b_.empty:
            continue
        rng = lambda v: float(v.max() - v.min()) if len(v) > 1 else np.nan
        bar = larger(rng(a_), rng(b_))
        rows.append(dict(arm=arm, gpu_minutes=float(t), arm_auroc=float(a_.mean()), default_auroc=float(b_.mean()),
                         difference=float(a_.mean() - b_.mean()), arm_seed_range=rng(a_), default_seed_range=rng(b_),
                         bar_seed_range=bar, clears_bar=bool(abs(a_.mean() - b_.mean()) > bar) if np.isfinite(bar) else None))
    return pd.DataFrame(rows)


def covariates_baseline(run):
    sys.path.insert(0, DEEPGP_REPO)
    cwd = os.getcwd()
    os.chdir(DEEPGP_REPO)
    try:
        import score_snapshots
        from widedeepgp.train import build_data
        cfg = SimpleNamespace(**run["config"]["args"])
        cfg.snp_stats = "store"                  # the covariates do not depend on the SNP scaling
        dm = build_data(cfg)
        m = score_snapshots.covariates_only(dm, "val", run["config"]["model"]["label_names"])
    finally:
        os.chdir(cwd)
    return dict(zip(m["labels"], [float(v) for v in m["auroc"]])), float(m["auroc_mean"]), float(m["bce_mean"])


def disease_table(runs, cov):
    rows = {}
    for run in runs:
        for name, m in run["results"]["val"].items():
            rows.setdefault(name, {}).setdefault(run["arm"], []).append(m["auroc"])
    out = []
    for name, by_arm in rows.items():
        row = dict(label=name, name=SHORT.get(name, name), covariates_only=cov.get(name, np.nan))
        for arm, v in by_arm.items():
            row[arm] = float(np.mean(v))
            row[arm + "_seed_range"] = float(np.max(v) - np.min(v)) if len(v) > 1 else np.nan
        if "default" in by_arm:
            row["default_gain_over_covariates"] = row["default"] - row["covariates_only"]
        out.append(row)
    return pd.DataFrame(out)


def markdown_table(rt, at, gs):
    """The arm table of results.md, one row per arm with its seeds side by side."""
    lines = ["| Arm | Best mean tuning AUROC, by seed | Mean | Cross-entropy | Genotype-dependent part, by seed (95% interval) "
             "| Minutes per epoch-equivalent | Peak GPU memory (GiB) |", "|---|---|---|---|---|---|---|"]
    for r in at.itertuples():
        v = rt[rt.arm == r.arm].sort_values("seed")
        seeds = "; ".join(f"{x:.4f}" for x in v.best_auroc_mean)
        part = "–"
        if gs is not None and (gs.arm == r.arm).any():
            g = gs[gs.arm == r.arm].sort_values("seed")
            part = "; ".join(f"{x.genetic_part:+.4f} ({x.ci_low:+.4f} to {x.ci_high:+.4f})" for x in g.itertuples())
        lines.append(f"| {r.label} | {seeds} | {r.best_auroc_mean:.4f} | {r.best_bce_mean:.4f} | {part} | "
                     f"{r.train_minutes_per_epoch_equiv:.1f} | {r.gpu_peak_train_gib:.1f} |")
    lines.append(f"| *Covariates only (logistic model)* | | {at.covariates_only_auroc_mean.iloc[0]:.4f} | "
                 f"{at.covariates_only_bce_mean.iloc[0]:.4f} | | | |")
    return "\n".join(lines) + "\n"


def capacity_table(trials, n_train=215_872):
    """Per configuration: the largest batch that fits on one GPU, memory per participant (slope
    of peak memory on batch size over the trials that fitted) and speed at the largest batch."""
    rows = []
    for name, d in trials.groupby("config", sort=False):
        ok = d[d.ok.astype(str) == "True"].sort_values("batch")
        if ok.empty:
            continue
        top = ok.iloc[-1]
        bad = d[d.ok.astype(str) != "True"].sort_values("batch")
        failed = bad.batch
        why = "" if bad.empty else ("kernel" if "error" in bad and str(bad.iloc[0].get("error", "")).startswith("kernel") else "memory")
        slope, icpt = (np.polyfit(ok.batch, ok.peak_gib, 1) if len(ok) > 1 else (np.nan, np.nan))
        best = ok.loc[ok.samples_per_s.idxmax()]
        rows.append(dict(config=name, mixer=top.mixer, precision=str(top.precision), grad_ckpt=bool(top.grad_ckpt),
                         annot=bool(top.annot), mode=top["mode"], n_parameters=int(top.n_parameters),
                         largest_batch=int(top.batch), smallest_batch_that_failed=int(failed.min()) if len(failed) else None,
                         limited_by=why,
                         peak_gib_at_largest=float(top.peak_gib), gib_per_participant=float(slope), gib_fixed=float(icpt),
                         step_s_at_largest=float(top.step_s), samples_per_s_at_largest=float(top.samples_per_s),
                         fastest_batch=int(best.batch), samples_per_s_fastest=float(best.samples_per_s),
                         gpu_hours_per_epoch_equiv=float(n_train / best.samples_per_s / 3600.0),
                         gpu_total_gib=float(top.gpu_total_gib)))
    return pd.DataFrame(rows)


# ---- figures ----------------------------------------------------------------------------------
def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "lines.linewidth": 1.4, "legend.frameon": False, "pdf.fonttype": 42,
        "savefig.dpi": 450, "figure.dpi": 150})


def save(fig, name, data, legend):
    fig.savefig(RES / f"{name}.png")
    fig.savefig(RES / f"{name}.pdf")
    plt.close(fig)
    data.to_csv(RES / f"{name}_data.csv", index=False, float_format="%.6g")
    with open(RES / f"{name}_legend.md", "w") as f:
        f.write(legend.strip() + "\n")


def fig_curves(cur, n_train):
    arms = [a for a in ARMS if a in set(cur.arm)]
    cols = 4
    rows_ = int(np.ceil(len(arms) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(183 * MM, (8 + 40 * rows_) * MM), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    base = cur[cur.arm == "default"]
    lo, hi = cur.auroc_mean.min(), cur.auroc_mean.max()
    for ax, arm in zip(axes.ravel(), arms):
        if arm != "default" and not base.empty:
            for _, d in base.groupby("seed"):
                ax.plot(d.epoch_equiv, d.auroc_mean, color=MUTED, lw=0.6)
            b = base.groupby("validation")[["epoch_equiv", "auroc_mean"]].mean()
            ax.plot(b.epoch_equiv, b.auroc_mean, color=INK2, lw=1.4)
        c = cur[cur.arm == arm]
        colour = INK2 if arm == "default" else BLUE
        for _, d in c.groupby("seed"):
            ax.plot(d.epoch_equiv, d.auroc_mean, color=MUTED if arm == "default" else BLUE, lw=0.6, alpha=1 if arm == "default" else 0.55)
        m = c.groupby("validation")[["epoch_equiv", "auroc_mean"]].mean()
        ax.plot(m.epoch_equiv, m.auroc_mean, color=colour, lw=1.4)
        ax.set_title(ARMS[arm][0], color=INK, loc="left", pad=3)
        ax.set_ylim(lo - 0.004, hi + 0.004)
    for ax in axes.ravel()[len(arms):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Epoch-equivalents")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean tuning AUROC")
    handles = [plt.Line2D([], [], color=INK2, lw=1.4, label="default"),
               plt.Line2D([], [], color=BLUE, lw=1.4, label="the arm named above the panel"),
               plt.Line2D([], [], color=MUTED, lw=0.6, label="single seeds")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    legend = f"""
**Figure 1. Mean tuning AUROC during training, by arm.** Each panel shows one arm of the pilot
(blue) against the default configuration (dark grey); thin lines are the two seeds and thick
lines their mean. The y-axis is the mean, over the 30 diseases, of the AUROC on the proxy's
10,000 tuning participants; the x-axis is training progress in epoch-equivalents (one
epoch-equivalent = {n_train:,} participants drawn). Every arm differs from the default in one
respect, named in the panel title. Data: `fig1_curves_data.csv`.
"""
    save(fig, "fig1_curves", cur, legend)


def fig_arms(rt, at, cov_mean):
    arms = [a for a in ARMS if a in set(rt.arm)]
    fig, ax = plt.subplots(figsize=(89 * MM, (14 + 7.5 * len(arms)) * MM), constrained_layout=True)
    base = rt[rt.arm == "default"].best_auroc_mean
    if len(base) > 1:
        ax.axvspan(base.min(), base.max(), color=GRID, lw=0, zorder=0)
    if len(base):
        ax.axvline(base.mean(), color=INK2, lw=0.8, zorder=1)
    lo_, hi_ = rt.best_auroc_mean.min(), rt.best_auroc_mean.max()
    if lo_ - 0.5 * (hi_ - lo_) - 0.002 <= cov_mean <= hi_ + 0.5 * (hi_ - lo_) + 0.002:   # drawn when it is near the arms
        ax.axvline(cov_mean, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(cov_mean, -0.75, "covariates only", color=INK2, fontsize=7, ha="right", va="bottom")
    for i, arm in enumerate(arms):
        v = rt[rt.arm == arm].best_auroc_mean
        ax.plot(v, [i] * len(v), "o", color=BLUE, ms=4.5, mec="white", mew=0.7, zorder=3)
        ax.plot([v.mean()] * 2, [i - 0.28, i + 0.28], color=INK, lw=1.2, zorder=4)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([ARMS[a][0] for a in arms])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Best mean tuning AUROC")
    legend = f"""
**Figure 2. Best mean tuning AUROC of each arm.** Dots are the two seeds of an arm and the
black tick their mean. The vertical line is the mean of the default configuration and the
shaded band the range of its two seeds; an arm whose mean lies outside the band, by more than
its own seed range, differs from the default by more than seed-to-seed variation. The dashed
line is a logistic model on the 12 covariates alone, fitted on the same training participants
(mean tuning AUROC {cov_mean:.4f}). Data: `fig2_arms_data.csv`.
"""
    data = rt[["arm", "seed", "best_auroc_mean"]].merge(at[["arm", "label", "best_auroc_mean", "seed_range",
                                                            "difference_from_default"]].rename(
        columns={"best_auroc_mean": "arm_mean"}), on="arm")
    data["covariates_only_mean_auroc"] = cov_mean
    save(fig, "fig2_arms", data, legend)


def fig_diseases(dt):
    d = dt.sort_values("default", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(120 * MM, (14 + 4.3 * len(d)) * MM), constrained_layout=True)
    y = np.arange(len(d))
    ax.hlines(y, d.covariates_only, d.default, color=AXIS, lw=0.8, zorder=1)
    ax.plot(d.covariates_only, y, "o", color=INK2, ms=4.2, mec="white", mew=0.7, zorder=3, label="12 covariates only")
    ax.plot(d.default, y, "o", color=BLUE, ms=4.2, mec="white", mew=0.7, zorder=4, label="default model")
    ax.set_yticks(y)
    ax.set_yticklabels(d.name)
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Tuning AUROC")
    ax.legend(loc="lower right")
    legend = """
**Figure 3. Tuning AUROC of each disease: default model against the covariates alone.** Blue
dots are the default configuration of the pilot at its best checkpoint (mean of two seeds);
grey dots are a logistic model on the 12 covariates (year of birth, sex, ten genetic principal
components) fitted on the same training participants. Diseases are ordered by the model's
AUROC. The pilot model sees an eighth of the SNPs and 40,000 participants, so the distance
between the dots is not an estimate of what the full model adds. Data: `fig3_diseases_data.csv`.
"""
    save(fig, "fig3_diseases", d, legend)


def fig_genetic(gs):
    arms = [a for a in ARMS if a in set(gs.arm)]
    fig, ax = plt.subplots(figsize=(89 * MM, (16 + 7.5 * len(arms)) * MM), constrained_layout=True)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ref = RES / "gwas_annotation_reference.csv"
    linear = float(pd.read_csv(ref).gain_from_prs.mean()) if ref.exists() else None
    if linear is not None:                              # what a linear score adds, for scale
        ax.axvline(1e3 * linear, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.text(1e3 * linear, -0.8, "clumped polygenic score", color=INK2, fontsize=7, ha="right", va="bottom")
    for i, arm in enumerate(arms):
        v = gs[gs.arm == arm].sort_values("seed")
        for k, r in enumerate(v.itertuples()):
            yk = i + (k - (len(v) - 1) / 2) * 0.3
            ax.plot([1e3 * r.ci_low, 1e3 * r.ci_high], [yk, yk], color=AXIS, lw=1.2, zorder=2, solid_capstyle="round")
            ax.plot(1e3 * r.genetic_part, yk, "o", color=BLUE, ms=4.5, mec="white", mew=0.7, zorder=3)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([ARMS[a][0] for a in arms])
    ax.set_ylim(len(arms) - 0.4, -0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUROC lost when genotypes are permuted\n(thousandths; mean over 30 diseases)")
    legend = """
**Figure 4. The part of each model's tuning AUROC that depends on the genotypes.** For every run
the best checkpoint was scored on the 10,000 tuning participants as they are and again after
giving each participant another participant's genotypes (three permutations), keeping
covariates and labels in place. Dots show the resulting loss of AUROC, averaged over the 30
diseases, in thousandths of an AUROC unit, for each seed of an arm; bars are 95% bootstrap
intervals over participants. A model that takes nothing from the SNPs sits at zero.
"""
    if linear is not None:
        legend += f"""The dashed line is what a clumped polygenic score built from 165,872 other participants adds to
the covariates on the same tuning participants ({1e3 * linear:.1f} thousandths).
"""
    legend += "Data: `fig4_genetic_data.csv`.\n"
    save(fig, "fig4_genetic", gs, legend)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="pilot")
    ap.add_argument("--runs", default=RUNS)
    a = ap.parse_args(argv)
    RES.mkdir(exist_ok=True)
    runs = load_runs(Path(a.runs) / a.experiment)
    if not runs:
        sys.exit("no finished runs")
    cur = curves(runs)
    rt = run_table(runs, cur)
    at = arm_table(rt)
    et = equal_time(cur, rt)
    cov, cov_mean, cov_bce = covariates_baseline(runs[0])
    dt = disease_table(runs, cov)
    at["covariates_only_auroc_mean"] = cov_mean
    at["covariates_only_bce_mean"] = cov_bce
    gs = pd.read_csv(RES / "genetic_signal.csv") if (RES / "genetic_signal.csv").exists() else None
    if gs is not None:
        g = gs.groupby("arm").agg(genetic_part=("genetic_part", "mean"), genetic_part_min=("genetic_part", "min"),
                                  genetic_part_max=("genetic_part", "max"), genetic_ci_low_min=("ci_low", "min")).reset_index()
        at = at.merge(g, on="arm", how="left")
    cur.to_csv(RES / "pilot_curves.csv", index=False, float_format="%.6g")
    rt.to_csv(RES / "pilot_runs.csv", index=False, float_format="%.6g")
    at.to_csv(RES / "pilot_arms.csv", index=False, float_format="%.6g")
    et.to_csv(RES / "pilot_equal_time.csv", index=False, float_format="%.6g")
    dt.to_csv(RES / "pilot_per_disease.csv", index=False, float_format="%.6g")
    with open(RES / "pilot_table.md", "w") as f:
        f.write(markdown_table(rt, at, gs))
    trials = RES / "capacity_trials.csv"
    if trials.exists():
        cap = capacity_table(pd.read_csv(trials))
        cap.to_csv(RES / "capacity_summary.csv", index=False, float_format="%.6g")
        print(cap[["config", "n_parameters", "largest_batch", "peak_gib_at_largest", "gib_per_participant",
                   "samples_per_s_fastest", "gpu_hours_per_epoch_equiv"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    style()
    fig_curves(cur, runs[0]["config"]["n_train"])
    fig_arms(rt, at, cov_mean)
    if "default" in dt:
        fig_diseases(dt)
    if gs is not None:
        fig_genetic(gs)
    pd.set_option("display.width", 250)
    print(at.drop(columns=["differs", "seeds"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(et):
        print(et.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"covariates-only mean tuning AUROC {cov_mean:.4f}, mean cross-entropy {cov_bce:.4f}; runs {len(runs)}")


if __name__ == "__main__":
    main()
