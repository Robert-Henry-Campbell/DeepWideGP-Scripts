"""Figures for the DeepGP replication (results.md). Collects the aggregate AUROCs from the run folders into
figures/*.csv, then draws each figure from its CSV (Nature widths 89 / 183 mm, Arial 7 pt; PDF and 300 dpi PNG).
Usage: python make_figures.py"""
import json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "/mnt/sdg/robert/deepwidegp/runs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
MM = 1 / 25.4
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": INK, "text.color": INK,
                     "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 1.0})


def j(p):
    return json.load(open(os.path.join(RUNS, p)))


def row(fig, panel, setting, model, a):
    return {"figure": fig, "panel": panel, "setting": setting, "model": model, "auroc": a["auroc"],
            "ci_low": a["ci95"][0], "ci_high": a["ci95"][1]}


def collect():
    e11 = j("deepgp_replication_e11/results.json")
    cM, cR = j("deepgp_no_learning/compare_M.json")["auroc"], j("deepgp_no_learning/compare_R.json")["auroc"]
    cG = j("deepgp_replication_e11_cohort_gwas/compare_R.json")["auroc"]
    af = j("deepgp_replication/results.json")
    ceil = j("deepgp_no_learning/ceiling.json")
    infl = j("deepgp_replication_e11_cohort_gwas/inflation.json")
    one = j("deepgp_replication_e11_cohort_gwas/first_epoch.json")[0]
    curve = j("deepgp_no_learning/test_curve_R.json")

    f1 = []
    for setting, c in (("1:1, controls matched on age and sex", cM), ("1:1, random controls", cR)):
        for model, key in (("DeepGP, minor-allele dosages", "DeepGP, minor-allele dosages"),
                           ("DeepGP, risk-allele dosages", "DeepGP, risk-allele dosages"),
                           ("PRS + covariates, paper's recipe", "PRS + covariates, paper's recipe"),
                           ("Covariates alone", "covariates alone")):
            f1.append(row(1, "a", setting, model, c[key]))
    for model, key in (("A weight per SNP (the PRS)", "a weight per SNP, from the GWAS (the PRS)"),
                       ("One weight shared by all SNPs,\nrisk-allele dosages", "one weight shared by all SNPs, risk-allele dosages"),
                       ("One weight shared by all SNPs,\nminor-allele dosages", "one weight shared by all SNPs, minor-allele dosages"),
                       ("One weight per chromosome,\nminor-allele dosages", "one weight per chromosome (22), minor-allele dosages"),
                       ("Position code used linearly,\nminor-allele dosages", "the position code used linearly (22 x 33 weights), minor-allele dosages")):
        f1.append(row(1, "b", "1:1, controls matched on age and sex", model,
                      {"auroc": ceil[key]["test"], "ci95": ceil[key]["test_ci95"]}))
    pd.DataFrame(f1).to_csv(f"{OUT}/fig1.csv", index=False)

    main = infl["prs"]["C+T, cohort's own GWAS, all imputed (98,440 lead SNPs)"]["recipe_auroc"]
    f2a = [{"model": "PRS + covariates, paper's recipe", **main},
           {"model": "DeepGP, risk allele (first check)", "training part": one["training part"],
            "validation part": one["validation part"], "test": one["test"]},
           {"model": "Covariates alone", **infl["covariates alone"]}]
    pd.DataFrame(f2a).to_csv(f"{OUT}/fig2a.csv", index=False)
    f2b = []
    for setting, c, keys in (
            ("Type 2 diabetes, population prevalence;\nGWAS of the same 215,872 people", e11["N"]["auroc"],
             ("covariates alone", "PRS + covariates, fitted on training rows (paper)", "PRS + covariates, cross-fitted on test rows")),
            ("Type 2 diabetes, 1:1 random controls;\nGWAS of 215,872 people", cR,
             ("covariates alone", "PRS + covariates, paper's recipe", "PRS + covariates, joined on test people (5 folds)")),
            ("Type 2 diabetes, 1:1 random controls;\nGWAS of the cohort's 34,690 people", cG,
             ("covariates alone", "PRS + covariates, paper's recipe", "PRS + covariates, joined on test people (5 folds)")),
            ("Atrial fibrillation, 1:1 random controls;\nGWAS of 215,872 people", af["R"]["auroc"],
             ("covariates alone", "PRS + covariates", "PRS + covariates, fitted on test folds"))):
        for model, key in zip(("Covariates alone", "PRS + cov., joined on the GWAS people (paper's recipe)",
                               "PRS + cov., joined on people outside the GWAS"), keys):
            f2b.append(row(2, "b", setting, model, c[key]))
    pd.DataFrame(f2b).to_csv(f"{OUT}/fig2b.csv", index=False)
    f2c = []
    for name, v in infl["prs"].items():
        gwas = "215,872 people" if "215,872" in name else "the cohort's 34,690 people"
        method = "LDpred2-auto" if name.startswith("LDpred2") else "C+T"
        n = int(re.findall(r"\(([\d,]+)", name)[-1].replace(",", ""))     # the SNP count in the name's last brackets
        f2c.append({"prs": name, "method": method, "gwas_people": gwas, "snps_in_prs": n,
                    "prs_weight_recipe": v["prs_weight_recipe"], "prs_weight_on_new_people": v["prs_weight_on_new_people"],
                    "over_weighting": v["inflation"], "recipe_test_auroc": v["recipe_auroc"]["test"]})
    pd.DataFrame(f2c).to_csv(f"{OUT}/fig2c.csv", index=False)
    pd.DataFrame([{"pass": r["epoch"] + 1, "training_auroc_per_batch": r["train_auroc"], "validation_auroc": r["val_auroc"],
                   "test_auroc": r["test_auroc"]} for r in curve]).to_csv(f"{OUT}/fig3.csv", index=False)
    pd.DataFrame([{"reference": k, "test_auroc": cR[v]["auroc"]} for k, v in (
        ("Covariates alone", "covariates alone"), ("PRS + covariates, paper's recipe", "PRS + covariates, paper's recipe"),
        ("PRS + covariates, joined on people outside the GWAS", "PRS + covariates, joined on test people (5 folds)"))]
                 ).to_csv(f"{OUT}/fig3_references.csv", index=False)


def letter(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="right", va="bottom")


def dots(ax, labels, d, color, filled=True, dy=0.0, size=4):
    y = np.arange(len(labels))[::-1] + dy
    ax.hlines(y, d.ci_low, d.ci_high, color=color, lw=1.0)
    ax.plot(d.auroc, y, "o", ms=size, color=color, mfc=color if filled else "white", mew=1.0, zorder=3)
    return y


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300)
    plt.close(fig)


def figure1():
    d = pd.read_csv(f"{OUT}/fig1.csv")
    fig = plt.figure(figsize=(183 * MM, 105 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.5, wspace=0.08, left=0.255, right=0.98, top=0.93, bottom=0.09)
    models = ["DeepGP, minor-allele dosages", "DeepGP, risk-allele dosages", "PRS + covariates, paper's recipe", "Covariates alone"]
    colors = {models[0]: BLUE, models[1]: BLUE, models[2]: ORANGE, models[3]: GREY}
    for k, setting in enumerate(["1:1, controls matched on age and sex", "1:1, random controls"]):
        ax = fig.add_subplot(gs[0, k])
        for i, m in enumerate(models):
            r = d[(d.panel == "a") & (d.setting == setting) & (d.model == m)]
            y = len(models) - 1 - i
            ax.hlines(y, r.ci_low, r.ci_high, color=colors[m], lw=1.0)
            ax.plot(r.auroc, y, "o", ms=4, color=colors[m], mfc="white" if "minor" in m else colors[m], mew=1.0, zorder=3)
            ax.text(r.ci_high.iloc[0] + 0.004, y, f"{r.auroc.iloc[0]:.3f}", va="center", fontsize=7, color=MUTED)
        for x, lab, c in ((0.64, "Paper's DeepGP (0.64)", BLUE), (0.60, "Paper's PRS + covariates (0.60)", ORANGE)):
            ax.axvline(x, color=c, lw=0.6, ls=(0, (3, 2)), zorder=1, label=lab)
        ax.axvline(0.5, color=GREY, lw=0.5, zorder=0)
        ax.set_xlim(0.46, 0.73)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models[::-1] if k == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_title(setting, loc="left", color=INK)
        ax.set_xlabel("AUROC on new people")
        if k == 0:
            letter(ax, "a")
            first = ax
    pos = ax.get_position()
    fig.legend(*first.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(pos.x1, pos.y0 - 0.075), ncol=2,
               handlelength=1.6)
    ax = fig.add_subplot(gs[1, :])
    b = d[d.panel == "b"].reset_index(drop=True)
    for i, r in b.iterrows():
        y = len(b) - 1 - i
        minor = "minor" in r.model
        ax.hlines(y, r.ci_low, r.ci_high, color=MUTED, lw=1.0)
        ax.plot(r.auroc, y, "o", ms=4, color=MUTED, mfc="white" if minor else MUTED, mew=1.0, zorder=3)
        ax.text(r.ci_high + 0.003, y, f"{r.auroc:.3f}", va="center", fontsize=7, color=MUTED)
    ax.axvline(0.5, color=GREY, lw=0.5, zorder=0)
    ax.set_yticks(range(len(b)))
    ax.set_yticklabels(b.model[::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.46, 0.66)
    ax.set_ylim(-0.6, len(b) - 0.4)
    ax.set_xlabel("AUROC on new people")
    ax.set_title("Linear models of the same SNPs, matched cohort: what one weight per SNP buys", loc="left")
    letter(ax, "b")
    save(fig, "fig1_allele_coding")


def figure2():
    a = pd.read_csv(f"{OUT}/fig2a.csv")
    b = pd.read_csv(f"{OUT}/fig2b.csv")
    c = pd.read_csv(f"{OUT}/fig2c.csv")
    fig = plt.figure(figsize=(183 * MM, 125 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1, 1], hspace=0.5, wspace=0.35,
                          left=0.075, right=0.98, top=0.94, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0])
    groups = ["training part", "validation part", "test"]
    ax.axvspan(-0.4, 1.5, color="#f1f0ec", zorder=0)
    ax.text(0.55, 0.515, "in the GWAS", color=MUTED, ha="center", fontsize=7)
    ax.text(2.0, 0.515, "outside it", color=MUTED, ha="center", fontsize=7)
    for (_, r), col, mk in zip(a.iterrows(), (ORANGE, BLUE, GREY), ("o", "s", "D")):
        ax.plot(range(3), [r[g] for g in groups], marker=mk, ms=3.5, color=col, label=r.model, zorder=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Training\npeople", "Validation\npeople", "New\npeople"])
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUROC")
    ax.set_title("Type 2 diabetes, 1:1 random controls, GWAS of the cohort itself", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.56), handlelength=1.6)
    letter(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    styles = {"C+T|the cohort's 34,690 people": (ORANGE, "o", "white"), "C+T|215,872 people": (ORANGE, "s", ORANGE),
              "LDpred2-auto|the cohort's 34,690 people": (AQUA, "D", "white")}
    for key, g in c.groupby(c.method + "|" + c.gwas_people, sort=False):
        col, mk, fc = styles[key]
        g = g.sort_values("snps_in_prs")
        ax.plot(g.snps_in_prs, g.over_weighting, marker=mk, ms=4, color=col, mfc=fc, mew=1.0, lw=0.8 if len(g) > 1 else 0)
    full = c[(c.method == "C+T") & (c.snps_in_prs == c[c.method == "C+T"].snps_in_prs.max())].iloc[0]
    big = c[c.gwas_people == "215,872 people"].iloc[0]
    ld = c[c.method == "LDpred2-auto"].iloc[0]
    ax.text(5000, 13.0, "C+T, GWAS of the cohort's 34,690 people\n(random 5-100% of its lead SNPs)", fontsize=7, color=INK, va="bottom")
    ax.text(big.snps_in_prs, big.over_weighting - 1.3, "C+T, GWAS of\n215,872 people", fontsize=7, ha="center", va="top")
    ax.text(ld.snps_in_prs, ld.over_weighting - 1.3, "LDpred2-auto,\ncohort's GWAS", fontsize=7, ha="center", va="top")
    ax.set_xscale("log")
    ax.set_xlim(3000, 3e6)
    ax.set_ylim(0, 16)
    ax.set_xlabel("SNPs in the PRS")
    ax.set_ylabel("PRS weight in the recipe ÷\nweight new people support")
    ax.set_title("PRS over-weighting: set by GWAS size, not SNP count", loc="left")
    letter(ax, "b")

    ax = fig.add_subplot(gs[1, :])
    settings = list(dict.fromkeys(b.setting))
    models = list(dict.fromkeys(b.model))
    cols, offs = (GREY, ORANGE, AQUA), (0.22, 0.0, -0.22)
    for m, col, off in zip(models, cols, offs):
        d = b[b.model == m].set_index("setting").loc[settings].reset_index()
        y = np.arange(len(settings))[::-1] + off
        ax.hlines(y, d.ci_low, d.ci_high, color=col, lw=1.0)
        ax.plot(d.auroc, y, "o", ms=4, color=col, label=m, zorder=3)
        for yy, v, hi in zip(y, d.auroc, d.ci_high):
            ax.text(hi + 0.002, yy, f"{v:.3f}", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(len(settings)))
    ax.set_yticklabels(settings[::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.63, 0.87)
    ax.set_ylim(-0.55, len(settings) - 0.45)
    ax.set_xlabel("AUROC on new people")
    ax.set_title("Joining the PRS to the covariates on the GWAS people pushes PRS + covariates below covariates alone", loc="left")
    ax.legend(loc="upper right", handlelength=1.0, borderaxespad=0.1)
    fig.subplots_adjust(left=0.075)
    pos = ax.get_position()
    ax.set_position([0.30, pos.y0, 0.68, pos.height])
    letter(ax, "c")
    save(fig, "fig2_prs_recipe")


def figure3():
    d = pd.read_csv(f"{OUT}/fig3.csv")
    ref = pd.read_csv(f"{OUT}/fig3_references.csv").set_index("reference").test_auroc
    fig, (top, bot) = plt.subplots(2, 1, figsize=(89 * MM, 95 * MM), sharex=True, gridspec_kw={"hspace": 0.25})
    chosen = int(d.loc[d.validation_auroc.idxmax(), "pass"])
    top.plot(d["pass"], d.training_auroc_per_batch, color=GREY, marker="o", ms=2.5, label="Training people (batch mean)")
    top.plot(d["pass"], d.validation_auroc, color=BLUE, marker="o", ms=2.5, label="Validation people")
    top.set_ylim(0.9, 1.0)
    top.set_ylabel("AUROC, people in the GWAS")
    top.legend(loc="upper left", handlelength=1.6)
    letter(top, "a")
    for (k, col), lab in zip(((("Covariates alone"), GREY), ("PRS + covariates, paper's recipe", ORANGE),
                              ("PRS + covariates, joined on people outside the GWAS", AQUA)),
                             ("Covariates alone", "PRS + cov., paper's recipe", "PRS + cov., joined outside the GWAS")):
        bot.axhline(ref[k], color=col, lw=0.8, ls=(0, (3, 2)))
        bot.text(d["pass"].max() + 0.3, ref[k] + 0.0008, f"{lab} {ref[k]:.3f}", va="bottom", ha="right", fontsize=7, color=INK)
    bot.plot(d["pass"], d.test_auroc, color=BLUE, marker="o", ms=2.5)
    bot.axvline(chosen, color=MUTED, lw=0.5, ls=":")
    bot.text(chosen + 0.3, 0.712, f"pass the code\nkeeps ({chosen})", fontsize=7, ha="left", va="bottom", color=MUTED)
    bot.set_xticks([1, 5, 10, 15, 20])
    bot.set_ylim(0.62, 0.73)
    bot.set_ylabel("AUROC, new people")
    bot.set_xlabel("Training pass")
    bot.set_xlim(0.5, d["pass"].max() + 0.5)
    letter(bot, "b")
    fig.subplots_adjust(left=0.17, right=0.97, top=0.95, bottom=0.1)
    save(fig, "fig3_training_passes")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    collect()
    figure1()
    figure2()
    figure3()
    print("written to", OUT)
