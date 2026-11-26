"""Tables and figures of the DeepGP summary for its authors (results.md): the E11 replication's overfitting table, the
LDpred2 in-sample check across the ladder, and the five-rung comparison of WideDeepGP, classic DeepGP and C+T.
Collects aggregate AUROCs from the run folders and the PGI arm into figures/*.csv, then draws the ladder figures
(Nature widths 89 / 183 mm, Arial 7 pt; PDF and 300 dpi PNG). Usage: python make_summary.py"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/sdg/robert/deepwidegp"
RUNS, PGI, SCRIPTS = f"{ROOT}/runs", f"{ROOT}/PGI/results", f"{ROOT}/DeepWideGP-Scripts/scripts"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RUNGS = {"hm3plus_n13199": 13199, "hm3plus_n26287": 26287, "hm3plus_n53498": 53498, "hm3plus_n107743": 107743,
         "hm3plus": 215872}
NAMES = {"I10": "Hypertension", "E78": "Lipid disorders", "K57": "Diverticular disease", "K21": "Reflux disease",
         "K29": "Gastritis", "K44": "Diaphragmatic hernia", "I25": "Ischaemic heart disease",
         "M19": "Other arthrosis", "I84+K64": "Haemorrhoids", "I48": "Atrial fibrillation", "M17": "Knee arthrosis",
         "C44": "Other skin cancer", "E11": "Type 2 diabetes", "E66": "Obesity", "M54": "Back pain",
         "D12": "Benign bowel tumour", "I20": "Angina", "E03": "Hypothyroidism", "K40": "Inguinal hernia",
         "M16": "Hip arthrosis", "K80": "Gallstones", "N17": "Acute kidney failure", "N18": "Chronic kidney disease",
         "J44": "COPD", "M23": "Knee derangement", "M47": "Spondylosis", "I95": "Hypotension", "I50": "Heart failure",
         "M81": "Osteoporosis", "I21": "Heart attack"}
MM = 1 / 25.4
BLUE, ORANGE, AQUA, INK, MUTED, RULE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#c9c8c3"
SERIES = [("gain_widedeepgp", "WideDeepGP", BLUE, "o"), ("gain_classic_deepgp", "Classic DeepGP", ORANGE, "s"),
          ("gain_ct", "C+T PRS + covariates", AQUA, "^")]
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "xtick.minor.size": 0, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "text.color": INK, "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "pdf.fonttype": 42, "lines.linewidth": 1.0})


def j(path):
    return json.load(open(path))


def scores(path):
    """Tuning AUROC per disease of a run's best model and of its covariates-only model, and tuning cases."""
    d = pd.read_csv(path)
    d = d[(d.split == "val") & (d.metric == "auroc") & (d.label != "MEAN")]
    return {s: dict(zip(g.label, g.value)) for s, g in d.groupby("snapshot")}, dict(zip(d.label, d.n_cases))


def ladder():
    rows = []
    for rung, n in RUNGS.items():
        runs = [scores(f"{RUNS}/widedeepgp_ladder/{rung}_s{s}/scores_val.csv") for s in (1, 2, 3)]
        cov = runs[0][0]["covariates_only"]
        assert all(abs(r[0]["covariates_only"][d] - cov[d]) < 1e-9 for r in runs for d in cov)
        ct = pd.read_csv(f"{PGI}/{rung}/tune_metrics.csv")
        ct = ct[(ct.method == "c_t") & (ct.basis == "linear") & (ct.status == "ok")].set_index("trait")
        ridge = pd.read_csv(f"{SCRIPTS}/widedeepgp-blocks/results/{rung}/blocks_linear.csv").set_index("label")
        assert len(cov) == 30 and set(ct.index) == set(cov) == set(ridge.index)
        for d in cov:
            wd = [r[0]["best"][d] for r in runs]
            rows.append({"rung": rung, "training_people": n, "disease": d, "name": NAMES[d],
                         "tuning_cases": int(runs[0][1][d]), "covariates": cov[d],
                         "classic_deepgp": j(f"{RUNS}/classic_deepgp_ladder/results/{rung}__{d}.json")["val_auroc"],
                         "widedeepgp_s1": wd[0], "widedeepgp_s2": wd[1], "widedeepgp_s3": wd[2],
                         "widedeepgp": float(np.mean(wd)), "ridge_prs_plus_covariates": ridge.loc[d, "auroc_val_ridge_prs"],
                         "covariates_tuning_fit": ct.loc[d, "auroc_covariates"],
                         "ct_plus_covariates": ct.loc[d, "auroc_combined"], "ct_alone": ct.loc[d, "auroc_pgi"]})
    t = pd.DataFrame(rows)
    t["gain_widedeepgp"] = t.widedeepgp - t.covariates
    t["gain_classic_deepgp"] = t.classic_deepgp - t.covariates
    t["gain_ct"] = t.ct_plus_covariates - t.covariates_tuning_fit
    t.to_csv(f"{OUT}/ladder_per_disease.csv", index=False, float_format="%.4f")
    num = [c for c in t.columns if c not in ("rung", "disease", "name", "tuning_cases", "training_people")]
    m = t.groupby("training_people")[num].mean()
    m["widedeepgp_seed_sd"] = m[["widedeepgp_s1", "widedeepgp_s2", "widedeepgp_s3"]].std(axis=1, ddof=1)
    m["classic_deepgp_above_covariates"] = (t.gain_classic_deepgp > 0).groupby(t.training_people).sum()
    m["widedeepgp_above_covariates"] = (t.gain_widedeepgp > 0).groupby(t.training_people).sum()
    m["widedeepgp_above_classic"] = (t.widedeepgp > t.classic_deepgp).groupby(t.training_people).sum()
    m.reset_index().to_csv(f"{OUT}/ladder_means.csv", index=False, float_format="%.4f")
    return t, m.reset_index()


def overfit():
    """E11, 1:1 random-controls cohort: AUROC on the GWAS people and on new people, and the recipe's over-weighting."""
    c = f"{RUNS}/deepgp_replication_e11_cohort_gwas"
    infl, po, one = j(f"{c}/inflation.json")["prs"], j(f"{c}/pgi_overfit.json"), j(f"{c}/first_epoch.json")[0]
    kept = (0.995, 0.992)  # the checkpoint the code chose (pass 15), as recorded in that run's results.md
    first215 = j(f"{RUNS}/deepgp_no_learning/test_curve_R.json")[0]
    test = lambda path: j(path)["auroc"]["DeepGP, risk-allele dosages"]["auroc"]
    own = infl["C+T, cohort's own GWAS, all imputed (98,440 lead SNPs)"]
    ld = infl["LDpred2-auto, cohort's own GWAS, HapMap3+ (1,258,200 SNPs)"]
    big = infl["C+T, GWAS of 215,872 people, HapMap3+ (47,341 lead SNPs)"]
    las = po["lassosum2, grid point chosen on the validation part"]

    def r(gwas, model, tr, va, te, prs=None):
        return {"gwas_people": gwas, "model": model, "training_part": tr, "validation_part": va, "new_people": te,
                "prs_overweighting": prs["inflation"] if prs else None,
                "covariate_share_recipe": prs["covariate_share_recipe"] if prs else None,
                "covariate_share_new_people": prs["covariate_share_on_new_people"] if prs else None}

    rows = [r(34690, "C+T PRS alone", own["prs_alone_auroc"]["gwas people"], None, own["prs_alone_auroc"]["test"]),
            r(34690, "C+T PRS + covariates, paper's recipe", *own["recipe_auroc"].values(), own),
            r(34690, "LDpred2-auto alone", *po["ldpred2_auto"]["score alone"].values()),
            r(34690, "LDpred2-auto + covariates, recipe", *po["ldpred2_auto"]["score + covariates"].values(), ld),
            r(34690, "lassosum2 alone", *las["score alone"].values()),
            r(34690, "lassosum2 + covariates, recipe", *las["score + covariates"].values()),
            r(34690, "DeepGP, risk allele, first check (300 batches)", one["training part"], one["validation part"],
              one["test"]),
            r(34690, "DeepGP, risk allele, model kept (pass 15 of 25)", *kept, test(f"{c}/compare_R.json")),
            r(34690, "Covariates alone", *po["covariates alone"].values()),
            r(215872, "C+T PRS alone", big["prs_alone_auroc"]["gwas people"], None, big["prs_alone_auroc"]["test"]),
            r(215872, "C+T PRS + covariates, paper's recipe", *big["recipe_auroc"].values(), big),
            r(215872, "DeepGP, risk allele, first pass", first215["train_auroc"], first215["val_auroc"],
              first215["test_auroc"]),
            r(215872, "DeepGP, risk allele, model kept", None, None, test(f"{RUNS}/deepgp_no_learning/compare_R.json"))]
    pd.DataFrame(rows).to_csv(f"{OUT}/e11_overfit.csv", index=False, float_format="%.4f")
    ck = pd.DataFrame(j(f"{RUNS}/kelemen_mlp/ladder_prs/check_prs_inputs.json"))
    ck["training_people"] = ck.rung.map(RUNGS)
    ck[["training_people", "trait", "auroc_training_rows", "auroc_tuning"]].to_csv(
        f"{OUT}/ldpred2_in_sample_by_rung.csv", index=False, float_format="%.4f")


def xaxis(ax, labels=True):
    ax.set_xscale("log")
    ax.set_xticks([13199, 53498, 215872] if not labels or labels == "short" else list(RUNGS.values()))
    ax.set_xticklabels(["13k", "53k", "216k"] if labels == "short" else
                       ["13,199", "26,287", "53,498", "107,743", "215,872"] if labels else [])
    ax.minorticks_off()
    ax.set_xlim(11000, 260000)


def fig_means(m):
    fig, ax = plt.subplots(figsize=(89 * MM, 62 * MM))
    ax.axhline(0, color=RULE, lw=0.75, zorder=1)
    ax.text(12500, -0.004, "covariates alone", ha="left", va="top", color=MUTED)
    x = m.training_people.values
    for col, label, color, marker in SERIES:
        y = m[col].values
        ax.plot(x, y, color=color, marker=marker, ms=3.5, mec="white", mew=0.5, lw=1.25, zorder=3)
        ax.annotate(label, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", color=INK)
    xaxis(ax)
    ax.set_xlim(11000, 215872 * 1.04)
    ax.tick_params(axis="x", rotation=0)
    ax.set_xlabel("Training participants")
    ax.set_ylabel("Mean AUROC gain over covariates alone\n(tuning set, 30 diseases)")
    ax.grid(axis="y", color=RULE, lw=0.4, alpha=0.6)
    fig.subplots_adjust(left=0.2, right=0.68, bottom=0.17, top=0.97)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_ladder_means.{ext}", dpi=300)
    plt.close(fig)


def fig_per_disease(t):
    lo, hi = t[[s[0] for s in SERIES]].min().min(), t[[s[0] for s in SERIES]].max().max()
    fig, axes = plt.subplots(6, 5, figsize=(183 * MM, 175 * MM), sharex=True, sharey=True)
    for ax, d in zip(axes.flat, t.disease.unique()):
        s = t[t.disease == d]
        ax.axhline(0, color=RULE, lw=0.6, zorder=1)
        for col, label, color, marker in SERIES:
            ax.plot(s.training_people, s[col], color=color, marker=marker, ms=2.2, mec="white", mew=0.3, lw=0.9,
                    label=label, zorder=3)
        ax.set_title(f"{d} {NAMES[d]}", loc="left", pad=2)
        xaxis(ax, "short")
        ax.set_ylim(lo - 0.004, hi + 0.004)
    for ax in axes[:, 0]:
        ax.set_ylabel("Gain over\ncovariates")
    for ax in axes[-1, :]:
        ax.set_xlabel("Training people")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.06, top=0.94, wspace=0.12, hspace=0.45)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_ladder_per_disease.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t, m = ladder()
    overfit()
    fig_means(m)
    fig_per_disease(t)
    print(m.round(4).to_string())
