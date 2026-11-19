"""Figure for the chromosome-22 haploblock pre-test: common haplotypes per haploblock, and the haplotype ridge's
AUROC gain over the SNP ridge per disease, with 95% paired bootstrap intervals (methods.md section 6).

    python make_figure.py --results results --chrom 22 --cell f0.01_c0.95

Writes results/figure_chr<chrom>_<cell>.{png,pdf}, the plotted numbers as _data.csv and a legend as _legend.md.
Nature single-column widths are 89 mm, double 183 mm; every label is 7 pt."""
import argparse
import os

import numpy as np
import pandas as pd

MM = 1 / 25.4
INK, INK2, AXIS, GRID = "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
SERIES = {"hap_minus_snp": ("#2a78d6", "o", "haplotype score (B) minus SNP score (A)"),
          "both_minus_snp": ("#d9731a", "s", "both (C) minus SNP score (A)")}
SETS = {"oof_train": "out-of-fold training rows", "val": "tuning set"}
METHODS = {"ridge": ("compare", "ridge, fitted per Privé block with the penalty chosen per disease on the out-of-fold training AUROC"),
           "ct": ("compare_ct", "clumping-and-thresholding score (marginal effects summed over the clumped columns passing the threshold, the r² and threshold chosen per disease on the out-of-fold training AUROC)")}


def style(plt):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
                         "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 450})


def long_table(table, pooled):
    """The plotted numbers in one long table: per disease and pooled, per set and comparison, with the interval."""
    rows = []
    for _, r in table.iterrows():
        for s in SETS:
            for c in SERIES:
                k = f"dauroc_{c}_{s}"
                rows.append(dict(label=r["label"], cases_train=int(r["cases_train"]), set=s, comparison=c,
                                 estimate=r[k], lo=r[f"{k}_lo"], hi=r[f"{k}_hi"]))
    for _, r in pooled.iterrows():
        rows.append(dict(label="pooled (mean of 30)", cases_train=np.nan, set=r["set"], comparison=r["comparison"],
                         estimate=r["mean_diff"], lo=r["lo"], hi=r["hi"]))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--chrom", default="22")
    ap.add_argument("--cell", default="f0.01_c0.95")
    ap.add_argument("--method", choices=list(METHODS), default="ridge")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    style(plt)
    seg = pd.read_csv(os.path.join(a.results, f"haploblocks_chr{a.chrom}_{a.cell}_segments.csv"))
    stem_in, what = METHODS[a.method]
    table = pd.read_csv(os.path.join(a.results, f"{stem_in}_chr{a.chrom}_{a.cell}.csv"))
    pooled = pd.read_csv(os.path.join(a.results, f"{stem_in}_chr{a.chrom}_{a.cell}_pooled.csv"))
    table = table[table["penalties"] == "chosen"].sort_values("cases_train", ascending=False).reset_index(drop=True)
    pooled = pooled[(pooled["penalties"] == "chosen") & (pooled["metric"] == "auroc")]
    data = long_table(table, pooled)
    counts = seg["H"].value_counts().sort_index()
    n_dis = len(table)
    height = 28 + 3.3 * (n_dis + 1)
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, height * MM), gridspec_kw=dict(width_ratios=[0.8, 1, 1], wspace=0.55))
    fig.subplots_adjust(left=0.06, right=0.99, top=1 - 7 / height, bottom=17 / height)
    ax = axes[0]
    ax.bar(counts.index, counts.values / counts.sum(), width=0.8, color="#2a78d6", linewidth=0)
    ax.axvspan(2.5, 10.5, color=GRID, alpha=0.6, zorder=0, linewidth=0)
    ax.set_xlabel("common haplotypes per haploblock (H)")
    ax.set_ylabel("share of haploblocks")
    ax.set_title(f"a  {len(seg):,} haploblocks, chr{a.chrom}", loc="left")
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    y = np.arange(n_dis + 1)[::-1]
    labels = list(table["label"]) + ["pooled"]
    for ax, (s, name) in zip(axes[1:], SETS.items()):
        for k, (c, (colour, marker, _)) in enumerate(SERIES.items()):
            d = data[(data["set"] == s) & (data["comparison"] == c)]
            off = 0.18 if k == 0 else -0.18
            ax.errorbar(d["estimate"], y + off, xerr=[d["estimate"] - d["lo"], d["hi"] - d["estimate"]], fmt=marker,
                        color=colour, ms=2.6, mew=0, elinewidth=0.7, capsize=0, zorder=3)
        ax.axvline(0, color=INK2, linewidth=0.6, zorder=2)
        ax.axhline(0.5, color=AXIS, linewidth=0.5, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_ylim(-0.6, n_dis + 0.6)
        ax.set_xlabel("ΔAUROC against the SNP score")
        ax.grid(axis="y", visible=False)
    axes[1].set_title(f"b  {SETS['oof_train']}", loc="left")
    axes[2].set_title(f"c  {SETS['val']}", loc="left")
    handles = [plt.Line2D([], [], color=v[0], marker=v[1], ms=2.6, mew=0, linestyle="none", label=v[2]) for v in SERIES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.6, 0.0), handletextpad=0.3, columnspacing=1.5)
    stem = os.path.join(a.results, f"figure_{a.method}_chr{a.chrom}_{a.cell}")
    fig.savefig(stem + ".png")
    fig.savefig(stem + ".pdf")
    data["H"] = np.nan
    hist = pd.DataFrame(dict(label="haploblocks", set="train", comparison="histogram", H=counts.index, estimate=counts.values))
    pd.concat([data, hist], ignore_index=True).to_csv(stem + "_data.csv", index=False, float_format="%.5f")
    n_boot = int(pooled["n_boot"].iloc[0])
    with open(stem + "_legend.md", "w") as f:
        f.write(f"**Haplotype against SNP {a.method} on chromosome {a.chrom} (cell {a.cell}).** "
                f"**a**, Number of common haplotypes H per haploblock (training frequency at least the cell's floor; the band marks "
                f"3 to 10, the range the design assumed), over the {len(seg):,} haploblocks cut from the chromosome's "
                f"{int(seg['n_snps'].sum()):,} HapMap3+ SNPs. **b**, **c**, Per disease, the gain in AUROC over the covariate-only model "
                f"of the haplotype-count score (B) and of the score on both feature sets (C), each minus the gain of the SNP-dosage "
                f"score (A) on the same Privé blocks, cross-fitted over five folds; each score is a {what}; "
                f"**b** on the out-of-fold training rows, **c** on the tuning set. Points are the "
                f"paired difference and bars its 95% participant bootstrap interval ({n_boot:,} resamples); the last row pools the "
                f"30 diseases (mean of the per-disease differences, bootstrap over the same resamples). Diseases are ordered by "
                f"training cases, most first. Data: `{os.path.basename(stem)}_data.csv`.\n")
    print(f"wrote {stem}.png/.pdf/_data.csv/_legend.md")


if __name__ == "__main__":
    main()
