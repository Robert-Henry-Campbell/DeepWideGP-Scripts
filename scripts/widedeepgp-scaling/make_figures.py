#!/usr/bin/env python3
"""Figures of the scaling study (plan section 6, "Figures"), each saved by pgi.report.save at a
Nature width (89 or 183 mm, every text >= 7 pt) with its data CSV and legend .md beside it.

    python make_figures.py --results <dir written by fit_curves.py fit> --tidy <tidy.csv>
                           [--compute <dir of compute.py CSVs>] --out <dir>

Figures: curves_by_family (gain and liability R2 against N_eff by family, pooled over diseases),
curves_per_disease, gap_vs_x (with the P2 fit, x* and the spline), size_axis (gain against mixer
size per rung with N_opt(D)), table_axis (d_e), blocks_axis (B_opt(D)), compute_fixed_data
(tuning AUROC against training FLOPs at the top rung), transfer_offset (held-out groups against
the tuning set by D), and the ablation table (Markdown and CSV).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

PGI = os.environ.get("PGI_REPO", "/mnt/sdg/robert/deepwidegp/PGI")
sys.path.insert(0, PGI)
from pgi import report as R  # noqa: E402

DEEP_KEYS = ["mixer", "n_mixer_params", "embed_dim", "n_blocks"]


def _style(family):
    if family == "deep":
        return dict(color=R.C1, lw=1.6, zorder=5, label="WideDeepGP (deep envelope)")
    if family == "lin":
        return dict(color=R.INK, lw=1.4, zorder=4, label="strongest linear method")
    if family.startswith("twin:"):
        return dict(color=R.C3, lw=1.2, ls="--", zorder=3, label=family)
    if family.startswith("deep:"):
        return dict(color=R.C1, lw=0.7, alpha=0.45, zorder=2, label=family)
    return dict(color=R.MUTED, lw=0.8, zorder=1, label=R.METHOD_LABELS.get(family, family))


def _pooled(cells):
    """Per family and rung: the mean over diseases of each value and the median N_eff."""
    agg = dict(n_eff=("n_eff", "median"), delta_auroc=("delta_auroc", "mean"), delta_r2=("delta_r2", "mean"))
    if "seed_lo" in cells:
        agg.update(seed_lo=("seed_lo", "mean"), seed_hi=("seed_hi", "mean"))
    for k in DEEP_KEYS:
        if k in cells:
            agg[k] = (k, "first")
    return cells.groupby(["family", "rung"]).agg(**agg).reset_index().sort_values(["family", "n_eff"])


def curves_by_family(cells, families=None):
    """(a) gain over the covariates and (b) liability R2 against median N_eff, one line per family."""
    p = _pooled(cells)
    families = families or [f for f in ("deep", "lin") if f in set(p.family)] + sorted(
        f for f in set(p.family) if not f.startswith("deep") and f != "lin")
    fig, axes = R.new_figure(183, 70, ncols=2)
    rows = []
    for ax, col, letter, ylab in zip(axes, ("delta_auroc", "delta_r2"), "ab", ("Gain in AUROC over the covariates", "Liability-scale R2 beyond the covariates")):
        for fam in families:
            s = p[p.family == fam]
            if not len(s):
                continue
            ax.plot(s.n_eff, s[col], marker="o", gid=f"family:{fam}", **_style(fam))
            if fam == "deep" and "seed_lo" in s and col == "delta_auroc":
                ax.fill_between(s.n_eff, s.seed_lo, s.seed_hi, color=R.C1, alpha=0.15, lw=0, gid="band:seeds")
            rows += [dict(panel=letter, family=fam, rung=r.rung, n_eff=r.n_eff, value=r[col],
                          seed_lo=r.get("seed_lo", np.nan) if col == "delta_auroc" else np.nan,
                          seed_hi=r.get("seed_hi", np.nan) if col == "delta_auroc" else np.nan) for _, r in s.iterrows()]
        ax.set_xscale("log")
        ax.set_xlabel("Median N_eff of the training set")
        ax.set_ylabel(ylab)
        R.panel(ax, letter)
    axes[0].legend(fontsize=7, loc="upper left")
    legend = ("**Figure | Learning curves by model family, pooled over diseases.** (**a**) Gain in tuning-set AUROC over the "
              "covariate-only model and (**b**) its liability-scale R2 against the median effective sample size of the training "
              "set, mean over the 30 diseases. The deep envelope (per rung the mixer size and table width with the highest mean "
              "tuning AUROC) is shown with its seed range; the strongest linear method is PGI methods 6.3's rule; the seven PGI "
              "methods are grey. Every value is a tuning-set number until the unlock. Data: the accompanying _data.csv.")
    return fig, pd.DataFrame(rows), legend


def curves_per_disease(cells, families=("deep", "lin"), ncols=6):
    """Gain against N_eff per disease, with bootstrap intervals where present."""
    diseases = sorted(cells.disease.unique())
    nrows = -(-len(diseases) // ncols)
    fig, axes = R.new_figure(183, 22 + 28 * nrows, nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    rows = []
    for ax, d in zip(axes, diseases):
        for fam in families:
            s = cells[(cells.disease == d) & (cells.family == fam)].sort_values("n_eff")
            if not len(s):
                continue
            ax.plot(s.n_eff, s.delta_auroc, marker="o", markersize=3, gid=f"{d}:{fam}", **_style(fam))
            if "delta_auroc_lo" in s and np.isfinite(s.delta_auroc_lo).all():
                ax.fill_between(s.n_eff, s.delta_auroc_lo, s.delta_auroc_hi, color=_style(fam)["color"], alpha=0.15, lw=0)
            rows += [dict(disease=d, family=fam, rung=r.rung, n_eff=r.n_eff, delta_auroc=r.delta_auroc,
                          lo=r.get("delta_auroc_lo", np.nan), hi=r.get("delta_auroc_hi", np.nan)) for _, r in s.iterrows()]
        ax.set_xscale("log")
        ax.set_title(d, loc="left", pad=2)
    for ax in axes[len(diseases):]:
        ax.set_visible(False)
    fig.supxlabel("N_eff of the training set", fontsize=7, color=R.INK2)
    fig.supylabel("Gain in AUROC over the covariates", fontsize=7, color=R.INK2)
    legend = ("**Figure | Learning curves per disease.** Tuning-set gain in AUROC over the covariate-only model against the "
              "disease's own N_eff at each rung, for the deep envelope (blue) and the strongest linear method (black); shaded "
              "bands are 95% bootstrap intervals over tuning participants where computed. Data: the accompanying _data.csv.")
    return fig, pd.DataFrame(rows), legend


def gap_vs_x(gap, pooled, spline=None):
    """The deep-minus-linear gap against x = N_eff h2_50 / M_e with the P2 line and x*."""
    fig, ax = R.new_figure(89, 65)
    ax.axhline(0, color=R.AXIS, lw=0.8)
    ax.scatter(gap.x, gap.delta, s=9, color=R.C1, alpha=0.7, lw=0, gid="points", label="disease x rung")
    rows = [dict(series="point", disease=r.disease, rung=r.rung, x=r.x, value=r.delta) for r in gap.itertuples()]
    p = pooled.set_index("statement")
    key = "P2 d (gap vs log x, random intercept)"
    if key in p.index and "intercept" in p and np.isfinite(p.loc[key, "intercept"]):
        c, d = float(p.loc[key, "intercept"]), float(p.loc[key, "estimate"])
        xs = np.logspace(np.log10(gap.x.min()), np.log10(gap.x.max()), 50)
        ax.plot(xs, c + d * np.log(xs), color=R.INK, lw=1.2, gid="fit:p2", label=f"P2 fit, d = {d:.4f}")
        rows += [dict(series="p2_fit", x=x, value=c + d * np.log(x)) for x in xs]
    if spline is not None and len(spline):
        ax.plot(spline.x, spline.fitted, color=R.C2, lw=1.0, ls="--", gid="fit:spline", label="spline sensitivity")
        rows += [dict(series="spline", x=r.x, value=r.fitted) for r in spline.itertuples()]
    xkey = "P2 x* (root of c + d log x)"
    if xkey in p.index and np.isfinite(p.loc[xkey, "estimate"]):
        xs_, lo, hi = (float(p.loc[xkey, k]) for k in ("estimate", "lo", "hi"))
        ax.axvline(xs_, color=R.C2, lw=0.8, gid="xstar")
        if np.isfinite(lo) and np.isfinite(hi):
            ax.axvspan(max(lo, 1e-6), hi, color=R.C2, alpha=0.1, lw=0)
        rows.append(dict(series="x_star", x=xs_, lo=lo, hi=hi))
    ax.set_xscale("log")
    ax.set_xlabel("x = N_eff h2_50 / M_e (M_e = 60,000)")
    ax.set_ylabel("Gap in AUROC gain, deep minus strongest linear")
    ax.legend(fontsize=7, loc="best")
    legend = ("**Figure | The deep-minus-linear gap against the information per locus.** Each point is one disease at one "
              "rung: the deep envelope's gain in tuning AUROC minus the strongest linear method's, against x = N_eff h2_50 / "
              "M_e with M_e = 60,000 as a unit. Black line: the P2 fit c + d log x with a random intercept per disease; dashed: "
              "the three-knot spline sensitivity; vertical line and band: x* with its bootstrap interval, where reported. Data: "
              "the accompanying _data.csv.")
    return fig, pd.DataFrame(rows), legend


def _axis_figure(cells, size, axis, xcol, xlabel, width, title):
    """Gain (mean over diseases) against one size axis, one panel per rung, N_opt marked."""
    p = _pooled(cells[cells.family.str.startswith("deep:")])
    p = p[np.isfinite(p[xcol])] if xcol in p else p.iloc[0:0]
    if p[xcol].nunique() < 2 if len(p) else True:              # one value on this axis: nothing to compare
        p = p.iloc[0:0]
    rungs = sorted(p.rung.unique(), key=lambda r: p[p.rung == r].n_eff.iloc[0])
    fig, axes = R.new_figure(width, 55, ncols=max(1, len(rungs)), sharey=True)
    axes = np.atleast_1d(axes)
    opt = size[size.axis == axis] if len(size) and "axis" in size else size.iloc[0:0]
    rows = []
    for k, (ax, rung) in enumerate(zip(axes, rungs)):
        s = p[p.rung == rung]
        for i, (mixer, sm) in enumerate(s.groupby("mixer")):
            sm = sm.sort_values(xcol)
            ax.plot(sm[xcol], sm.delta_auroc, marker="o", color=(R.C1, R.C2, R.C3)[i % 3], gid=f"{rung}:{mixer}", label=mixer)
            rows += [dict(panel=R._letter(k), rung=rung, mixer=mixer, family=r.family, x=r[xcol], value=r.delta_auroc)
                     for _, r in sm.iterrows()]
            o = opt[(opt.rung == rung) & (opt.mixer == mixer)]
            if len(o):
                best = sm[sm.family == o.n_opt.iloc[0]]
                if len(best):
                    ax.plot(best[xcol], best.delta_auroc, marker="*", markersize=9, color=R.INK, lw=0, gid=f"opt:{rung}:{mixer}")
                    rows.append(dict(panel=R._letter(k), rung=rung, mixer=mixer, family=o.n_opt.iloc[0], x=float(best[xcol].iloc[0]),
                                     value=float(best.delta_auroc.iloc[0]), series="n_opt", size_matters=bool(o.size_matters.iloc[0])))
        if xcol == "n_mixer_params":
            ax.set_xscale("log")
        ax.set_title(f"{rung} (median N_eff {s.n_eff.iloc[0]:,.0f})" if len(s) else rung, loc="left", pad=2)
        ax.set_xlabel(xlabel)
        R.panel(ax, R._letter(k))
    axes[0].set_ylabel("Gain in AUROC over the covariates")
    if rows:
        axes[0].legend(fontsize=7)
    return fig, pd.DataFrame(rows), title


def size_axis(cells, size):
    fig, data, _ = _axis_figure(cells, size, "mixer size", "n_mixer_params", "Mixer parameters (non-embedding)", 183, "")
    legend = ("**Figure | Gain against mixer size at each training size, N_opt(D).** Per rung (panels), the mean over diseases "
              "of the tuning-set gain in AUROC against the mixer's non-embedding parameter count, per mixer (colours), at the "
              "grid's table width and partition. The star marks N_opt(D), the size with the highest seed-mean AUROC; the data "
              "table says whether size matters at that rung (the gain over the smallest size exceeding its paired bootstrap "
              "interval and the largest within-size seed range). The total parameter count is in the data table. Data: the "
              "accompanying _data.csv.")
    return fig, data, legend


def table_axis(cells, size):
    fig, data, _ = _axis_figure(cells, size, "table width", "embed_dim", "Per-SNP table width d_e", 183, "")
    legend = ("**Figure | Gain against the per-SNP table width at each training size, d_e_opt(D).** As the mixer-size figure, "
              "for d_e in {4, 16, 64} at the grid's mixer size and partition. Data: the accompanying _data.csv.")
    return fig, data, legend


def blocks_axis(cells, size):
    fig, data, _ = _axis_figure(cells, size, "blocks", "n_blocks", "LD blocks (tokens)", 183, "")
    legend = ("**Figure | Gain against the number of LD blocks at each training size, B_opt(D).** As the mixer-size figure, "
              "for the three nested partitions at the grid's mixer size and table width. Data: the accompanying _data.csv.")
    return fig, data, legend


def compute_fixed_data(compute, rung=None):
    """Tuning mean AUROC against cumulative training FLOPs, one curve per run, at one rung."""
    v = compute[compute.kind == "validation"]
    rung = rung or (v.rung.iloc[int(v.n_train.to_numpy().argmax())] if len(v) else None)     # the top rung: fixed data
    v = v[v.rung == rung]
    fig, ax = R.new_figure(89, 65)
    rows = []
    for (run, n), s in v.groupby(["run", "n_mixer_params"]):
        s = s.sort_values("flops")
        ax.plot(s.flops, s.auroc_mean, lw=0.9, gid=f"run:{run}", label=f"{s.mixer.iloc[0]} N={n:.2g}")
        rows += [dict(run=run, mixer=r.mixer, n_mixer_params=n, n_params=r.n_params, flops=r.flops, samples_seen=r.samples_seen,
                      auroc_mean=r.auroc_mean) for r in s.itertuples()]
    ax.set_xscale("log")
    ax.set_xlabel("Training compute (FLOPs)")
    ax.set_ylabel("Mean tuning AUROC")
    ax.legend(fontsize=7, loc="lower right")
    legend = (f"**Figure | Performance against compute at fixed data ({rung}).** Mean tuning-set AUROC over the 30 diseases "
              "at each validation pass against the cumulative training compute (3 x forward FLOPs per participant x participants "
              "seen; compute.py), one curve per model size and mixer, all trained on the same participants with the data re-used "
              "over passes. This is not a compute-optimal frontier: data is not a free variable. Data: the accompanying _data.csv.")
    return fig, pd.DataFrame(rows), legend


GROUPS = ("asian", "black", "wb_related")      # the held-out groups; the test split is not a transfer group


def transfer_offset(tidy, family=None):
    """Mean AUROC on the tuning set minus on each held-out group, against median N_eff."""
    t = tidy[tidy.family.str.startswith("deep:")] if family is None else tidy[tidy.family == family]
    t = t.groupby(["family", "rung", "split", "disease"]).agg(auroc=("auroc", "mean"), n_eff=("n_eff", "first")).reset_index()
    val = t[t.split == "val"].rename(columns={"auroc": "auroc_val"}).drop(columns="split")
    other = t[t.split.isin(GROUPS)].merge(val, on=["family", "rung", "disease", "n_eff"])
    other["offset"] = other.auroc_val - other.auroc
    p = other.groupby(["split", "family", "rung"]).agg(offset=("offset", "mean"), n_eff=("n_eff", "median")).reset_index()
    fig, ax = R.new_figure(89, 60)
    ax.axhline(0, color=R.AXIS, lw=0.8)
    for i, (split, s) in enumerate(p.groupby("split")):
        for fam, sf in s.groupby("family"):
            sf = sf.sort_values("n_eff")
            ax.plot(sf.n_eff, sf.offset, marker="o", color=(R.C1, R.C2, R.C3)[i % 3], lw=0.9, gid=f"{split}:{fam}", label=split)
    ax.set_xscale("log")
    ax.set_xlabel("Median N_eff of the training set")
    ax.set_ylabel("AUROC, tuning set minus held-out group")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=7)
    legend = ("**Figure | Transfer offset against training size.** For each held-out group (the Asian and Black participants as "
              "the distribution shift; the related White British as a relatedness check), the mean over diseases of the deep "
              "model's AUROC on the tuning set minus its AUROC on the group, against the median N_eff of the rung. Kaplan et "
              "al. found transfer a constant offset; here it is measured per rung. Data: the accompanying _data.csv.")
    return fig, p, legend


def ablation_table(cells):
    """Deep families against rungs: the mean over diseases of the gain, with the seed range."""
    p = _pooled(cells[cells.family.str.startswith("deep:")])
    text = p.apply(lambda r: f"{r.delta_auroc:.4f}" + (f" ({r.seed_lo:.4f}-{r.seed_hi:.4f})" if "seed_lo" in p and np.isfinite(r.get("seed_lo", np.nan)) else ""), axis=1)
    wide = pd.DataFrame({"family": p.family, "rung": p.rung, "value": text}).pivot(index="family", columns="rung", values="value").reset_index()
    return wide, p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="the directory fit_curves.py fit wrote")
    ap.add_argument("--tidy", default=None, help="tidy.csv (for the transfer figure; every split)")
    ap.add_argument("--compute", default=None, help="directory of compute.py CSVs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    read = lambda name: pd.read_csv(os.path.join(a.results, f"{name}.csv")) if os.path.exists(os.path.join(a.results, f"{name}.csv")) else None  # noqa: E731
    cells, size, pooled, gap, spline = (read(n) for n in ("cells", "size", "pooled", "gap", "spline"))
    size = size if size is not None else pd.DataFrame(columns=["axis", "rung", "mixer", "n_opt", "size_matters"])
    written = {}
    for name, fn in (("curves_by_family", curves_by_family), ("curves_per_disease", curves_per_disease)):
        fig, data, legend = fn(cells)
        written[name] = R.save(fig, a.out, name, data, legend)
    if gap is not None and pooled is not None:
        fig, data, legend = gap_vs_x(gap, pooled, spline)
        written["gap_vs_x"] = R.save(fig, a.out, "gap_vs_x", data, legend)
    for name, fn in (("size_axis", size_axis), ("table_axis", table_axis), ("blocks_axis", blocks_axis)):
        fig, data, legend = fn(cells, size)
        if len(data):
            written[name] = R.save(fig, a.out, name, data, legend)
    if a.compute:
        files = sorted(glob.glob(os.path.join(a.compute, "*_compute.csv")))
        if files:
            fig, data, legend = compute_fixed_data(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
            written["compute_fixed_data"] = R.save(fig, a.out, "compute_fixed_data", data, legend)
    if a.tidy:
        tidy = pd.read_csv(a.tidy)
        if "split" in tidy and tidy.split.isin(GROUPS).any():
            fig, data, legend = transfer_offset(tidy)
            written["transfer_offset"] = R.save(fig, a.out, "transfer_offset", data, legend)
    wide, long = ablation_table(cells)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "ablations.md"), "w") as f:
        f.write("# Deep cells by rung: mean tuning gain in AUROC over the covariates (seed range)\n\n" + R.md_table(wide))
    long.to_csv(os.path.join(a.out, "ablations_data.csv"), index=False)
    print("\n".join(f"{k}: {v['png']}" for k, v in written.items()))


if __name__ == "__main__":
    main()
