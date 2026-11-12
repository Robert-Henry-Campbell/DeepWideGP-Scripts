#!/usr/bin/env python3
"""How much of a chromosome's SNP-effect profile the model's position code can represent.

The sequence encoders share their weights across SNPs and average over a chromosome, so the one
input that tells a SNP from the others is the position code of layers/Embed.py: 32 sinusoidal
features of the SNP's index on its chromosome (sin and cos of p * 10^(-6 * 2i / 32) for
i = 0..15, at the positions p = 0..L-1 that pos_mode "index" gives). Whatever weight of its own
the network gives a SNP has to be read off those 32 numbers. As a measure of how much room that
leaves, this script fits a chromosome's true effect profile b (one effect per SNP, in order) by
least squares on the 32 features and a constant, X = [1, code(0..L-1)] (L x 33), and
reports the share of the profile's variance that the fit reproduces,

    share = 1 - |b - X beta_hat|^2 / |b - mean(b)|^2,

pooled over the chromosomes of a trait (explained over total sums of squares, so chromosomes
without a causal SNP add nothing). The fit is least squares with the cut-off numpy.linalg.lstsq
uses with rcond=None (its default from NumPy 2.0; NumPy 1.26, installed here, defaults to
machine precision alone), in double precision, on features computed in float64: directions of X
whose singular value is below machine precision times the larger of L and 33, relative to the
largest (10^-14 to 10^-11 here), cannot be told from rounding and are dropped; the numerical
rank that remains (10 to 31 of the 33 columns) is in the output. In exact arithmetic the 33
columns are independent once a chromosome has at least 33 SNPs, but a read-out would need
weights more than 10^11 times larger to use the dropped directions, and the model holds the code
in float32. Projecting onto all 33 columns in floating point therefore projects partly onto
rounding noise; on the 26 to 267 selected SNPs per chromosome that moves the share by up to
seven percentage points from one rounding to another. A linear read-out is not the encoder, so
this is a heuristic measure of capacity, not a bound on what the network can learn. The share on
the GWAS-selected SNPs depends on the cut-off: about 9 to 14% with a float32-precision cut-off
(and then about 80%, not 100%, at ten SNPs), roughly 20 to 38% if every direction is kept. The
proxy's figure (0.3 to 0.5%) and the 57,000-SNP figure (about 0.05%) hold under any cut-off.

    proxy      the simulated traits' true effects (sim_truth.npz) on the 157,279 proxy SNPs,
               chromosome by chromosome as in the proxy's store.json regions
    selected   the same effects on the 2,476 GWAS-selected SNPs, indexed 0..k-1 per chromosome
               as in the hm3plus_proxy8_sim_selected dataset (its store's SNP list)
    random     standard-normal profiles of 10, 110, 7,000 and 57,000 SNPs, roughly a chromosome
               of the repository's synthetic test data, of the selected SNPs, of the proxy and
               of the full dataset; 100 draws each

Writes results/position_code_capacity.csv. Reads SNP-level arrays only (the true effects and the
selected store's SNP list), no participant data. CPU, seconds.

    python scripts/widedeepgp-pilot/position_code_capacity.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DATA = "/mnt/sdg/robert/deepwidegp/data/datasets"
D_MODEL, BASE = 32, 1e6                       # layers/Embed.py: d_model features, spacing log(1e6) / d_model
RANDOM_LENGTHS, DRAWS, SEED = (10, 110, 7000, 57000), 100, 20261112


def position_code(n, d_model=D_MODEL, base=BASE):
    """The sinusoidal code of layers/Embed.py (PositionalEmbedding) for positions 0..n-1, in
    float64: [n, d_model], sines in the even columns and cosines in the odd ones."""
    pos = np.arange(n, dtype=np.float64)[:, None]
    div = np.exp(np.arange(0, d_model, 2, dtype=np.float64) * -(np.log(base) / d_model))
    pe = np.zeros((n, d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def fit(profiles):
    """Least-squares fit of every column of ``profiles`` [L, k] on the position code and a
    constant, with the cut-off of numpy.linalg.lstsq(rcond=None): directions whose singular value
    is below eps * max(L, 33) * s_max are dropped. Returns the explained and the total sum of
    squares about the mean, each [k], and the numerical rank of the design matrix (the directions
    kept)."""
    n = profiles.shape[0]
    x = np.column_stack([np.ones(n), position_code(n)])
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    rank = int((s > s[0] * max(x.shape) * np.finfo(np.float64).eps).sum())    # numpy.linalg.lstsq's cut-off, rcond=None
    b = profiles - profiles.mean(axis=0)            # the constant is among the regressors
    return ((u[:, :rank].T @ b) ** 2).sum(axis=0), (b ** 2).sum(axis=0), rank


def pooled(chromosomes, traits):
    """One row per trait: the share of variance reproduced, pooled over chromosomes (each an
    [L_c, traits] profile), with the range over the chromosomes that carry any effect."""
    fits = [fit(c) for c in chromosomes]
    ess, tss = sum(f[0] for f in fits), sum(f[1] for f in fits)
    lengths, ranks = [len(c) for c in chromosomes], [f[2] for f in fits]
    rows = []
    for j, t in enumerate(traits):
        per = [f[0][j] / f[1][j] for f in fits if f[1][j] > 0]
        rows.append(dict(trait=t, chromosomes=len(chromosomes), snps=int(sum(lengths)),
                         snps_per_chromosome_min=min(lengths), snps_per_chromosome_max=max(lengths),
                         numerical_rank_min=min(ranks), numerical_rank_max=max(ranks),
                         share_of_variance=float(ess[j] / tss[j]), share_min=float(min(per)), share_max=float(max(per))))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--name", default="hm3plus_proxy8_sim")
    a = ap.parse_args(argv)
    src = Path(a.data) / "ard30" / a.name
    truth = np.load(src / "sim_truth.npz")
    beta, traits = truth["beta"].astype(np.float64), [str(t) for t in truth["traits"]]

    # proxy: one stretch per chromosome, in the order of the regions (the order of sim_truth's rows)
    spec = json.load(open(src / "store.json"))
    bounds = np.cumsum([0] + [seq[0][2] - seq[0][1] for seq in spec["regions"]])
    assert bounds[-1] == len(beta)
    rows = [dict(profile="proxy", **r) for r in pooled([beta[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)], traits)]

    # selected: the store of the selected SNPs names each by its index among the proxy SNPs
    sel = Path(a.data) / "ard30" / (a.name + "_selected")
    store = json.load(open(sel / "store.json"))["store"]
    counts = json.load(open(sel / "select_log.json"))["snps_by_chromosome"]
    chosen = []
    for c, k in counts.items():
        ids = pd.read_csv(Path(store) / f"chr{c}.variants.tsv", sep="\t", usecols=["ID"])["ID"]
        idx = ids.str.replace("proxy_snp_", "", regex=False).astype(np.int64).to_numpy()
        assert len(idx) == k and np.all(np.diff(idx) > 0)          # positions 0..k-1 follow the proxy order
        chosen.append(beta[idx])
    rows += [dict(profile="selected", **r) for r in pooled(chosen, traits)]

    # random profiles, for reference
    rng = np.random.default_rng(SEED)
    for n in RANDOM_LENGTHS:
        ess, tss, rank = fit(rng.standard_normal((n, DRAWS)))
        share = ess / tss
        rows.append(dict(profile="random", trait="standard normal", chromosomes=1, snps=n, snps_per_chromosome_min=n,
                         snps_per_chromosome_max=n, numerical_rank_min=rank, numerical_rank_max=rank, draws=DRAWS,
                         share_of_variance=float(share.mean()), share_min=float(share.min()), share_max=float(share.max())))
    out = pd.DataFrame(rows)
    RES.mkdir(exist_ok=True)
    out.to_csv(RES / "position_code_capacity.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    show = out.assign(**{c: 100 * out[c] for c in ("share_of_variance", "share_min", "share_max")})
    print("share of variance reproduced (%):")
    print(show.drop(columns=["draws"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
