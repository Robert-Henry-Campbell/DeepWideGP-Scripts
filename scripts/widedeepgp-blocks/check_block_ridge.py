#!/usr/bin/env python3
"""Checks of a built block ridge (methods.md section 11); aggregate output only.

1. One random block: its fold-0 and full weights against an independent numpy float64 ridge on dosages read
   from the store (same standardisation, residualisation and penalties), and its stored int8 channels against
   the exact block scores of the fold-0 training rows (the int8 round trip).
2. The PRS at every penalty against the sum of the dequantised block channels over all blocks (tuning rows).
3. Per disease: the out-of-fold training AUROC against the tuning AUROC with bootstrap standard errors, and
   the pseudo-fold variance ratio (tuning-set score variance / training out-of-fold score variance).
Writes results/<rung>/check_block_ridge.csv and .md beside this script.

    python check_block_ridge.py <dataset_dir> --partition <dir or npz> [--seed 1]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ridge_common import MISSING, Rung, auroc, inverse_sd, load_partition  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def exact_block(store, chrom, s, e, tr, folds, mean, std, lams, fold):
    """numpy ridge of one block for the model that excludes ``fold`` (None: the full model): the standardised
    dosages Z of every training row [n, n_b] and the weights [n_b, P n_labels] (column P j + p)."""
    rows = tr["rows"]
    q = np.concatenate([store.read_q(chrom, rows[i:i + 2048], s, e) for i in range(0, len(rows), 2048)])
    z = (q.astype(np.float64) * 0.01 - mean[s:e]) * inverse_sd(std[s:e])
    z[q == MISSING] = 0.0
    m = folds != (-1 if fold is None else fold)
    A = np.column_stack([np.ones(int(m.sum())), tr["cov"][m]])
    R = tr["y"][m] - A @ np.linalg.lstsq(A, tr["y"][m], rcond=None)[0]
    G, zr = z[m].T @ z[m], z[m].T @ R
    return z, np.stack([np.linalg.solve(G + l * np.eye(len(G)), zr) for l in lams], axis=2).reshape(len(G), -1)


def bootstrap_se(score, y, B, rng):
    """Standard error of the AUROC over B resamples of the rows."""
    n = len(y)
    return float(np.nanstd([auroc(score[i], y[i]) for i in (rng.integers(0, n, n) for _ in range(B))]))


def markdown(table):
    cells = [[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row] for row in table.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)] + ["| " + " | ".join(r) + " |" for r in cells])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir")
    ap.add_argument("--partition", required=True)
    ap.add_argument("--seed", type=int, default=1, help="chooses the block and the bootstrap draws")
    ap.add_argument("--bootstrap", type=int, default=200)
    a = ap.parse_args(argv)
    rung = Rung(a.dataset_dir)
    part = load_partition(a.partition, rung.store)
    with open(os.path.join(rung.blocks, "ridge_manifest.json")) as f:
        man = json.load(f)
    with open(os.path.join(rung.blocks, "channels_manifest.json")) as f:
        cman = json.load(f)
    tr = rung.table("train")
    folds = rung.folds("train", tr["iid"])
    _, mean, std = rung.snp_stats(tr["rows"])
    lams, h2s = man["lambda"], man["penalties_h2"]
    n_pen, labels = len(h2s), rung.labels
    chroms = list(rung.store.chroms)
    offsets = np.cumsum([0] + [rung.store.n_variants(c) for c in chroms])
    rng = np.random.default_rng(a.seed)
    flat = [(ci, c, b, s, e) for ci, c in enumerate(chroms) for b, s, e in part["blocks"][c]]
    ci, c, b, s, e = flat[rng.integers(len(flat))]
    out = {"block": b, "chrom": c, "n_snps": e - s}

    # 1. the random block: weights and the int8 round trip
    z, w0 = exact_block(rung.store, c, s, e, tr, folds, mean[ci], std[ci], lams, 0)
    _, wf = exact_block(rung.store, c, s, e, tr, folds, mean[ci], std[ci], lams, None)
    for name, w in (("fold0", w0), ("full", wf)):
        stored = np.load(os.path.join(rung.blocks, f"ridge_weights_{name}.npy"), mmap_mode="r")[offsets[ci] + s:offsets[ci] + e]
        out[f"max_rel_diff_{name}"] = float(np.abs(stored - w).max() / np.abs(w).max())
    cmean, csd = np.array(cman["mean"], np.float32), np.array(cman["sd"], np.float32)
    rows0 = np.flatnonzero(folds == 0)
    exact = z[rows0] @ w0
    ch = np.load(os.path.join(rung.blocks, "channels_train.npy"), mmap_mode="r")
    deq = ch[rows0, b, :].astype(np.float32) / cman["scale"] * csd[b] + cmean[b]
    std_units = (exact - cmean[b]) / csd[b]
    inside = np.abs(std_units) <= cman["clip_sd"]
    err = (deq - exact) / csd[b]
    out["int8_rms_error_sd_units_unclipped"] = float(np.sqrt(np.mean(err[inside] ** 2)))
    out["int8_max_error_sd_units_unclipped"] = float(np.abs(err[inside]).max())
    out["fraction_clipped"] = float(1 - inside.mean())

    # 2. the PRS against the sum of the dequantised channels (tuning rows)
    pv = np.load(os.path.join(rung.blocks, "prs_val.npz"))
    chv = np.load(os.path.join(rung.blocks, "channels_val.npy"), mmap_mode="r")
    idx = np.sort(rng.choice(len(pv["iid"]), min(2000, len(pv["iid"])), replace=False))
    summed = (chv[idx].astype(np.float32) / cman["scale"] * csd + cmean).sum(1)
    rel = np.sqrt(np.mean((summed - pv["score_all"][idx]) ** 2, 0)) / pv["score_all"].std(0)
    out["sum_of_channels_vs_prs_rel_rms_median"], out["sum_of_channels_vs_prs_rel_rms_max"] = float(np.median(rel)), float(rel.max())
    out["fraction_of_rows_with_a_clipped_channel"] = float((np.abs(chv[idx]) >= round(cman["clip_sd"] * cman["scale"])).any(axis=(1, 2)).mean())

    # 3. out-of-fold training AUROC against tuning AUROC; the pseudo-fold variance ratio
    pt = np.load(os.path.join(rung.blocks, "prs_train.npz"))
    rows = []
    for j, lab in enumerate(labels):
        at, av = auroc(pt["score"][:, j], pt["y"][:, j]), auroc(pv["score"][:, j], pv["y"][:, j])
        st, sv = bootstrap_se(pt["score"][:, j], pt["y"][:, j], a.bootstrap, rng), bootstrap_se(pv["score"][:, j], pv["y"][:, j], a.bootstrap, rng)
        row = dict(label=lab, cases_train=int(pt["y"][:, j].sum()), cases_val=int(pv["y"][:, j].sum()), chosen_h2=man["chosen_h2"][lab],
                   auroc_train_oof=at, se_train=st, auroc_val=av, se_val=sv, diff=av - at,
                   within_2se=bool(abs(av - at) <= 2 * np.sqrt(st ** 2 + sv ** 2)),
                   var_ratio_val_over_train=float(pv["score"][:, j].var() / pt["score"][:, j].var()))
        row.update({f"auroc_val_h2_{h}": auroc(pv["score_all"][:, n_pen * j + p], pv["y"][:, j]) for p, h in enumerate(h2s)})
        rows.append(row)
    table = pd.DataFrame(rows)
    res = os.path.join(HERE, "results", os.path.basename(rung.dir))
    os.makedirs(res, exist_ok=True)
    table.to_csv(os.path.join(res, "check_block_ridge.csv"), index=False, float_format="%.4f")
    secs = man.get("seconds", {})
    lines = [f"# Block ridge check: {os.path.basename(rung.dir)}", "",
             f"Training rows {man['n_train']}, folds {man['fold_counts']}, blocks {part['n_blocks']}, penalties h2 {h2s} "
             f"(lambda = M / h2, M = {man['M']}).", "",
             f"Random block {b} (chromosome {c}, {e - s} SNPs): max |stored - numpy| / max |numpy| = "
             f"{out['max_rel_diff_fold0']:.2e} (fold 0), {out['max_rel_diff_full']:.2e} (full). int8 round trip on its "
             f"fold-0 training rows: RMS error {out['int8_rms_error_sd_units_unclipped']:.4f} SD, max "
             f"{out['int8_max_error_sd_units_unclipped']:.4f} SD, {100 * out['fraction_clipped']:.2f}% of values clipped at "
             f"+-{cman['clip_sd']} SD.", "",
             f"Sum of the dequantised channels against the stored PRS ({len(idx)} tuning rows, {len(rel)} channels): relative RMS "
             f"error median {out['sum_of_channels_vs_prs_rel_rms_median']:.4f}, max {out['sum_of_channels_vs_prs_rel_rms_max']:.4f}; "
             f"{100 * out['fraction_of_rows_with_a_clipped_channel']:.2f}% of these rows have a clipped channel (the clip, not the "
             f"rounding, is what the sum loses).", "",
             f"Mean AUROC over {len(labels)} diseases at the chosen penalty: out-of-fold training {table.auroc_train_oof.mean():.4f}, "
             f"tuning {table.auroc_val.mean():.4f}; {int(table.within_2se.sum())} of {len(table)} agree within 2 SE "
             f"({a.bootstrap} bootstrap resamples). Pseudo-fold variance ratio (tuning / training): median "
             f"{table.var_ratio_val_over_train.median():.3f}, range {table.var_ratio_val_over_train.min():.3f} to "
             f"{table.var_ratio_val_over_train.max():.3f}.", "",
             "Mean tuning AUROC by penalty: " + ", ".join(f"h2 {h}: {table[f'auroc_val_h2_{h}'].mean():.4f}" for h in h2s)
             + "; chosen h2 counts: " + ", ".join(f"{h}: {int((table.chosen_h2 == h).sum())}" for h in h2s) + ".", "",
             "Build seconds: " + ", ".join(f"{k} {v}" for k, v in secs.items()) + f" (device {man.get('device')}).", "",
             markdown(table)]
    with open(os.path.join(res, "check_block_ridge.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:-1]))
    return out, table


if __name__ == "__main__":
    main()
