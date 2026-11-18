"""Tests of the closed-form block ridge on a synthetic store (DeepWideGP's generate_synthetic_multilabel; no
UK Biobank data). The scripts pick the device themselves: CPU, or the GPU that CUDA_VISIBLE_DEVICES exposes.

    /home/robert/envs/widedeepgp/bin/python -m pytest scripts/widedeepgp-blocks/test_block_ridge.py -q
"""
import hashlib
import json
import os
import shutil
import stat
import sys

import numpy as np
import pandas as pd
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_block_ridge as bbr  # noqa: E402
import check_block_ridge  # noqa: E402
import fit_block_linear  # noqa: E402
import make_folds  # noqa: E402
import make_warm_start  # noqa: E402
import ridge_common as rc  # noqa: E402

sys.path.insert(0, os.path.join(rc.REPO, "scripts"))
from generate_synthetic_multilabel import generate  # noqa: E402

N, SNPS, PER_BLOCK = 4000, 220, 4          # 10 SNPs per chromosome -> blocks of 4, 4 and 2 SNPs: 66 blocks
H2 = [0.05, 0.15, 0.45]


def write_partition(store, path, per_block=PER_BLOCK, digest=None, n_variants=None, ids_of=None):
    """A partition of ``per_block`` SNPs per block (the last block of a chromosome takes the rest), numbered
    genome-wide; the keyword arguments plant defects for the refusal tests."""
    chroms = list(store.chroms)
    out, nxt = {}, 0
    for c in chroms:
        b = np.arange(store.n_variants(c)) // per_block + nxt
        out[f"block_chr{c}"] = (ids_of(b) if ids_of else b).astype(np.int32)
        nxt = int(b[-1]) + 1
    ids = np.concatenate([store.variants(c)["id"] for c in chroms])
    np.savez(path, chroms=np.array(chroms), n_variants=np.array(n_variants or [store.n_variants(c) for c in chroms], dtype=np.int64),
             variant_digest=np.array(digest or hashlib.sha256("\n".join(ids).encode()).hexdigest()), n_blocks=np.int64(nxt), **out)
    return path


def block_of(snp):
    """Global block of a global synthetic SNP index (10 SNPs per chromosome, PER_BLOCK per block)."""
    per_chrom = SNPS // 22
    return (snp // per_chrom) * ((per_chrom + PER_BLOCK - 1) // PER_BLOCK) + (snp % per_chrom) // PER_BLOCK


def build(tmp, name="multi", seed=3):
    """A synthetic dataset, its partition, folds and block ridge; returns the paths and the truth."""
    base = str(tmp / "genes")
    data_dir, truth = generate(name, n_participants=N, n_snps=SNPS, base_dir=base, seed=seed)
    store = rc.GenoStore(os.path.join(data_dir, "genostore"))
    part = write_partition(store, str(tmp / f"partition_{name}.npz"))
    make_folds.main([data_dir, "--splits", "train", "val"])
    bbr.main([data_dir, "--partition", part, "--splits", "train", "val", "--penalties", *map(str, H2)])
    return dict(dir=data_dir, part=part, truth=truth, store=store, blocks=os.path.join(data_dir, "blocks"))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build(tmp_path_factory.mktemp("ridge"))


def numpy_ridge(d, rows_in, lams):
    """Independent float64 ridge per block: standardised dosages from the store, the training-set statistics,
    labels residualised on [1, covariates] over ``rows_in`` -> {block: weights [n_b, 3 n_labels]}, column 3 j + p."""
    rung = rc.Rung(d["dir"])
    tr = rung.table("train")
    _, mean, std = rung.snp_stats(tr["rows"])
    A = np.column_stack([np.ones(int(rows_in.sum())), tr["cov"][rows_in]])
    R = tr["y"][rows_in] - A @ np.linalg.lstsq(A, tr["y"][rows_in], rcond=None)[0]
    part = rc.load_partition(d["part"], rung.store)
    out = {}
    for ci, c in enumerate(rung.store.chroms):
        x = rung.store.read_dosage(c, tr["rows"]).astype(np.float64)
        z = np.where(np.isnan(x), 0.0, (x - mean[ci]) * rc.inverse_sd(std[ci]))[rows_in]
        for b, s, e in part["blocks"][c]:
            G, zr = z[:, s:e].T @ z[:, s:e], z[:, s:e].T @ R
            out[b] = np.stack([np.linalg.solve(G + l * np.eye(e - s), zr) for l in lams], axis=2).reshape(e - s, -1)
    return out


def stored_blocks(d, model):
    rung = rc.Rung(d["dir"])
    part = rc.load_partition(d["part"], rung.store)
    w = np.load(os.path.join(d["blocks"], f"ridge_weights_{model}.npy"))
    off, out = 0, {}
    for c in rung.store.chroms:
        for b, s, e in part["blocks"][c]:
            out[b] = w[off + s:off + e]
        off += rung.store.n_variants(c)
    return out


def test_training_folds_follow_the_ladder_permutation_and_nest(built, tmp_path):
    z = np.load(os.path.join(built["blocks"], "folds.npz"))
    iid = pd.read_csv(os.path.join(built["dir"], "samples_train.tsv"), sep="\t", dtype=str)["IID"].to_numpy()
    order = np.random.default_rng(1).permutation(np.sort(iid))
    pos = {s: i for i, s in enumerate(order)}
    assert np.array_equal(z["fold_train"], np.array([pos[s] % 5 for s in iid]))
    counts = np.bincount(z["fold_train"], minlength=5)
    assert counts.max() - counts.min() <= 1 and counts.sum() == len(iid)
    # a sub-rung (the first 1000 of the permutation, written in the parent's row order) gets the same folds
    sub = tmp_path / "sub" / "store"
    sub.mkdir(parents=True)
    keep = set(order[:1000])
    df = pd.read_csv(os.path.join(built["dir"], "samples_train.tsv"), sep="\t", dtype=str)
    df[df["IID"].isin(keep)].to_csv(sub / "samples_train.tsv", sep="\t", index=False)
    shutil.copy(os.path.join(built["dir"], "samples_val.tsv"), sub / "samples_val.tsv")
    spec = json.load(open(os.path.join(built["dir"], "store.json")))
    spec["store"] = os.path.join(built["dir"], "genostore")
    json.dump(spec, open(sub / "store.json", "w"))
    json.dump({"parent": built["dir"], "seed": 1, "n_train": 1000}, open(sub / "ladder.json", "w"))
    make_folds.main([str(sub), "--splits", "train"])
    zs = np.load(sub / "blocks" / "folds.npz")
    parent_fold = dict(zip(z["iid_train"], z["fold_train"]))
    assert len(zs["iid_train"]) == 1000
    assert all(parent_fold[s] == f for s, f in zip(zs["iid_train"], zs["fold_train"]))
    assert not np.array_equal(zs["fold_train"], np.arange(1000) % 5)       # row number mod 5 would not nest
    # a table that is not a permutation prefix is refused
    df.iloc[::4].to_csv(sub / "samples_train.tsv", sep="\t", index=False)
    with pytest.raises(SystemExit, match="not the first n participants"):
        make_folds.main([str(sub), "--splits", "train"])


def test_pseudo_folds_are_deterministic_and_balanced(built):
    ids = [f"p{i}" for i in range(20000)]
    f1, f2 = rc.pseudo_fold(ids), rc.pseudo_fold(ids)
    assert np.array_equal(f1, f2) and f1.dtype == np.int8 and set(f1) == set(range(5))
    assert not np.array_equal(f1, rc.pseudo_fold(ids, seed=2))
    assert np.abs(np.bincount(f1) / len(ids) - 0.2).max() < 0.015
    # the hash is the documented one, and the tuning split carries it
    assert f1[0] == int.from_bytes(hashlib.sha256(b"p0\t1").digest()[:8], "big") % 5
    z = np.load(os.path.join(built["blocks"], "folds.npz"))
    assert np.array_equal(z["fold_val"], rc.pseudo_fold(z["iid_val"]))


def test_the_block_ridge_equals_numpy_for_every_block_and_model(built):
    folds = np.load(os.path.join(built["blocks"], "folds.npz"))["fold_train"]
    lams = [SNPS / h for h in H2]
    for model, rows_in in [("fold0", folds != 0), ("fold3", folds != 3), ("full", np.ones(len(folds), bool))]:
        exact, stored = numpy_ridge(built, rows_in, lams), stored_blocks(built, model)
        assert set(exact) == set(stored) == set(range(66))
        for b in exact:
            assert np.allclose(stored[b], exact[b], rtol=1e-4, atol=1e-6 * np.abs(exact[b]).max()), (model, b)
    man = json.load(open(os.path.join(built["blocks"], "ridge_manifest.json")))
    assert man["lambda"] == pytest.approx(lams) and man["M"] == SNPS


def test_a_participants_own_fold_never_enters_their_score(built, tmp_path):
    """Flipping every label of fold 2 changes every model but fold 2's: fold-2 rows (and the tuning rows of
    pseudo-fold 2) keep their scores to the last bit, everybody else moves."""
    copy = str(tmp_path / "genes" / "multi" / "store")
    shutil.copytree(built["dir"], copy)
    shutil.rmtree(os.path.join(copy, "blocks"))
    folds = np.load(os.path.join(built["blocks"], "folds.npz"))
    flip = set(folds["iid_train"][folds["fold_train"] == 2])
    n_lab = len(json.load(open(os.path.join(copy, "store.json")))["labels"])
    with open(os.path.join(built["dir"], "samples_train.tsv")) as f:
        lines = f.read().splitlines()
    out = [lines[0]]
    for line in lines[1:]:
        p = line.split("\t")
        if p[0] in flip:
            p[1:1 + n_lab] = [str(1 - int(v)) for v in p[1:1 + n_lab]]
        out.append("\t".join(p))
    with open(os.path.join(copy, "samples_train.tsv"), "w") as f:
        f.write("\n".join(out) + "\n")
    make_folds.main([copy, "--splits", "train", "val"])
    bbr.main([copy, "--partition", built["part"], "--splits", "train", "val", "--penalties", *map(str, H2)])
    for split, key in (("train", "fold_train"), ("val", "fold_val")):
        a, b = (np.load(os.path.join(d, "blocks", f"prs_{split}.npz"))["score_all"] for d in (built["dir"], copy))
        own = folds[key] == 2
        assert np.array_equal(a[own], b[own])
        assert (np.abs(a[~own] - b[~own]).max(1) > 0).mean() > 0.99
    for m in bbr.MODELS:
        a, b = (np.load(os.path.join(d, "blocks", f"ridge_weights_{m}.npy")) for d in (built["dir"], copy))
        assert np.array_equal(a, b) == (m == "fold2"), m


def test_causal_blocks_predict_and_the_null_trait_does_not(built):
    labels = json.load(open(os.path.join(built["dir"], "store.json")))["labels"]
    man = json.load(open(os.path.join(built["blocks"], "ridge_manifest.json")))
    pv, pt = (np.load(os.path.join(built["blocks"], f"prs_{s}.npz")) for s in ("val", "train"))
    for j, lab in enumerate(labels):
        for z in (pv, pt):
            au = rc.auroc(z["score"][:, j], z["y"][:, j])
            assert (abs(au - 0.5) < 0.1) if lab == "null_f" else (au > 0.6), (lab, au)
    j = labels.index("solo_d")
    p = man["chosen_penalty_index"][j]
    ch = np.load(os.path.join(built["blocks"], "channels_val.npy"))
    causal = {block_of(s) for s in built["truth"]["own_causal"]["solo_d"]}
    every_causal = {block_of(s) for s in built["truth"]["shared_causal"] + sum(built["truth"]["own_causal"].values(), [])}
    causal_aus = [rc.auroc(ch[:, b, 3 * j + p].astype(float), pv["y"][:, j]) for b in causal]
    assert max(causal_aus) > 0.58 and min(causal_aus) > 0.45       # the two causal SNPs have random effect sizes
    quiet = [b for b in range(66) if b not in every_causal][:20]
    aus = np.array([rc.auroc(ch[:, b, 3 * j + p].astype(float), pv["y"][:, j]) for b in quiet])
    assert abs(aus.mean() - 0.5) < 0.03 and np.abs(aus - 0.5).max() < 0.15
    assert all(man["chosen_h2"][lab] in H2 for lab in labels)


def test_int8_channels_round_trip_and_sum_to_the_prs(built):
    rung = rc.Rung(built["dir"])
    tr, va = rung.table("train"), rung.table("val")
    _, mean, std = rung.snp_stats(tr["rows"])
    folds = np.load(os.path.join(built["blocks"], "folds.npz"))["fold_val"]
    cman = json.load(open(os.path.join(built["blocks"], "channels_manifest.json")))
    part = rc.load_partition(built["part"], rung.store)
    weights = [stored_blocks(built, f"fold{f}") for f in range(5)]
    exact = np.zeros((len(va["rows"]), part["n_blocks"], 3 * len(rung.labels)), np.float32)
    for ci, c in enumerate(rung.store.chroms):
        x = rung.store.read_dosage(c, va["rows"]).astype(np.float64)
        z = np.where(np.isnan(x), 0.0, (x - mean[ci]) * rc.inverse_sd(std[ci]))
        for b, s, e in part["blocks"][c]:
            for f in range(5):
                exact[folds == f, b] = z[folds == f, s:e] @ weights[f][b]
    stored = np.load(os.path.join(built["blocks"], "channels_val.npy"))
    cm, cs = np.array(cman["mean"], np.float32), np.array(cman["sd"], np.float32)
    assert stored.dtype == np.int8 and stored.shape == exact.shape and cm.shape == cs.shape == exact.shape[1:]
    std_units = (exact - cm) / cs
    clipped = np.abs(std_units) > cman["clip_sd"]
    assert np.array_equal(stored[clipped], np.sign(std_units[clipped]) * round(cman["clip_sd"] * cman["scale"]))
    err = stored[~clipped] / cman["scale"] - std_units[~clipped]
    assert np.abs(err).max() <= 0.5 / cman["scale"] + 1e-5
    # the training channels are standardised on the training rows
    ctr = np.load(os.path.join(built["blocks"], "channels_train.npy")).astype(np.float32) / cman["scale"]
    assert np.abs(ctr.mean(0)).max() < 0.1 and abs(np.median(ctr.std(0)) - 1) < 0.1
    pv = np.load(os.path.join(built["blocks"], "prs_val.npz"))
    assert np.allclose(exact.sum(1), pv["score_all"], rtol=1e-4, atol=1e-5)
    assert pv["score"].shape == (len(va["rows"]), len(rung.labels)) and pv["fold"].dtype == np.int8
    assert pv["iid"].dtype.kind == "U" and list(pv["labels"]) == rung.labels


def test_zero_variance_snps_missing_genotypes_and_the_solver():
    q = torch.tensor([[0, 100, 255], [200, 100, 0]], dtype=torch.uint8)
    mean, inv = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64), torch.tensor(rc.inverse_sd([0.5, 0.0, 0.25]))
    z = rc.standardise(q, mean, inv)
    assert z.dtype == torch.float64 and np.allclose(z.numpy(), [[-2, 0, 0], [2, 0, -2]])
    assert list(rc.inverse_sd([0.0, 2.0])) == [0.0, 0.5]
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 4))
    x[:, 1] = 0.0                                     # a SNP without variation
    y = rng.standard_normal((50, 2))
    folds = np.arange(50) % 5
    G = torch.stack([torch.as_tensor(x[folds == f].T @ x[folds == f]) for f in range(5)])
    T = torch.stack([torch.as_tensor(x[folds == f].T @ np.column_stack([y, np.ones(50)])[folds == f]) for f in range(5)])
    B = torch.zeros(6, 1, 2, dtype=torch.float64)      # intercept only, zero: R = Y
    w = bbr.solve_block(G, T, B, [3.0, 30.0]).numpy()
    assert w.shape == (6, 4, 4)
    for f in range(6):
        m = folds != f
        for p, lam in enumerate([3.0, 30.0]):
            exact = np.linalg.solve(x[m].T @ x[m] + lam * np.eye(4), x[m].T @ y[m])
            assert np.allclose(w[f][:, [p, 2 + p]], exact, atol=1e-6)
    assert np.abs(w[:, 1]).max() < 1e-12


def test_manifest_hashes_verify_and_files_are_private(built):
    man = json.load(open(os.path.join(built["blocks"], "ridge_manifest.json")))
    cman = json.load(open(os.path.join(built["blocks"], "channels_manifest.json")))
    listed = json.load(open(os.path.join(built["dir"], "blocks.json")))["files"]
    for name, sha in {**man["files"], **{k: v["sha256"] for k, v in cman["files"].items()}}.items():
        assert rc.sha256_file(os.path.join(built["blocks"], name)) == sha == listed[name]
    present = sorted(f for f in os.listdir(built["blocks"]) if f != "test_access.log")
    assert present == sorted(listed) and "partial" not in present
    assert set(present) >= {"folds.npz", "ridge_weights_full.npy", "prs_train.npz", "channels_val.npy", "channels_manifest.json"}
    summary = pd.read_csv(os.path.join(built["blocks"], "ridge_summary.csv"))          # what e0_summary.py reads
    pv = np.load(os.path.join(built["blocks"], "prs_val.npz"))
    assert list(summary.label) == list(pv["labels"]) and summary.chosen_h2.isin(H2).all()
    assert np.allclose(summary.auroc_val, [rc.auroc(pv["score"][:, j], pv["y"][:, j]) for j in range(len(summary))], atol=1e-4)
    for f in present:
        assert stat.S_IMODE(os.stat(os.path.join(built["blocks"], f)).st_mode) == 0o600, f
    assert stat.S_IMODE(os.stat(built["blocks"]).st_mode) == 0o700
    assert man["standardisation"]["file"].startswith("snp_stats_train_") and man["fold_counts"] == [560] * 5
    assert man["train_iids_sha256"] == rc.sha256_iids(np.load(os.path.join(built["blocks"], "folds.npz"))["iid_train"])
    assert json.load(open(os.path.join(built["dir"], "store.json"))).get("blocks") is None    # never inside store.json


def test_the_test_split_is_locked(built):
    log = os.path.join(built["blocks"], "test_access.log")
    with pytest.raises(SystemExit, match="the test split is locked"):
        make_folds.main([built["dir"], "--splits", "test"])
    with pytest.raises(SystemExit, match="the test split is locked"):
        bbr.main([built["dir"], "--partition", built["part"], "--splits", "test"])
    assert not os.path.exists(log) and not os.path.exists(os.path.join(built["blocks"], "prs_test.npz"))
    make_folds.main([built["dir"], "--splits", "test", "--unlock_test"])
    bbr.main([built["dir"], "--partition", built["part"], "--splits", "test", "--unlock_test", "--penalties", *map(str, H2)])
    assert os.path.exists(os.path.join(built["blocks"], "prs_test.npz")) and os.path.exists(os.path.join(built["blocks"], "channels_test.npy"))
    lines = open(log).read().strip().splitlines()
    assert len(lines) == 2 and lines[0].endswith("\tmake_folds") and lines[1].endswith("\tbuild_block_ridge")
    z = np.load(os.path.join(built["blocks"], "folds.npz"))
    assert np.array_equal(z["fold_test"], rc.pseudo_fold(z["iid_test"])) and "fold_train" in z.files


def test_the_build_resumes_per_chromosome(built, tmp_path, monkeypatch):
    copy = str(tmp_path / "genes" / "multi" / "store")
    shutil.copytree(built["dir"], copy)
    shutil.rmtree(os.path.join(copy, "blocks"))
    make_folds.main([copy, "--splits", "train", "val"])
    real, calls = bbr.solve_chromosome, []

    def failing(*args, **kw):
        if len(calls) == 3:
            raise RuntimeError("interrupted")
        calls.append(args[2])
        return real(*args, **kw)

    monkeypatch.setattr(bbr, "solve_chromosome", failing)
    with pytest.raises(RuntimeError, match="interrupted"):
        bbr.main([copy, "--partition", built["part"], "--splits", "train", "val", "--penalties", *map(str, H2)])
    partial = sorted(os.listdir(os.path.join(copy, "blocks", "partial")))
    assert len(partial) == 18 and all(f.startswith("weights_chr") for f in partial)
    calls.clear()
    monkeypatch.setattr(bbr, "solve_chromosome", lambda *a, **k: (calls.append(a[2]), real(*a, **k))[1])
    bbr.main([copy, "--partition", built["part"], "--splits", "train", "val", "--penalties", *map(str, H2)])
    assert len(calls) == 19 and not os.path.exists(os.path.join(copy, "blocks", "partial"))
    for m in bbr.MODELS:
        assert np.array_equal(*(np.load(os.path.join(d, "blocks", f"ridge_weights_{m}.npy")) for d in (built["dir"], copy)))
    a, b = (np.load(os.path.join(d, "blocks", "prs_val.npz"))["score"] for d in (built["dir"], copy))
    assert np.array_equal(a, b)
    # a second run with everything in place recomputes nothing
    monkeypatch.setattr(bbr, "solve_chromosome", lambda *a, **k: pytest.fail("recomputed a chromosome"))
    monkeypatch.setattr(bbr, "score_split", lambda *a, **k: pytest.fail("rescored a split"))
    bbr.main([copy, "--partition", built["part"], "--splits", "train", "val", "--penalties", *map(str, H2)])


def test_a_partition_for_other_variants_or_without_contiguous_blocks_is_refused(built, tmp_path):
    store = built["store"]
    with pytest.raises(SystemExit, match="other variants"):
        rc.load_partition(write_partition(store, str(tmp_path / "p1.npz"), digest="0" * 64), store)
    bad = [store.n_variants(c) for c in store.chroms]
    bad[0] += 1
    with pytest.raises(SystemExit, match="other chromosomes or variant counts"):
        rc.load_partition(write_partition(store, str(tmp_path / "p2.npz"), n_variants=bad), store)
    with pytest.raises(SystemExit, match="contiguous runs"):
        rc.load_partition(write_partition(store, str(tmp_path / "p3.npz"), ids_of=lambda b: b[::-1]), store)
    with pytest.raises(SystemExit, match="run make_folds.py again"):
        rc.Rung(built["dir"]).folds("train", np.load(os.path.join(built["blocks"], "folds.npz"))["iid_train"][::-1])
    with pytest.raises(SystemExit, match="run make_folds.py first"):
        rc.Rung(built["dir"]).folds("wb_related", np.array(["x"]))


def test_make_warm_start_is_the_truncated_svd_of_the_full_weights(built):
    out = make_warm_start.main([built["dir"], "--embed_dim", "4"])
    saved = torch.load(out)
    man, state = saved["manifest"], saved["state"]
    seq_len = [built["store"].n_variants(c) for c in built["store"].chroms]
    assert man["kind"] == "ridge" and man["snp_code"] == "dosage" and man["embed_dim"] == 4
    assert man["seq_len"] == seq_len and man["n_blocks"] == 66
    assert sorted(state) == sorted(f"tables.{i}" for i in range(22))
    assert all(state[f"tables.{i}"].shape == (seq_len[i], 4) and state[f"tables.{i}"].dtype == torch.float32 for i in range(22))
    w = np.load(os.path.join(built["blocks"], "ridge_weights_full.npy"))
    t = np.concatenate([state[f"tables.{i}"].numpy() for i in range(22)])
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    assert np.allclose(t.T @ t, np.diag(s[:4] ** 2), rtol=1e-4, atol=1e-6 * s[0] ** 2)      # orthogonal columns of length s_k
    assert np.allclose(np.abs(t), np.abs(u[:, :4] * s[:4]), rtol=1e-4, atol=1e-6 * s[0])
    assert man["source"]["variance_kept"] == pytest.approx(float((s[:4] ** 2).sum() / (s ** 2).sum()))
    assert man["source"]["weights_sha256"] == rc.sha256_file(os.path.join(built["blocks"], "ridge_weights_full.npy"))
    assert (man["partition_digest"] is None and "TODO" in man["source"]) or len(man["partition_digest"]) == 16
    assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    with pytest.raises(SystemExit, match="--embed_dim"):
        make_warm_start.main([built["dir"], "--embed_dim", "91"])


def test_the_logistic_ridge_solver_matches_sklearn_and_leaves_the_covariates_free():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(1)
    x = rng.standard_normal((600, 25)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 3] + 0.3 * x[:, 20] + rng.standard_normal(600) > 0.8).astype(np.float32)
    for C in (0.01, 1.0):                                   # sklearn: C sum(loss) + ||beta||^2 / 2  <=>  lam = 1 / C
        w, nit = fit_block_linear.logistic_ridge(x, y, 1.0 / C, 0)
        ref = LogisticRegression(C=C, max_iter=2000, tol=1e-8).fit(x, y)
        assert np.allclose(w[:-1], ref.coef_[0], atol=2e-3) and abs(w[-1] - ref.intercept_[0]) < 2e-3 and nit > 0
    # a huge penalty on the channels leaves the free columns at their unpenalised covariates-only fit
    w, _ = fit_block_linear.logistic_ridge(x, y, 1e7, 2)
    ref = LogisticRegression(C=1e9, max_iter=5000, tol=1e-10).fit(x[:, :2], y)
    assert np.allclose(w[:2], ref.coef_[0], atol=2e-3) and abs(w[-1] - ref.intercept_[0]) < 2e-3 and np.abs(w[2:-1]).max() < 1e-4
    w0, _ = fit_block_linear.logistic_ridge(x, y, 0.0, 25)   # no penalty at all: the plain logistic fit
    ref = LogisticRegression(C=1e9, max_iter=5000, tol=1e-10).fit(x, y)
    assert np.allclose(w0[:-1], ref.coef_[0], atol=5e-3)


def test_fit_block_linear_writes_scores_in_the_deep_arms_format(built):
    table = fit_block_linear.main([built["dir"], "--lam_grid", "10", "1000"])
    labels = json.load(open(os.path.join(built["dir"], "store.json")))["labels"]
    for method in ("logistic_ridge", "ridge_prs"):
        path = os.path.join(built["dir"], "scores", "blocks_linear", method, "val_best.npz")
        z = np.load(path)
        assert sorted(z.files) == ["iid", "labels", "logit_population", "y"] and z["iid"].dtype.kind == "U"
        assert list(z["labels"]) == labels and z["logit_population"].dtype == z["y"].dtype == np.float32
        assert z["logit_population"].shape == z["y"].shape == (600, len(labels))
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        pv = np.load(os.path.join(built["blocks"], "prs_val.npz"))
        assert np.array_equal(z["iid"], pv["iid"]) and np.array_equal(z["y"], pv["y"])
        for j, lab in enumerate(labels):
            au = rc.auroc(z["logit_population"][:, j], z["y"][:, j])
            assert (abs(au - 0.5) < 0.12) if lab == "null_f" else (au > 0.6), (method, lab, au)
    assert len(table) == len(labels) and set(table.lam) <= {10.0, 1000.0}
    csv = pd.read_csv(os.path.join(HERE, "results", "store", "blocks_linear.csv"))
    assert list(csv.label) == labels and {"auroc_val_logistic_ridge", "auroc_val_ridge_prs", "lam_at_grid_edge", "n_iter"} <= set(csv.columns)


def test_check_block_ridge_finds_agreement(built):
    out, table = check_block_ridge.main([built["dir"], "--partition", built["part"], "--bootstrap", "50"])
    assert out["max_rel_diff_fold0"] < 1e-4 and out["max_rel_diff_full"] < 1e-4
    assert out["int8_rms_error_sd_units_unclipped"] < 0.5 / 36
    # four-SNP channels are discrete and heavy-tailed, so here the clip at 3.5 SD, not the rounding, sets the summed error
    assert out["sum_of_channels_vs_prs_rel_rms_max"] < 0.25 and 0 < out["fraction_of_rows_with_a_clipped_channel"] <= 1
    assert len(table) == 6 and table.within_2se.sum() >= 5 and (table.var_ratio_val_over_train.between(0.5, 2.0)).all()
    res = os.path.join(HERE, "results", "store")
    assert os.path.exists(os.path.join(res, "check_block_ridge.md")) and os.path.exists(os.path.join(res, "check_block_ridge.csv"))
    text = open(os.path.join(res, "check_block_ridge.md")).read()
    assert "s0" not in text.split("Random block")[0]             # no participant identifier in the report


@pytest.fixture(scope="module", autouse=True)
def remove_synthetic_results():
    """The synthetic runs write results/store/ beside the real rungs' folders; drop it afterwards."""
    yield
    shutil.rmtree(os.path.join(HERE, "results", "store"), ignore_errors=True)


def test_fit_block_linear_resumes_per_disease(built, monkeypatch):
    """Interrupted after two diseases, a rerun continues from the partial file and reproduces the uninterrupted run."""
    ref = fit_block_linear.main([built["dir"], "--lam_grid", "10", "1000"])
    score = os.path.join(built["dir"], "scores", "blocks_linear", "logistic_ridge", "val_best.npz")
    ref_logit = np.load(score)["logit_population"].copy()
    part = os.path.join(built["blocks"], "blocks_linear_partial.npz")
    assert not os.path.exists(part)
    saved, real = [], fit_block_linear.save_npz

    def interrupted(path, **arrays):
        saved.append(path)
        if len(saved) == 3:                                   # the third disease's save: the process dies before it
            raise RuntimeError("interrupted")
        return real(path, **arrays)
    monkeypatch.setattr(fit_block_linear, "save_npz", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        fit_block_linear.main([built["dir"], "--lam_grid", "10", "1000"])
    z = np.load(part)
    assert len(json.loads(str(z["rows"]))) == 2 and stat.S_IMODE(os.stat(part).st_mode) == 0o600
    monkeypatch.setattr(fit_block_linear, "save_npz", real)
    table = fit_block_linear.main([built["dir"], "--lam_grid", "10", "1000"])
    assert list(table.label) == list(ref.label)
    for col in ("lam", "auroc_val_logistic_ridge", "auroc_val_ridge_prs", "n_iter"):
        assert np.allclose(table[col].to_numpy(float), ref[col].to_numpy(float)), col
    assert np.array_equal(np.load(score)["logit_population"], ref_logit)
    assert not os.path.exists(part)
