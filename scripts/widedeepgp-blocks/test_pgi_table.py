"""Tests of make_pgi_table.py and make_complement.py on DeepWideGP's synthetic six-trait store, with made-up PGI weights
in the PGI cache layout (CPU, a minute or two; no UK Biobank data): the table's block sums give back each PGI exactly,
zero-weight SNPs leave the model with the kept-SNP divisor per block, every channel has unit RMS, wrong alleles and
in-sample weights are refused unless allowed, a blocks run loads the file frozen and caches it, and the complement keeps
the parent's other training lines unchanged and is leak-free for the excluded rung's weights.

    PYTHONPATH=<DeepWideGP>:<DeepWideGP>/scripts python -m pytest test_pgi_table.py -q
"""
import gzip
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_complement  # noqa: E402
import make_pgi_table  # noqa: E402

METHOD = "lassosum2"


def _dm(ds, workers=0):
    from args_generator import args_initial
    from widedeepgp.train import build_data
    cfg = args_initial(["--label", "multi", "--rd_mode", os.path.basename(ds), "--dm", "snps_covs", "--input_mode", "blocks",
                        "--partition", "uniform:2", "--snp_stats", "train", "--num_workers", str(workers), "--wide", "off"])
    cfg.data_dir = os.path.dirname(os.path.dirname(ds))
    return build_data(cfg)


def _dosages(dm, split):
    rows = dm.splits[split]["rows"]
    return np.concatenate([dm.store.read_dosage(c, rows, v0, v1) for seq in dm.regions for c, v0, v1 in seq], axis=1)


def _fake_pgi(world, ds, n_train, weights):
    """Weights in the PGI cache layout for the rung ``ds`` (GWAS of n_train), its tune_metrics row per trait and its
    tuning-set scores (sum of weight x ALT dosage)."""
    dm = _dm(ds)
    rows = []
    for j, t in enumerate(dm.label_names):
        key = f"{abs(hash((ds, t))):016x}".ljust(64, "0")[:64]
        entry = world / "cache" / key[:2] / key
        entry.mkdir(parents=True, exist_ok=True)
        w = weights[t]
        with gzip.open(entry / "weights.tsv.gz", "wt") as f:
            w.to_csv(f, sep="\t", index=False, float_format="%.17g")
        (entry / "manifest.json").write_text(json.dumps({"method": METHOD, "trait": t, "status": "ok", "key": key,
                                                         "n_weights": len(w), "trait_info": {"N": n_train},
                                                         "choice": {"candidate": "c1"}}))
        rows.append({"method": METHOD, "trait": t, "status": "ok", "pgi_key": key[:16], "auroc_combined": 0.5,
                     "auroc_covariates": 0.5})
    res = world / "results" / os.path.basename(ds)
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(res / "tune_metrics.csv", index=False)
    ids = np.concatenate([dm.store.variants(c)["id"][v0:v1] for seq in dm.regions for c, v0, v1 in seq]).astype(str)
    W = np.zeros((len(ids), len(dm.label_names)))
    for j, t in enumerate(dm.label_names):
        W[pd.Index(ids).get_indexer(weights[t]["SNP"]), j] = weights[t]["w"]
    val = pd.read_csv(os.path.join(ds, "samples_val.tsv"), sep="\t", dtype={"IID": str})
    out = os.path.join(ds, "scores", "pgi", "f" * 64, METHOD)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "val.npz"), iid=val["IID"].to_numpy(str), pgi=_dosages(dm, "val") @ W,
             status=np.array(["ok"] * len(dm.label_names)))
    return W


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    from generate_synthetic_multilabel import generate
    world = tmp_path_factory.mktemp("w")
    generate("multi", n_participants=1200, n_snps=132, base_dir=str(world / "genes"), seed=3)
    ds = str(world / "genes" / "multi" / "store")
    dm = _dm(ds)
    rng = np.random.default_rng(5)
    tabs = [dm.store.variants(c) for c in dm.store.chroms]
    ids = np.concatenate([t["id"] for t in tabs]).astype(str)
    alt, ref = (np.concatenate([t[k] for t in tabs]).astype(str) for k in ("alt", "ref"))
    weights = {}
    for t in dm.label_names:
        pick = np.sort(rng.choice(len(ids), size=20, replace=False))
        weights[t] = pd.DataFrame({"SNP": ids[pick], "CHR": 1, "POS": pick, "A1": alt[pick], "A2": ref[pick],
                                   "w": rng.normal(0, 0.1, size=20)})
    W = _fake_pgi(world, ds, len(dm.splits["train"]["y"]), weights)
    return {"dir": world, "ds": ds, "dm": dm, "weights": weights, "W": W}


def _build(world, ds, *extra, weights_dataset=None, out="t.pt"):
    args = [ds, "--method", METHOD, "--partition", "uniform:2", "--pgi_results", str(world["dir"] / "results"),
            "--pgi_cache", str(world["dir"] / "cache"), "--check_rows", "100000", "--num_workers", "0",
            "--out", str(world["dir"] / out)] + list(extra)
    if weights_dataset:
        args += ["--weights_dataset", weights_dataset]
    return make_pgi_table.main(args)


def _pool(dm, path, split):
    from widedeepgp.blockencoder import BlockEncoder
    f = torch.load(path, weights_only=False)
    enc = BlockEncoder(dm.partition, f["manifest"]["embed_dim"], 8, "dosage", 0.0, 1)
    enc.load_state_dict({**enc.state_dict(), **f["state"]})
    return make_pgi_table.pooled(dm, enc, split, np.arange(len(dm.splits[split]["y"])), "cpu"), f["manifest"]


@pytest.mark.parametrize("drop_zero", [True, False])
def test_block_sums_give_back_each_pgi_and_channels_have_unit_rms(world, drop_zero):
    dm, W = world["dm"], world["W"]
    path = _build(world, world["ds"], "--in_sample", *(["--drop_zero"] if drop_zero else []), out=f"t{drop_zero}.pt")
    p, man = _pool(dm, path, "val")
    src = man["source"]
    assert (man["kind"], man["embed_dim"], man["n_folds"], man["snp_code"]) == ("pgi", 6, 1, "dosage")
    assert src["training_overlap"] == len(dm.splits["train"]["y"]) and src["tuning_correlation_min"] > 0.9999
    kept = W.any(axis=1)
    assert src["snps_kept"] == (kept.sum() if drop_zero else len(W)) and src["snps_all"] == len(W)
    ids = np.concatenate(dm.partition["ids"])
    blocks, inv = np.unique(ids, return_inverse=True)
    n = np.bincount(inv, weights=kept if drop_zero else np.ones(len(W)))
    assert src["blocks_empty"] == int((n == 0).sum()) and (src["blocks_empty"] > 0) == drop_zero
    # pooled x sqrt(divisor) x the channel's unscaled RMS, summed over blocks = the PGI less its training-mean term
    got = (p * np.sqrt(n)[None, :, None]).sum(axis=1) * np.array(src["channel_rms_unscaled"])[None, :]
    mean = np.concatenate(dm.snp_mean)
    want = _dosages(dm, "val") @ W - mean @ W
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4 * np.abs(want).max())
    table = np.concatenate([torch.load(path, weights_only=False)["state"][f"tables.{i}"].numpy() for i in range(len(dm.regions))])
    assert not table[~kept].any()                                   # SNPs without a weight carry nothing
    ptr, _ = _pool(dm, path, "train")
    np.testing.assert_allclose(np.sqrt((ptr.astype(np.float64) ** 2).mean(axis=(0, 1))), 1.0, rtol=1e-5)


def test_in_sample_weights_and_wrong_alleles_are_refused(world):
    with pytest.raises(SystemExit, match="labels would be in their own inputs"):
        _build(world, world["ds"])
    other = world["dir"] / "flip"
    other.mkdir(exist_ok=True)
    t0 = world["dm"].label_names[0]
    for name in ("results", "cache"):
        shutil.copytree(world["dir"] / name, other / name, dirs_exist_ok=True)
    flipped = world["weights"][t0].copy()
    flipped.loc[flipped.index[0], "A1"] = flipped.loc[flipped.index[0], "A2"]
    key = pd.read_csv(other / "results" / "store" / "tune_metrics.csv", dtype={"pgi_key": str}).set_index("trait").loc[t0, "pgi_key"]
    entry = next((other / "cache" / key[:2]).glob(key + "*"))
    with gzip.open(entry / "weights.tsv.gz", "wt") as f:
        flipped.to_csv(f, sep="\t", index=False, float_format="%.17g")
    with pytest.raises(SystemExit, match="not for the store's ALT allele"):
        make_pgi_table.main([world["ds"], "--method", METHOD, "--partition", "uniform:2", "--in_sample", "--num_workers", "0",
                             "--pgi_results", str(other / "results"), "--pgi_cache", str(other / "cache"),
                             "--out", str(world["dir"] / "x.pt")])


def test_a_complement_keeps_the_other_lines_and_its_table_is_leak_free(world):
    ds, dm = world["ds"], world["dm"]
    rung = os.path.join(os.path.dirname(ds), "rung")
    os.makedirs(rung, exist_ok=True)
    lines = open(os.path.join(ds, "samples_train.tsv")).readlines()
    half = len(lines) // 2
    open(os.path.join(rung, "samples_train.tsv"), "w").writelines(lines[:half])
    out = os.path.join(os.path.dirname(ds), "comp")
    make_complement.main([ds, rung, out])
    assert open(os.path.join(out, "samples_train.tsv")).readlines() == [lines[0]] + lines[half:]
    assert os.stat(os.path.join(out, "samples_train.tsv")).st_mode & 0o777 == 0o600
    assert os.path.realpath(os.path.join(out, "samples_val.tsv")) == os.path.realpath(os.path.join(ds, "samples_val.tsv"))
    assert json.load(open(os.path.join(out, "complement.json")))["n_train"] == len(lines) - half
    with pytest.raises(SystemExit, match="exists"):
        make_complement.main([ds, rung, out])
    bad = os.path.join(os.path.dirname(ds), "bad")
    os.makedirs(bad, exist_ok=True)
    open(os.path.join(bad, "samples_train.tsv"), "w").writelines(lines[:2] + ["\t".join(["nobody"] + lines[1].split("\t")[1:])])
    with pytest.raises(SystemExit, match="not all in the parent"):
        make_complement.main([ds, bad, os.path.join(os.path.dirname(ds), "comp2")])
    # the rung's own weights (its GWAS of half - 1 people): no overlap with the complement, so no --in_sample needed
    _fake_pgi(world["dir"], rung_ds := _as_dataset(ds, rung), half - 1, world["weights"])
    path = _build(world, out, weights_dataset=rung_ds, out="comp.pt")
    src = torch.load(path, weights_only=False)["manifest"]["source"]
    assert src["training_overlap"] == 0 and src["n_training"] == len(lines) - half


def _as_dataset(parent, rung):
    """Give the made-up rung the parent's other files, so that it is a dataset of its own."""
    for name in ("store.json", "labels.json", "samples_val.tsv", "samples_test.tsv", "annotations.npz"):
        src, dst = os.path.join(parent, name), os.path.join(rung, name)
        if os.path.exists(src) and not os.path.exists(dst):
            if name == "store.json":
                spec = json.load(open(src))
                spec["store"] = os.path.join(parent, spec["store"])
                json.dump(spec, open(dst, "w"))
            else:
                os.symlink(os.path.realpath(src), dst)
    return rung


def test_a_blocks_run_loads_the_table_frozen_and_caches_it(world, tmp_path):
    import main_multilabel
    path = _build(world, world["ds"], "--in_sample", "--drop_zero", out="run.pt")
    args = ["--label", "multi", "--rd_mode", "store", "--data_dir", str(world["dir"] / "genes"), "--run_dir", str(tmp_path),
            "--experiment", "t", "--exp_name", "pgi", "--partition", "uniform:2", "--mixer", "transformer", "--d_model", "8",
            "--n_heads", "2", "--embed_dim", "6", "--batch_size", "16", "--num_workers", "0", "--max_epoch_equiv", "1",
            "--val_per_epoch", "2", "--min_participants", "0", "--val_min_participants", "0", "--stop_patience", "0",
            "--tune_trials", "0", "--eval_batch_size", "64", "--input_mask", "0", "--init_encoder", path, "--freeze_encoder"]
    out = main_multilabel.main(args)
    config = json.load(open(os.path.join(out["run_dir"], "config.json")))
    assert config["first_layer_cache_seconds"] is not None and config["blocks"]["embed_dim"] == 6
    ck = torch.load(out["best_checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
    state = torch.load(path, weights_only=False)["state"]
    assert all(torch.equal(ck[f"encoder.{k}"], v) for k, v in state.items())
