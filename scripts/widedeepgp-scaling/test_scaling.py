"""Tests of compute.py, fit_curves.py and make_figures.py on synthetic inputs in tmp_path: FLOPs
against a hand count for a tiny configuration of each mixer, the compute table of a fake run, the
gathering of a fake run tree with a fake PGI results folder, the table-only fit, the per-person fit
on a simulated tuning set in the real file layout (and the test split's lock and log), and every
figure. No real run, dataset or score file is read.

    /home/robert/envs/widedeepgp/bin/python -m pytest scripts/widedeepgp-scaling/test_scaling.py -q
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute  # noqa: E402
import fit_curves as F  # noqa: E402
import make_figures as M  # noqa: E402

LABELS = ["I10", "E11", "K80"]
K = {"I10": 0.3, "E11": 0.1, "K80": 0.08}
COVARIATES = ["birth_year", "sex"] + [f"pc{k}" for k in range(1, 11)]
FINGERPRINT = "abcdef0123456789"


@pytest.fixture(autouse=True)
def _few_blas_threads():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=4):
        yield


def tiny_config(mixer, dataset_dir, seed=1, d_model=8, e_layers=2, embed_dim=4, n_blocks=5, exp_name="cell", n_mixer=1000,
                snp_code="dosage", init_encoder=None, freeze_encoder=False):
    """config.json as train.py writes it: the ablation flags in args and model (init_encoder a path or
    None, freeze_encoder a bool), the file's digest in blocks."""
    return {"args": {"seed": seed, "experiment": "blocks", "exp_name": exp_name, "snp_code": snp_code, "init_encoder": init_encoder,
                     "freeze_encoder": freeze_encoder},
            "model": {"d_model": d_model, "e_layers": e_layers, "n_heads": 2, "mixer": mixer, "label_names": LABELS,
                      "init_encoder": init_encoder, "freeze_encoder": freeze_encoder},
            "blocks": {"partition": "prive431", "partition_digest": "abc", "n_blocks": n_blocks, "n_tokens": n_blocks + 1,
                       "embed_dim": embed_dim, "snp_code": snp_code,
                       "n_parameters_by_module": {"table": 100 * embed_dim, "identity": 7 * embed_dim, "block_mlp": 300,
                                                  "mixer": n_mixer, "covariates": 50, "head": 200},
                       "n_embedding_parameters": 107 * embed_dim, "n_non_embedding_parameters": n_mixer + 550,
                       "init_encoder_sha256": "ab" * 32 if init_encoder else None},
            "n_snps": 100, "n_parameters": 107 * embed_dim + n_mixer + 550, "dataset_dir": str(dataset_dir), "n_train": 8400}


def test_flops_match_a_hand_count_for_a_tiny_configuration_of_each_mixer(tmp_path):
    # transformer: M = 100, codes 1, d_e = 4, K = 5, T = 6, d = 8, L = 2, N_mixer = 1000
    f = compute.flops_per_participant(tiny_config("transformer", tmp_path))
    assert f["table"] == 2 * 100 * 1 * 4 == 800
    assert f["block_mlp"] == 2 * 5 * (4 * 16 + 4 * 4 * 8) == 1920
    assert f["mixer"] == 2 * 1000 * 6 == 12000
    assert f["attention_or_scan"] == 4 * 2 * 36 * 8 == 2304
    assert f["forward"] == 800 + 1920 + 12000 + 2304 == 17024 and f["train"] == 3 * 17024
    g = compute.flops_per_participant(tiny_config("bimamba1", tmp_path))
    assert g["attention_or_scan"] == 2 * 6 * 16 * 16 * 18 * 2 == 110592
    assert g["forward"] == 800 + 1920 + 12000 + 110592
    geno = compute.flops_per_participant(tiny_config("transformer", tmp_path, snp_code="genotype"))
    assert geno["table"] == 3 * 800                              # three vectors per SNP
    with pytest.raises(ValueError, match="no FLOPs formula"):
        compute.flops_per_participant(tiny_config("mamba3", tmp_path))


def fake_run(root, mixer, dataset_dir, seed=1, exp_name="cell", gain=0.02, splits=("val",), **kw):
    """A run directory with config.json, snapshots.csv, metrics.csv and scores_<split>_summary.csv."""
    run = root / exp_name / f"seed{seed}"
    run.mkdir(parents=True)
    config = tiny_config(mixer, dataset_dir, seed=seed, exp_name=exp_name, **kw)
    json.dump(config, open(run / "config.json", "w"))
    pd.DataFrame({"step": [10, 20, 30], "batches": [10, 20, 30], "samples_seen": [2560, 5120, 7680],
                  "epoch_equiv": [0.3, 0.6, 0.9], "train_seconds": [36.0, 72.0, 108.0],
                  "file": [f"snapshots/step_{s}.ckpt" for s in (10, 20, 30)]}).to_csv(run / "snapshots.csv", index=False)
    rows = []
    for v, step, seen, auc in ((1, 15, 3840, 0.60), (2, 30, 7680, 0.62)):
        for label in LABELS + ["MEAN"]:
            for metric, value in (("auroc", auc), ("ap", 0.3), ("bce", 0.4)):
                rows.append(dict(validation=v, step=step, samples_seen=seen, epoch_equiv=seen / 8400, label=label, metric=metric,
                                 value=value, n=1800, n_cases=500))
    pd.DataFrame(rows).to_csv(run / "metrics.csv", index=False)
    cov = {"I10": 0.66, "E11": 0.70, "K80": 0.62}
    for split in splits:
        pd.DataFrame([dict(snapshot="best", label=l, n=20000, n_cases=2000, auroc=cov[l] + gain + 0.001 * seed,
                           auroc_covariates_only=cov[l], auroc_gain=gain + 0.001 * seed, bce=0.3, bce_covariates_only=0.31)
                      for l in LABELS]).to_csv(run / f"scores_{split}_summary.csv", index=False)
    return run


def test_the_compute_table_of_a_fake_run_and_the_cli(tmp_path):
    run = fake_run(tmp_path / "runs", "transformer", tmp_path / "datasets" / "hm3plus_n13199")
    t = compute.compute_table(str(run))
    assert list(t.kind) == ["snapshot"] * 3 + ["validation"] * 2
    assert np.allclose(t.flops, 3 * 17024 * t.samples_seen)
    assert t.gpu_hours.iloc[2] == pytest.approx(108 / 3600) and np.isnan(t.gpu_hours.iloc[3])
    assert t.auroc_mean.iloc[4] == 0.62 and np.isnan(t.auroc_mean.iloc[0])
    assert (t.n_mixer_params == 1000).all() and (t.rung == "hm3plus_n13199").all() and (t.seed == 1).all()
    assert t.flops_per_participant_train.iloc[0] == 3 * 17024 and (t.n_train == 8400).all()
    compute.main([str(run), "--out", str(tmp_path / "compute")])
    out = pd.read_csv(tmp_path / "compute" / "hm3plus_n13199__cell__seed1_compute.csv")
    assert len(out) == 5 and out.run.iloc[0] == "hm3plus_n13199__cell__seed1"
    empty = tmp_path / "empty_run"
    empty.mkdir()
    json.dump(tiny_config("bimamba1", tmp_path), open(empty / "config.json", "w"))
    e = compute.compute_table(str(empty))
    assert len(e) == 1 and np.isnan(e.flops.iloc[0]) and e.mixer.iloc[0] == "bimamba1"


def fake_people(n=700, seed=5):
    """A simulated tuning set and test set (the same for every rung, as in the real design): the
    covariates, three diseases from a liability model and each person's genetic value."""
    rng = np.random.default_rng(seed)
    out = {}
    for split in ("val", "test"):
        cov = pd.DataFrame({"birth_year": rng.uniform(1940, 1970, n).round(), "sex": rng.integers(0, 2, n).astype(float)})
        for k in range(1, 11):
            cov[f"pc{k}"] = rng.normal(size=n)
        g = rng.normal(size=(n, 3))
        liab = 0.03 * (cov.birth_year.to_numpy() - 1955)[:, None] + 0.3 * cov.sex.to_numpy()[:, None] + g + rng.normal(size=(n, 3))
        y = np.column_stack([(liab[:, j] > np.quantile(liab[:, j], 1 - K[l])).astype(float) for j, l in enumerate(LABELS)])
        table = pd.concat([pd.DataFrame({"IID": [f"{split}{i:05d}" for i in range(n)]}), pd.DataFrame(y, columns=LABELS), cov], axis=1)
        out[split] = (table, g)
    return out


def write_people(datasets, rung, rung_i, people, runs, seed=5):
    """The rung's tables, store.json and per-person score files in the real layout: the deep runs'
    <split>_best.npz, the PGI arm's <fingerprint>/<method>/<split>.npz (c_t and ldpred2_auto only;
    lassosum2 has rows in tune_metrics.csv but no files) and the twin's <split>_best.npz, with less
    noise at each larger rung."""
    rng = np.random.default_rng(seed + rung_i)
    d = datasets / rung
    json.dump({"store": "fake", "pos_mode": "index", "labels": LABELS, "covariates": COVARIATES}, open(d / "store.json", "w"))
    for split, (table, g) in people.items():
        table.to_csv(d / f"samples_{split}.tsv", sep="\t", index=False)
        iid, y = table.IID.to_numpy(str), table[LABELS].to_numpy(np.float32)
        files = {d / "scores" / "blocks" / name / f"{split}_best.npz": (0.7, "logit_population") for name in runs}
        files.update({d / "scores" / "pgi" / (FINGERPRINT + "aaaa") / m / f"{split}.npz": (sd, "pgi") for m, sd in (("c_t", 2.0), ("ldpred2_auto", 1.0))})
        files[d / "scores" / "blocks_linear" / "ridge" / f"{split}_best.npz"] = (1.0, "logit_population")
        for path, (sd, key) in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            s = (g + rng.normal(0, sd / (rung_i + 1), g.shape)).astype(np.float32)
            extra = {"logit_population": np.zeros_like(s), "status": np.array(["ok"] * 3)} if key == "pgi" else {}
            np.savez(path, iid=iid, labels=np.array(LABELS), y=y, **{key: s}, **extra)


def fake_study(tmp_path, people=False):
    """Five rungs x two mixer sizes x two seeds of deep runs (summaries for the tuning and the test
    split), a PGI results folder with three methods and the disease tables; with ``people``, the
    simulated tables and every per-person score file too. Returns (roots, pgi results, datasets,
    sldsc, power)."""
    runs, datasets, pgi = tmp_path / "runs", tmp_path / "datasets", tmp_path / "pgi_results"
    sim = fake_people() if people else None
    for rung_i, (rung, n_train) in enumerate(F.RUNGS.items()):
        (datasets / rung).mkdir(parents=True)
        names = []
        for size, n_mixer, gain in (("S1", 1000, 0.01), ("S3", 20000, 0.02)):
            for seed in (1, 2):
                names.append(size if seed == 1 else f"{size}_seed{seed}")           # the chain's naming of the seed retrains
                fake_run(runs / rung, "transformer", datasets / rung, seed=seed, exp_name=names[-1], n_mixer=n_mixer,
                         gain=gain + 0.004 * np.log2(n_train / 13199), d_model=8 if size == "S1" else 16, splits=("val", "test"))
        if people:
            write_people(datasets, rung, rung_i, sim, names)
        (pgi / rung).mkdir(parents=True)
        rows = []
        for method, q in (("c_t", 0.5), ("ldpred2_auto", 0.9), ("lassosum2", 0.7)):
            for label in LABELS:
                cov = {"I10": 0.66, "E11": 0.70, "K80": 0.62}[label]
                g = q * (0.01 + 0.004 * np.log2(n_train / 13199))
                rows.append(dict(fingerprint=FINGERPRINT, dataset=rung, method=method, trait=label, status="ok",
                                 evaluation="tuning, in-sample", basis="linear", n=20000, n_cases=2000, auroc_pgi=0.55,
                                 auroc_combined=cov + g, ce_combined=0.3, auroc_covariates=cov, ce_covariates=0.31,
                                 delta_auroc=g, delta_ce=-0.01, z_pgi=5.0, combined_converged=True, combined_separation=False,
                                 pgi_key="k", detail=""))
        rows.append(dict(rows[-1], trait="K80", method="c_t", status="failed", delta_auroc=np.nan))     # dropped
        pd.DataFrame(rows).to_csv(pgi / rung / "tune_metrics.csv", index=False)
    if not people:
        twin = datasets / "hm3plus" / "scores" / "blocks_linear" / "ridge" / "val_best.npz"
        twin.parent.mkdir(parents=True)
        np.savez(twin, iid=np.array(["a"]), labels=np.array(LABELS), logit_population=np.zeros((1, 3), np.float32), y=np.zeros((1, 3), np.float32))
        one = datasets / "hm3plus" / "scores" / "pgi" / (FINGERPRINT + "aaaa") / "c_t" / "val.npz"
        one.parent.mkdir(parents=True)
        np.savez(one, iid=np.array(["a"]), pgi=np.zeros((1, 3), np.float32))
    sldsc = tmp_path / "sldsc_h2.csv"
    pd.DataFrame([dict(label=l, status="ok", N_eff=1e5, K_train=K[l], h2_obs50=0.1, h2_obs50_se=0.01, h2_liab=0.15) for l in LABELS]).to_csv(sldsc, index=False)
    power = tmp_path / "rung_power.csv"
    pd.DataFrame([dict(n_train=n, median_neff=n / 4, M_e=me, label=l, neff_train=4 * n * K[l] * (1 - K[l]), power=0.5)
                  for n in F.RUNGS.values() for me in (30000, 60000, 120000) for l in LABELS]).to_csv(power, index=False)
    return [str(runs)], str(pgi), str(datasets), str(sldsc), str(power)


def test_gather_builds_the_tidy_table_and_the_manifest_from_a_fake_study(tmp_path):
    roots, pgi, datasets, sldsc, power = fake_study(tmp_path)
    t, m = F.gather(roots, pgi, datasets, sldsc, power)
    deep = t[t.family.str.startswith("deep:")]
    assert set(deep.family) == {"deep:transformer:d8x2:de4:prive431", "deep:transformer:d16x2:de4:prive431"}
    assert len(deep) == 5 * 2 * 2 * 3 * 2 and set(deep.seed) == {1, 2} and set(deep.split) == {"val", "test"}
    assert (deep.n_mixer_params.isin([1000, 20000])).all() and (deep.n_blocks == 5).all()
    lin = t[~t.family.str.startswith("deep:")]
    assert set(lin.family) == {"c_t", "ldpred2_auto", "lassosum2"} and len(lin) == 5 * 3 * 3        # the failed cell is dropped
    assert np.isfinite(t.n_eff).all() and np.isfinite(t.delta_r2).all() and (t.delta_r2 > 0).all()
    i10 = t[(t.disease == "I10") & (t.rung == "hm3plus")].iloc[0]
    assert i10.n_eff == pytest.approx(4 * 215872 * 0.3 * 0.7) and i10.K == 0.3 and i10.h2_50 == 0.1 and i10.h2_liab == 0.15
    assert (t.score_path.dropna().str.endswith("_best.npz")).all()
    assert set(m.family) == {"deep:transformer:d8x2:de4:prive431", "deep:transformer:d16x2:de4:prive431", "c_t", "ldpred2_auto",
                             "lassosum2", "twin:ridge"}
    assert set(m.split) == {"val", "test"} and len(m[m.split == "test"]) == len(m[m.split == "val"]) - 1   # one twin file, val only
    val = m[m.split == "val"]
    assert val[val.family == "twin:ridge"].exists.all() and val[(val.family == "c_t") & (val.rung == "hm3plus")].exists.iloc[0]
    assert not val[(val.family == "c_t") & (val.rung == "hm3plus_n13199")].exists.iloc[0]
    assert not val[val.family.str.startswith("deep:")].exists.any() and not m[m.split == "test"].exists.any()
    assert len(val[val.family.str.startswith("deep:")]) == 5 * 2 * 2
    assert m[m.split == "test"].path.str.contains("test.npz|test_best.npz").all()


def test_family_names_carry_the_ablation_flags_as_the_model_records_them(tmp_path):
    d = tmp_path / "datasets" / "hm3plus_n13199"
    fam, rung, cols = F.family_of(tiny_config("bimamba1", d, snp_code="genotype"))
    assert fam == "deep:bimamba1:d8x2:de4:prive431:snp_code=genotype" and rung == "hm3plus_n13199" and cols["embed_dim"] == 4
    plain = tiny_config("transformer", d)
    own = tiny_config("transformer", d, init_encoder=str(d / "blocks" / "init_ridge_d16.pt"))                 # the rung's own warm start
    top = tiny_config("transformer", d, init_encoder=str(tmp_path / "datasets" / "hm3plus" / "blocks" / "pretrained_prive431_dosage_e16.pt"))
    frozen = tiny_config("transformer", d, init_encoder=str(d / "blocks" / "init_ridge_d16.pt"), freeze_encoder=True)
    assert F.family_of(plain)[0] == "deep:transformer:d8x2:de4:prive431"
    assert F.family_of(own)[0] == "deep:transformer:d8x2:de4:prive431:init=init_ridge_d16"
    assert F.family_of(top)[0] == "deep:transformer:d8x2:de4:prive431:init=pretrained_prive431_dosage_e16@hm3plus"
    assert F.family_of(frozen)[0] == "deep:transformer:d8x2:de4:prive431:init=init_ridge_d16:frozen"
    at_top = tiny_config("transformer", tmp_path / "datasets" / "hm3plus", init_encoder=top["args"]["init_encoder"])
    assert F.family_of(at_top)[0].endswith(":init=pretrained_prive431_dosage_e16")     # the same file is the top rung's own
    model_only = tiny_config("transformer", d, init_encoder=own["args"]["init_encoder"], freeze_encoder=True)
    model_only["args"] = {k: v for k, v in model_only["args"].items() if k not in ("init_encoder", "freeze_encoder")}
    assert F.family_of(model_only)[0] == F.family_of(frozen)[0]
    snp_mode = {"args": {"seed": 1}, "model": {"mixer": "bimamba1", "d_model": 32, "e_layers": 1}, "n_parameters": 5, "dataset_dir": str(d)}
    assert F.family_of(snp_mode) == ("deep:bimamba1:d32x1:snps", "hm3plus_n13199", dict(mixer="bimamba1", d_model=32, e_layers=1, n_params=5))


def test_gather_refuses_two_runs_that_would_share_a_cell(tmp_path):
    roots, pgi, datasets, sldsc, power = fake_study(tmp_path)
    rung = "hm3plus_n26287"
    fake_run(tmp_path / "runs" / rung / "trials", "transformer", Path(datasets) / rung, seed=1, exp_name="S3", n_mixer=20000, d_model=16,
             splits=("val", "test"))
    with pytest.raises(ValueError, match="runs that would share a cell") as e:
        F.gather(roots, pgi, datasets, sldsc, power)
    assert str(e.value).count("deep:transformer:d16x2:de4:prive431") == 4 and "trials/S3/seed1" in str(e.value)   # val and test, both runs


def test_fit_on_the_table_alone_writes_every_result_table_and_the_figures_draw(tmp_path):
    roots, pgi, datasets, sldsc, power = fake_study(tmp_path)
    F.main(["gather", "--runs", roots[0], "--pgi_results", pgi, "--datasets", datasets, "--sldsc", sldsc, "--power", power,
            "--out", str(tmp_path / "gathered")])
    assert (tmp_path / "gathered" / "tidy.csv").exists() and (tmp_path / "gathered" / "scores_manifest.csv").exists()
    results = F.fit(str(tmp_path / "gathered" / "tidy.csv"), str(tmp_path / "fit"))
    for name in ("cells", "pooled", "null", "envelope", "slopes", "rho", "gap", "size", "size_summary", "daetwyler", "eligibility"):
        assert (tmp_path / "fit" / f"{name}.csv").exists(), name
    assert results["null"].null.iloc[0] is None                   # no intervals without per-person scores
    assert set(results["envelope"].family) == {"deep:transformer:d16x2:de4:prive431"}
    assert len(results["size"]) == 5 and (results["size"].n_opt == "deep:transformer:d16x2:de4:prive431").all()
    assert set(results["cells"].rung) == set(F.RUNGS) and "test" not in set(pd.read_csv(tmp_path / "fit" / "cells.csv").get("split", []))
    runs = [str(p) for p in (tmp_path / "runs").glob("*/S3/seed1")]
    compute.main(runs + ["--out", str(tmp_path / "compute")])
    tidy = pd.read_csv(tmp_path / "gathered" / "tidy.csv")
    extra = tidy[tidy.family.str.startswith("deep:") & (tidy.split == "val")].assign(split="asian", auroc=lambda d: d.auroc - 0.02)
    pd.concat([tidy, extra]).to_csv(tmp_path / "gathered" / "tidy.csv", index=False)
    M.main(["--results", str(tmp_path / "fit"), "--tidy", str(tmp_path / "gathered" / "tidy.csv"),
            "--compute", str(tmp_path / "compute"), "--out", str(tmp_path / "figures")])
    for name in ("curves_by_family", "curves_per_disease", "gap_vs_x", "size_axis", "compute_fixed_data", "transfer_offset"):
        for ext in (".png", ".pdf", "_data.csv", "_legend.md"):
            assert (tmp_path / "figures" / f"{name}{ext}").exists(), name + ext
    assert (tmp_path / "figures" / "ablations.md").exists() and (tmp_path / "figures" / "ablations_data.csv").exists()
    assert not (tmp_path / "figures" / "table_axis.png").exists()  # one table width only: nothing to draw
    assert not (tmp_path / "figures" / "blocks_axis.png").exists()
    assert len(list((tmp_path / "compute").glob("*_compute.csv"))) == 5
    data = pd.read_csv(tmp_path / "figures" / "curves_by_family_data.csv")
    assert set(data.family) >= {"deep", "lin", "c_t", "ldpred2_auto"} and set(data.panel) == {"a", "b"}
    size = pd.read_csv(tmp_path / "figures" / "size_axis_data.csv")
    assert (size[size.series == "n_opt"].family == "deep:transformer:d16x2:de4:prive431").all()
    offset = pd.read_csv(tmp_path / "figures" / "transfer_offset_data.csv")
    assert np.allclose(offset.offset, 0.02) and set(offset.split) == {"asian"}     # the test split is not a transfer group
    comp = pd.read_csv(tmp_path / "figures" / "compute_fixed_data_data.csv")
    assert len(comp) == 2 and comp.flops.is_monotonic_increasing


def test_the_per_person_fit_runs_on_the_real_layout_and_the_test_split_is_locked_and_logged(tmp_path, capsys):
    roots, pgi, datasets, sldsc, power = fake_study(tmp_path, people=True)
    t, m = F.gather(roots, pgi, datasets, sldsc, power)
    gathered = tmp_path / "gathered"
    gathered.mkdir()
    t.to_csv(gathered / "tidy.csv", index=False)
    m.to_csv(gathered / "scores_manifest.csv", index=False)
    assert m[m.family != "lassosum2"].exists.all() and not m[m.family == "lassosum2"].exists.any()
    rung = os.path.join(datasets, "hm3plus_n13199")
    args = (str(gathered / "tidy.csv"), str(tmp_path / "fit_val"), str(gathered / "scores_manifest.csv"), rung, 40)
    r = F.fit(*args)
    cells = r["cells"]
    deep = {"deep:transformer:d8x2:de4:prive431", "deep:transformer:d16x2:de4:prive431"}
    assert set(cells.family) == deep | {"c_t", "ldpred2_auto", "twin:ridge", "lin", "deep"}      # lassosum2 has no scores
    assert np.isfinite(cells[["n_eff", "h2_50", "K", "delta_auroc", "delta_auroc_lo", "delta_auroc_hi"]]).all().all()
    assert (cells[cells.family.isin(deep | {"deep"})].seeds == 2).all() and (cells[~cells.family.isin(deep | {"deep"})].seeds == 1).all()
    assert len(cells[cells.family == "twin:ridge"]) == 15 and len(r["p4"]) == 5 and np.isfinite(r["p4"].lo).all()
    assert len(r["size_summary"]) == 1 and r["size_summary"].axis.iloc[0] == "mixer size"
    assert r["null"].null.iloc[0] in (True, False)                 # every clause measured: sizes, a twin and intervals
    assert (tmp_path / "fit_val" / "pooled.csv").exists() and not (tmp_path / "fit_val" / "test_access.log").exists()
    capsys.readouterr()
    with pytest.raises(SystemExit, match="locked"):
        F.fit(*args[:1], str(tmp_path / "fit_test"), *args[2:], split="test")
    assert not (tmp_path / "fit_test").exists()
    rt = F.fit(*args[:1], str(tmp_path / "fit_test"), *args[2:], split="test", unlock_test=True)
    log = (tmp_path / "fit_test" / "test_access.log").read_text()
    assert log.count("\n") == 1 and "fit --split test" in log and "dataset=" + rung in log
    assert set(rt["cells"].family) == set(cells.family) and len(rt["cells"]) == len(cells)
    assert not np.allclose(rt["cells"].delta_auroc, cells.delta_auroc)               # other people, other values
    F.main(["fit", "--table", args[0], "--out", str(tmp_path / "fit_cli")])
    assert "TABLE ONLY" in capsys.readouterr().out


def test_a_tune_metrics_table_with_two_covariate_bases_is_refused(tmp_path):
    """The shared protocol has one basis (linear since 18 November 2026); a mixed table is a pipeline error, not a filter."""
    rung = next(iter(F.RUNGS))
    (tmp_path / rung).mkdir()
    row = dict(fingerprint="f", dataset=rung, method="c_t", trait="I10", status="ok", basis="linear", n=20000, n_cases=2000,
               auroc_combined=0.67, auroc_covariates=0.66, delta_auroc=0.01, pgi_key="k")
    pd.DataFrame([row, dict(row, method="lassosum2")]).to_csv(tmp_path / rung / "tune_metrics.csv", index=False)
    assert len(F.pgi_rows(str(tmp_path))) == 2
    pd.DataFrame([row, dict(row, method="lassosum2", basis="spline")]).to_csv(tmp_path / rung / "tune_metrics.csv", index=False)
    with pytest.raises(ValueError, match="several covariate bases"):
        F.pgi_rows(str(tmp_path))
