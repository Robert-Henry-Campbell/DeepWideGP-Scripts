"""Tests of the E0 simulator and read-out on a tiny synthetic store (no GPU, no UK Biobank data).

    /home/robert/envs/widedeepgp/bin/python -m pytest scripts/widedeepgp-blocks/test_e0.py -q
"""
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
DEEPGP = Path(os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DEEPGP / "tests"))
import ml_helpers  # noqa: E402  (puts the model repository and its scripts on sys.path)
import e0_summary  # noqa: E402
import simulate_traits as st  # noqa: E402
from genostore.store import GenoStore  # noqa: E402

# the real specs scaled to a store of 220 SNPs (10 per chromosome; blocks of 5 SNPs)
TRAITS = [
    dict(name="add_lo", kind="add", causal=20, h2=0.3),
    dict(name="add_hi", kind="add", causal=60, h2=0.5),
    dict(name=("shared_a", "shared_b"), kind="shared", causal=80, shared=60, rg=0.7, h2=0.3),
    dict(name="dom", kind="dom", causal=20, h2=0.2, planted=5, h2_planted=0.1),
    dict(name="epi", kind="epi", causal=20, h2=0.2, planted=5, h2_planted=0.1),
    dict(name="hetero", kind="hetero", causal=20, h2=0.2, planted=5, h2_planted=0.1, modifier_maf=0.1),
    dict(name="null", kind="null", h2=0.0, prevalence=0.1),
]
NAMES = ["add_lo", "add_hi", "shared_a", "shared_b", "dom", "epi", "hetero", "null"]
PLANTED = ("dom", "epi", "hetero")


def _blocks(store, k=5):
    return np.concatenate([np.arange(store.n_variants(c)) // k + 100 * i for i, c in enumerate(store.chroms)])


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e0")
    data_dir, _ = ml_helpers.synthetic_dataset(tmp, name="multi", n=4000, snps=220, seed=3)
    src = Path(data_dir) / "multi" / "store"
    # the rungs also have a related-participants table: split the synthetic test table into two disjoint ones
    test = pd.read_csv(src / "samples_test.tsv", sep="\t", dtype=str, keep_default_na=False)
    test.iloc[:300].to_csv(src / "samples_wb_related.tsv", sep="\t", index=False)
    test.iloc[300:].to_csv(src / "samples_test.tsv", sep="\t", index=False)
    store = GenoStore(str(src / "genostore"))
    read, orig = [], GenoStore.read_raw

    def recording(self, chrom, rows, v0=0, v1=None):
        read.append(np.asarray(rows))
        return orig(self, chrom, rows, v0, v1)

    GenoStore.read_raw = recording
    try:
        summary = st.build(src, src.parent / "sim", _blocks(store), traits=TRAITS, seed=1, device="cpu")
    finally:
        GenoStore.read_raw = orig
    row_of = {s: i for i, s in enumerate(store.samples)}
    rows = {s: np.array([row_of[i] for i in pd.read_csv(src / f"samples_{s}.tsv", sep="\t", dtype={"IID": str})["IID"]])
            for s in ("train", "val", "wb_related", "test")}
    return SimpleNamespace(src=src, out=src.parent / "sim", data_dir=data_dir, store=store, block=_blocks(store), summary=summary,
                           rows=rows, read=np.unique(np.concatenate(read)), truth=dict(np.load(src.parent / "sim" / "sim_truth.npz")),
                           val=slice(2800, 3400))


def _dosages(s):
    """Standardised dosages [n, 220] of the simulated rows (train, val, related), with the saved training statistics."""
    rows = np.concatenate([s.rows[k] for k in st.SPLITS])
    stats = np.load(next(s.out.glob("snp_stats_train_*.npz")))
    x = np.concatenate([s.store.read_dosage(c, rows) for c in s.store.chroms], axis=1).astype(np.float64)
    mean = np.concatenate([stats[f"mean{i}"] for i in range(22)])
    std = np.concatenate([stats[f"std{i}"] for i in range(22)])
    return x, (x - mean) / std, mean


def _r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1.0 - np.var(y - X @ b) / np.var(y)


def _g(s, trait):
    return np.concatenate([s.truth[f"g_{k}"][:, NAMES.index(trait)] for k in st.SPLITS]).astype(np.float64)


def _source_copy(sim, dst):
    """A copy of the synthetic source (store.json with an absolute store path, the four tables) to tamper with."""
    dst.mkdir()
    spec = json.load(open(sim.src / "store.json"))
    spec["store"] = spec["store"] if os.path.isabs(spec["store"]) else str((sim.src / spec["store"]).resolve())
    json.dump(spec, open(dst / "store.json", "w"))
    for s in st.SPLITS + ("test",):
        shutil.copy(sim.src / f"samples_{s}.tsv", dst / f"samples_{s}.tsv")
    return dst, spec


# ---- the dataset ------------------------------------------------------------------------------------------
def test_layout_tables_and_permissions(sim):
    out, src = sim.out, sim.src
    assert oct(out.stat().st_mode & 0o777) == "0o700"
    for f in ("store.json", "labels.json", "sim_truth.npz", "sim_log.json", "samples_train.tsv", "samples_val.tsv",
              "samples_wb_related.tsv", "samples_test.tsv"):
        assert oct((out / f).stat().st_mode & 0o777) == "0o600"
    spec, old = json.load(open(out / "store.json")), json.load(open(src / "store.json"))
    assert set(spec) == {"store", "pos_mode", "labels", "covariates"} and "regions" not in spec
    assert spec["labels"] == NAMES and spec["covariates"] == old["covariates"] and spec["pos_mode"] == old["pos_mode"]
    assert os.path.isabs(spec["store"]) and Path(spec["store"]).samefile(src / "genostore")
    for s in ("train", "val", "wb_related", "test"):
        new = pd.read_csv(out / f"samples_{s}.tsv", sep="\t", dtype=str, keep_default_na=False)
        was = pd.read_csv(src / f"samples_{s}.tsv", sep="\t", dtype=str, keep_default_na=False)
        assert list(new.columns) == ["IID"] + NAMES + old["covariates"]
        assert new["IID"].tolist() == was["IID"].tolist()                      # same people, same order
        assert new[old["covariates"]].equals(was[old["covariates"]])           # covariate text copied verbatim
        if s == "test":
            assert (new[NAMES] == "").all().all()                             # copied, never simulated
        else:
            assert set(np.unique(new[NAMES].to_numpy())) <= {"0", "1"}
    labels = json.load(open(out / "labels.json"))
    assert [d["label"] for d in labels] == NAMES
    assert labels[0]["cases_train"] == int(pd.read_csv(out / "samples_train.tsv", sep="\t")["add_lo"].sum())
    assert len(list(out.glob("snp_stats_train_*.npz"))) == 1


def test_the_test_split_is_never_read_and_the_others_are(sim):
    simulated = np.concatenate([sim.rows[k] for k in st.SPLITS])
    assert set(sim.read.tolist()) == set(simulated.tolist())
    assert not set(sim.read.tolist()) & set(sim.rows["test"].tolist())


def test_the_log_holds_aggregates_only(sim):
    text = (sim.out / "sim_log.json").read_text()
    ids = pd.read_csv(sim.src / "samples_train.tsv", sep="\t", dtype={"IID": str})["IID"]
    assert not any(i in text for i in ids)

    def longest(o):
        """The longest list anywhere in the document (dictionaries are keyed aggregates)."""
        if isinstance(o, dict):
            return max([0] + [longest(v) for v in o.values()])
        if isinstance(o, list):
            return max([len(o)] + [longest(v) for v in o])
        return 0
    log = json.loads(text)
    assert longest(log) <= len(NAMES)                                         # nothing per person
    assert log["n"] == {"train": 2800, "val": 600, "wb_related": 300, "test": 300}
    assert log["standardisation"]["from"].startswith("the training rows")
    assert set(log["traits"]) == set(NAMES)
    assert {t: v["n_planted"] for t, v in log["traits"].items()} == {t: (5 if t in PLANTED else 0) for t in NAMES}


def test_h2_prevalence_and_the_ceiling(sim):
    traits, n_all = sim.summary["traits"], 3700
    for sp in TRAITS:
        for nm in st.names(sp):
            t = traits[nm]
            target = sp["h2"] + sp.get("h2_planted", 0)
            assert t["h2_target"] == pytest.approx(target)
            assert t["h2_realised"] == pytest.approx(target, abs=0.02)
            cases = sum(t["cases"].values())
            assert abs(cases / n_all - t["prevalence_target"]) < 1.0 / n_all   # the quantile makes it exact overall
            for s in st.SPLITS:
                assert t["prevalence"][s] == pytest.approx(t["prevalence_target"], abs=0.06)
    assert traits["null"]["prevalence_target"] == 0.1 and traits["add_lo"]["prevalence_target"] == 0.2
    assert traits["null"]["ceiling_auroc_val"] == 0.5 and traits["null"]["h2_realised"] == 0.0
    assert traits["null"]["additive_r2_in_sample"] is None and traits["null"]["additive_ceiling_auroc_val"] is None
    assert traits["null"]["n_predictors"] == 0 and 0.4 <= traits["null"]["gblup_auroc_val"] <= 0.6
    for nm in ("add_lo", "add_hi", "shared_a", "dom", "epi", "hetero"):
        assert traits[nm]["ceiling_auroc_val"] > 0.65 and traits[nm]["ceiling_auroc_val"] > traits[nm]["gblup_auroc_val"]
    assert traits["add_hi"]["gblup_auroc_val"] > 0.6 and traits["add_lo"]["gblup_auroc_val"] > 0.6
    assert traits["add_hi"]["ceiling_auroc_val"] > traits["add_lo"]["ceiling_auroc_val"]  # h2 0.5 against 0.3


def test_the_additive_ceiling(sim):
    """The additive ceiling is the AUROC of the trait's additive part: the whole ceiling for an additive trait,
    below it by the planted share for a planted one."""
    _, z, _ = _dosages(sim)
    y = pd.read_csv(sim.out / "samples_val.tsv", sep="\t")
    for nm in ("add_lo", "add_hi", "shared_a", "shared_b"):
        t = sim.summary["traits"][nm]
        assert t["additive_ceiling_auroc_val"] == pytest.approx(t["ceiling_auroc_val"], abs=1e-3)
    for nm in PLANTED:
        t = sim.summary["traits"][nm]
        additive = z[:, sim.truth[f"{nm}_causal"]] @ sim.truth[f"{nm}_beta"]
        assert t["additive_ceiling_auroc_val"] == pytest.approx(st.auroc(additive[sim.val], y[nm]), abs=0.01)
        assert t["ceiling_auroc_val"] - t["additive_ceiling_auroc_val"] > 0.02     # the planted gap
        assert t["additive_ceiling_auroc_val"] > t["gblup_auroc_val"]


def test_additive_traits_are_linear_in_their_causal_snps(sim):
    _, z, _ = _dosages(sim)
    for nm in ("add_lo", "add_hi"):
        causal = sim.truth[f"{nm}_causal"]
        assert len(causal) == dict(add_lo=20, add_hi=60)[nm] and len(np.unique(causal)) == len(causal)
        assert _r2(_g(sim, nm), z[:, causal]) > 0.999
        assert sim.summary["traits"][nm]["additive_r2_in_sample"] > 0.999
        assert sim.summary["traits"][nm]["h2_components"] == pytest.approx({"additive": sim.summary["traits"][nm]["h2_realised"]}, abs=1e-6)


def test_dominance_components(sim):
    x, z, _ = _dosages(sim)
    causal, loci = sim.truth["dom_causal"], sim.truth["dom_loci"]
    assert len(loci) == len(sim.truth["dom_effect"]) == 5 and np.isin(loci, causal).all() and len(np.unique(loci)) == 5
    g, za = _g(sim, "dom"), z[:, causal]
    h = 1.0 - np.abs(x[:, loci] - 1.0)
    d = np.column_stack([h[:, j] - np.polyval(np.polyfit(x[:, c], h[:, j], 1), x[:, c]) for j, c in enumerate(loci)])
    assert _r2(g, za) == pytest.approx(0.2 / 0.3, abs=0.03)                  # the additive codes see 2/3
    assert _r2(g, np.column_stack([za, d])) > 0.999                           # plus orthogonal dominance: all
    shares = sim.summary["traits"]["dom"]["h2_components"]
    assert shares["additive"] == pytest.approx(0.2, abs=0.02) and shares["planted"] == pytest.approx(0.1, abs=0.02)
    assert sim.summary["traits"]["dom"]["additive_r2_in_sample"] == pytest.approx(0.2 / 0.3, abs=0.03)


def test_interaction_components(sim):
    _, z, _ = _dosages(sim)
    causal, pairs = sim.truth["epi_causal"], sim.truth["epi_pairs"]
    assert pairs.shape == (5, 2) and len(np.unique(pairs)) == 10 and len(sim.truth["epi_effect"]) == 5
    assert np.isin(pairs, causal).all() and (sim.block[pairs[:, 0]] != sim.block[pairs[:, 1]]).all()
    g = _g(sim, "epi")
    prod = z[:, pairs[:, 0]] * z[:, pairs[:, 1]]
    assert _r2(g, z[:, causal]) == pytest.approx(0.2 / 0.3, abs=0.03)
    assert _r2(g, np.column_stack([z[:, causal], prod])) > 0.999
    shares = sim.summary["traits"]["epi"]["h2_components"]
    assert shares["additive"] == pytest.approx(0.2, abs=0.02) and shares["planted"] == pytest.approx(0.1, abs=0.02)


def test_heterogeneity_components(sim):
    x, z, mean = _dosages(sim)
    causal, loci, mod = sim.truth["hetero_causal"], sim.truth["hetero_loci"], sim.truth["hetero_modifier"]
    assert len(loci) == len(mod) == 5 and np.isin(loci, causal).all()
    assert (sim.block[mod] == sim.block[loci]).all() and (mod != loci).all()
    g = _g(sim, "hetero")
    carrier = np.where(mean[mod] <= 1.0, x[:, mod] >= 0.5, x[:, mod] <= 1.5).astype(float)
    both = np.union1d(causal, mod)
    r2_add = _r2(g, z[:, both])
    assert r2_add == pytest.approx(0.2 / 0.3, abs=0.03)                        # the switch is orthogonal to both dosages
    assert _r2(g, np.column_stack([z[:, both], z[:, loci] * carrier])) > 0.999
    shares = sim.summary["traits"]["hetero"]["h2_components"]
    assert shares["additive"] == pytest.approx(0.2, abs=0.02) and shares["planted"] == pytest.approx(0.1, abs=0.02)
    assert sim.summary["traits"]["hetero"]["additive_r2_in_sample"] == pytest.approx(r2_add, abs=0.01)
    assert sim.summary["traits"]["hetero"]["n_predictors"] == len(both)


def test_the_shared_pair(sim):
    a, b = sim.truth["shared_causal_a"], sim.truth["shared_causal_b"]
    assert len(a) == len(b) == 80 and len(np.intersect1d(a, b)) == 60
    ba, bb = sim.truth["shared_beta_a"], sim.truth["shared_beta_b"]
    assert np.corrcoef(ba[:60], bb[:60])[0, 1] > 0.8 and abs(np.corrcoef(ba[60:], bb[60:])[0, 1]) < 0.4
    assert sim.summary["genetic_correlation"]["shared_a_shared_b"] == pytest.approx(0.7, abs=0.1)
    assert np.corrcoef(_g(sim, "shared_a"), _g(sim, "shared_b"))[0, 1] == pytest.approx(0.7, abs=0.1)


def test_the_saved_statistics_are_the_data_modules(sim):
    """The deep runs standardise with snp_stats_train_<hash>.npz; the simulator's file must be the one they look for."""
    from genostore.datamodule import SNPStoreDataModule
    files = list(sim.out.glob("snp_stats_train_*.npz"))
    assert len(files) == 1
    cfg = ml_helpers.make_cfg(sim.data_dir, "multi", rd_mode="sim", snp_stats="train", scaling=True)
    dm = SNPStoreDataModule(cfg)
    assert len(list(sim.out.glob("snp_stats_train_*.npz"))) == 1              # found by name: nothing recomputed
    z = np.load(files[0])
    x = np.concatenate([sim.store.read_dosage(c, sim.rows["train"]) for c in sim.store.chroms], axis=1).astype(np.float64)
    mean, std = np.concatenate([z[f"mean{i}"] for i in range(22)]), np.concatenate([z[f"std{i}"] for i in range(22)])
    assert mean == pytest.approx(x.mean(0), abs=1e-9) and std == pytest.approx(x.std(0), abs=1e-9)
    assert np.concatenate(dm.snp_mean) == pytest.approx(mean, abs=1e-6)
    assert dm.label_names == NAMES and dm.splits["val"]["y"].shape == (600, 8)
    assert np.isnan(dm.splits["test"]["y"]).all()


def test_a_second_build_refuses_to_overwrite(sim):
    with pytest.raises(SystemExit):
        st.build(sim.src, sim.out, sim.block, traits=TRAITS, seed=1, device="cpu")


def test_a_source_with_regions_is_refused(sim, tmp_path):
    d, spec = _source_copy(sim, tmp_path / "src")
    json.dump({**spec, "regions": [["1", 0, 5]]}, open(d / "store.json", "w"))
    with pytest.raises(ValueError, match="regions"):
        st.build(d, tmp_path / "out", sim.block, traits=TRAITS, device="cpu")
    assert not (tmp_path / "out").exists()


def test_an_identifier_absent_from_the_store_is_refused(sim, tmp_path):
    d, _ = _source_copy(sim, tmp_path / "src")
    t = pd.read_csv(d / "samples_val.tsv", sep="\t", dtype=str, keep_default_na=False)
    t.loc[0, "IID"] = "nobody"
    t.to_csv(d / "samples_val.tsv", sep="\t", index=False)
    with pytest.raises(KeyError, match="samples_val.tsv: 1 identifiers"):
        st.build(d, tmp_path / "out", sim.block, traits=TRAITS, device="cpu")


def test_na_cells_are_copied_verbatim(sim, tmp_path):
    d, spec = _source_copy(sim, tmp_path / "src")
    cov = spec["covariates"][0]
    t = pd.read_csv(d / "samples_test.tsv", sep="\t", dtype=str, keep_default_na=False)
    t.loc[0, cov] = "NA"
    t.to_csv(d / "samples_test.tsv", sep="\t", index=False)
    st.build(d, tmp_path / "out", sim.block, traits=TRAITS, device="cpu")
    new = pd.read_csv(tmp_path / "out" / "samples_test.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert new.loc[0, cov] == "NA" and new.loc[1, cov] == t.loc[1, cov]


# ---- the pieces ----------------------------------------------------------------------------------------
def test_missing_genotypes_count_as_the_mean():
    q = torch.tensor([[50, 255], [255, 100], [150, 200], [150, 0]], dtype=torch.uint8)
    mean, std = st.moments(q)
    x0, x1 = np.array([0.5, 1.5, 1.5]), np.array([1.0, 2.0, 0.0])            # the observed dosages only
    assert mean == pytest.approx([x0.mean(), x1.mean()]) and std == pytest.approx([x0.std(), x1.std()])
    g = st.Geno(q, np.array([0, 1]), mean, st.inverse_sd(std))
    x = g.x(np.array([0, 1])).numpy()
    assert x[1, 0] == pytest.approx(mean[0]) and x[0, 1] == pytest.approx(mean[1]) and x[0, 0] == 0.5 and x[3, 1] == 0.0
    z = g.z(np.array([0, 1])).numpy()
    assert z[1, 0] == 0 and z[0, 1] == 0 and z[0, 0] == pytest.approx((0.5 - mean[0]) / std[0], abs=1e-5)
    assert z[2:, 0] == pytest.approx((1.5 - mean[0]) / std[0], abs=1e-5)


def test_auroc_by_hand():
    assert st.auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    assert st.auroc(np.zeros(6), np.array([0, 1, 0, 1, 0, 1])) == 0.5           # a constant score: the null's ceiling
    assert np.isnan(st.auroc(np.arange(3.0), np.ones(3)))                      # one class: undefined


def test_residual_and_unit_by_hand():
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    assert torch.allclose(st.residual(2 * x + 3, x), torch.zeros(4, 1), atol=1e-6)
    y = x ** 2                                                                 # fit of [1,4,9,16] on x: 5x - 5
    assert torch.allclose(st.residual(y, x), y - (5 * x - 5), atol=1e-5)
    two = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [2.0, 1.0]])
    r = st.residual(two[:, :1] * two[:, 1:], two[:, :1], two[:, 1:])
    assert abs(float((r * (two[:, :1] - two[:, :1].mean())).sum())) < 1e-5     # orthogonal to both regressors
    assert abs(float((r * (two[:, 1:] - two[:, 1:].mean())).sum())) < 1e-5 and abs(float(r.sum())) < 1e-5
    u = st.unit(torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]))
    assert u[:, 0].std() == pytest.approx(1.0) and torch.all(u[:, 1] == 0) and torch.isfinite(u).all()   # no 0/0


def test_gblup_equals_a_direct_solve():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((30, 50))
    k = torch.tensor(z @ z.T / 50)
    kx = torch.tensor(rng.standard_normal((7, 50)) @ z.T / 50)
    y = torch.tensor(rng.binomial(1, 0.3, 30).astype(np.float64))
    before = k.clone()
    pred = st.gblup(k, kx, y, 0.25)
    want = kx.numpy() @ np.linalg.solve(k.numpy() + 3.0 * np.eye(30), (y - y.mean()).numpy())   # lambda = 0.75 / 0.25
    assert pred.numpy() == pytest.approx(want, rel=1e-8)
    assert torch.allclose(k, before, rtol=1e-12, atol=1e-12)                   # the penalty was removed again


def test_liability_scaling_by_hand():
    rng = np.random.default_rng(1)
    v = torch.tensor(rng.standard_normal(20000)) * 7 + 3
    y, g, liab, shares = st.liability({"additive": (v, 0.4)}, 20000, 0.2, rng)
    assert g.var() == pytest.approx(0.4, rel=1e-9) and abs(g.mean()) < 1e-9
    assert liab.var() == pytest.approx(1.0, abs=0.03) and y.mean() == pytest.approx(0.2, abs=1e-4)
    assert shares["additive"] == pytest.approx(0.4, abs=0.02)
    y0, g0, liab0, s0 = st.liability({}, 500, 0.1, rng)
    assert not g0.any() and y0.mean() == pytest.approx(0.1, abs=0.01) and s0 == {}


def test_draw_refuses_impossible_specs():
    rng, block = np.random.default_rng(0), np.repeat(np.arange(4), 10)
    with pytest.raises(ValueError, match="genetic correlation"):
        st.draw(dict(name=("a", "b"), kind="shared", causal=10, shared=5, rg=0.7, h2=0.3), 40, block, np.full(40, 0.3), rng)
    with pytest.raises(ValueError, match="between-block pairs"):
        st.draw(dict(name="epi", kind="epi", causal=10, h2=0.2, planted=5, h2_planted=0.1), 40, np.zeros(40, dtype=int), np.full(40, 0.3), rng)
    with pytest.raises(ValueError, match="modifier"):
        st.draw(dict(name="h", kind="hetero", causal=3, h2=0.2, planted=3, h2_planted=0.1, modifier_maf=0.1), 3, np.arange(3), np.full(3, 0.3), rng)
    d = st.draw(dict(name="h", kind="hetero", causal=4, h2=0.2, planted=4, h2_planted=0.1, modifier_maf=0.1), 40, block, np.full(40, 0.05), rng)
    assert (block[d["modifier"]] == block[d["loci"]]).all() and (d["modifier"] != d["loci"]).all()     # the MAF fallback
    assert sorted(d) == ["beta", "causal", "effect", "loci", "modifier"] and len(d["effect"]) == 4


def test_read_blocks_checks_the_variants(sim, tmp_path):
    ids = np.concatenate([sim.store.variants(c)["id"] for c in sim.store.chroms])
    t = pd.DataFrame({"ID": ids, "block_id": np.arange(len(ids)) // 10 + 1, "other": 0})
    t.to_csv(tmp_path / "ok.tsv", sep="\t", index=False)
    assert np.array_equal(st.read_blocks(tmp_path / "ok.tsv", sim.store), np.arange(220) // 10 + 1)
    t.loc[5, "ID"] = "rs_other"
    t.to_csv(tmp_path / "bad.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="store order"):
        st.read_blocks(tmp_path / "bad.tsv", sim.store)
    t.iloc[:-1].to_csv(tmp_path / "short.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="store order"):
        st.read_blocks(tmp_path / "short.tsv", sim.store)
    with pytest.raises(ValueError, match="block ids"):
        st.build(sim.src, tmp_path / "x", np.zeros(3, dtype=int), traits=TRAITS, device="cpu")


# ---- the read-out ----------------------------------------------------------------------------------------
RIDGE = dict(add_lo=0.75, shared_a=0.70, dom=0.70, epi=0.70, null=0.50)
TWIN = dict(add_lo=0.80, shared_a=0.75, dom=0.70, epi=0.65, null=0.50)


def _scores(path, key, aucs):
    """Nine people (4 controls, 5 cases) whose scores have exactly the AUROC ``aucs[label]`` (a multiple of 0.05)."""
    cols = []
    for auc in aucs.values():
        k, wins = round(auc * 20), []
        for _ in range(5):
            wins.append(min(4, k))
            k -= wins[-1]
        cols.append(np.r_[[10, 20, 30, 40], 10 * np.array(wins) + 5])        # a case with w wins beats w of the controls
    y = np.tile(np.array([0] * 4 + [1] * 5, dtype=np.float32)[:, None], (1, len(aucs)))
    np.savez(path, labels=np.array(list(aucs)), y=y, **{key: np.column_stack(cols).astype(np.float32)})


def _dataset(tmp_path, ridge=True, twin=False, stale=False):
    d = tmp_path / "sim"
    d.mkdir()
    kinds = dict(add_lo="add", shared_a="shared", dom="dom", epi="epi", null="null")
    ceiling = dict(add_lo=0.78, shared_a=0.77, dom=0.80, epi=0.79, null=0.5)
    additive = dict(add_lo=0.78, shared_a=0.77, dom=0.74, epi=0.75, null=None)
    traits = {t: {"kind": k, "ceiling_auroc_val": ceiling[t], "additive_ceiling_auroc_val": additive[t], "gblup_auroc_val": 0.6}
              for t, k in kinds.items()}
    if stale:
        for v in traits.values():
            del v["additive_ceiling_auroc_val"]
    json.dump({"built": "now", "seed": 1, "n_snps": 220, "n": {"train": 10, "val": 5}, "traits": traits}, open(d / "sim_log.json", "w"))
    if ridge:
        (d / "blocks").mkdir()
        _scores(d / "blocks" / "prs_val.npz", "score", RIDGE)
    if twin:
        (d / "scores" / "blocks_linear" / "logistic_ridge").mkdir(parents=True)
        _scores(d / "scores" / "blocks_linear" / "logistic_ridge" / "val_best.npz", "logit_population", TWIN)
    return d


def _run(tmp_path, name, aurocs, search=False, mixer="transformer"):
    r = tmp_path / name
    d = r / "trials" / "t3" if search else r
    d.mkdir(parents=True)
    val = {t: {"auroc": a, "ap": 0.3, "bce": 0.5} for t, a in aurocs.items()}
    json.dump({"val": val} if search else {"val": val, "val_mean": {"auroc": 0.7}}, open(d / "results.json", "w"))
    if search:
        json.dump({"tuning": True, "best_trial_dir": "trials/t3"}, open(r / "results.json", "w"))
    if mixer:
        json.dump({"model": {"mixer": mixer, "d_model": 16}}, open(d / "config.json", "w"))
    return str(r)


# thresholds: ridge - 0.01 = 0.74, 0.69; ridge + 0.5 (ceiling - additive ceiling) = 0.70 + 0.03, 0.70 + 0.02
GOOD = dict(add_lo=0.741, shared_a=0.691, dom=0.731, epi=0.721, null=0.55)


def test_scores_by_hand(tmp_path):
    _scores(tmp_path / "s.npz", "score", RIDGE)
    assert e0_summary.scores(tmp_path / "s.npz", "score").to_dict() == pytest.approx(RIDGE)


def test_e0_summary_passes_a_run_that_clears_every_criterion(tmp_path):
    d = _dataset(tmp_path)
    runs = [_run(tmp_path, "mamba_s2", GOOD, mixer="bimamba1"), _run(tmp_path, "search_tf", GOOD, search=True)]
    assert e0_summary.main(runs + ["--dataset", str(d), "--out", str(tmp_path / "res")]) == 0
    text = (tmp_path / "res" / "e0_results.md").read_text()
    assert "Gate: passed" in text and "FAIL" not in text and "0.7410" in text and "0.7500" in text and "0.7400" in text
    assert "| mamba_s2 | bimamba1 | pass | pass | pass | pass | pass | pass |" in text
    assert "| search_tf | transformer | pass | pass | pass | pass | pass | pass |" in text


@pytest.mark.parametrize("change, trait", [(dict(add_lo=0.739), "add_lo"), (dict(shared_a=0.689), "shared_a"),
                                           (dict(dom=0.729), "dom"), (dict(epi=0.719), "epi"), (dict(null=0.61), "null"),
                                           (dict(null=0.39), "null")])
def test_e0_summary_fails_each_criterion(tmp_path, change, trait):
    d = _dataset(tmp_path)
    runs = [_run(tmp_path, "good", GOOD), _run(tmp_path, "bad", {**GOOD, **change}, mixer="bimamba1")]
    assert e0_summary.main(runs + ["--dataset", str(d), "--out", str(tmp_path / "res")]) == 1
    vt = [l for l in (tmp_path / "res" / "e0_results.md").read_text().splitlines() if l.startswith("| bad |")][0]
    cells = dict(zip(["mixer", "add_lo", "shared_a", "dom", "epi", "null", "all"], [c.strip() for c in vt.split("|")[2:-1]]))
    assert cells[trait] == "FAIL" and cells["all"] == "FAIL" and cells["mixer"] == "bimamba1"
    assert all(v == "pass" for t, v in cells.items() if t not in (trait, "all", "mixer"))


def test_the_gate_wants_a_passing_run_per_mixer(tmp_path):
    d = _dataset(tmp_path)
    tf, ablation, mamba = (_run(tmp_path, "tf", GOOD), _run(tmp_path, "tf_ablation", {**GOOD, "dom": 0.70}),
                           _run(tmp_path, "mamba", GOOD, mixer="bimamba1"))
    assert e0_summary.main([tf, ablation, mamba, "--dataset", str(d), "--out", str(tmp_path / "a")]) == 0   # a failing ablation beside a pass
    text = (tmp_path / "a" / "e0_results.md").read_text()
    assert "| tf_ablation | transformer | pass | pass | FAIL | pass | pass | FAIL |" in text and "Gate: passed" in text
    bad = _run(tmp_path, "mamba_bad", {**GOOD, "epi": 0.70}, mixer="bimamba1")
    assert e0_summary.main([tf, bad, "--dataset", str(d), "--out", str(tmp_path / "b")]) == 1              # a mixer without a pass
    assert "Gate: not passed" in (tmp_path / "b" / "e0_results.md").read_text()
    unknown = _run(tmp_path, "old", GOOD, mixer=None)                                                        # no config.json
    assert e0_summary.main([unknown, "--dataset", str(d), "--out", str(tmp_path / "c")]) == 0
    assert "| old | ? | pass |" in (tmp_path / "c" / "e0_results.md").read_text()


def test_e0_summary_without_the_ridge_reference_is_undetermined(tmp_path):
    d = _dataset(tmp_path, ridge=False)
    assert e0_summary.main([_run(tmp_path, "r", GOOD), "--dataset", str(d), "--out", str(tmp_path / "res")]) == 1
    text = (tmp_path / "res" / "e0_results.md").read_text()
    assert "MISSING" in text and "undetermined" in text and "Gate: not passed" in text
    assert "| r | transformer | undetermined | undetermined | undetermined | undetermined | pass | FAIL |" in text   # the null needs no ridge


def test_e0_summary_marks_a_trait_a_run_lacks(tmp_path):
    d = _dataset(tmp_path)
    missing = {t: a for t, a in GOOD.items() if t != "dom"}
    assert e0_summary.main([_run(tmp_path, "r", missing), "--dataset", str(d), "--out", str(tmp_path / "res")]) == 1
    assert "| r | transformer | pass | pass | undetermined | pass | pass | FAIL |" in (tmp_path / "res" / "e0_results.md").read_text()


def test_e0_summary_shows_the_block_linear_twin(tmp_path):
    d = _dataset(tmp_path, twin=True)
    assert e0_summary.main([_run(tmp_path, "r", GOOD), "--dataset", str(d), "--out", str(tmp_path / "res")]) == 0
    text = (tmp_path / "res" / "e0_results.md").read_text()
    assert "twin_logistic_ridge" in text
    assert [l for l in text.splitlines() if l.startswith("| add_lo |")][0].split("|")[7].strip() == "0.8000"


def test_e0_summary_refuses_a_dataset_without_the_additive_ceiling(tmp_path):
    d = _dataset(tmp_path, stale=True)
    with pytest.raises(SystemExit, match="additive ceiling"):
        e0_summary.main([_run(tmp_path, "r", GOOD), "--dataset", str(d), "--out", str(tmp_path / "res")])


def test_verdict_by_hand():
    v = e0_summary.verdict
    assert v("dom", 0.73, 0.70, 0.80, 0.74) is True and v("dom", 0.7299, 0.70, 0.80, 0.74) is False
    assert v("add", 0.74, 0.75, 0.78, 0.78) is True and v("add", 0.7399, 0.75, 0.78, 0.78) is False
    assert v("null", 0.40, 0.5, 0.5, None) is True and v("null", 0.601, 0.5, 0.5, None) is False
    assert v("add", 0.75, float("nan"), 0.80, 0.80) is None and v("null", None, 0.5, 0.5, None) is None
