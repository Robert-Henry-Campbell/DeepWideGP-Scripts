"""Unit tests of the checks added to the pilot after its audit, on made-up data (no GPU, no UK
Biobank data): the gain over the covariates, the guard on the third table, the owner-only
genotype store, the reading of figure 5 and the position-code fit.

    /home/robert/envs/widedeepgp/bin/python -m pytest scripts/widedeepgp-pilot/test_pilot_checks.py -q
"""
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_pilot as ap  # noqa: E402
import position_code_capacity as pc  # noqa: E402
import simulated_control as sc  # noqa: E402


def test_every_arm_is_also_read_as_a_gain_over_the_covariates():
    rt = pd.DataFrame(dict(arm=["default", "default", "bf16", "bf16"], seed=[1, 2, 1, 2], best_auroc_mean=[0.70, 0.72, 0.74, 0.75]))
    at = pd.DataFrame(dict(arm=["default", "bf16"], best_auroc_mean=[0.71, 0.745]))
    dt = pd.DataFrame(dict(label=["a", "b", "c"], covariates_only=[0.70, 0.75, 0.80], default=[0.69, 0.76, 0.79],
                           bf16=[0.71, 0.76, 0.81]))
    rt, at = ap.against_covariates(rt, at, dt, cov_mean=0.73)
    assert np.allclose(rt.gain_over_covariates, [-0.03, -0.01, 0.01, 0.02])
    at = at.set_index("arm")
    assert at.loc["default", "gain_over_covariates"] == pytest.approx(-0.02) and at.loc["bf16", "gain_over_covariates"] == pytest.approx(0.015)
    assert at.loc["default", "diseases_below_covariates"] == 2 and at.loc["bf16", "diseases_below_covariates"] == 0


def test_the_scripts_refuse_a_dataset_whose_third_table_may_be_the_real_test_set(tmp_path):
    proxy, sim, full, odd = (tmp_path / n for n in ("proxy", "sim", "full", "odd"))
    for d in (proxy, sim, full, odd):
        d.mkdir()
    json.dump({"test_table_is": "a draw from the training split, disjoint from the proxy's training participants"},
              open(proxy / "proxy_log.json", "w"))
    json.dump({"source": str(proxy)}, open(sim / "sim_log.json", "w"))                 # made from the proxy
    json.dump({"test_table_is": "the real test set"}, open(odd / "proxy_log.json", "w"))
    sc.require_training_draw(proxy)
    sc.require_training_draw(sim)
    assert sc.third_table(full) is None
    for d in (full, odd):                                                              # nothing recorded, or not a draw
        with pytest.raises(SystemExit):
            sc.require_training_draw(d)


def test_a_new_genotype_store_is_made_owner_only(tmp_path):
    store = tmp_path / "store"
    (store / "sub").mkdir(parents=True)
    for f in (store / "chr1.base.bin", store / "sub" / "x.json"):
        f.write_bytes(b"0")
        os.chmod(f, 0o664)
    os.chmod(store, 0o775)
    sc.make_private(store)
    modes = {p.name: stat.S_IMODE(p.stat().st_mode) for p in [store, *store.rglob("*")]}
    assert modes == {"store": 0o700, "sub": 0o700, "chr1.base.bin": 0o600, "x.json": 0o600}


def _metrics(path, means, per_trait, step=0.5):
    """A run's metrics.csv: the mean AUROC and two traits' AUROCs (the second 0.01 lower) at every
    validation pass."""
    rows = []
    for i, (m, t) in enumerate(zip(means, per_trait)):
        e = round(step * (i + 1), 4)
        rows += [dict(validation=i + 1, step=i + 1, samples_seen=i + 1, epoch_equiv=e, label=label, metric="auroc", value=v)
                 for label, v in (("MEAN", m), ("t1", t), ("t2", t - 0.01))]
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path / "metrics.csv", index=False)


def test_figure_5_reads_every_run_at_the_end_of_training(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "RES", tmp_path / "results")
    (tmp_path / "results").mkdir()
    pd.DataFrame([dict(trait=t, causal_snps=20, enrichment=1.0, heritability=0.3, ceiling_auroc_tune=0.78, prs_auroc_tune=0.66,
                       clumped_auroc_tune=0.75, gblup_auroc_tune=0.67) for t in ("t1", "t2")]).to_csv(
        tmp_path / "results" / "simulated_control_references.csv", index=False)
    # twelve passes of half an epoch-equivalent; the long run goes on to 30 and peaks at 13.5 (pass 27)
    _metrics(tmp_path / "runs" / "pilot_sim" / "selected_s1", np.linspace(0.55, 0.60, 12), np.linspace(0.60, 0.65, 12))
    means = np.r_[np.linspace(0.55, 0.62, 27), np.linspace(0.61, 0.59, 33)]
    trait = np.r_[np.linspace(0.60, 0.68, 27), np.linspace(0.67, 0.64, 33)]
    _metrics(tmp_path / "runs" / "pilot_sim" / "selected_long_s1", means, trait)
    sc.summarise(type("A", (), dict(runs=str(tmp_path / "runs"), experiment="pilot_sim"))())
    out = pd.read_csv(tmp_path / "results" / "simulated_control_runs.csv").query("trait == 't1'").set_index("arm")
    assert out.loc["selected_long", "auroc_last"] == pytest.approx(0.64) and out.loc["selected", "auroc_last"] == pytest.approx(0.65)
    assert out.loc["selected_long", "best_pass_epoch_equivalents"] == pytest.approx(13.5)
    assert out.loc["selected_long", "auroc_at_best_pass"] == pytest.approx(0.68)
    text = " ".join((tmp_path / "results" / "fig5_control_legend.md").read_text().split())
    assert "at the end of training" in text and "training was 6 epoch-equivalents" in text
    assert 'and 30 for the row "trained longer"' in text and "highest at 13.5 epoch-equivalents" in text
    assert all(len(line) <= 96 for line in (tmp_path / "results" / "fig5_control_legend.md").read_text().splitlines())


def test_the_position_code_is_the_models_and_a_fit_on_it_reproduces_what_it_can(monkeypatch):
    embed = Path(os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")) / "layers" / "Embed.py"
    if embed.exists():                                                  # the code of layers/Embed.py, in float64
        monkeypatch.syspath_prepend(str(embed.parent.parent))
        from layers.Embed import PositionalEmbedding
        model = PositionalEmbedding(32, position=np.arange(1000)).pe[0].numpy()
        assert np.allclose(pc.position_code(1000), model, atol=1e-4)                # float32 rounding of the arguments
    rng = np.random.default_rng(0)
    ess, tss, rank = pc.fit(rng.standard_normal((10, 4)))               # ten SNPs: every profile is reproduced
    assert rank == 10 and np.allclose(ess / tss, 1.0)
    code = pc.position_code(500)
    ess, tss, _ = pc.fit(np.column_stack([3 * code[:, 4] - code[:, 9], rng.standard_normal(500)]))
    assert ess[0] / tss[0] == pytest.approx(1.0) and ess[1] / tss[1] < 0.2   # a profile in the code's span, and noise
    rows = pc.pooled([np.column_stack([np.zeros(50), rng.standard_normal(50)]), np.column_stack([rng.standard_normal(60), np.zeros(60)])],
                     ["a", "b"])
    assert [r["chromosomes"] for r in rows] == [2, 2] and all(0 < r["share_of_variance"] < 1 for r in rows)
