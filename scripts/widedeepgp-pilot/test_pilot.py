"""Unit tests of the pilot's bookkeeping on made-up numbers (no GPU, no UK Biobank data).

    /home/robert/envs/widedeepgp/bin/python -m pytest scripts/widedeepgp-pilot/test_pilot.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_pilot as ap  # noqa: E402
import run_pilot  # noqa: E402


def _runs(values):
    """{arm: [best mean AUROC per seed]} -> the per-run table."""
    return pd.DataFrame([dict(arm=arm, seed=i + 1, best_auroc_mean=v, best_bce_mean=0.3, train_minutes_per_epoch_equiv=7.0,
                              gpu_peak_train_gib=4.0, n_parameters=1000)
                         for arm, vs in values.items() for i, v in enumerate(vs)])


def test_an_arm_differs_only_beyond_the_larger_seed_range():
    at = ap.arm_table(_runs({"default": [0.70, 0.72], "annot": [0.75, 0.76], "gamma1": [0.70, 0.74], "bf16": [0.705, 0.715]}))
    at = at.set_index("arm")
    assert at.loc["default", "seed_range"] == pytest.approx(0.02) and at.loc["default", "clears_bar"] is None
    assert at.loc["annot", "difference_from_default"] == pytest.approx(0.045) and bool(at.loc["annot", "clears_bar"])
    # the arm's own two seeds are 0.04 apart: a difference of 0.01 is within seed-to-seed variation
    assert at.loc["gamma1", "bar_seed_range"] == pytest.approx(0.04) and not bool(at.loc["gamma1", "clears_bar"])
    assert at.loc["bf16", "difference_from_default"] == pytest.approx(0.0) and not bool(at.loc["bf16", "clears_bar"])
    assert list(at.index) == ["default", "gamma1", "bf16", "annot"]            # the order of ARMS, not of the input


def _curve(arm, seed, minutes, values):
    best, rows = -1.0, []
    for i, (t, v) in enumerate(zip(minutes, values)):
        best = max(best, v)
        rows.append(dict(arm=arm, seed=seed, validation=i + 1, gpu_minutes=t, auroc_mean=v, auroc_mean_best_so_far=best))
    return rows


def test_arms_of_different_speed_are_compared_at_the_same_gpu_time():
    cur = pd.DataFrame(_curve("default", 1, [10, 20, 30], [0.60, 0.64, 0.66]) + _curve("default", 2, [10, 20, 30], [0.60, 0.65, 0.67])
                       + _curve("mamba3", 1, [20, 40, 60], [0.63, 0.67, 0.70]) + _curve("mamba3", 2, [20, 40, 60], [0.62, 0.68, 0.71]))
    et = ap.equal_time(cur, None).set_index("arm")
    # both have reached 30 minutes; by then Mamba-3 has made one validation pass, the default three
    assert et.loc["mamba3", "gpu_minutes"] == 30
    assert et.loc["mamba3", "arm_auroc"] == pytest.approx(0.625) and et.loc["mamba3", "default_auroc"] == pytest.approx(0.665)
    assert et.loc["mamba3", "difference"] == pytest.approx(-0.04) and bool(et.loc["mamba3", "clears_bar"])


def test_capacity_table_reads_the_limit_and_the_memory_per_participant():
    row = lambda cfg, b, ok, peak=None, rate=None, error="": dict(config=cfg, mixer="x", precision="32", grad_ckpt=False, annot=False,
                                                        mode="train", n_parameters=10, batch=b, ok=ok, error=error, peak_gib=peak,
                                                        step_s=None if rate is None else b / rate, samples_per_s=rate,
                                                        gpu_total_gib=48.0)
    t = pd.DataFrame([row("a", 8, True, 10.0, 8.0), row("a", 16, True, 18.0, 10.0), row("a", 32, False), row("a", 24, False),
                      row("a", 20, True, 22.0, 9.0), row("b", 8, False),
                      row("c", 8, True, 5.0, 8.0), row("c", 16, False, error="kernel: illegal memory access")])
    c = ap.capacity_table(t, n_train=36_000).set_index("config")
    assert list(c.index) == ["a", "c"]                                         # nothing fitted for "b"
    assert c.loc["a", "limited_by"] == "memory" and c.loc["c", "limited_by"] == "kernel"
    assert c.loc["a", "largest_batch"] == 20 and c.loc["a", "smallest_batch_that_failed"] == 24
    assert c.loc["a", "gib_per_participant"] == pytest.approx(1.0) and c.loc["a", "gib_fixed"] == pytest.approx(2.0)
    assert c.loc["a", "fastest_batch"] == 16 and c.loc["a", "gpu_hours_per_epoch_equiv"] == pytest.approx(1.0)


def test_every_arm_changes_one_thing_and_shares_the_rest():
    a = SimpleNamespace(data="D", runs="R", experiment="pilot", epoch_equiv=6.0, rd_mode="hm3plus_proxy8")
    ref = run_pilot.command("default", 2, a)
    assert ref[0] == run_pilot.PY["fork"] and ref[ref.index("--exp_name") + 1] == "default_s2"
    assert ref[ref.index("--seed") + 1] == "2" and run_pilot.ARMS["default"][1] == []
    for arm, (env, extra) in run_pilot.ARMS.items():
        cmd = run_pilot.command(arm, 2, a)
        assert cmd[0] == run_pilot.PY[env]
        # the same command as the default but for the run's name, followed by the arm's own flags
        assert cmd[1:len(ref)] == [x.replace("default_s2", f"{arm}_s2") for x in ref[1:]]
        assert cmd[len(ref):] == extra
    assert set(run_pilot.ARMS) == set(ap.ARMS)


def test_permuting_an_informative_input_lowers_the_auroc_and_an_uninformative_one_does_not():
    import numpy as np

    import genetic_signal as gs
    rng = np.random.default_rng(0)
    n = 4000
    cov, gen = rng.normal(size=(n, 1)), rng.normal(size=(n, 1))
    y = (rng.random((n, 2)) < 1 / (1 + np.exp(-(1.2 * cov + np.hstack([1.0 * gen, 0.0 * gen]) - 2.0)))).astype(float)
    # a "model" that uses the covariate and the genotype; in the second column the genotype carries nothing
    score = lambda g: np.hstack([1.2 * cov + 1.0 * g, 1.2 * cov + 1.0 * g])
    base = score(gen)
    perms = [score(gen[rng.permutation(n)]) for _ in range(3)]
    au, pu, lo, hi, se = gs.drop_with_interval(base, perms, y, n_boot=100)
    from sklearn.metrics import roc_auc_score
    assert au[0] == pytest.approx(roc_auc_score(y[:, 0], base[:, 0]))          # the rank formula
    assert au[0] - pu[0] > 0.03                                                # genotype informative: a clear drop
    assert abs(au[1] - pu[1]) < 0.02                                           # genotype is noise: no drop
    assert lo < (au - pu).mean() < hi and se > 0


def test_simulated_traits_have_the_stated_architecture():
    import numpy as np

    import simulated_control as sc
    rng = np.random.default_rng(0)
    n_snps, n = 5000, 20000
    enriched = np.zeros(n_snps, dtype=bool)
    enriched[:250] = True                                                       # 5% of SNPs
    traits = [("a", 20, 0.3, 1.0), ("b", 500, 0.5, 1.0), ("c", 200, 0.5, 20.0)]
    beta = sc.draw_effects(n_snps, traits, enriched, rng)
    assert [(beta[:, j] != 0).sum() for j in range(3)] == [20, 500, 200]
    # 5% of SNPs weighted 20 times: about half of the causal SNPs of the enriched trait fall among them
    assert 70 <= (beta[enriched, 2] != 0).sum() <= 130 and (beta[enriched, 1] != 0).sum() <= 50
    x = rng.standard_normal((n, n_snps)).astype(np.float32)
    y, g = sc.liability_to_labels((x @ beta).astype(np.float64), np.array([0.3, 0.5, 0.5]), 0.2, rng)
    assert np.allclose(g.var(axis=0), [0.3, 0.5, 0.5], rtol=1e-6) and np.allclose(y.mean(axis=0), 0.2, atol=1e-3)
    # the true genetic value predicts as well as the heritability allows at a prevalence of 20%:
    # an AUROC of about 0.77 at a heritability of 0.3 and 0.84 at 0.5
    au = [sc.auroc(g[:, j], y[:, j]) for j in range(3)]
    assert 0.75 < au[0] < 0.80 and 0.82 < au[1] < 0.86 and 0.82 < au[2] < 0.86
