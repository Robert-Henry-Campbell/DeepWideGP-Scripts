"""evaluate.py on small synthetic files: the 6.1 protocol against an independent fit, alignment by IDs, refusals,
coverage rows and small counts.

Run:  /home/robert/envs/widedeepgp/bin/python -m pytest tests/test_evaluate.py -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import evaluate as E  # noqa: E402

COVS = ["birth_year", "sex"] + [f"pc{i}" for i in range(1, 11)]
RUNG = "hm3plus_n13199"


def people(rng, n, prefix, labels, signal):
    t = pd.DataFrame({"IID": [f"{prefix}{i}" for i in range(n)], "birth_year": rng.normal(1952, 8, n),
                      "sex": rng.integers(0, 2, n).astype(float), **{f"pc{i}": rng.normal(0, 1, n) for i in range(1, 11)}})
    g = rng.normal(0, 1, (n, len(labels)))
    for j, lab in enumerate(labels):
        eta = -1.0 + signal * g[:, j] + 0.03 * (t.birth_year - 1952) + 0.4 * t.sex
        t[lab] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return t, g


def save(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    labels = ["D1", "D2"]
    ds = tmp_path / "ard30"
    val, g_val = people(rng, 1500, "V", labels, 1.0)
    asian, g_as = people(rng, 900, "A", labels, 0.6)
    os.makedirs(ds / RUNG)
    os.makedirs(ds / "hm3plus_asian_black")
    val.to_csv(ds / RUNG / "samples_val.tsv", sep="\t", index=False)
    asian.to_csv(ds / "hm3plus_asian_black" / "samples_asian.tsv", sep="\t", index=False)
    noise = lambda n: rng.normal(0, 0.5, n)            # noqa: E731
    s_val = np.column_stack([g_val[:, j] + noise(len(val)) for j in range(2)])
    s_as = np.column_stack([g_as[:, j] + noise(len(asian)) for j in range(2)])
    order = rng.permutation(len(asian))                # a group file in its own order, with IDs
    k = ds / RUNG / "scores" / "kelemen_mlp" / f"ladder_prs_{RUNG}"
    for j, lab in enumerate(labels):                   # one disease per run, the tuning file without IDs
        save(str(k / f"{lab}_large" / "val.npz"), logits=s_val[:, j:j + 1].astype(np.float32),
             y=val[[lab]].to_numpy(np.float32), labels=np.array([lab]))
        save(str(k / f"{lab}_large" / "asian.npz"), iid=asian.IID.to_numpy(str)[order],
             logits=s_as[order, j:j + 1].astype(np.float32), y=asian[[lab]].to_numpy(np.float32)[order], labels=np.array([lab]))
    w = ds / RUNG / "scores" / "widedeepgp_ladder" / f"{RUNG}_s1"     # all diseases in one run, with IDs
    save(str(w / "val_best.npz"), iid=val.IID.to_numpy(str), labels=np.array(labels), y=val[labels].to_numpy(np.float32),
         logit_population=s_val.astype(np.float32))
    p = ds / RUNG / "scores" / "pgi" / "fp" / "c_t"                   # a PGI file: its own index, not its combined logit
    save(str(p / "val.npz"), iid=val.IID.to_numpy(str), labels=np.array(labels), y=val[labels].to_numpy(np.float32),
         pgi=s_val, logit_population=np.zeros_like(s_val, dtype=np.float32))
    save(str(p / "asian.npz"), iid=asian.IID.to_numpy(str), labels=np.array(labels), y=asian[labels].to_numpy(np.float32),
         pgi=s_as, logit_population=np.zeros_like(s_as, dtype=np.float32))
    monkeypatch.setattr(E, "DATASETS", str(ds))
    return dict(ds=ds, val=val, asian=asian, s_val=s_val, s_as=s_as, labels=labels)


def independent(w, j, with_score=True):
    """sklearn's unpenalised logistic regression on the tuning set, z-scored as the arm's basis, applied to the group."""
    v, a = w["val"], w["asian"]
    mu, sd = v[COVS].mean(), v[COVS].std(ddof=0)
    xv, xa = ((v[COVS] - mu) / sd).to_numpy(), ((a[COVS] - mu) / sd).to_numpy()
    if with_score:
        sm, ss = w["s_val"][:, j].mean(), w["s_val"][:, j].std()
        xv = np.column_stack([(w["s_val"][:, j] - sm) / ss, xv])
        xa = np.column_stack([(w["s_as"][:, j] - sm) / ss, xa])
    fit = LogisticRegression(penalty=None, max_iter=10000, tol=1e-12).fit(xv, v[w["labels"][j]])
    return roc_auc_score(a[w["labels"][j]], fit.decision_function(xa))


def test_the_protocol_matches_an_independent_fit_on_the_group(world):
    df = E.evaluate([RUNG], ["kelemen_mlp", "pgi"], ["asian"])
    ok = df[df.status == "ok"].set_index(["model", "disease"])
    for j, lab in enumerate(world["labels"]):
        for m in ("kelemen_mlp", "pgi"):
            r = ok.loc[(m, lab)]
            assert r.auroc == pytest.approx(independent(world, j), abs=1e-6)
            assert r.auroc_covariates_only == pytest.approx(independent(world, j, with_score=False), abs=1e-6)
            assert r.auroc_score == pytest.approx(roc_auc_score(world["asian"][lab], world["s_as"][:, j]), abs=1e-9)
            assert r.gain == pytest.approx(r.auroc - r.auroc_covariates_only, abs=1e-12)
            assert r.auroc_lo < r.auroc < r.auroc_hi and 0 <= r.gain_p <= 1 and r.n == 900
            assert r.n_cases == str(int(world["asian"][lab].sum()))
            assert r.auroc_tuning_score == pytest.approx(roc_auc_score(world["val"][lab], world["s_val"][:, j]), abs=1e-9)
    assert set(ok.reset_index().variant) == {"large", "c_t"}


def test_runs_without_a_group_file_are_listed_as_not_scored(world):
    df = E.evaluate([RUNG], ["widedeepgp"], ["asian"])
    assert len(df) == 2 and set(df.status) == {"not scored"} and set(df.variant) == {"s1"}


def test_a_file_whose_labels_are_not_the_tables_is_refused(world):
    f = world["ds"] / RUNG / "scores" / "kelemen_mlp" / f"ladder_prs_{RUNG}" / "D1_large" / "asian.npz"
    z = dict(np.load(f))
    z["y"] = z["y"][::-1].copy()                    # labels in another order than its IDs
    np.savez(f, **z)
    with pytest.raises(ValueError, match="not the table's"):
        E.evaluate([RUNG], ["kelemen_mlp"], ["asian"])


def test_a_tuning_file_without_ids_must_have_the_tables_length(world):
    f = world["ds"] / RUNG / "scores" / "kelemen_mlp" / f"ladder_prs_{RUNG}" / "D2_large" / "val.npz"
    z = dict(np.load(f))
    np.savez(f, **{k: (v[:-1] if k in ("logits", "y") else v) for k, v in z.items()})
    with pytest.raises(ValueError, match="without IDs"):
        E.evaluate([RUNG], ["kelemen_mlp"], ["asian"])


def test_small_counts_are_written_below_ten():
    assert [E.small(n) for n in (0, 1, 9, 10, 18)] == ["0", "<10", "<10", "10", "18"]


def test_the_variant_drops_the_disease_or_the_rung():
    assert E.variant("kelemen_mlp", "C44_large_linear", RUNG, ["C44"]) == "large_linear"
    assert E.variant("delphi", "M17_code", RUNG, ["M17"]) == "code"
    assert E.variant("scprs", "K57", RUNG, ["K57"]) == "scprs"
    assert E.variant("widedeepgp", f"{RUNG}_s2", RUNG, ["a", "b"]) == "s2"
    assert E.variant("prsformer", f"{RUNG}_linear", RUNG, ["a", "b"]) == "linear"
    assert E.variant("pgi", "lassosum2", RUNG, ["a", "b"]) == "lassosum2"
    assert E.variant("block_ridge_prs", "blocks", RUNG, ["a", "b"]) == "ridge_prs"


def test_the_main_writes_aggregates_only(world, tmp_path):
    out = str(tmp_path / "res" / "heldout_auroc.csv")
    E.main(["--rungs", RUNG, "--models", "kelemen_mlp", "--cohorts", "asian", "--out", out])
    df = pd.read_csv(out)
    assert {"IID", "iid"}.isdisjoint(df.columns) and len(df) == 2 and (df.status == "ok").all()
