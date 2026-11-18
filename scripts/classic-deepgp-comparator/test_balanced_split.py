"""The 1:1 split and its reading.  python -m pytest scripts/classic-deepgp-comparator/test_balanced_split.py -q"""
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                     # balanced_eval imports make_balanced_split


def _mod(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_balanced_rows_keeps_every_case_and_draws_as_many_controls():
    m = _mod("make_balanced_split")
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], float)
    rows = m.balanced_rows(y, 1)
    assert list(rows) == sorted(rows) and len(rows) == 6 == len(set(rows))
    assert set(np.flatnonzero(y == 1)) <= set(rows) and (y[rows] == 0).sum() == 3


def test_balanced_rows_is_reproducible_by_seed_or_generator_and_seed_dependent():
    m = _mod("make_balanced_split")
    y = np.r_[np.ones(5), np.zeros(200)]
    assert np.array_equal(m.balanced_rows(y, 1), m.balanced_rows(y, 1))
    assert np.array_equal(m.balanced_rows(y, 1), m.balanced_rows(y, np.random.default_rng(1)))
    assert not np.array_equal(m.balanced_rows(y, 1), m.balanced_rows(y, 2))
    rng = np.random.default_rng(1)                 # a generator moves on between draws
    assert not np.array_equal(m.balanced_rows(y, rng), m.balanced_rows(y, rng))


def test_balanced_rows_skips_unlabelled_rows_and_samples_whichever_class_is_larger():
    m = _mod("make_balanced_split")
    y = np.array([np.nan, 1, 1, 1, 0, 0, np.nan], float)          # more cases than controls
    rows = m.balanced_rows(y, 1)
    assert len(rows) == 4 and (y[rows] == 0).sum() == 2 and (y[rows] == 1).sum() == 2
    with pytest.raises(ValueError):
        m.balanced_rows(np.array([1, 1, np.nan]), 1)


def _dataset(tmp_path, n=300, p=0.2):
    rng = np.random.default_rng(0)
    src = tmp_path / "src"
    src.mkdir()

    def table(k, start):
        df = pd.DataFrame({"IID": [f"P{start + i}" for i in range(k)], "I10": rng.binomial(1, p, k),
                           "E78": rng.binomial(1, 0.1, k).astype(float),
                           "birth_year": rng.integers(1940, 1970, k), "sex": rng.integers(0, 2, k)})
        df.loc[df.index[:2], "E78"] = np.nan                       # an empty cell
        return df

    for split, k, s in (("train", n, 0), ("val", 50, 10000), ("test", 60, 20000)):
        table(k, s).to_csv(src / f"samples_{split}.tsv", sep="\t", index=False)
    (src / "store.json").write_text(json.dumps({"store": "/nowhere", "labels": ["I10", "E78"],
                                                "covariates": ["birth_year", "sex"]}))
    return src


def test_main_builds_a_dataset_that_shares_everything_but_the_training_table(tmp_path):
    m = _mod("make_balanced_split")
    src = _dataset(tmp_path)
    m.main(["--dataset", str(src), "--label", "I10", "--seed", "1"])
    out = tmp_path / "src_cc11_I10"
    assert oct(out.stat().st_mode & 0o777) == "0o700"
    tr = pd.read_csv(out / "samples_train.tsv", sep="\t", dtype={"IID": str})
    full = pd.read_csv(src / "samples_train.tsv", sep="\t", dtype={"IID": str})
    n1 = int((full["I10"] == 1).sum())
    assert len(tr) == 2 * n1 and (tr["I10"] == 1).sum() == n1 and (tr["I10"] == 0).sum() == n1
    assert tr["IID"].is_unique and set(tr["IID"]) <= set(full["IID"])
    assert list(tr.columns) == list(full.columns)
    merged = tr.merge(full, on="IID", suffixes=("", "_src"))
    for c in ("E78", "birth_year", "sex"):                          # rows copied verbatim, empty cells included
        assert merged[c].equals(merged[f"{c}_src"])
    for name in ("samples_val.tsv", "samples_test.tsv", "store.json"):
        assert os.path.islink(out / name) and os.path.realpath(out / name) == os.path.realpath(src / name)
    log = json.load(open(out / "build_log.json"))
    assert log["train"] == {"rows": 2 * n1, "cases": n1, "controls": n1}
    assert log["source_train"] == {"rows": 300, "cases": n1, "controls": 300 - n1, "unlabelled": 0}
    with pytest.raises(FileExistsError):                            # never overwrites
        m.main(["--dataset", str(src), "--label", "I10"])


def test_balanced_aurocs_agree_with_the_whole_split_in_expectation_and_exactly_at_the_extremes():
    be = _mod("balanced_eval")
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.3).astype(float)
    logit = y + rng.normal(size=y.size)
    b = be.balanced_aurocs(logit, y, 50, 1)
    assert len(b) == 50 and abs(b.mean() - roc_auc_score(y, logit)) < 0.005 and b.std() < 0.01
    assert np.array_equal(be.balanced_aurocs(logit, y, 3, 1), be.balanced_aurocs(logit, y, 3, 1))
    assert np.all(be.balanced_aurocs(10 * y, y, 5, 1) == 1.0)
    assert np.all(be.balanced_aurocs(np.zeros_like(y), y, 5, 1) == 0.5)
    assert be.n_eff(3, 7) == pytest.approx(8.4) and be.n_eff(5, 5) == 10.0


def test_balanced_eval_main_reads_one_label_and_drops_its_unlabelled_rows(tmp_path, capsys):
    be = _mod("balanced_eval")
    rng = np.random.default_rng(0)
    y = np.column_stack([(rng.random(400) < 0.25), (rng.random(400) < 0.5)]).astype(np.float32)
    y[:10, 0] = np.nan
    logit = (y + rng.normal(size=y.shape)).astype(np.float32)
    np.savez(tmp_path / "val_best.npz", iid=np.array([f"P{i}" for i in range(400)]), labels=np.array(["I10", "E78"]),
             logit_population=logit, y=y)
    be.main(["--scores", str(tmp_path / "val_best.npz"), "--label", "I10", "--draws", "5"])
    out = capsys.readouterr().out
    n1 = int(np.nansum(y[:, 0]))
    assert out.startswith(f"I10: {n1} cases, {390 - n1} controls, N_eff {be.n_eff(n1, 390 - n1):.0f}")
    assert "AUROC on the whole split" in out and "5 random 1:1 subsets (seed 1)" in out
