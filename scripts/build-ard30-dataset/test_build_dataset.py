"""Unit tests on made-up records for build_dataset.py (no UK Biobank data).

    /home/robert/envs/widedeepgp/bin/python -m pytest scripts/build-ard30-dataset/test_build_dataset.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_dataset as bd  # noqa: E402


def _records():
    rec = lambda rows: pd.DataFrame(rows, columns=["eid", "cat", "sub"])
    return {
        "hes": rec([("1", "I10", "I10"), ("1", "E11", "E119"), ("2", "I84", "I849"), ("9", "I10", "I10")]),
        "death": rec([("3", "E11", "E119"), ("2", "I10", "I10")]),
        "cancer_reg": rec([("4", "C44", "C449"), ("5", "K64", "K649"), ("1", "Z99", "Z999")]),
    }


def test_label_matrix_unites_sources_and_codes():
    phenos = [("I10", ("I10",), "t"), ("E11", ("E11",), "t"), ("I84+K64", ("I84", "K64"), "t"), ("C44", ("C44",), "t")]
    eids = np.array(["1", "2", "3", "4", "5", "6"])            # "9" is not a participant; "6" has no record
    y = bd.label_matrix(_records(), eids, phenos)
    assert y.dtype == np.uint8 and y.shape == (6, 4)
    assert y.tolist() == [[1, 1, 0, 0],                        # inpatient I10 and E11 (Z99 is not a phenotype)
                          [1, 0, 1, 0],                        # death-registry I10, inpatient I84
                          [0, 1, 0, 0],                        # death-registry E11 only
                          [0, 0, 0, 1],                        # cancer registry
                          [0, 0, 1, 0],                        # K64 counts toward the united phenotype
                          [0, 0, 0, 0]]


def test_selection_table(tmp_path):
    p = tmp_path / "sel.csv"
    pd.DataFrame({"phenotype": ["I10", "I84+K64"], "icd10_codes": ["I10", "I84;K64"], "icd10_title": ["a", "b"],
                  "n_cases__geno_wb_unrel": [3, 2]}).to_csv(p, index=False)
    phenos, table = bd.read_selected(p)
    assert phenos == [("I10", ("I10",), "a"), ("I84+K64", ("I84", "K64"), "b")] and table.loc["I84+K64", "icd10_title"] == "b"
    pd.DataFrame({"phenotype": ["I10", "X"], "icd10_codes": ["I10", "I10;K64"], "icd10_title": ["a", "b"],
                  "n_cases__geno_wb_unrel": [3, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="more than one phenotype"):
        bd.read_selected(p)


def test_birth_year_is_within_half_a_year():
    rng = np.random.default_rng(0)
    born = pd.to_datetime("1940-01-01") + pd.to_timedelta(rng.integers(0, 30 * 365, size=2000), unit="D")
    seen = pd.to_datetime("2006-03-13") + pd.to_timedelta(rng.integers(0, 4 * 365, size=2000), unit="D")
    age = ((seen.year - born.year) - ((seen.month < born.month) | ((seen.month == born.month) & (seen.day < born.day)))
           ).astype(int)                                        # completed years, as UK Biobank records it
    est = bd.birth_year(pd.Series(seen.strftime("%Y-%m-%d")), pd.Series(age.astype(str))).to_numpy()
    true = bd.decimal_year(pd.Series(born)).to_numpy()
    assert np.abs(est - true).max() <= 0.51 and abs(np.mean(est - true)) < 0.02
    assert np.isnan(bd.birth_year(pd.Series(["", "2008-01-01"]), pd.Series(["60", ""]))).all()


def test_split_is_fixed_disjoint_and_ignores_input_order():
    ids = [str(1000000 + i) for i in range(5000)]
    a = bd.split_ids(ids, n_tune=300, n_test=700, seed=5)
    b = bd.split_ids(ids[::-1], n_tune=300, n_test=700, seed=5)
    assert a == b and len(a["test"]) == 700 and len(a["val"]) == 300 and len(a["train"]) == 4000
    assert not (a["train"] & a["val"]) and not (a["train"] & a["test"]) and not (a["val"] & a["test"])
    assert a["test"] != bd.split_ids(ids, n_tune=300, n_test=700, seed=6)["test"]


def test_annotations_follow_store_order(tmp_path):
    ann = tmp_path / "annot.txt"
    pd.DataFrame({"SNP": ["rs3", "rs1", "rs9"], "Intercept": [1, 1, 1], "coding": [1, 0, 1], "gerp": [0.5, -1.0, 2.0]}
                 ).to_csv(ann, sep="\t", index=False)
    snps = pd.DataFrame({"CHROM": ["1", "1", "2"], "ID": ["rs1", "rs2", "rs3"]})
    names, covered = bd.build_annotations(ann, snps, tmp_path / "a.npz")
    z = np.load(tmp_path / "a.npz")
    assert names == ["coding", "gerp", "no_annotation"] and covered == 2
    assert z["chr1"].tolist() == [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]       # rs2 has no annotation: zeros and the flag
    assert z["chr2"].tolist() == [[1.0, 0.5, 0.0]]


# ---- describe_dataset.py ---------------------------------------------------------------------
def _made_up_labels(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    frailty = rng.normal(size=n)                               # a shared factor makes the labels co-occur
    cut = np.array([0.6, 1.2, 1.8, 2.3])
    return (frailty[:, None] + rng.normal(size=(n, 4)) > cut).astype(np.float64)


def test_rotation_case_share_matches_the_label_matrix():
    import describe_dataset as dd
    y = _made_up_labels()
    share = dd.rotation_case_share(y.T @ y, len(y))
    for i in range(4):
        cases, others = y[y[:, i] == 1], y[y[:, i] == 0]
        direct = 0.5 * cases.mean(axis=0) + 0.5 * others.mean(axis=0)
        assert np.allclose(share[i], direct)
    assert np.allclose(np.diag(share), 0.5)                    # the target itself is half cases


def test_describe_reports_sampling_rates_that_average_one():
    import describe_dataset as dd
    y = _made_up_labels()
    per, s = dd.describe(y, ["a", "b", "c", "d"])
    assert abs(s["omega"]["mean"] - 1.0) < 1e-9 and s["omega"]["min"] == pytest.approx(s["omega"]["none_of_the_30"])
    assert 0.0 < s["effective_sample_share_with_correction"] < 1.0
    assert s["diseases_per_person"]["mean"] == pytest.approx(y.sum(axis=1).mean())
    # labels that co-occur are over-represented over a rotation, beyond what independence gives
    independent = (0.5 + 3 * per["prevalence_train"]) / 4
    assert (per["case_share_over_rotation"] > independent).all()
    assert (per["case_share_over_rotation"] > per["prevalence_train"]).all()
    assert per["class_weight_sqrt"].tolist() == pytest.approx(np.sqrt(per["class_weight_parity"]).tolist())


# ---- make_proxy_dataset.py -------------------------------------------------------------------
def test_proxy_stretches_tile_a_chromosome():
    import make_proxy_dataset as mp
    for m in (17844, 105260, 8, 1001):
        parts = [mp.stretch(m, k) for k in range(8)]
        assert parts[0][0] == 0 and parts[-1][1] == m
        assert all(a[1] == b[0] for a, b in zip(parts, parts[1:]))          # contiguous, no SNP twice
        assert max(v1 - v0 for v0, v1 in parts) - min(v1 - v0 for v0, v1 in parts) <= 1
    with pytest.raises(ValueError):
        mp.stretch(100, 8)


def test_proxy_draws_are_disjoint_and_repeatable():
    import make_proxy_dataset as mp
    ids = [str(i) for i in range(1000)]
    a, b = mp.draw(ids, [300, 200], seed=1)
    assert len(a) == 300 and len(b) == 200 and not set(a) & set(b)
    a2, b2 = mp.draw(list(reversed(ids)), [300, 200], seed=1)                # the input order does not matter
    assert (a == a2).all() and (b == b2).all()
    assert not (mp.draw(ids, [300], seed=2)[0] == a).all()
    with pytest.raises(ValueError):
        mp.draw(ids, [900, 200], seed=1)
