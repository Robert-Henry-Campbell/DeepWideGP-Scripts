"""The ladder's SNP lists: store indices and risk-allele signs.  python -m pytest scripts/classic-deepgp-comparator/test_build_lists.py -q"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _bl():
    spec = importlib.util.spec_from_file_location("build_lists", HERE / "build_lists.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def variants(ids, ref, alt):
    return {"id": np.array(ids, dtype=object), "ref": np.array(ref, dtype=object), "alt": np.array(alt, dtype=object)}


VAR = {"1": variants(["a", "b", "c", "d", "e"], list("AAAAA"), list("GGGGG")),
       "2": variants(["f", "g", "h", "i"], list("CCCC"), list("TTTT"))}


def sumstats(rows):
    return pd.DataFrame(rows, columns=["SNP", "A1", "A2", "BETA"])


def test_indices_and_signs_follow_the_risk_allele():
    bl = _bl()
    ss = sumstats([["a", "G", "A", 0.2], ["b", "G", "A", -0.1], ["d", "A", "G", 0.3], ["e", "A", "G", -0.05],
                   ["c", "G", "A", 0.0], ["f", "T", "C", 0.1], ["h", "C", "T", 0.4], ["i", "T", "C", -0.2],
                   ["g", "T", "C", 0.9]])
    out, n_neg = bl.risk_list({"e", "a", "b", "d", "f", "h", "i"}, ss, VAR)
    assert list(out["idx_1"]) == [0, 1, 3, 4] and list(out["idx_2"]) == [0, 2, 3]   # store order, unlisted SNPs left out
    # +1: ALT raises risk (A1 = ALT, beta > 0; or A1 = REF, beta < 0); -1: REF raises risk
    assert list(out["sign_1"]) == [1, -1, -1, 1] and list(out["sign_2"]) == [1, -1, -1]
    assert n_neg == 4 and out["sign_1"].dtype == np.float32 and out["idx_1"].dtype == np.int64


def test_lists_need_three_snps_per_chromosome_and_matching_alleles(monkeypatch):
    bl = _bl()
    ok = sumstats([[s, "G", "A", 0.1] for s in "abcde"] + [[s, "T", "C", 0.1] for s in "fghi"])
    with pytest.raises(ValueError, match="fewer than 3"):
        bl.risk_list({"a", "b", "c", "f", "g"}, ok, VAR)
    bad = ok.copy()
    bad.loc[bad.SNP == "b", "A1"] = "T"                                       # neither of the store's alleles
    with pytest.raises(ValueError, match="alleles do not match"):
        bl.risk_list({"a", "b", "c", "f", "g", "h"}, bad, VAR)
    with pytest.raises(ValueError, match="not in the store"):
        bl.risk_list({"a", "b", "c", "f", "g", "h", "zz"}, ok, VAR)
