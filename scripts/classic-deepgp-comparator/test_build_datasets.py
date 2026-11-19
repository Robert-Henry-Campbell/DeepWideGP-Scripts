"""The classic comparator's SNP selection helpers.  python -m pytest scripts/classic-deepgp-comparator/test_build_datasets.py -q"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

HERE = pathlib.Path(__file__).resolve().parent


def _bd():
    spec = importlib.util.spec_from_file_location("build_datasets", HERE / "build_datasets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def table(**p):
    return pd.DataFrame({"SNP": list(p), "P": list(p.values())})


def test_min_p_takes_each_snps_smallest_finite_p_over_the_diseases():
    bd = _bd()
    a = table(rs1=0.5, rs2=0.01, rs3=np.nan, rs4=np.nan)
    b = table(rs2=0.2, rs1=0.04, rs4=np.nan, rs3=0.03)                     # the same SNPs, another order
    out = bd.min_p([a, b])
    assert list(out["SNP"]) == ["rs1", "rs2", "rs3", "rs4"]                 # the first table's order
    p = out.set_index("SNP")["P"]
    assert (p["rs1"], p["rs2"], p["rs3"]) == (0.04, 0.01, 0.03) and np.isnan(p["rs4"])
    assert bd.min_p([a]).equals(a)                                          # one disease: the classic selection's input
    assert bd.min_p([a, a]).equals(a)


def test_min_p_refuses_tables_of_different_snps():
    bd = _bd()
    with pytest.raises(ValueError):
        bd.min_p([table(rs1=0.1, rs2=0.2), table(rs1=0.1, rs3=0.2)])
    with pytest.raises(ValueError):
        bd.min_p([table(rs1=0.1, rs2=0.2), table(rs1=0.1)])


def test_the_classic_parameters_are_pinned():
    bd = _bd()
    assert (bd.DISEASE, bd.P_MAX, bd.R2, bd.KB, bd.MIN_PER_CHROM) == ("I10", 0.05, 0.1, 250, 3)
    assert list(bd.SIZES.values()) == [13199, 26287, 53498, 107743, 215872]


def test_threshold_only_selection_keeps_every_snp_with_a_finite_p_at_or_below_p_max():
    bd = _bd()
    ss = table(rs1=0.01, rs2=0.0100001, rs3=np.nan, rs4=0.0, rs5=0.5)
    assert bd.threshold_ids(ss, 0.01) == {"rs1", "rs4"}                      # at the threshold counts, just above not
    assert bd.threshold_ids(ss, 1.0) == {"rs1", "rs2", "rs4", "rs5"}         # never a SNP without a p
    assert bd.selection_rule(0.01, False) == "p <= 0.01 in any disease's GWAS, not clumped"
    assert bd.selection_rule(0.05, True) == "p <= 0.05 in any disease's GWAS, clumped once on the smallest p"


def test_the_parser_defaults_to_the_classic_rule_from_the_main_store():
    bd = _bd()
    a = bd.parser().parse_args([])
    assert (a.p_max, a.no_clump, a.source, a.name, a.diseases, a.rungs) == \
        (0.05, False, bd.MAIN, "classic_i10", ["I10"], list(bd.SIZES))
    a = bd.parser().parse_args(["--p_max", "0.01", "--no_clump", "--source", "/x", "--rungs", "hm3plus"])
    assert (a.p_max, a.no_clump, a.source, a.rungs) == (0.01, True, "/x", ["hm3plus"])
