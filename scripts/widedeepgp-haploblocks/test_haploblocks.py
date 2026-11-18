"""Tests of the haploblock pre-test scripts (CPU, synthetic data): the hard calls and PLINK packing, the genetic
map, the haploblock rule and codes, the three ridges' algebra and the cross-fitted routing."""
import gzip
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import export_chr22 as ex  # noqa: E402


def test_hard_calls_round_within_the_threshold_and_miss_otherwise():
    q = np.array([[0, 10, 11, 100, 95, 189, 190, 200, 255]], dtype=np.uint8)
    calls = ex.hard_calls(q, 0.1)
    assert calls.tolist() == [[0, 0, -1, 1, 1, -1, 2, 2, -1]]
    assert ex.hard_calls(q, 0.2).tolist() == [[0, 0, 0, 1, 1, 2, 2, 2, -1]]
    assert calls.dtype == np.int8


def test_bed_packing_round_trips_and_follows_the_plink_codes():
    rng = np.random.default_rng(0)
    calls = rng.integers(-1, 3, size=(11, 7)).astype(np.int8)      # 11 samples: three padded slots in the last byte
    data = ex.pack_bed(calls)
    assert data[:3] == b"\x6c\x1b\x01" and len(data) == 3 + 7 * 3
    assert (ex.unpack_bed(data, 11, 7) == calls).all()
    one = ex.pack_bed(np.array([[0], [1], [2], [-1]], dtype=np.int8))   # hom A2 = 11, het = 10, hom A1 = 00, missing = 01
    assert one[3] == (0b11 | 0b10 << 2 | 0b00 << 4 | 0b01 << 6)


def test_bed_export_reads_back_through_plink2_with_alt_as_allele_1(tmp_path):
    plink2 = "/mnt/sdg/robert/miniforge3/envs/ard-gwas/bin/plink2"
    if not os.path.exists(plink2):
        pytest.skip("plink2 not found")
    calls = np.array([[2, 0, 1], [1, -1, 0], [0, 2, 2], [2, 1, 1], [1, 1, 0]], dtype=np.int8)   # 5 samples x 3 SNPs
    p = tmp_path / "t"
    (p.with_suffix(".bed")).write_bytes(ex.pack_bed(calls))
    (p.with_suffix(".bim")).write_text("".join(f"22\trs{i}\t0\t{1000 + i}\tA\tC\n" for i in range(3)))   # A1 = ALT = A
    (p.with_suffix(".fam")).write_text("".join(f"0\ts{i}\t0\t0\t0\t-9\n" for i in range(5)))
    subprocess.run([plink2, "--bfile", str(p), "--export", "vcf", "--out", str(p), "--threads", "1"],
                   check=True, capture_output=True)
    rows = [l.split("\t") for l in (p.with_suffix(".vcf")).read_text().splitlines() if not l.startswith("#")]
    assert [r[3] + r[4] for r in rows] == ["CA"] * 3                   # REF = A2 = C, ALT = A1 = A
    alt_counts = np.array([[(-1 if g == "./." else g.count("1")) for g in r[9:]] for r in rows]).T
    assert (alt_counts == calls).all()


def test_the_genetic_map_is_rewritten_in_shapeit_format(tmp_path):
    src = tmp_path / "omni.gz"
    with gzip.open(src, "wt") as f:
        f.write("rs1 100 0.0\nrs2 200 0.5\nrs3 200 0.5\nrs4 300 1.25\n")
    n = ex.write_gmap(str(src), "22", str(tmp_path / "out.gz"))
    with gzip.open(tmp_path / "out.gz", "rt") as f:
        lines = f.read().splitlines()
    assert n == 3 and lines[0] == "pos\tchr\tcM" and lines[1:] == ["100\t22\t0.000000", "200\t22\t0.500000", "300\t22\t1.250000"]
    with gzip.open(src, "wt") as f:
        f.write("rs1 100 1.0\nrs2 200 0.5\n")
    with pytest.raises(SystemExit):
        ex.write_gmap(str(src), "22", str(tmp_path / "bad.gz"))


import haploblocks as hb  # noqa: E402


def _blocks(rng, n_chrom, n_blocks=3, n_snps=8, common=(0.5, 0.3, 0.16), rare=0.04):
    """Chromosomes over n_blocks independent blocks of 8 SNPs: three common haplotypes per block covering 96%, the
    rest spread over random 8-SNP patterns (each far below 1%). Returns alleles [m, n_chrom] and the block ranges."""
    pats = np.array([[0] * 8, [1, 1, 0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 1, 1, 1]], dtype=np.uint8)
    H = np.zeros((n_blocks * n_snps, n_chrom), dtype=np.uint8)
    for b in range(n_blocks):
        which = rng.choice(4, size=n_chrom, p=list(common) + [rare])
        hap = pats[np.minimum(which, 2)].copy()
        rare_rows = which == 3
        hap[rare_rows] = rng.integers(0, 2, size=(rare_rows.sum(), n_snps))
        H[b * n_snps:(b + 1) * n_snps] = hap.T
    return H, [(b, b * n_snps, (b + 1) * n_snps) for b in range(n_blocks)]


def _coverage(H, start, end, is_train, freq):
    """Independent twin of the rule: share of training chromosomes whose sub-haplotype has frequency >= freq."""
    sub = np.ascontiguousarray(H[start:end][:, is_train].T)
    _, inv, cnt = np.unique(sub, axis=0, return_inverse=True, return_counts=True)
    return (cnt[inv.ravel()] >= freq * is_train.sum()).mean()


# ten training chromosomes over five SNPs: 4 x 00000, 3 x 11000, 2 x 01110, 1 x 10100 (rare); the fifth SNP has one
# minor allele (10%); freq 0.2 = two chromosomes
HAND = np.array([[0, 0, 0, 0, 0]] * 4 + [[1, 1, 0, 0, 0]] * 3 + [[0, 1, 1, 1, 0]] * 2 + [[1, 0, 1, 0, 1]], dtype=np.uint8).T


def test_the_hand_worked_segments_at_two_coverages():
    is_train = np.ones(10, bool)
    segs = hb.segment_block(HAND, 0, 4, is_train, freq=0.2, cover=0.9)   # 00, 11, 01 cover 9 of 10 at every length
    assert [(s[0], s[1], s[3], s[4]) for s in segs] == [(0, 4, 3, 0.9)]
    assert segs[0][2].tolist() == [0] * 4 + [1] * 3 + [2] * 2 + [3]      # codes by decreasing frequency, other = 3
    segs = hb.segment_block(HAND, 0, 5, is_train, freq=0.2, cover=0.95)  # every pair of SNPs covers only 0.9
    assert [(s[0], s[1], s[3]) for s in segs] == [(0, 1, 2), (1, 2, 2), (2, 3, 2), (3, 4, 2), (4, 5, 1)]
    assert segs[4][4] == 0.9 and segs[4][2].tolist() == [0] * 9 + [1]    # a single SNP stands even below the coverage


def test_a_tuning_chromosome_outside_the_training_dictionary_is_other():
    H = np.concatenate([HAND, np.array([[1, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=np.uint8).T], axis=1)
    is_train = np.r_[np.ones(10, bool), False, False]
    seg, codes = hb.segment_chromosome(H, [(7, 0, 4)], is_train, 0.2, 0.9)
    assert seg["H"].tolist() == [3] and seg["block"].tolist() == [7] and codes.shape == (6, 1, 2)
    assert codes[5, 0].tolist() == [3, 0]                                 # unseen haplotype -> other, 0000 -> code 0
    assert codes[:5, 0].ravel().tolist() == [0] * 4 + [1] * 3 + [2] * 2 + [3]


def test_segments_are_valid_and_maximal_on_random_blocks_and_codes_are_haplotypes():
    rng = np.random.default_rng(1)
    H, true = _blocks(rng, 20000)
    is_train = np.ones(H.shape[1], bool)
    is_train[-200:] = False
    segs = hb.segment_block(H, 0, H.shape[0], is_train, freq=0.01, cover=0.95)
    assert 3 <= len(segs) <= 6 and segs[0][0] == 0 and segs[-1][1] == H.shape[0]
    for (start, end, codes, n_common, cover), nxt in zip(segs, segs[1:] + [None]):
        assert cover == pytest.approx(_coverage(H, start, end, is_train, 0.01)) and (cover >= 0.95 or end - start == 1)
        if nxt is not None:
            assert nxt[0] == end and _coverage(H, start, end + 1, is_train, 0.01) < 0.95   # maximal
        for code in range(n_common):
            pat = H[start:end][:, codes == code].T
            assert len(pat) and (pat == pat[0]).all()                     # one haplotype per common code
        counts = np.bincount(codes[is_train], minlength=n_common + 1)
        assert (np.diff(counts[:n_common]) <= 0).all() and counts[n_common] / is_train.sum() == pytest.approx(1 - cover)


def test_summarise_counts_features_and_projects_to_the_genome():
    seg = dict(start=np.array([0, 4, 8]), end=np.array([4, 8, 12]), block=np.array([0, 0, 1]),
               H=np.array([3, 5, 1]), cover=np.array([0.96, 0.95, 1.0]))
    pos = np.arange(12) * 1000
    row = hb.summarise(seg, pos, 12, "f0.01_c0.95", 0.01, 0.95)
    assert row["n_segments"] == 3 and row["features"] == 4 + 6 + 2 and row["features_per_snp"] == 1.0
    assert row["features_genome_projected"] == hb.M_GENOME and row["share_H_2_to_15"] == pytest.approx(2 / 3)
    assert row["kb_per_segment_median"] == 3.0 and row["H_max"] == 5


def test_read_haplotypes_parses_a_phased_bcf_and_checks_order(tmp_path):
    bcftools = "/mnt/sdg/robert/envs/phasing/bin/bcftools"
    if not os.path.exists(bcftools):
        pytest.skip("bcftools not found")
    os.environ["PATH"] = os.path.dirname(bcftools) + os.pathsep + os.environ["PATH"]
    vcf = tmp_path / "t.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n##contig=<ID=22>\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\ts2\n"
                   "22\t100\trs1\tA\tC\t.\t.\t.\tGT\t0|1\t1|1\t0|0\n22\t200\trs2\tA\tC\t.\t.\t.\tGT\t1|0\t0|0\t0|1\n")
    bcf = tmp_path / "t.bcf"
    subprocess.run([bcftools, "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    H = hb.read_haplotypes(str(bcf), np.array(["s0", "s1", "s2"]), np.array([100, 200]))
    assert H.tolist() == [[0, 1, 1, 1, 0, 0], [1, 0, 0, 0, 0, 1]]
    with pytest.raises(SystemExit):
        hb.read_haplotypes(str(bcf), np.array(["s0", "s2", "s1"]), np.array([100, 200]))
    with pytest.raises(SystemExit):
        hb.read_haplotypes(str(bcf), np.array(["s0", "s1", "s2"]), np.array([100, 201]))
    vcf.write_text(vcf.read_text().replace("0|0\n22", "0/0\n22"))
    subprocess.run([bcftools, "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    with pytest.raises(SystemExit):
        hb.read_haplotypes(str(bcf), np.array(["s0", "s1", "s2"]), np.array([100, 200]))


import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import compare_ridge as cr  # noqa: E402
from build_block_ridge import residual_coefficients, solve_block  # noqa: E402
from ridge_common import K_FOLDS  # noqa: E402


def test_hap_counts_hold_two_copies_per_segment_and_standardise_on_the_training_rows():
    codes = np.array([[[0, 0], [1, 2]], [[0, 1], [2, 2]], [[1, 1], [0, 2]]])          # 3 participants, 2 segments
    X = cr.hap_counts(codes, np.array([1, 2]))                                       # features: 2 + 3
    assert X.tolist() == [[2, 0, 0, 1, 1], [1, 1, 0, 0, 2], [0, 2, 1, 0, 1]]
    assert (X[:, :2].sum(1) == 2).all() and (X[:, 2:].sum(1) == 2).all()
    Z = cr.standardise_columns(X, 2)
    assert np.allclose(Z[:2].mean(0), 0) and np.allclose(Z[:2, 0] ** 2, 1) and (Z[:, 2] == 0).all()   # constant on the training rows


def _ridge_case(seed=0, n=600, ms=6, mh=4, n_lab=3, n_cov=2):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, ms + mh))
    cov = rng.standard_normal((n, n_cov))
    y = (Z[:, :2].sum(1, keepdims=True) + cov[:, :1] + rng.standard_normal((n, n_lab)) > 0.5).astype(float)
    folds = rng.integers(0, K_FOLDS, n)
    return Z, cov, y, folds, ms


def test_the_block_ridge_equals_numpy_for_every_fold_model_and_penalty():
    Z, cov, y, folds, _ = _ridge_case()
    t = np.column_stack([y, np.ones(len(y)), cov])
    B = residual_coefficients(y, cov, folds)
    G, T = cr.block_grams(Z, folds, t)
    lams = [50.0, 5.0]
    W = solve_block(G, T, torch.from_numpy(B), lams).numpy()
    for f in range(K_FOLDS + 1):
        keep = folds != f if f < K_FOLDS else np.ones(len(y), bool)
        A = Z[keep]
        R = y[keep] - np.column_stack([np.ones(keep.sum()), cov[keep]]) @ B[f]
        for p, lam in enumerate(lams):
            w = np.linalg.solve(A.T @ A + lam * np.eye(Z.shape[1]), A.T @ R)
            assert np.allclose(W[f][:, len(lams) * np.arange(y.shape[1]) + p], w, atol=1e-5)


def test_scaling_the_haplotype_features_is_a_class_specific_penalty():
    Z, cov, y, folds, ms = _ridge_case(1)
    s = 0.37
    Zt = Z.copy(); Zt[:, ms:] *= s
    t = np.column_stack([y, np.ones(len(y)), cov])
    B = residual_coefficients(y, cov, folds)
    G, T = cr.block_grams(Zt, folds, t)
    lam = 20.0
    W = solve_block(G, T, torch.from_numpy(B), [lam]).numpy()[K_FOLDS]
    R = y - np.column_stack([np.ones(len(y)), cov]) @ B[K_FOLDS]
    pen = np.r_[np.full(ms, lam), np.full(Z.shape[1] - ms, lam / s ** 2)]
    w = np.linalg.solve(Z.T @ Z + np.diag(pen), Z.T @ R)
    assert np.allclose(W[:ms], w[:ms], atol=1e-5) and np.allclose(W[ms:] * s, w[ms:], atol=1e-5)
    assert np.allclose(Zt @ W, Z @ w, atol=1e-5)


def test_scores_are_routed_to_each_rows_fold_model():
    Z = np.ones((7, 3)); Z[:, 1] = 2
    folds = np.array([0, 1, 2, 3, 4, 0, 2])
    W = np.stack([(f + 1) * np.ones((3, 2)) for f in range(K_FOLDS + 1)])
    S = cr.route_scores(Z, folds, W)
    assert S[:, 0].tolist() == [4 * (f + 1) for f in folds] and (S[:, 0] == S[:, 1]).all()


def test_the_bootstrap_auroc_and_correlation_match_their_plain_twins_on_replicated_rows():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = y + rng.standard_normal(400)
    r = y - 0.5 + 0.3 * rng.standard_normal(400)
    Wt = cr.boot_weights(400, 3)
    assert (Wt[0] == 1).all() and (Wt[1:].sum(1) == 400).all()
    a, c = cr.boot_auroc(s, y, Wt, chunk=2), cr.boot_corr(s, r, Wt)
    assert a[0] == pytest.approx(roc_auc_score(y, s)) and c[0] == pytest.approx(np.corrcoef(s, r)[0, 1])
    for b in (1, 2, 3):
        rep = np.repeat(np.arange(400), Wt[b])
        assert a[b] == pytest.approx(roc_auc_score(y[rep], s[rep])) and c[b] == pytest.approx(np.corrcoef(s[rep], r[rep])[0, 1])


def test_choose_picks_the_best_penalty_and_pick_slices_it():
    y = np.array([[0, 1], [1, 0], [1, 1], [0, 0]], float)
    S = np.array([[0.1, 0.9, 0.2, 0.1], [0.9, 0.1, 0.8, 0.9], [0.8, 0.5, 0.9, 0.9], [0.2, 0.3, 0.1, 0.2]])   # columns 2 j + p
    choice = cr.choose(S, y, 2)
    assert choice.tolist() == [0, 0] or choice.tolist() == [0, 1]
    assert (cr.pick(S, np.array([1, 0]), 2) == S[:, [1, 2]]).all()


def test_the_logistic_combination_scores_out_of_sample():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 3))
    y = (x[:, 0] + 0.5 * x[:, 2] + rng.standard_normal(2000) > 0).astype(float)
    tr, te = cr.logistic(x[:1500], y[:1500], [x[:1500], x[1500:]])
    assert roc_auc_score(y[1500:], te) > 0.75 and len(tr) == 1500
