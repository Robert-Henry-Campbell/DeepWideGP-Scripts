"""Tests of check_partition.py and make_partition.R on tiny made-up stores and LD matrices (CPU, seconds; no
UK Biobank data). The expected splits are worked by hand in the comments.

    /home/robert/envs/widedeepgp/bin/python -m pytest scripts/widedeepgp-blocks/test_blocks.py -q
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_partition as cp  # noqa: E402

RSCRIPT = "/usr/bin/Rscript"
IDS = {"1": [f"rs{i}" for i in range(1, 6)], "2": [f"rs{i}" for i in range(6, 9)]}      # 5 + 3 variants


def make_store(d, ids_by_chrom, pos=None):
    """A store directory with just what GenoStore needs for .chroms, .n_variants and .variants."""
    d = Path(d)
    d.mkdir(exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"n_samples": 3, "block_variants": 16384}))
    for c, ids in ids_by_chrom.items():
        (d / f"chr{c}.json").write_text(json.dumps({"chrom": c, "n_variants": len(ids), "complete": True}))
        p = pos[c] if pos else range(1, len(ids) + 1)
        (d / f"chr{c}.variants.tsv").write_text("#CHROM\tPOS\tID\tREF\tALT\n"
                                               + "".join(f"{c}\t{q}\t{i}\tA\tG\n" for q, i in zip(p, ids)))
    return str(d)


def write_tsv(path, rows, header=("CHROM", "ID", "block")):
    Path(path).write_text("\t".join(header) + "\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
    return str(path)


def labelled(ids_by_chrom, labels_by_chrom):
    return [(c, i, b) for c in ids_by_chrom for i, b in zip(ids_by_chrom[c], labels_by_chrom[c])]


@pytest.fixture
def store(tmp_path):
    return make_store(tmp_path / "store", IDS)


def build(tmp_path, store, name, rows, parent=None, **kw):
    tsv = write_tsv(tmp_path / f"{name}.tsv", rows)
    argv = [name, "--tsv", tsv, "--store", store, "--partitions", str(tmp_path / "parts")]
    cp.main(argv + (["--parent", parent] if parent else []))
    return tmp_path / "parts" / name


def test_a_valid_partition_is_written_in_the_contract_format(tmp_path, store):
    out = build(tmp_path, store, "p", labelled(IDS, {"1": [0, 0, 1, 1, 1], "2": [2, 2, 3]}))
    z = np.load(out / "partition.npz")                                       # no pickles needed
    assert sorted(z.files) == ["block_chr1", "block_chr2", "chroms", "n_blocks", "n_variants", "variant_digest"]
    assert z["chroms"].tolist() == ["1", "2"] and z["chroms"].dtype.kind == "U"
    assert z["n_variants"].tolist() == [5, 3] and z["n_variants"].dtype == np.int64
    assert z["block_chr1"].tolist() == [0, 0, 1, 1, 1] and z["block_chr1"].dtype == np.int32
    assert z["block_chr2"].tolist() == [2, 2, 3]
    assert z["n_blocks"].shape == () and z["n_blocks"] == 4 and z["n_blocks"].dtype == np.int64
    assert str(z["variant_digest"]) == hashlib.sha256("\n".join(IDS["1"] + IDS["2"]).encode()).hexdigest()
    info = json.loads((out / "partition.json").read_text())
    assert info["name"] == "p" and info["n_blocks"] == 4 and info["n_variants"] == 8
    assert info["variant_digest"] == str(z["variant_digest"])
    assert info["partition_npz_sha256"] == hashlib.sha256((out / "partition.npz").read_bytes()).hexdigest()
    assert info["block_size"] == {"min": 1, "q05": 1, "q25": 2, "median": 2, "q75": 2, "q95": 3, "max": 3, "mean": 2.0}
    assert [(r["chrom"], r["n_variants"], r["n_blocks"], r["min"], r["max"]) for r in info["per_chromosome"]] == [
        ("1", 5, 2, 2, 3), ("2", 3, 2, 1, 2)]
    assert info["built"]["column"] == "block" and info["built"]["parent"] is None
    assert info["built"]["source_sha256"] == hashlib.sha256((tmp_path / "p.tsv").read_bytes()).hexdigest()


def test_labels_are_renumbered_by_order_of_appearance(tmp_path, store):
    out = build(tmp_path, store, "p", labelled(IDS, {"1": [7, 7, 3, 3, 3], "2": [9, 9, 12]}))
    z = np.load(out / "partition.npz")
    assert z["block_chr1"].tolist() == [0, 0, 1, 1, 1] and z["block_chr2"].tolist() == [2, 2, 3]


def test_a_block_that_returns_after_another_is_refused(tmp_path, store):
    with pytest.raises(SystemExit, match="chr1: not contiguous: block 0 returns after block 1"):
        build(tmp_path, store, "p", labelled(IDS, {"1": [0, 0, 1, 0, 0], "2": [2, 2, 3]}))
    with pytest.raises(SystemExit, match="chr2: not contiguous: block 2 returns after block 3"):
        build(tmp_path, store, "p", labelled(IDS, {"1": [0, 0, 1, 1, 1], "2": [2, 3, 2]}))


def test_a_block_spanning_two_chromosomes_is_refused(tmp_path, store):
    with pytest.raises(SystemExit, match="block 1 spans chr1 and chr2"):
        build(tmp_path, store, "p", labelled(IDS, {"1": [0, 0, 1, 1, 1], "2": [1, 2, 2]}))


def test_a_tsv_that_is_not_the_store_in_store_order_is_refused(tmp_path, store):
    rows = labelled(IDS, {"1": [0, 0, 1, 1, 1], "2": [2, 2, 3]})
    with pytest.raises(SystemExit, match="7 rows, but the store has 8 variants"):
        build(tmp_path, store, "p", rows[:-1])
    with pytest.raises(SystemExit, match="9 rows, but the store has 8"):
        build(tmp_path, store, "p", rows + [("2", "rs9", 3)])
    with pytest.raises(SystemExit, match=r"2 rows are not the store's variant in store order \(first at row 2\)"):
        build(tmp_path, store, "p", [rows[0], rows[2], rows[1]] + rows[3:])
    with pytest.raises(SystemExit, match="1 rows are not the store's variant"):
        build(tmp_path, store, "p", [("2",) + r[1:] if r[1] == "rs5" else r for r in rows])
    with pytest.raises(SystemExit, match="1 rows are not the store's variant"):
        build(tmp_path, store, "p", [("1", "rs99", 0) if r[1] == "rs1" else r for r in rows])


def test_nesting_is_checked_against_the_parent(tmp_path, store):
    build(tmp_path, store, "coarse", labelled(IDS, {"1": [0, 0, 1, 1, 1], "2": [2, 2, 3]}))
    out = build(tmp_path, store, "fine", labelled(IDS, {"1": [0, 1, 2, 2, 3], "2": [4, 4, 5]}), parent="coarse")
    assert json.loads((out / "partition.json").read_text())["built"]["parent"] == "coarse"
    with pytest.raises(SystemExit, match="not nested in .*coarse.*: 1 blocks straddle"):   # block 0 covers rs2 and rs3
        build(tmp_path, store, "bad", labelled(IDS, {"1": [0, 0, 0, 1, 1], "2": [2, 2, 3]}), parent="coarse")
    with pytest.raises(SystemExit, match="2 blocks straddle"):                 # one chromosome-wide block each
        build(tmp_path, store, "bad", labelled(IDS, {"1": [0, 0, 0, 0, 0], "2": [1, 1, 1]}), parent="coarse")
    other = make_store(tmp_path / "other", {"1": IDS["1"], "2": ["rs6", "rs7", "rsX"]})
    tsv = write_tsv(tmp_path / "o.tsv", labelled({"1": IDS["1"], "2": ["rs6", "rs7", "rsX"]}, {"1": [0, 0, 1, 1, 1], "2": [2, 2, 3]}))
    with pytest.raises(SystemExit, match="built on other variants"):
        cp.main(["o", "--tsv", tsv, "--store", other, "--partitions", str(tmp_path / "parts"), "--parent", "coarse"])


def test_prive_blocks_come_from_the_reference_block_id_column(tmp_path, store):
    header = ("CHROM", "POS", "ID", "REF", "ALT", "block_id")
    rows = [(c, k, i, "A", "G", b) for c, i, b in labelled(IDS, {"1": [1, 1, 2, 2, 2], "2": [3, 3, 3]}) for k in [0]]
    ref = write_tsv(tmp_path / "ref.tsv", rows, header)
    cp.main(["prive", "--from_reference", ref, "--store", store, "--partitions", str(tmp_path / "parts")])
    z = np.load(tmp_path / "parts" / "prive" / "partition.npz")
    assert z["block_chr1"].tolist() == [0, 0, 1, 1, 1] and z["block_chr2"].tolist() == [2, 2, 2] and z["n_blocks"] == 3
    info = json.loads((tmp_path / "parts" / "prive" / "partition.json").read_text())
    assert info["built"]["column"] == "block_id" and info["built"]["source"] == ref
    bad = write_tsv(tmp_path / "bad.tsv", [r if r[2] != "rs7" else r[:2] + ("rs70",) + r[3:] for r in rows], header)
    with pytest.raises(SystemExit, match="1 rows are not the store's variant in store order"):
        cp.main(["prive", "--from_reference", bad, "--store", store, "--partitions", str(tmp_path / "parts")])


def test_the_csv_table_lists_each_partition_per_chromosome_and_overall(tmp_path, store):
    build(tmp_path, store, "a", labelled(IDS, {"1": [0, 0, 1, 1, 1], "2": [2, 2, 3]}))
    build(tmp_path, store, "b", labelled(IDS, {"1": [0] * 5, "2": [1] * 3}))
    cp.main(["a", "b", "--csv", str(tmp_path / "t.csv"), "--partitions", str(tmp_path / "parts")])
    lines = (tmp_path / "t.csv").read_text().splitlines()
    assert lines[0] == "partition,chrom,n_variants,n_blocks,size_min,size_q25,size_median,size_q75,size_max"
    assert lines[1:] == ["a,1,5,2,2,2,2,3,3", "a,2,3,2,1,1,1,2,2", "a,all,8,4,1,2,2,2,3",   # nearest rank; a half
                         "b,1,5,1,5,5,5,5,5", "b,2,3,1,3,3,3,3,3", "b,all,8,2,3,3,3,5,5"]   # index takes the lower


# --- make_partition.R on synthetic LD: one 4-SNP cluster, r = 0.9 among the first three and 0.3 to the fourth,
# so its cheapest 2-way split is (3, 1) at cost 3 x 0.09 = 0.27 (1.71 and 1.80 for the other cuts); clusters are
# uncorrelated, so splitting between clusters costs 0.

C4 = np.array([[1, .9, .9, .3], [.9, 1, .9, .3], [.9, .9, 1, .3], [.3, .3, .3, 1]])


def blockdiag(*ms):
    out = np.zeros((sum(len(m) for m in ms),) * 2)
    k = 0
    for m in ms:
        out[k:k + len(m), k:k + len(m)] = m
        k += len(m)
    return out


def write_ld(ld_dir, chrom, dense, snps):
    """chr<c>.corr.rds (a dsCMatrix, as the PGI arm's) and chr<c>.snpinfo.tsv with the SNPs in the matrix's order."""
    ld_dir.mkdir(exist_ok=True)
    csv = ld_dir / f"chr{chrom}.csv"
    np.savetxt(csv, dense, delimiter=",")
    subprocess.run([RSCRIPT, "-e", f'm <- as.matrix(read.csv("{csv}", header = FALSE)); dimnames(m) <- NULL; '
                    f'x <- Matrix::Matrix(m, sparse = TRUE); stopifnot(is(x, "dsCMatrix")); saveRDS(x, "{ld_dir}/chr{chrom}.corr.rds")'],
                   check=True, capture_output=True)
    (ld_dir / f"chr{chrom}.snpinfo.tsv").write_text("SNP\tLD\n" + "".join(f"{s}\t1\n" for s in snps))


def run_r(**kw):
    r = subprocess.run([RSCRIPT, str(HERE / "make_partition.R")] + [f"{k}={v}" for k, v in kw.items()], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stderr


def test_make_partition_splits_at_the_weakest_ld_nests_and_resumes(tmp_path):
    # Prive blocks: A = chr1, 3 clusters (12 SNPs); B = 1 cluster and C = 2 clusters on chr2. Targets 6 and 10
    # of 24 SNPs. Largest-remainder shares: level 1 (3, 1, 2) and level 2 (5, 2, 3) blocks; inside A the 5 go
    # (2, 2, 1) to its three level-1 blocks, inside C the 3 go (2, 1). Level 1 therefore cuts between clusters
    # (cost 0) and level 2 cuts (3, 1) inside four clusters (cost 0.27 each).
    ids = {"1": [f"a{i}" for i in range(12)], "2": [f"b{i}" for i in range(4)] + [f"c{i}" for i in range(8)]}
    store = make_store(tmp_path / "store", ids, pos={"1": range(100, 1300, 100), "2": range(100, 1300, 100)})
    ref = write_tsv(tmp_path / "ref.tsv", [(c, 0, i, b) for c, i, b in labelled(ids, {"1": [1] * 12, "2": [2] * 4 + [3] * 8})],
                    ("CHROM", "POS", "ID", "block_id"))
    ld = tmp_path / "ld"
    perm = np.arange(12)[::-1]                                       # chr1's LD matrix lists the SNPs backwards
    write_ld(ld, "1", blockdiag(C4, C4, C4)[np.ix_(perm, perm)], [ids["1"][i] for i in perm])
    write_ld(ld, "2", blockdiag(C4, C4, C4), ids["2"])
    out = tmp_path / "parts"
    common = dict(store=store, reference=ref, ld_dir=ld, out=out, targets="6,10", cores=2)
    log = run_r(chroms="1", **common)
    assert "chr1: 12 variants, 0 absent from the LD matrix, blocks 3/5" in log and "no assembly yet" in log
    assert (out / "work" / "chr1.tsv").exists() and not (out / "ldsplit6").exists()
    os.utime(out / "work" / "chr1.tsv", (1e9, 1e9))
    run_r(**common)                                                 # resumes: chr1 kept, chr2 done, then assembled
    assert os.stat(out / "work" / "chr1.tsv").st_mtime == 1e9
    rows = [l.split("\t") for l in (out / "ldsplit6" / "blocks.tsv").read_text().splitlines()]
    assert rows[0] == ["CHROM", "ID", "block"]
    assert [r[:2] for r in rows[1:]] == [[c, i] for c in ids for i in ids[c]]
    assert [int(r[2]) for r in rows[1:]] == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4
    rows10 = [l.split("\t") for l in (out / "ldsplit10" / "blocks.tsv").read_text().splitlines()][1:]
    assert [int(r[2]) for r in rows10] == [0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4] + [5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 9]
    j6 = json.loads((out / "ldsplit6" / "make_partition.json").read_text())
    assert j6["n_blocks"] == 6 and j6["blocks_per_chromosome"] == {"1": 3, "2": 3} and j6["n_absent_from_ld"]["total"] == 0
    assert j6["prive_blocks_off_target"] == 0 and j6["total_cost"] == 0
    assert [(b["block_id"], b["m"], b["k_target"], b["k"]) for b in j6["prive_blocks"]] == [(1, 12, 3, 3), (2, 4, 1, 1), (3, 8, 2, 2)]
    assert j6["parameters"]["thr_r2"] == 0.02 and j6["parameters"]["max_r2"] == 1 and j6["inputs"]["targets"] == [6, 10]
    j10 = json.loads((out / "ldsplit10" / "make_partition.json").read_text())
    assert j10["n_blocks"] == 10 and j10["block_size"] == {"0%": 1, "5%": 1, "25%": 1, "50%": 3, "75%": 3, "95%": 4, "100%": 4}
    assert [(b["block_id"], b["k_target"], b["k"]) for b in j10["prive_blocks"]] == [(1, 5, 5), (2, 2, 2), (3, 3, 3)]
    assert np.allclose([b["cost"] for b in j10["prive_blocks"]], [0.54, 0.27, 0.27])
    # check_partition accepts the chain prive -> ldsplit6 -> ldsplit10 and embeds the builder's aggregates
    parts = ["--store", store, "--partitions", str(out)]
    cp.main(["prive", "--from_reference", ref] + parts)
    cp.main(["ldsplit6", "--tsv", str(out / "ldsplit6" / "blocks.tsv"), "--parent", "prive", "--built", str(out / "ldsplit6" / "make_partition.json")] + parts)
    cp.main(["ldsplit10", "--tsv", str(out / "ldsplit10" / "blocks.tsv"), "--parent", "ldsplit6"] + parts)
    z = np.load(out / "ldsplit10" / "partition.npz")
    assert z["n_blocks"] == 10 and z["block_chr2"].tolist() == [5, 5, 5, 6, 7, 7, 7, 8, 9, 9, 9, 9]
    info = json.loads((out / "ldsplit6" / "partition.json").read_text())
    assert info["built"]["parent"] == "prive" and info["built"]["target"] == 6 and info["built"]["parameters"]["max_K"] == "2 K"
    with pytest.raises(SystemExit, match="4 blocks straddle"):      # the coarse one is not nested in the fine one: 4 split clusters
        cp.main(["back", "--tsv", str(out / "ldsplit6" / "blocks.tsv"), "--parent", "ldsplit10"] + parts)


def test_variants_absent_from_the_ld_matrix_join_their_nearest_neighbour_and_are_counted(tmp_path):
    # One block of 5 store variants; rsX (position 410) is not in the LD matrix. Targets 1 and 2: level 1 is the
    # whole block; level 2 splits the four present SNPs (3, 1) and rsX joins rs4, its nearest neighbour.
    ids = {"1": ["rs1", "rs2", "rs3", "rs4", "rsX"]}
    store = make_store(tmp_path / "store", ids, pos={"1": [100, 200, 300, 400, 410]})
    ref = write_tsv(tmp_path / "ref.tsv", [("1", i, 1) for i in ids["1"]], ("CHROM", "ID", "block_id"))
    write_ld(tmp_path / "ld", "1", C4, ids["1"][:4])
    out = tmp_path / "parts"
    log = run_r(store=store, reference=ref, ld_dir=tmp_path / "ld", out=out, targets="1,2", cores=1)
    assert "chr1: 5 variants, 1 absent from the LD matrix, blocks 1/2" in log
    rows = [l.split("\t") for l in (out / "ldsplit2" / "blocks.tsv").read_text().splitlines()][1:]
    assert [int(r[2]) for r in rows] == [0, 0, 0, 1, 1]
    assert [int(r[2]) for r in [l.split("\t") for l in (out / "ldsplit1" / "blocks.tsv").read_text().splitlines()][1:]] == [0] * 5
    j = json.loads((out / "ldsplit2" / "make_partition.json").read_text())
    assert j["n_absent_from_ld"] == {"total": 1, "per_chromosome": {"1": 1}} and j["n_blocks"] == 2
    assert j["prive_blocks"][0]["m"] == 5 and j["prive_blocks"][0]["k"] == 2 and abs(j["prive_blocks"][0]["cost"] - 0.27) < 1e-6  # float32 r
