"""Shared pieces of the closed-form block ridge scripts (make_folds, build_block_ridge, check_block_ridge,
fit_block_linear, make_warm_start; methods.md section 11): the rung's tables and stores, the fold rules, the
cached SNP statistics, the partition, owner-only files and the test-split lock."""
import hashlib
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from genostore.store import GenoStore  # noqa: E402

MISSING = 255                           # the store's code for a missing genotype
SEED = 1                                # the pseudo-fold hash seed
K_FOLDS = 5
HELD_OUT = "hm3plus_asian_black"        # sibling dataset of the held-out groups: its own store, the same variants
LOCKED = "the test split is locked: pass --unlock_test once every choice is recorded"


def chmod600(path):
    os.chmod(path, 0o600)
    return path


def save_npy(path, a):
    np.save(path, a)
    return chmod600(path)


def save_npz(path, **arrays):
    np.savez(path, **arrays)
    return chmod600(path)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    return chmod600(path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_iids(iids):
    """The ladder's digest of a participant set (PGI make_ladder.py): sha256 of the sorted IIDs joined by newlines."""
    return hashlib.sha256("\n".join(sorted(iids)).encode()).hexdigest()


def pseudo_fold(iids, seed=SEED):
    """Fold of a participant outside the training table: sha256 of "<IID>\\t<seed>", its first 8 bytes as a
    big-endian integer, modulo K_FOLDS. Deterministic and independent of every table."""
    return np.array([int.from_bytes(hashlib.sha256(f"{i}\t{seed}".encode()).digest()[:8], "big") % K_FOLDS
                     for i in iids], dtype=np.int8)


def auroc(score, y):
    """Area under the ROC curve (ties averaged); NaN unless both classes occur."""
    y = np.asarray(y) > 0.5
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def prefetch(fn, items, threads):
    """fn(item) for every item, in order, computed up to ``threads`` items ahead on a thread pool."""
    with ThreadPoolExecutor(threads) as ex:
        pending = deque()
        for item in items:
            pending.append(ex.submit(fn, item))
            if len(pending) >= threads:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def unlock(args, splits, blocks_dir, what):
    """Refuse the test split without --unlock_test; with it, append the access to blocks/test_access.log."""
    if "test" not in splits:
        return
    if not args.unlock_test:
        raise SystemExit(LOCKED)
    path = os.path.join(blocks_dir, "test_access.log")
    with open(path, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{what}\n")
    chmod600(path)


def inverse_sd(std):
    """1 / SD, and 0 for a SNP without variation: its standardised dosage is then 0, and so is its weight."""
    std = np.asarray(std, dtype=np.float64)
    return np.where(std > 0, 1.0 / np.where(std > 0, std, 1.0), 0.0)


def standardise(q, mean, inv_std):
    """uint8 q [B, n] (255 = missing) -> (dosage - mean) * inv_std in mean's dtype on its device; a missing
    genotype counts as the mean (0 once standardised)."""
    q = torch.as_tensor(q, device=mean.device)
    z = (q.to(mean.dtype) * 0.01 - mean) * inv_std
    return z.masked_fill_(q == MISSING, 0.0)


def load_partition(spec, store):
    """partition.npz (or its folder; the format of methods.md section 5) checked against the store's variants:
    {"path", "sha256", "n_blocks", "blocks": {chrom: [(block, v0, v1), ...]}}, blocks numbered 0..n_blocks-1 as
    contiguous runs in store order."""
    path = spec if spec.endswith(".npz") else os.path.join(spec, "partition.npz")
    z = np.load(path)
    chroms = [str(c) for c in z["chroms"]]
    if chroms != list(store.chroms) or [int(n) for n in z["n_variants"]] != [store.n_variants(c) for c in chroms]:
        raise SystemExit(f"{path}: built for other chromosomes or variant counts than the store's")
    ids = np.concatenate([store.variants(c)["id"] for c in chroms])
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != str(z["variant_digest"]):
        raise SystemExit(f"{path}: built for other variants than the store's")
    blocks, seen = {}, []
    for c in chroms:
        b = z[f"block_chr{c}"].astype(np.int64)
        cuts = np.flatnonzero(np.diff(b) != 0) + 1
        starts, stops = np.r_[0, cuts], np.r_[cuts, len(b)]
        blocks[c] = [(int(b[s]), int(s), int(e)) for s, e in zip(starts, stops)]
        seen += [int(b[s]) for s in starts]
    if seen != list(range(int(z["n_blocks"]))):
        raise SystemExit(f"{path}: block ids must be contiguous runs numbered 0..n_blocks-1 in store order")
    return {"path": os.path.abspath(path), "sha256": sha256_file(path), "n_blocks": int(z["n_blocks"]), "blocks": blocks}


class Rung:
    """One dataset directory of the ladder: store.json, its tables, and blocks/ for the ridge's products."""

    def __init__(self, dataset_dir):
        self.dir = os.path.abspath(dataset_dir)
        with open(os.path.join(self.dir, "store.json")) as f:
            spec = json.load(f)
        self.labels, self.covariates = list(spec["labels"]), list(spec["covariates"])
        self.store = self._store(self.dir, spec["store"])
        self.blocks = os.path.join(self.dir, "blocks")
        os.makedirs(self.blocks, mode=0o700, exist_ok=True)

    @staticmethod
    def _store(d, path):
        return GenoStore(path if os.path.isabs(path) else os.path.join(d, path))

    def table(self, split):
        """samples_<split>.tsv as {iid: str [n], y: float32 [n, labels], cov: float64 [n, covariates], rows: the
        store rows, store}. A split without a table here is a held-out group of the sibling dataset HELD_OUT."""
        d, store = self.dir, self.store
        if not os.path.exists(os.path.join(d, f"samples_{split}.tsv")):
            d = os.path.join(os.path.dirname(self.dir), HELD_OUT)
            with open(os.path.join(d, "store.json")) as f:
                store = self._store(d, json.load(f)["store"])
            if [store.n_variants(c) for c in store.chroms] != [self.store.n_variants(c) for c in self.store.chroms]:
                raise SystemExit(f"the store of {split} holds other variants than the rung's store")
        df = pd.read_csv(os.path.join(d, f"samples_{split}.tsv"), sep="\t", dtype={"IID": str})
        y = df[self.labels].to_numpy(np.float32)
        if np.isnan(y).any():
            raise ValueError(f"samples_{split}.tsv: every label must be 0 or 1")
        row_of = {s: i for i, s in enumerate(store.samples)}
        iid = df["IID"].to_numpy(str)
        return dict(iid=iid, y=y, cov=df[self.covariates].to_numpy(np.float64),
                    rows=np.array([row_of[s] for s in iid], dtype=np.int64), store=store)

    def folds(self, split, iid):
        """The split's folds from blocks/folds.npz, which must list the table's IIDs in its order."""
        z = np.load(os.path.join(self.blocks, "folds.npz"))
        if f"fold_{split}" not in z.files:
            raise SystemExit(f"no folds for the split {split}: run make_folds.py first")
        if len(z[f"iid_{split}"]) != len(iid) or (z[f"iid_{split}"] != iid).any():
            raise SystemExit(f"folds.npz does not match samples_{split}.tsv: run make_folds.py again")
        return z[f"fold_{split}"]

    def snp_stats(self, train_rows):
        """Mean and SD of the dosage per SNP over the training rows, per chromosome: the data module's cached
        snp_stats_train_<hash>.npz (same hash, same statistics), computed and cached here when it is absent."""
        store = self.store
        regions = [[[c, 0, store.n_variants(c)]] for c in store.chroms]
        h = hashlib.sha256(np.ascontiguousarray(train_rows, dtype=np.int64).tobytes())
        h.update(json.dumps(regions).encode())
        path = os.path.join(self.dir, f"snp_stats_train_{h.hexdigest()[:16]}.npz")
        if not os.path.exists(path):
            stats = {}
            for i, c in enumerate(store.chroms):
                s0, s1, cnt = 0.0, 0.0, 0.0
                for rows in np.array_split(train_rows, max(1, len(train_rows) // 512)):
                    q = store.read_q(c, rows)
                    ok = q != MISSING
                    x = np.where(ok, q, 0).astype(np.float64) * 0.01
                    s0, s1, cnt = s0 + x.sum(0), s1 + (x * x).sum(0), cnt + ok.sum(0)
                m = s0 / np.maximum(cnt, 1)
                stats[f"mean{i}"], stats[f"std{i}"] = m, np.sqrt(np.maximum(s1 / np.maximum(cnt, 1) - m * m, 0.0))
            np.savez(path, **stats)
        z = np.load(path)
        k = len(store.chroms)
        return path, [z[f"mean{i}"] for i in range(k)], [z[f"std{i}"] for i in range(k)]
