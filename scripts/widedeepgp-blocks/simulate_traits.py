#!/usr/bin/env python3
"""Simulated liability-threshold traits on the real genotypes of a ladder rung: the E0 gate's dataset
(methods.md, section "E0 gate").

Writes <data>/ard30/<name>/ with the source rung's participants, SNPs (all, whole chromosomes), covariates
and tables, the 30 disease columns replaced by the traits of TRAITS; sim_truth.npz (the draws and the
per-person genetic values of the simulated splits); sim_log.json (aggregates only: realised h2, prevalence,
the ceiling, the additive ceiling and GBLUP on the tuning set). The test table is copied with empty trait
columns: its genotypes are never read. Everything is owner-only.

    CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python simulate_traits.py
"""
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ridge_common import MISSING, auroc, inverse_sd, standardise  # puts the model repository on sys.path
from genostore import torch_decode as td  # noqa: E402
from genostore.store import GenoStore  # noqa: E402

DATA = "/mnt/sdg/robert/deepwidegp/data/datasets"
BLOCKS = "/mnt/sdg/robert/deepwidegp/data/reference/hm3plus_in_ukb_pgen.tsv"
PREVALENCE = 0.2
SPLITS = ("train", "val", "wb_related")           # simulated; "test" is copied with empty trait columns
# kind: add | shared (a pair with `shared` causal SNPs in common, genetic correlation rg) | null | a planted
# kind: additive over `causal` SNPs plus a purely non-additive component of variance h2_planted at `planted`
# loci, concentrated so that it is learnable at the bottom rung: dom (an orthogonal dominance code at a
# causal SNP), epi (the centred product of two causal SNPs in different blocks), hetero (a causal SNP's
# effect switched by a same-block modifier's minor allele)
TRAITS = [
    dict(name="add_lo", kind="add", causal=2000, h2=0.3),
    dict(name="add_hi", kind="add", causal=20000, h2=0.5),
    dict(name=("shared_a", "shared_b"), kind="shared", causal=2000, shared=1500, rg=0.7, h2=0.3),
    dict(name="dom", kind="dom", causal=2000, h2=0.2, planted=20, h2_planted=0.1),
    dict(name="epi", kind="epi", causal=2000, h2=0.2, planted=20, h2_planted=0.1),
    dict(name="hetero", kind="hetero", causal=2000, h2=0.2, planted=20, h2_planted=0.1, modifier_maf=0.1),
    dict(name="null", kind="null", h2=0.0, prevalence=0.1),
]


def names(spec):
    return list(spec["name"]) if isinstance(spec["name"], tuple) else [spec["name"]]


def source(src):
    """(store.json, the store directory, the store) of a rung; whole chromosomes only."""
    spec = json.load(open(Path(src) / "store.json"))
    if "regions" in spec:
        raise ValueError(f"{src}: the source selects regions; the simulation needs whole chromosomes")
    store_dir = spec["store"] if os.path.isabs(spec["store"]) else str((Path(src) / spec["store"]).resolve())
    return spec, store_dir, GenoStore(store_dir)


def read_blocks(path, store):
    """Block id per SNP in store order, from a table whose ID column is the store's variant ids."""
    t = pd.read_csv(path, sep="\t", usecols=["ID", "block_id"], dtype={"ID": str})
    ids = np.concatenate([store.variants(c)["id"] for c in store.chroms])
    if len(t) != len(ids) or not np.array_equal(t["ID"].to_numpy(), ids):
        raise ValueError(f"{path}: its ID column is not the store's variants in store order")
    return t["block_id"].to_numpy(dtype=np.int64)


def draw(spec, m, block, maf, rng):
    """Causal SNPs (genome-wide indices, store order) and effects of one spec."""
    k, d = spec.get("causal", 0), {}
    if spec["kind"] == "null":
        return d
    if spec["kind"] == "shared":
        s = spec["shared"]
        rho = spec["rg"] * k / s                       # r_g = rho x shared share (independent SNPs, equal variances)
        if rho > 1:
            raise ValueError("genetic correlation above the shared share of causal SNPs is not reachable")
        pick = rng.choice(m, size=2 * k - s, replace=False)
        d["causal_a"], d["causal_b"] = pick[:k], np.concatenate([pick[:s], pick[k:]])
        d["beta_a"], d["beta_b"] = rng.standard_normal(k), rng.standard_normal(k)
        d["beta_b"][:s] = rho * d["beta_a"][:s] + np.sqrt(1 - rho ** 2) * d["beta_b"][:s]
        return d
    d["causal"] = np.sort(rng.choice(m, size=k, replace=False))
    d["beta"] = rng.standard_normal(k)
    if spec["kind"] == "add":
        return d
    d["effect"] = rng.standard_normal(spec["planted"])
    if spec["kind"] == "epi":
        perm, used, pairs = rng.permutation(d["causal"]), np.zeros(k, dtype=bool), []
        for i in range(k):
            if used[i] or len(pairs) == spec["planted"]:
                continue
            j = next((j for j in range(i + 1, k) if not used[j] and block[perm[j]] != block[perm[i]]), None)
            if j is not None:
                pairs.append((perm[i], perm[j]))
                used[i] = used[j] = True
        if len(pairs) < spec["planted"]:
            raise ValueError(f"only {len(pairs)} between-block pairs among the causal SNPs")
        d["pairs"] = np.array(pairs, dtype=np.int64)
        return d
    d["loci"] = np.sort(rng.choice(d["causal"], size=spec["planted"], replace=False))
    if spec["kind"] == "hetero":
        is_causal = np.zeros(m, dtype=bool)
        is_causal[d["causal"]] = True
        mods = []
        for s_ in d["loci"]:
            cand = np.flatnonzero(block == block[s_])
            ok = cand[~is_causal[cand] & (maf[cand] >= spec["modifier_maf"])]
            if not len(ok):                            # a tiny block: any other SNP of it
                ok = cand[cand != s_]
            if not len(ok):
                raise ValueError("a causal SNP's block has no other SNP to act as its modifier")
            mods.append(rng.choice(ok))
        d["modifier"] = np.array(mods, dtype=np.int64)
    return d


class Geno:
    """Standardised dosages of the gathered SNP columns ``cols`` (sorted genome-wide indices), by index."""

    def __init__(self, q, cols, mean, inv_std):
        self.q, self.cols = q, cols                    # q: uint8 [n, len(cols)] on the device
        self.mean = torch.as_tensor(mean[cols], dtype=torch.float32, device=q.device)
        self.inv = torch.as_tensor(inv_std[cols], dtype=torch.float32, device=q.device)

    def _pos(self, idx):
        return torch.as_tensor(np.searchsorted(self.cols, idx), device=self.q.device)

    def z(self, idx, rows=slice(None)):
        pos = self._pos(idx)
        return standardise(self.q[rows][:, pos], self.mean[pos], self.inv[pos])

    def x(self, idx):
        """Raw dosage, missing -> the training mean."""
        pos = self._pos(idx)
        q = self.q[:, pos]
        return torch.where(q == MISSING, self.mean[pos], q.float() * 0.01)

    def zb(self, idx, beta, chunk=8192):
        """z(idx) @ beta in row chunks: 20,000 columns at biobank scale do not fit at once."""
        return torch.cat([self.z(idx, slice(i, i + chunk)) @ beta for i in range(0, len(self.q), chunk)])


def residual(y, *xs):
    """Per column: y minus its least-squares fit on a constant and the matching columns of each x."""
    y, basis = y - y.mean(0), []
    for x in xs:
        x = x - x.mean(0)
        for b in basis:
            x = x - b * (x * b).sum(0) / (b * b).sum(0).clamp_min(1e-12)
        basis.append(x)
        y = y - x * (y * x).sum(0) / (x * x).sum(0).clamp_min(1e-12)
    return y


def unit(a):
    """Columns scaled to unit variance; a zero-variance column is left as it is (no division by zero)."""
    sd = a.std(0)
    return a / torch.where(sd > 0, sd, torch.ones_like(sd))


def components(spec, d, g, dev):
    """{trait: {component: (value [n], h2)}} of one spec's traits; the value is unscaled."""
    t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)
    if spec["kind"] == "null":
        return {spec["name"]: {}}
    if spec["kind"] == "shared":
        return {n: {"additive": (g.zb(d[f"causal_{s}"], t(d[f"beta_{s}"])), spec["h2"])}
                for n, s in zip(spec["name"], "ab")}
    out = {"additive": (g.zb(d["causal"], t(d["beta"])), spec["h2"])}
    if spec["kind"] == "dom":                          # heterozygote weight 1 - |x - 1|, orthogonal to the dosage
        x = g.x(d["loci"])
        code = residual(1.0 - (x - 1.0).abs(), x)
    elif spec["kind"] == "epi":                        # product of two standardised dosages, orthogonal to both
        zs, zt = g.z(d["pairs"][:, 0]), g.z(d["pairs"][:, 1])
        code = residual(zs * zt, zs, zt)
    elif spec["kind"] == "hetero":                     # the effect in carriers of the modifier's minor allele only
        zs, xm = g.z(d["loci"]), g.x(d["modifier"])
        carrier = torch.where(g.mean[g._pos(d["modifier"])] <= 1.0, xm >= 0.5, xm <= 1.5).float()
        code = residual(zs * carrier, zs, xm)
    if spec["kind"] in ("dom", "epi", "hetero"):
        out["planted"] = (unit(code) @ t(d["effect"]), spec["h2_planted"])
    return {spec["name"]: out}


def liability(comps, n, prevalence, rng):
    """Genetic value scaled to var h2 (the sum of the components' h2), noise to 1 - h2, cases above the
    prevalence quantile. Returns (labels uint8, genetic value, liability, realised share of each component)."""
    h2 = sum(h for _, h in comps.values())
    g = np.zeros(n)
    parts = {k: (v - v.mean()).double().cpu().numpy() / max(float(v.std()), 1e-12) * np.sqrt(h) for k, (v, h) in comps.items()}
    for p in parts.values():
        g += p
    f = np.sqrt(h2) / g.std() if h2 > 0 else 0.0
    g = (g - g.mean()) * f
    liab = g + rng.standard_normal(n) * np.sqrt(1.0 - h2)
    y = (liab > np.quantile(liab, 1.0 - prevalence)).astype(np.uint8)
    shares = {k: float((f * p).var() / liab.var()) for k, p in parts.items()}
    return y, g, liab, shares


def gblup(k_tr, k_x, y_tr, h2):
    """Best linear prediction of the tuning rows through the training GRM; penalty (1 - h2) / h2."""
    lam = (1.0 - h2) / h2
    k_tr.diagonal().add_(lam)
    a = torch.linalg.solve(k_tr, y_tr - y_tr.mean())
    k_tr.diagonal().sub_(lam)
    return k_x @ a


def additive_fit(g, idx, value, chunk=8192):
    """Fitted values of ``value`` on a constant and the standardised dosages of the SNPs ``idx`` over all rows
    (a tiny ridge): the best additive model of the trait's own SNPs."""
    k, dev = len(idx) + 1, g.q.device
    zz, zy = torch.zeros((k, k), dtype=torch.float64, device=dev), torch.zeros(k, dtype=torch.float64, device=dev)
    v = torch.as_tensor(value, dtype=torch.float32, device=dev)
    aug = lambda r: torch.cat([g.z(idx, r), torch.ones((len(v[r]), 1), device=dev)], 1)
    rows = [slice(i, i + chunk) for i in range(0, len(v), chunk)]
    for r in rows:
        z = aug(r)
        zz += (z.T @ z).double()
        zy += (z.T @ v[r]).double()
    zz.diagonal().add_(1e-6 * zz.diagonal().mean())
    b = torch.linalg.solve(zz, zy).float()
    return torch.cat([aug(r) @ b for r in rows]).cpu().numpy().astype(np.float64)


def stats_path(src, rows, store):
    """The data module's cache of training-set SNP statistics for these rows and whole chromosomes."""
    h = hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    h.update(json.dumps([[[c, 0, store.n_variants(c)]] for c in store.chroms]).encode())
    return Path(src) / f"snp_stats_train_{h.hexdigest()[:16]}.npz"


def batches(store, c, rows, dev, size=4096):
    """(positions in rows, uint8 q [B, M_c]) of chromosome c, decoded on dev, read in file order."""
    for i in range(0, len(rows), size):
        idx = np.arange(i, min(len(rows), i + size))
        idx = idx[np.argsort(rows[idx])]
        r = store.read_raw(c, rows[idx])
        parts = [torch.empty(0, dtype=torch.uint8, device=dev) if a is None else torch.from_numpy(np.ascontiguousarray(a)).to(dev)
                 for a in (r.full, r.full_ovr, r.tail, r.tail_ovr)]
        yield idx, td.decode_segment(*parts, store.w, r.tail_width, r.skip, r.n)


def moments(q, chunk=2048):
    """Per-SNP mean and std of the dosage over non-missing rows, as the data module computes them."""
    s0, s1, cnt = (torch.zeros(q.shape[1], dtype=torch.float64, device=q.device) for _ in range(3))
    for i in range(0, len(q), chunk):
        ok = q[i:i + chunk] != MISSING
        x = torch.where(ok, q[i:i + chunk], 0).double() * 0.01
        s0 += x.sum(0); s1 += (x * x).sum(0); cnt += ok.sum(0)
    mean = (s0 / cnt.clamp(min=1)).cpu().numpy()
    return mean, np.sqrt(np.maximum((s1 / cnt.clamp(min=1)).cpu().numpy() - mean * mean, 0.0))


def build(src, out, block, traits=TRAITS, seed=1, device="cuda"):
    """The dataset of simulated traits beside its source rung; returns the aggregate log."""
    t0, src, out, dev = time.time(), Path(src), Path(out), torch.device(device)
    if (out / "sim_log.json").exists():
        raise SystemExit(f"{out} already holds a simulated dataset; remove it to rebuild")
    spec, store_dir, store = source(src)
    m, n_c = sum(store.n_variants(c) for c in store.chroms), [store.n_variants(c) for c in store.chroms]
    if len(block) != m:
        raise ValueError(f"{len(block)} block ids for {m} SNPs")
    tables = {s: pd.read_csv(src / f"samples_{s}.tsv", sep="\t", dtype=str, keep_default_na=False) for s in SPLITS + ("test",)}
    row_of = {s: i for i, s in enumerate(store.samples)}
    rows = {}
    for s in SPLITS:
        unknown = sum(i not in row_of for i in tables[s]["IID"])
        if unknown:
            raise KeyError(f"samples_{s}.tsv: {unknown} identifiers are not in the store")
        rows[s] = np.array([row_of[i] for i in tables[s]["IID"]], dtype=np.int64)
    n = {s: len(r) for s, r in rows.items()}
    sl, at = {}, 0
    for s in SPLITS:
        sl[s] = slice(at, at + n[s]); at += n[s]
    n_all = at

    store_mean = np.concatenate([store.stats(c)[0] for c in store.chroms]) / 2.0
    maf = np.minimum(store_mean, 1.0 - store_mean)
    rng = np.random.default_rng(seed)
    draws = [draw(sp, m, block, maf, rng) for sp in traits]
    cols = np.unique(np.concatenate([a.ravel() for d in draws for k, a in d.items() if a.dtype.kind == "i"]))

    # one pass per chromosome: gather the causal columns; GRM of the training rows and training x tuning
    cache = stats_path(src, rows["train"], store)
    cached = np.load(cache) if cache.exists() else None
    qc = torch.empty((n_all, len(cols)), dtype=torch.uint8, device=dev)
    k_tr = torch.zeros((n["train"], n["train"]), dtype=torch.float64, device=dev)
    k_x = torch.zeros((n["val"], n["train"]), dtype=torch.float64, device=dev)
    mean, std, off = [], [], 0
    for i, c in enumerate(store.chroms):
        in_c = (cols >= off) & (cols < off + n_c[i])
        pos = torch.as_tensor(np.flatnonzero(in_c), device=dev)
        loc = torch.as_tensor(cols[in_c] - off, device=dev)
        q_tr = torch.empty((n["train"], n_c[i]), dtype=torch.uint8, device=dev)
        for idx, q in batches(store, c, rows["train"], dev):
            q_tr[idx] = q
        mu_c, sd_c = (cached[f"mean{i}"], cached[f"std{i}"]) if cached is not None else moments(q_tr)
        mean.append(mu_c); std.append(sd_c)
        mu = torch.as_tensor(mu_c, dtype=torch.float32, device=dev)
        inv = torch.as_tensor(inverse_sd(sd_c), dtype=torch.float32, device=dev)
        qc[sl["train"], pos] = q_tr[:, loc]
        z_tr = torch.empty((n["train"], n_c[i]), dtype=torch.float32, device=dev)
        for r0 in range(0, n["train"], 8192):
            z_tr[r0:r0 + 8192] = standardise(q_tr[r0:r0 + 8192], mu, inv)
        del q_tr
        k_tr += (z_tr @ z_tr.T).double()
        for idx, q in batches(store, c, rows["val"], dev):
            qc[torch.as_tensor(sl["val"].start + idx, device=dev)[:, None], pos] = q[:, loc]
            k_x[idx] += (standardise(q, mu, inv) @ z_tr.T).double()
        for idx, q in batches(store, c, rows["wb_related"], dev):
            qc[torch.as_tensor(sl["wb_related"].start + idx, device=dev)[:, None], pos] = q[:, loc]
        del z_tr
        off += n_c[i]
        print(f"chr{c}: {n_c[i]} SNPs, {int(in_c.sum())} causal columns, {time.time() - t0:.0f} s", flush=True)
    k_tr /= m
    k_x /= m
    mean, std = np.concatenate(mean), np.concatenate(std)
    g = Geno(qc, cols, mean, inverse_sd(std))

    # genetic values, liabilities, labels; references on the tuning set
    log, y, gv, truth = {}, {}, {}, {}
    for sp, d in zip(traits, draws):
        for name, comps in components(sp, d, g, dev).items():
            prev, h2 = sp.get("prevalence", PREVALENCE), float(sum(h for _, h in comps.values()))
            y[name], gv[name], liab, shares = liability(comps, n_all, prev, rng)
            idx = {"shared": lambda: d["causal_" + "ab"[sp["name"].index(name)]], "null": lambda: None,
                   "hetero": lambda: np.union1d(d["causal"], d["modifier"])}.get(sp["kind"], lambda: d["causal"])()
            yv, ytr = y[name][sl["val"]], torch.as_tensor(y[name][sl["train"]], dtype=torch.float64, device=dev)
            fit = None if idx is None else additive_fit(g, idx, gv[name])
            log[name] = dict(
                kind=sp["kind"], n_causal=int(sp.get("causal", 0)), n_planted=int(sp.get("planted", 0)), h2_target=h2,
                h2_realised=float(gv[name].var() / liab.var()), h2_components=shares, prevalence_target=prev,
                prevalence={s: float(y[name][sl[s]].mean()) for s in SPLITS}, cases={s: int(y[name][sl[s]].sum()) for s in SPLITS},
                ceiling_auroc_val=auroc(gv[name][sl["val"]], yv),
                additive_ceiling_auroc_val=None if fit is None else auroc(fit[sl["val"]], yv),
                gblup_auroc_val=auroc(gblup(k_tr, k_x, ytr, max(h2, 0.01)).cpu().numpy(), yv),
                additive_r2_in_sample=None if fit is None else float(1.0 - (gv[name] - fit).var() / gv[name].var()),
                n_predictors=0 if idx is None else int(len(idx)))
        tag = "shared" if sp["kind"] == "shared" else names(sp)[0]
        for k, a in d.items():
            truth[f"{tag}_{k}"] = a
    pairs = [sp["name"] for sp in traits if sp["kind"] == "shared"]
    rg = {f"{a}_{b}": float(np.corrcoef(gv[a], gv[b])[0, 1]) for a, b in pairs}
    labels = [nm for sp in traits for nm in names(sp)]

    # the dataset: tables in the source's row order, the trait columns in place of the diseases
    out.mkdir(mode=0o700, exist_ok=True)
    for s, df in tables.items():
        lab = pd.DataFrame({nm: (y[nm][sl[s]] if s in sl else np.full(len(df), np.nan)) for nm in labels})
        pd.concat([df[["IID"]], lab, df[spec["covariates"]]], axis=1).to_csv(out / f"samples_{s}.tsv", sep="\t", index=False)
    with open(out / "store.json", "w") as f:
        json.dump({"store": store_dir, "pos_mode": spec.get("pos_mode", "index"), "labels": labels, "covariates": spec["covariates"]}, f, indent=1)
    with open(out / "labels.json", "w") as f:
        json.dump([dict(label=nm, definition=f"simulated liability-threshold trait ({log[nm]['kind']}), seed {seed}",
                        **{f"cases_{s}": log[nm]["cases"][s] for s in SPLITS}, prevalence_train=log[nm]["prevalence"]["train"],
                        h2=log[nm]["h2_target"], n_causal=log[nm]["n_causal"]) for nm in labels], f, indent=1)
    if (src / "ladder.json").exists():
        shutil.copy2(src / "ladder.json", out / "ladder.json")
    if cached is not None:
        shutil.copy2(cache, out / cache.name)
    else:
        np.savez(out / cache.name, **{f"mean{i}": a for i, a in enumerate(np.split(mean, np.cumsum(n_c)[:-1]))},
                 **{f"std{i}": a for i, a in enumerate(np.split(std, np.cumsum(n_c)[:-1]))})
    np.savez_compressed(out / "sim_truth.npz", traits=np.array(labels), block=block,
                        **{f"iid_{s}": tables[s]["IID"].to_numpy().astype(str) for s in SPLITS},
                        **{f"g_{s}": np.stack([gv[nm][sl[s]] for nm in labels], 1).astype(np.float32) for s in SPLITS}, **truth)
    summary = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "source": str(src), "store": store_dir, "seed": seed, "n_snps": m,
               "n_blocks": int(len(np.unique(block))), "n": {**n, "test": len(tables["test"])},
               "test_table": "identifiers and covariates copied from the source; trait columns empty; genotypes not read",
               "standardisation": {"file": cache.name, "from": "the source's cache" if cached is not None else "the training rows (computed)"},
               "gblup": "ridge on all SNPs through the training GRM, penalty (1 - h2) / h2 with h2 floored at 0.01",
               "additive_ceiling": "AUROC on the tuning rows of the in-sample additive fit of the genetic value on the standardised "
                                   "dosages of its SNPs (causal, plus modifiers); additive_r2_in_sample is that fit's R2",
               "genetic_correlation": rg, "traits": log, "seconds": round(time.time() - t0, 1)}
    with open(out / "sim_log.json", "w") as f:
        json.dump(summary, f, indent=1)
    for p in out.iterdir():
        os.chmod(p, 0o600)
    os.chmod(out, 0o700)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="hm3plus_n13199")
    ap.add_argument("--name", default=None, help="default: <source>_sim")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--blocks", default=BLOCKS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    torch.set_num_threads(8)
    src = Path(a.data) / "ard30" / a.source
    block = read_blocks(a.blocks, source(src)[2])
    summary = build(src, Path(a.data) / "ard30" / (a.name or a.source + "_sim"), block, seed=a.seed, device=a.device)
    rows = [dict(trait=nm, kind=t["kind"], causal=t["n_causal"], planted=t["n_planted"], h2=t["h2_target"], h2_realised=t["h2_realised"],
                 prevalence_train=t["prevalence"]["train"], ceiling=t["ceiling_auroc_val"], additive_ceiling=t["additive_ceiling_auroc_val"],
                 gblup=t["gblup_auroc_val"], additive_r2=t["additive_r2_in_sample"]) for nm, t in summary["traits"].items()]
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"genetic correlation {summary['genetic_correlation']}; {summary['seconds']} s")


if __name__ == "__main__":
    main()
