#!/usr/bin/env python3
"""The experiment chain of the LD-block WideDeepGP study (design document sections 5 and 7; plan section 7,
step 10). One process runs every cell in order from a frozen export of DeepWideGP's `widening` branch:

    lin        the closed-form block ridge, the linear twins and the ridge warm start, per dataset (GPU 2)
    e0         the gate: searches on the simulated traits of hm3plus_n13199_sim; e0_summary.py must pass
    t0         the pre-test: S1, S3, S5 once each at the top rung
    e3top      the partition sweep at the top rung, S3, both mixers  -> the grid's partition (decisions.json)
    abl        the ablations at S3, bottom and top rung                -> the headline configuration
    grid       5 sizes x 5 rungs x 2 mixers, S3 row and top-rung column first
    tab        embed_dim 4 and 64 at S3 across the rungs
    e2b        three shapes at S3's size, top rung
    e3         the partition sweep at the other rungs
    score      every finished cell's chosen model and seed retrains scored on the tuning set (--save_scores)

A cell is one run of main_multilabel.py (a 12-trial search with seed 1) under runs/blocks/<name>; cells of
e3top, abl, grid, tab, e2b and e3 are retrained with seeds 2 and 3 at the chosen settings (--tune_trials 0),
two retrains at a time beside the next search. A finished cell (results.json) is skipped, so rerunning
continues; the chain stops at the first failure, and after the gate when it fails. Every flag is explicit.
Log: runs/blocks/chain.log. Started detached by run_all.sh (which freezes the code).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = "/mnt/sdg/robert/deepwidegp"
PY = "/home/robert/envs/widedeepgp/bin/python"
RUNS = f"{ROOT}/runs"
OUT = f"{RUNS}/blocks"
DATA = f"{ROOT}/data/datasets"
PART = f"{ROOT}/data/reference/partitions"
SCRIPTS = f"{ROOT}/DeepWideGP-Scripts/scripts/widedeepgp-blocks"
CODE = os.environ.get("BLOCKS_CODE")                     # the frozen export (run_all.sh)
GPUS = ["2", "3", "4"]
RUNGS = ["hm3plus_n13199", "hm3plus_n26287", "hm3plus_n53498", "hm3plus_n107743", "hm3plus"]
TOP, BOTTOM, SIM = RUNGS[-1], RUNGS[0], "hm3plus_n13199_sim"
PARTITIONS = ["prive431", "ldsplit2000", "ldsplit5000"]
MIXERS = ["transformer", "bimamba1"]
# d_model, layers per mixer (Mamba depth matched to the transformer's parameter count within a factor 1.3)
SIZES = {"S1": (16, {"transformer": 1, "bimamba1": 1}), "S2": (32, {"transformer": 2, "bimamba1": 2}),
         "S3": (64, {"transformer": 4, "bimamba1": 6}), "S4": (128, {"transformer": 6, "bimamba1": 9}),
         "S5": (256, {"transformer": 8, "bimamba1": 14})}
SHAPES = {"S3w": (32, {"transformer": 16, "bimamba1": 20}), "S3d": (128, {"transformer": 1, "bimamba1": 2})}
DEFAULTS = {"lr": 1e-3, "l2": 1e-4, "dropout": 0.1, "table_var": 1.0}    # single runs (T0); searches tune these


def log(msg):
    with open(f"{OUT}/chain.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def sh(cmd, log_name, gpu=None, cwd=None, env=None):
    """Run a command, its output appended to runs/blocks/logs/<log_name>.log; returns the exit code."""
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    e = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", **(env or {}))
    if gpu is not None:
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(f"{OUT}/logs/{log_name}.log", "a") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=e).returncode


def spawn(cmd, log_name, gpu, cwd=None):
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    e = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", CUDA_VISIBLE_DEVICES=str(gpu))
    f = open(f"{OUT}/logs/{log_name}.log", "a")
    f.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(cmd)}\n")
    f.flush()
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=e)


def dataset(ds):
    return f"{DATA}/ard30/{ds}"


def results(run):
    path = f"{run}/results.json"
    return json.load(open(path)) if os.path.exists(path) else None


# ---- the linear comparators (Scripts step 7) -------------------------------------------------------
def ridge_summary(ds):
    """<dataset>/blocks/ridge_summary.csv (label, auroc_val): the cross-fitted ridge PRS's tuning AUROC, which
    e0_summary.py reads. Aggregates only; the per-person scores stay in memory."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    d = dataset(ds)
    out = f"{d}/blocks/ridge_summary.csv"
    if os.path.exists(out):
        return
    z = np.load(f"{d}/blocks/prs_val.npz", allow_pickle=False)
    table = pd.read_csv(f"{d}/samples_val.tsv", sep="\t", dtype={"IID": str}).set_index("IID")
    labels = [str(x) for x in z["labels"]]
    y = table.loc[[str(i) for i in z["iid"]], labels].to_numpy(dtype=float)
    rows = [{"label": name, "auroc_val": float(roc_auc_score(y[:, j], z["score"][:, j])) if 0 < y[:, j].sum() < len(y) else float("nan")}
            for j, name in enumerate(labels)]
    pd.DataFrame(rows).to_csv(out, index=False)
    os.chmod(out, 0o600)


def lin(ds, gpu, splits):
    """Folds, the block ridge, the linear twins and the warm start of one dataset; skipped where done."""
    d = dataset(ds)
    if os.path.exists(f"{d}/blocks/init_ridge_d16.pt") and os.path.exists(f"{d}/blocks/ridge_summary.csv"):
        return 0
    log(f"lin {ds}: starting on GPU {gpu}")
    for cmd in ([PY, f"{SCRIPTS}/make_folds.py", d, "--splits", *splits],
                [PY, f"{SCRIPTS}/build_block_ridge.py", d, "--partition", f"{PART}/prive431", "--splits", *splits],
                [PY, f"{SCRIPTS}/check_block_ridge.py", d, "--partition", f"{PART}/prive431"],
                [PY, f"{SCRIPTS}/fit_block_linear.py", d],
                [PY, f"{SCRIPTS}/make_warm_start.py", d, "--embed_dim", "16"]):
        rc = sh(cmd, f"lin_{ds}", gpu=gpu)
        if rc:
            log(f"lin {ds}: {os.path.basename(cmd[1])} exit {rc}; chain stopped")
            return rc
    ridge_summary(ds)
    log(f"lin {ds}: done")
    return 0


# ---- deep cells --------------------------------------------------------------------------------------
def cell_name(mixer, size, partition, code, init, rung, embed_dim=16):
    e = "" if embed_dim == 16 else f"_e{embed_dim}"
    return f"{mixer}_{size}_{partition}_{code}_{init}_{rung}{e}"


def pretrained_file(rung, partition, embed_dim, code):
    return f"{dataset(rung)}/blocks/pretrained_{partition}_{code}_e{embed_dim}.pt"


def ensure_pretrained(rung, partition, embed_dim, code, gpu):
    out = pretrained_file(rung, partition, embed_dim, code)
    if os.path.exists(out):
        return out
    log(f"pretraining {rung} {partition} {code} e{embed_dim} on GPU {gpu}")
    rc = sh([PY, "pretrain_blocks.py", "--data_dir", DATA, "--label", "ard30", "--rd_mode", rung, "--partition",
             f"{PART}/{partition}", "--embed_dim", str(embed_dim), "--snp_code", code, "--batch_size", "256",
             "--num_workers", "8", "--lr", "1e-3", "--epochs", "1", "--seed", "1", "--out", out],
            f"pretrain_{rung}_{partition}_{code}_e{embed_dim}", gpu=gpu, cwd=CODE)
    if rc or not os.path.exists(out):
        raise RuntimeError(f"pretraining {rung} {partition} failed (exit {rc})")
    return out


def init_file(init, rung, partition, embed_dim, code, gpu):
    """The --init_encoder file of a cell's init: none, pre (that rung's masked pretraining), preall (the top
    rung's), ridge (the rung's closed-form ridge warm start)."""
    if init == "none":
        return None
    if init == "ridge":
        path = f"{dataset(rung)}/blocks/init_ridge_d{embed_dim}.pt"
        if not os.path.exists(path):
            rc = sh([PY, f"{SCRIPTS}/make_warm_start.py", dataset(rung), "--embed_dim", str(embed_dim)], f"lin_{rung}", gpu=gpu)
            if rc or not os.path.exists(path):
                raise RuntimeError(f"warm start of {rung} at embed_dim {embed_dim} failed")
        return path
    return ensure_pretrained(TOP if init == "preall" else rung, partition, embed_dim, code, gpu)


def flags(mixer, size, partition, code, init, rung, embed_dim=16, shape=None):
    d, layers = (shape or SIZES[size])
    n_blocks = {"prive431": 431, "ldsplit2000": 2000, "ldsplit5000": 5000}[partition]
    big = size in ("S4", "S5") and n_blocks >= 2000               # the attention term: batch 128 x 2 (test_ml_blocks_gpu)
    f = ["--data_dir", DATA, "--label", "ard30", "--rd_mode", rung, "--input_mode", "blocks", "--wide", "off",
         "--partition", f"{PART}/{partition}", "--mixer", mixer, "--snp_code", code, "--embed_dim", str(embed_dim),
         "--table_kappa", "0.1", "--d_model", str(d), "--e_layers", str(layers[mixer]), "--n_heads", str(max(1, d // 16)),
         "--head_hidden", str(4 * d), "--n_fcn", "2", "--final_pool", "mean", "--snp_stats", "train",
         "--batch_size", "128" if big else "256", "--accumulate", "2" if big else "1", "--eval_batch_size", "1024",
         "--precision", "bf16", "--num_workers", "8", "--max_epoch_equiv", "20", "--val_per_epoch", "4",
         "--val_min_participants", "20000", "--val_max_epoch_equiv", "1", "--min_participants", "128000",
         "--stop_patience", "8", "--lr_schedule", "cosine", "--warmup_steps", "200", "--snapshots_per_decade", "6",
         "--snapshots_max", "20" if size == "S3" or rung == TOP else "6", "--run_dir", RUNS, "--experiment", "blocks"]
    if size == "S5" and n_blocks >= 5000:
        f.append("--grad_ckpt")
    return f


def search(name, base_flags, init_path, trials=12):
    """One cell's seed-1 search on the three GPUs; returns its results.json."""
    run = f"{OUT}/{name}"
    r = results(run)
    if r is not None:
        return r
    log(f"cell {name}: search starting")
    cmd = [PY, "main_multilabel.py", *base_flags, "--exp_name", name, "--seed", "1", "--tune_trials", str(trials),
           "--tune_gpus", ",".join(GPUS)]
    if init_path:
        cmd += ["--init_encoder", init_path]
    rc = sh(cmd, f"cell_{name}", cwd=CODE)
    r = results(run)
    if rc or r is None:
        raise RuntimeError(f"cell {name}: exit {rc}, no results.json")
    log(f"cell {name}: search done, best tuning AUROC {r['best_val_auroc_mean']:.4f}, settings {r['best_params']}")
    return r


class Seeds:
    """Seed retrains (2 and 3) of finished searches, two at a time on GPUs 2 and 3, beside the next search."""

    def __init__(self):
        self.running = []

    def start(self, name, base_flags, init_path):
        r = results(f"{OUT}/{name}")
        p = r["best_params"]
        for seed, gpu in ((2, GPUS[0]), (3, GPUS[1])):
            run = f"{OUT}/{name}_seed{seed}"
            if results(run) is not None:
                continue
            self.wait(max_running=1)
            cmd = [PY, "main_multilabel.py", *base_flags, "--exp_name", f"{name}_seed{seed}", "--seed", str(seed),
                   "--tune_trials", "0", "--lr", str(p["lr"]), "--l2", str(p["l2"]), "--dropout", str(p["dropout"]),
                   "--table_var", str(p["table_var"])]
            if init_path:
                cmd += ["--init_encoder", init_path]
            log(f"cell {name}: seed {seed} retrain starting on GPU {gpu}")
            self.running.append((f"{name}_seed{seed}", spawn(cmd, f"cell_{name}_seed{seed}", gpu, cwd=CODE)))

    def wait(self, max_running=0):
        while len(self.running) > max_running:
            name, proc = self.running.pop(0)
            rc = proc.wait()
            if rc or results(f"{OUT}/{name}") is None:
                raise RuntimeError(f"{name}: exit {rc}, no results.json")
            log(f"{name}: done, tuning AUROC {results(f'{OUT}/{name}')['best_val_auroc_mean']:.4f}")


def seed_mean(name):
    """Mean best tuning AUROC over the three seeds of a finished cell (NaN while a seed is missing)."""
    vals = [results(f"{OUT}/{n}") for n in (name, f"{name}_seed2", f"{name}_seed3")]
    return float(np.mean([v["best_val_auroc_mean"] for v in vals])) if all(vals) else float("nan")


def run_cells(cells, seeds, with_seeds=True):
    """cells: [(name, base_flags, init_path)]; each search, then its seed retrains beside the next search."""
    for name, base_flags, init_path in cells:
        search(name, base_flags, init_path)
        if with_seeds:
            seeds.start(name, base_flags, init_path)
    if with_seeds:
        seeds.wait()


def decisions():
    path = f"{OUT}/decisions.json"
    return json.load(open(path)) if os.path.exists(path) else {}


def decide(key, value, detail):
    d = decisions()
    d[key] = {"value": value, "detail": detail, "written": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(f"{OUT}/decisions.json", "w") as f:
        json.dump(d, f, indent=1)
    log(f"decision {key} = {value}: {detail}")
    return value


def main():
    if not CODE or not os.path.isdir(CODE):
        raise SystemExit("BLOCKS_CODE must name the frozen export of DeepWideGP (run_all.sh)")
    os.makedirs(OUT, exist_ok=True)
    log(f"chain starting from {CODE}, pid {os.getpid()}")
    seeds = Seeds()
    held = ["train", "val", "wb_related"]

    # lin: the simulated dataset's references first (E0 needs them), then the bottom rung
    for ds in (SIM, BOTTOM):
        if lin(ds, GPUS[0], held):
            raise SystemExit(1)

    # e0: the gate on the simulated traits, no seed retrains
    def e0_cell(mixer, size, partition, code, init):
        name = cell_name(mixer, size, partition, code, init, SIM)
        return name, flags(mixer, size, partition, code, init, SIM), init_file(init, SIM, partition, 16, code, GPUS[2])
    e0 = [e0_cell(m, s, p, "dosage", "none") for p in PARTITIONS for m in MIXERS for s in ("S1", "S3")]
    e0 += [e0_cell(m, "S3", "prive431", "genotype", "none") for m in MIXERS]
    e0 += [e0_cell(m, "S3", "prive431", "dosage", "pre") for m in MIXERS]
    e0 += [e0_cell(m, "S3", "prive431", "dosage", "ridge") for m in MIXERS]
    run_cells(e0, seeds, with_seeds=False)
    rc = sh([PY, f"{SCRIPTS}/e0_summary.py", "--dataset", dataset(SIM), "--out", f"{OUT}/e0", *[f"{OUT}/{n}" for n, _, _ in e0]],
            "e0_summary")
    decide("e0_gate", "pass" if rc == 0 else "fail", f"e0_summary.py exit {rc}; table in {OUT}/e0/e0_results.md")
    if rc:
        log("E0 gate failed; the chain stops here for review")
        raise SystemExit(2)

    # the other rungs' linear comparators run on GPU 2 in the background from here on
    lin_proc = spawn([PY, __file__, "lin_rest"], "lin_rest", GPUS[0])

    # t0: three single runs at the top rung, one per GPU
    procs = []
    for size, gpu in zip(("S1", "S3", "S5"), GPUS):
        name = f"t0_transformer_{size}_prive431_{TOP}"
        if results(f"{OUT}/{name}") is None:
            cmd = [PY, "main_multilabel.py", *flags("transformer", size, "prive431", "dosage", "none", TOP), "--exp_name", name,
                   "--seed", "1", "--tune_trials", "0", *[a for k, v in DEFAULTS.items() for a in (f"--{k}", str(v))]]
            procs.append((name, spawn(cmd, f"cell_{name}", gpu, cwd=CODE)))
    for name, proc in procs:
        if proc.wait() or results(f"{OUT}/{name}") is None:
            raise RuntimeError(f"{name} failed")
        log(f"t0 {name}: tuning AUROC {results(f'{OUT}/{name}')['best_val_auroc_mean']:.4f}")

    # e3top: the partition sweep at the top rung -> the grid's partition
    e3top = [(cell_name(m, "S3", p, "dosage", "none", TOP), flags(m, "S3", p, "dosage", "none", TOP), None)
             for p in PARTITIONS for m in MIXERS]
    run_cells(e3top, seeds)
    score = {p: float(np.mean([seed_mean(cell_name(m, "S3", p, "dosage", "none", TOP)) for m in MIXERS])) for p in PARTITIONS}
    partition = decisions().get("partition", {}).get("value") or decide("partition", max(score, key=score.get), json.dumps(score))

    # abl: code and initialisation at S3, bottom and top rung -> the headline configuration
    variants = [("dosage", "none"), ("genotype", "none"), ("dosage", "pre"), ("dosage", "ridge")]
    abl = []
    for rung in (BOTTOM, TOP):
        for code, init in variants + ([("dosage", "preall")] if rung == BOTTOM else []):
            for m in MIXERS:
                abl.append((cell_name(m, "S3", partition, code, init, rung), flags(m, "S3", partition, code, init, rung),
                            init_file(init, rung, partition, 16, code, GPUS[2])))
    run_cells(abl, seeds)
    score = {f"{c}/{i}": float(np.mean([seed_mean(cell_name(m, "S3", partition, c, i, TOP)) for m in MIXERS])) for c, i in variants}
    head = decisions().get("headline", {}).get("value") or decide("headline", max(score, key=score.get), json.dumps(score))
    code, init = head.split("/")

    # grid: S3 row and top-rung column first, then the rest, rungs smallest first
    lin_proc.wait()
    order = [("S3", r) for r in RUNGS] + [(s, TOP) for s in SIZES if s != "S3"] + \
            [(s, r) for r in RUNGS[:-1] for s in SIZES if s != "S3"]
    grid = [(cell_name(m, s, partition, code, init, r), flags(m, s, partition, code, init, r),
             init_file(init, r, partition, 16, code, GPUS[2])) for s, r in order for m in MIXERS]
    run_cells(grid, seeds)

    # tab: the table's width at S3
    tab = [(cell_name(m, "S3", partition, code, init, r, e), flags(m, "S3", partition, code, init, r, e),
            init_file(init, r, partition, e, code, GPUS[2])) for e in (4, 64) for r in RUNGS for m in MIXERS]
    run_cells(tab, seeds)

    # e2b: shapes at S3's size, top rung
    e2b = [(cell_name(m, k, partition, code, init, TOP), flags(m, "S3", partition, code, init, TOP, shape=SHAPES[k]),
            init_file(init, TOP, partition, 16, code, GPUS[2])) for k in SHAPES for m in MIXERS]
    run_cells(e2b, seeds)

    # e3: the other partitions at the other rungs
    e3 = [(cell_name(m, "S3", p, code, init, r), flags(m, "S3", p, code, init, r), init_file(init, r, p, 16, code, GPUS[2]))
          for r in RUNGS[:-1] for p in PARTITIONS if p != partition for m in MIXERS]
    run_cells(e3, seeds)

    # score: the chosen model of every cell and its seed retrains on the tuning set, scores saved under data/
    for name in sorted(os.listdir(OUT)):
        run = f"{OUT}/{name}"
        if results(run) is None or os.path.exists(f"{run}/scores_val_summary.csv") or name in ("e0", "logs"):
            continue
        target = run
        if results(run).get("tuning"):
            target = os.path.join(run, results(run)["best_trial_dir"])
        if os.path.exists(f"{target}/scores_val_summary.csv"):
            continue
        rc = sh([PY, "score_snapshots.py", "--run", run, "--split", "val", "--snapshots", "best", "--save_scores",
                 "--batch_size", "1024"], f"score_{name}", gpu=GPUS[2], cwd=CODE)
        log(f"score {name}: exit {rc}")
    log("chain finished: every planned cell done; E5 (held-out groups), E6 (h2 scan) and E7 are separate steps")


def lin_rest():
    for ds in RUNGS[1:]:
        if lin(ds, GPUS[0], ["train", "val", "wb_related"]):
            raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:] == ["lin_rest"]:
        lin_rest()
    else:
        main()
