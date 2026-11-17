#!/usr/bin/env python3
"""The experiment chain of the LD-block WideDeepGP study on the real diseases and real genotypes (design
document sections 5, 7 and 12; Robert, 17 November 2026: no simulated-trait gate, the top rung first, then down
the rungs; the side studies after the grid). One process runs every cell in order from a frozen export of
DeepWideGP's `widening` branch:

    lin        the closed-form block ridge, the linear twins and the ridge warm start per rung, top rung first,
               in the background on GPU 2 (the comparators and the warm start; nothing waits for them but the
               ablations)
    t0         the pre-test: S1, S3, S5 once each at the top rung, fixed settings, one per GPU
    grid       5 sizes x 2 mixers per rung, rung by rung from 215,872 down to 13,199 participants, on the Prive
               partition with the plain configuration (dosage code, no initialisation)
    side       at the top rung: the partition sweep (ldsplit2000, ldsplit5000), the ablations (genotype code,
               masked pretraining, ridge warm start), the table width (4 and 64), the shapes; then the partition
               sweep and the table width at the other rungs
    score      every finished cell's chosen model and seed retrains scored on the tuning set (--save_scores)

A cell is one run of main_multilabel.py (a 12-trial search with seed 1) under runs/blocks/<name>, retrained with
seeds 2 and 3 at the chosen settings (--tune_trials 0); the retrains of three cells run together in waves that
fill the GPUs. GPUs (Robert, 17 November 2026): as many as are free, up to three, picked at every launch; a GPU
with another session's process is not free, one with only this chain's processes is (the linear comparators
share one); ten CPU threads per GPU. Nothing with simulated phenotypes is queued. A finished cell
(results.json) is skipped, so rerunning continues; the chain stops at the first failure. Every flag is explicit.
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
MAX_GPUS, THREADS = 3, "10"
RUNGS = ["hm3plus_n13199", "hm3plus_n26287", "hm3plus_n53498", "hm3plus_n107743", "hm3plus"]
TOP, BOTTOM = RUNGS[-1], RUNGS[0]
SPLITS = ["train", "val", "wb_related", "asian", "black"]
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
    e = dict(os.environ, OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, **(env or {}))
    if gpu is not None:
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with open(f"{OUT}/logs/{log_name}.log", "a") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=e).returncode


def spawn(cmd, log_name, gpu, cwd=None):
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    e = dict(os.environ, OMP_NUM_THREADS=THREADS, MKL_NUM_THREADS=THREADS, CUDA_VISIBLE_DEVICES=str(gpu))
    f = open(f"{OUT}/logs/{log_name}.log", "a")
    f.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {' '.join(cmd)}\n")
    f.flush()
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=e)


def dataset(ds):
    return f"{DATA}/ard30/{ds}"


def results(run):
    path = f"{run}/results.json"
    return json.load(open(path)) if os.path.exists(path) else None


def free_gpus(limit=MAX_GPUS):
    """Up to ``limit`` GPUs without another session's compute process, the least used by this chain first;
    waits (five-minute polls) until there is one."""
    while True:
        try:
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=60).stdout.splitlines()
            rows = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=60).stdout.splitlines()
        except (OSError, subprocess.SubprocessError):
            time.sleep(300)
            continue
        mine, foreign, ours = os.getsid(0), set(), {}
        for line in apps:
            if "," not in line:
                continue
            uuid, pid = (x.strip() for x in line.split(",")[:2])
            try:
                same = os.getsid(int(pid)) == mine
            except (ProcessLookupError, PermissionError, ValueError):
                continue
            if same:
                ours[uuid] = ours.get(uuid, 0) + 1
            else:
                foreign.add(uuid)
        free = [(ours.get(uuid.strip(), 0), idx.strip()) for idx, uuid in (r.split(",") for r in rows) if uuid.strip() not in foreign]
        if free:
            return [idx for _, idx in sorted(free)][:limit]
        log("waiting for a GPU without another session's process")
        time.sleep(300)


# ---- the linear comparators (Scripts step 7) -------------------------------------------------------
def lin(ds, gpu, splits):
    """Folds, the block ridge, the linear twins and the warm start of one dataset; skipped where done."""
    d = dataset(ds)
    if os.path.exists(f"{d}/blocks/init_ridge_d16.pt") and os.path.exists(f"{d}/blocks/prs_val.npz"):
        return 0
    log(f"lin {ds}: starting on GPU {gpu}")
    for cmd in ([PY, f"{SCRIPTS}/make_folds.py", d, "--splits", *splits],
                [PY, f"{SCRIPTS}/build_block_ridge.py", d, "--partition", f"{PART}/prive431", "--splits", *splits, "--threads", THREADS],
                [PY, f"{SCRIPTS}/check_block_ridge.py", d, "--partition", f"{PART}/prive431"],
                [PY, f"{SCRIPTS}/fit_block_linear.py", d],
                [PY, f"{SCRIPTS}/make_warm_start.py", d, "--embed_dim", "16"]):
        rc = sh(cmd, f"lin_{ds}", gpu=gpu)
        if rc:
            log(f"lin {ds}: {os.path.basename(cmd[1])} exit {rc}; chain stopped")
            return rc
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
             "--num_workers", THREADS, "--lr", "1e-3", "--epochs", "1", "--seed", "1", "--out", out],
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
         "--precision", "bf16", "--num_workers", THREADS, "--max_epoch_equiv", "20", "--val_per_epoch", "4",
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
           "--tune_gpus", ",".join(free_gpus())]
    if init_path:
        cmd += ["--init_encoder", init_path]
    rc = sh(cmd, f"cell_{name}", cwd=CODE)
    r = results(run)
    if rc or r is None:
        raise RuntimeError(f"cell {name}: exit {rc}, no results.json")
    log(f"cell {name}: search done, best tuning AUROC {r['best_val_auroc_mean']:.4f}, settings {r['best_params']}")
    return r


class Seeds:
    """Seed retrains (2 and 3) of finished searches, run in waves that fill the free GPUs (flushed after every
    three cells and at the end of a phase)."""

    def __init__(self):
        self.pending = []

    def start(self, name, base_flags, init_path):
        p = results(f"{OUT}/{name}")["best_params"]
        for seed in (2, 3):
            if results(f"{OUT}/{name}_seed{seed}") is None:
                cmd = [PY, "main_multilabel.py", *base_flags, "--exp_name", f"{name}_seed{seed}", "--seed", str(seed),
                       "--tune_trials", "0", "--lr", str(p["lr"]), "--l2", str(p["l2"]), "--dropout", str(p["dropout"]),
                       "--table_var", str(p["table_var"])] + (["--init_encoder", init_path] if init_path else [])
                self.pending.append((f"{name}_seed{seed}", cmd))
        if len(self.pending) >= 6:
            self.flush()

    def flush(self):
        while self.pending:
            gpus = free_gpus()
            wave, self.pending = self.pending[:len(gpus)], self.pending[len(gpus):]
            procs = []
            for (name, cmd), gpu in zip(wave, gpus):
                log(f"{name}: retrain starting on GPU {gpu}")
                procs.append((name, spawn(cmd, f"cell_{name}", gpu, cwd=CODE)))
            for name, proc in procs:
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
        seeds.flush()


def main():
    if not CODE or not os.path.isdir(CODE):
        raise SystemExit("BLOCKS_CODE must name the frozen export of DeepWideGP (run_all.sh)")
    os.makedirs(OUT, exist_ok=True)
    log(f"chain starting from {CODE}, pid {os.getpid()}")
    seeds = Seeds()
    down = RUNGS[::-1]                                       # the top rung first
    lin_proc = spawn([PY, __file__, "lin_all"], "lin_all", free_gpus()[0])

    # t0: three single runs at the top rung, the slowest first, in waves over the free GPUs
    todo = [z for z in ("S5", "S3", "S1") if results(f"{OUT}/t0_transformer_{z}_prive431_{TOP}") is None]
    while todo:
        gpus = free_gpus()
        wave, todo = todo[:len(gpus)], todo[len(gpus):]
        procs = []
        for size, gpu in zip(wave, gpus):
            name = f"t0_transformer_{size}_prive431_{TOP}"
            cmd = [PY, "main_multilabel.py", *flags("transformer", size, "prive431", "dosage", "none", TOP), "--exp_name", name,
                   "--seed", "1", "--tune_trials", "0", *[a for k, v in DEFAULTS.items() for a in (f"--{k}", str(v))]]
            procs.append((name, spawn(cmd, f"cell_{name}", gpu, cwd=CODE)))
        for name, proc in procs:
            if proc.wait() or results(f"{OUT}/{name}") is None:
                raise RuntimeError(f"{name} failed")
            log(f"t0 {name}: tuning AUROC {results(f'{OUT}/{name}')['best_val_auroc_mean']:.4f}")

    # grid: rung by rung from the top, five sizes x two mixers, the plain configuration on the Prive partition
    grid = [(cell_name(m, s, "prive431", "dosage", "none", r), flags(m, s, "prive431", "dosage", "none", r), None)
            for r in down for s in SIZES for m in MIXERS]
    run_cells(grid, seeds)

    # side studies at the top rung: partition sweep, ablations (the warm start needs the top rung's ridge), table width, shapes
    side = [(cell_name(m, "S3", p, "dosage", "none", TOP), flags(m, "S3", p, "dosage", "none", TOP), None)
            for p in PARTITIONS if p != "prive431" for m in MIXERS]
    run_cells(side, seeds)
    if lin_proc.wait():
        raise RuntimeError("the linear comparators failed; see logs/lin_all.log")
    abl = [(cell_name(m, "S3", "prive431", code, init, TOP), flags(m, "S3", "prive431", code, init, TOP),
            init_file(init, TOP, "prive431", 16, code, free_gpus()[0]))
           for code, init in (("genotype", "none"), ("dosage", "pre"), ("dosage", "ridge")) for m in MIXERS]
    run_cells(abl, seeds)
    tab = [(cell_name(m, "S3", "prive431", "dosage", "none", TOP, e), flags(m, "S3", "prive431", "dosage", "none", TOP, e), None)
           for e in (4, 64) for m in MIXERS]
    run_cells(tab, seeds)
    e2b = [(cell_name(m, k, "prive431", "dosage", "none", TOP), flags(m, "S3", "prive431", "dosage", "none", TOP, shape=SHAPES[k]), None)
           for k in SHAPES for m in MIXERS]
    run_cells(e2b, seeds)

    # the partition sweep and the table width at the other rungs, top-down
    rest = [(cell_name(m, "S3", p, "dosage", "none", r), flags(m, "S3", p, "dosage", "none", r), None)
            for r in down[1:] for p in PARTITIONS if p != "prive431" for m in MIXERS]
    rest += [(cell_name(m, "S3", "prive431", "dosage", "none", r, e), flags(m, "S3", "prive431", "dosage", "none", r, e), None)
             for r in down[1:] for e in (4, 64) for m in MIXERS]
    run_cells(rest, seeds)

    # score: the chosen model of every cell and its seed retrains on the tuning set, scores saved under data/
    for name in sorted(os.listdir(OUT)):
        run = f"{OUT}/{name}"
        r = results(run)
        if r is None or name in ("logs",):
            continue
        target = os.path.join(run, r["best_trial_dir"]) if r.get("tuning") else run
        if os.path.exists(f"{target}/scores_val_summary.csv"):
            continue
        rc = sh([PY, "score_snapshots.py", "--run", run, "--split", "val", "--snapshots", "best", "--save_scores",
                 "--batch_size", "1024"], f"score_{name}", gpu=free_gpus()[0], cwd=CODE)
        log(f"score {name}: exit {rc}")
    log("chain finished: every queued cell done; E5 (held-out groups), E6 (h2 scan) and E7 are separate steps")


def lin_all():
    for ds in RUNGS[::-1]:
        if lin(ds, free_gpus()[0], SPLITS):
            raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:] == ["lin_all"]:
        lin_all()
    else:
        main()
