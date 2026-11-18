#!/usr/bin/env python3
"""The experiment chain of the LD-block WideDeepGP study on the real diseases and real genotypes (design
document sections 5, 7 and 12; Robert, 17 November 2026: no simulated-trait gate, the top rung first, then down
the rungs; the side studies after the grid). One process runs every cell in order from a frozen export of
DeepWideGP's `widening` branch:

    lin        the comparators per rung in two phases. On a GPU, reserved for the build and released when the phase
               ends: the folds, the closed-form block ridge and the ridge warm start, the top rung first and the
               other rungs in turn after T0 (their warm starts are needed days later). On the CPU, beside the training: the ridge check and the
               linear twins. T0 and a rung's grid wait for that rung's warm start only; a step whose product
               exists is skipped (18 November 2026)
    t0         the pre-test: S1, S3, S5 once each at the top rung, fixed settings, from the ridge warm start, each
               on a GPU of its own as one frees; the first thing after the warm start
    grid       5 sizes x 2 mixers per rung, rung by rung from 215,872 down to 13,199 participants, on the Prive
               partition, dosage code, from the ridge warm start
    side       at the top rung: the partition sweep (ldsplit2000, ldsplit5000), the ablations (no initialisation,
               genotype code, masked pretraining, the rotation sampler with the square-root class weight), the
               table width (4 and 64), the shapes; then the partition sweep and the table width at the other rungs
    score      every finished cell's chosen model and seed retrains scored on the tuning set (--save_scores)

A cell is one run of main_multilabel.py (a 12-trial search with seed 1) under runs/blocks/<name>, retrained with
seeds 2 and 3 at the chosen settings (--tune_trials 0); the retrains of three cells run together in waves that
fill the GPUs. GPUs (Robert, 17 November 2026): as many as are free, up to three, picked at every launch; a GPU
with another session's process is not free, one with only this chain's processes is (the linear comparators
share one); ten CPU threads per GPU. The recipe (Robert, 17 November, night; design document section 12, item 11):
the ridge warm start as every cell's starting point (so each rung's linear comparators are built first), the
settings of Sigurdsson et al. 2023 fixed (learning rate 1e-4, weight decay 1e-3, dropout 0.5, batch 64, 40% input
masking, plateau schedule, early stopping after 16 passes once 2,000 steps are done), the table's own Adam group
(epsilon 2.5e-7, decay) and a search over the table's pace and decay only (SEARCH). The objective (Robert, 18 November
2026; design document section 12, item 12): plain cross-entropy at natural prevalence, uniform shuffled batches and no
class weight (OBJECTIVE); the rotation of disease-balanced batches with the square-root class weight, the objective of
every run before, is one paired ablation cell at the top rung (ROTATION). Nothing with simulated phenotypes is queued. A finished cell
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
RESERVED = {}                                             # gpu -> process that owns it alone (a comparator build needs ~45 GB)
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
DEFAULTS = {"table_kappa": 0.03, "table_decay": 1.0}   # single runs (T0); the search tunes these two (SEARCH)
SEARCH = ("table", 6)                                     # --tune_space and trials per cell; ("run", 12) tunes the network's too
INIT = "ridge"                                            # the grid's starting point; "none" is the ablation
OBJECTIVE = ["--sampler", "shuffled", "--gamma", "0"]     # plain cross-entropy at natural prevalence (18 November 2026)
ROTATION = ["--sampler", "rotation", "--gamma", "0.5"]    # the objective before: disease-balanced batches, square-root weight


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


def single(cmd, name):
    """A single run's command (--tune_trials 0), continuing from its last checkpoint when a stopped attempt left one."""
    return cmd + (["--resume"] if os.path.exists(f"{OUT}/{name}/checkpoints/last.ckpt") else [])


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


def free_gpus(limit=MAX_GPUS, exclusive=False):
    """Up to ``limit`` GPUs without another session's compute process and not reserved for a comparator build,
    the least used by this chain first (``exclusive``: none of this chain's processes either); waits
    (five-minute polls) until there is one."""
    while True:
        for gpu, proc in list(RESERVED.items()):
            if proc.poll() is not None:
                del RESERVED[gpu]
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
        free = [(ours.get(uuid.strip(), 0), idx.strip()) for idx, uuid in (r.split(",") for r in rows)
                if uuid.strip() not in foreign and idx.strip() not in RESERVED and not (exclusive and ours.get(uuid.strip()))]
        if free:
            return [idx for _, idx in sorted(free)][:limit]
        log("waiting for a GPU without another session's process")
        time.sleep(300)


# ---- the linear comparators (Scripts step 7) -------------------------------------------------------
def lin_steps(ds, phase):
    """One dataset's comparator steps as (product, command): the ``gpu`` phase (the folds, the block ridge, the
    warm start) or the ``cpu`` phase (the ridge check, the linear twins). A step whose product exists is skipped."""
    d, res = dataset(ds), f"{SCRIPTS}/results/{ds}"
    if phase == "gpu":
        return [(f"{d}/blocks/folds.npz", [PY, f"{SCRIPTS}/make_folds.py", d, "--splits", *SPLITS]),
                (f"{d}/blocks.json", [PY, f"{SCRIPTS}/build_block_ridge.py", d, "--partition", f"{PART}/prive431",
                                      "--splits", *SPLITS, "--threads", THREADS]),
                (f"{d}/blocks/init_ridge_d16.pt", [PY, f"{SCRIPTS}/make_warm_start.py", d, "--embed_dim", "16"])]
    if phase == "cpu":
        return [(f"{res}/check_block_ridge.md", [PY, f"{SCRIPTS}/check_block_ridge.py", d, "--partition", f"{PART}/prive431"]),
                (f"{res}/blocks_linear.csv", [PY, f"{SCRIPTS}/fit_block_linear.py", d])]
    raise ValueError(f"unknown phase {phase}")


def lin(ds, phase):
    """Runs the steps of one dataset's phase whose product is missing (the cpu phase once the block ridge exists);
    returns the exit code (1 when a step ends without its product)."""
    todo = [(p, cmd) for p, cmd in lin_steps(ds, phase) if not os.path.exists(p)]
    if not todo:
        return 0
    while phase == "cpu" and not os.path.exists(f"{dataset(ds)}/blocks.json"):
        if os.getppid() == 1:                                 # the chain is gone: no ridge will come
            return 1
        time.sleep(300)
    log(f"lin {ds} {phase}: starting" + (f" on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '?')}" if phase == "gpu" else ""))
    for product, cmd in todo:
        rc = sh(cmd, f"lin_{ds}_{phase}")
        if rc or not os.path.exists(product):
            log(f"lin {ds} {phase}: {os.path.basename(cmd[1])} exit {rc}, {os.path.basename(product)} "
                f"{'written' if os.path.exists(product) else 'missing'}; chain stopped")
            return rc or 1
    log(f"lin {ds} {phase}: done")
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


def flags(mixer, size, partition, code, init, rung, embed_dim=16, shape=None, objective=OBJECTIVE):
    d, layers = (shape or SIZES[size])
    n_blocks = {"prive431": 431, "ldsplit2000": 2000, "ldsplit5000": 5000}[partition]
    f = ["--data_dir", DATA, "--label", "ard30", "--rd_mode", rung, "--input_mode", "blocks", "--wide", "off",
         "--partition", f"{PART}/{partition}", "--mixer", mixer, "--snp_code", code, "--embed_dim", str(embed_dim),
         "--table_eps", "2.5e-7", "--d_model", str(d), "--e_layers", str(layers[mixer]), "--n_heads", str(max(1, d // 16)),
         "--head_hidden", str(4 * d), "--n_fcn", "2", "--final_pool", "mean", "--snp_stats", "train",
         "--lr", "1e-4", "--l2", "1e-3", "--dropout", "0.5", "--input_mask", "0.4",
         "--batch_size", "64", "--accumulate", "1", "--eval_batch_size", "1024",
         "--precision", "bf16", "--num_workers", THREADS, "--max_epoch_equiv", "20", "--val_per_epoch", "4",
         "--val_min_participants", "20000", "--val_max_epoch_equiv", "1", "--min_participants", "128000",
         "--stop_patience", "16", "--stop_burnin", "2000", "--tune_min_validations", "8", "--lr_schedule", "plateau",
         "--warmup_steps", "0", "--snapshots_per_decade", "6",
         "--snapshots_max", "20" if size == "S3" or rung == TOP else "6", "--run_dir", RUNS, "--experiment", "blocks",
         *objective]
    if size == "S5" and n_blocks >= 5000:
        f.append("--grad_ckpt")
    return f


def search(name, base_flags, init_path, trials=SEARCH[1]):
    """One cell's seed-1 search on the three GPUs; returns its results.json."""
    run = f"{OUT}/{name}"
    r = results(run)
    if r is not None:
        return r
    log(f"cell {name}: search starting")
    cmd = [PY, "main_multilabel.py", *base_flags, "--exp_name", name, "--seed", "1", "--tune_trials", str(trials),
           "--tune_space", SEARCH[0], "--tune_gpus", ",".join(free_gpus())]
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
                       "--tune_trials", "0", *[a for k, v in p.items() for a in (f"--{k}", str(v))]] + \
                      (["--init_encoder", init_path] if init_path else [])
                self.pending.append((f"{name}_seed{seed}", single(cmd, f"{name}_seed{seed}")))
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
    # the top rung's comparator GPU phase (folds, block ridge, warm start) on a GPU reserved for it and released when
    # the phase ends; the CPU phases (the ridge check, the linear twins) of every rung beside the training
    gpu = free_gpus(exclusive=True)[0]
    lin_top = spawn([PY, __file__, "lin", "gpu", "one", TOP], f"lin_{TOP}_gpu", gpu)
    RESERVED[gpu] = lin_top                                   # a build fills its GPU (eigh of a 15,000-SNP block)
    twins = [spawn([PY, __file__, "lin", "cpu", "one", TOP], f"lin_{TOP}_cpu", ""),
             spawn([PY, __file__, "lin", "cpu", "rest"], "lin_rest_cpu", "")]
    if lin_top.wait():
        raise RuntimeError(f"the block ridge or the warm start of {TOP} failed; see logs/lin_{TOP}_gpu.log")
    log(f"lin {TOP}: warm start ready")
    top_init = init_file(INIT, TOP, "prive431", 16, "dosage", free_gpus()[0])

    # t0: three single runs at the top rung from the warm start, the slowest first, each on a GPU of its own as one frees
    procs, running = [], {}
    for size in ("S5", "S3", "S1"):
        name = f"t0_transformer_{size}_prive431_{TOP}"
        if results(f"{OUT}/{name}") is not None:
            continue
        while True:
            running = {g: p for g, p in running.items() if p.poll() is None}
            gpus = [g for g in free_gpus(exclusive=True) if g not in running]      # not one this loop just filled
            if gpus:
                break
            time.sleep(300)
        cmd = [PY, "main_multilabel.py", *flags("transformer", size, "prive431", "dosage", INIT, TOP), "--exp_name", name,
               "--seed", "1", "--tune_trials", "0", *[a for k, v in DEFAULTS.items() for a in (f"--{k}", str(v))],
               "--init_encoder", top_init]
        log(f"t0 {name}: starting on GPU {gpus[0]}")
        running[gpus[0]] = spawn(single(cmd, name), f"cell_{name}", gpus[0], cwd=CODE)
        procs.append((name, running[gpus[0]]))
    for name, proc in procs:
        if proc.wait() or results(f"{OUT}/{name}") is None:
            raise RuntimeError(f"{name} failed")
        log(f"t0 {name}: tuning AUROC {results(f'{OUT}/{name}')['best_val_auroc_mean']:.4f}")

    # the other rungs' GPU phases, in turn on a GPU reserved for them: their warm starts are needed only when the grid
    # reaches those rungs, days later, so they wait until T0 is done rather than take a GPU from it
    gpu = free_gpus(exclusive=True)[0]
    lin_rest = spawn([PY, __file__, "lin", "gpu", "rest"], "lin_rest_gpu", gpu)
    RESERVED[gpu] = lin_rest

    # grid: rung by rung from the top, five sizes x two mixers, the plain configuration on the Prive partition
    for r in down:                                           # a rung's cells wait for its warm start
        if r != TOP:
            while not os.path.exists(f"{dataset(r)}/blocks/init_ridge_d16.pt"):
                if lin_rest.poll() is not None:
                    raise RuntimeError("the other rungs' block ridge or warm start failed; see logs/lin_rest_gpu.log")
                time.sleep(300)
        init_r = init_file(INIT, r, "prive431", 16, "dosage", free_gpus()[0])
        run_cells([(cell_name(m, z, "prive431", "dosage", INIT, r), flags(m, z, "prive431", "dosage", INIT, r), init_r)
                   for z in SIZES for m in MIXERS], seeds)
    if lin_rest.wait():
        raise RuntimeError("the other rungs' block ridge or warm start failed; see logs/lin_rest_gpu.log")

    # side studies at the top rung: the partition sweep (the warm-start table is per SNP, so the same file serves every
    # partition), the ablations (no initialisation, which ran first while the warm start was a day away, now here), the
    # table width, the shapes
    side = [(cell_name(m, "S3", p, "dosage", INIT, TOP), flags(m, "S3", p, "dosage", INIT, TOP), top_init)
            for p in PARTITIONS if p != "prive431" for m in MIXERS]
    run_cells(side, seeds)
    abl = [(cell_name(m, "S3", "prive431", code, init, TOP), flags(m, "S3", "prive431", code, init, TOP),
            init_file(init, TOP, "prive431", 16, code, free_gpus()[0]))
           for code, init in (("dosage", "none"), ("genotype", "none"), ("dosage", "pre")) for m in MIXERS]
    abl.append((cell_name("transformer", "S3", "prive431", "dosage", INIT, TOP) + "_rotation",      # the old objective
                flags("transformer", "S3", "prive431", "dosage", INIT, TOP, objective=ROTATION), top_init))
    run_cells(abl, seeds)
    tab = [(cell_name(m, "S3", "prive431", "dosage", INIT, TOP, e), flags(m, "S3", "prive431", "dosage", INIT, TOP, e),
            init_file(INIT, TOP, "prive431", e, "dosage", free_gpus()[0])) for e in (4, 64) for m in MIXERS]
    run_cells(tab, seeds)
    e2b = [(cell_name(m, k, "prive431", "dosage", INIT, TOP), flags(m, "S3", "prive431", "dosage", INIT, TOP, shape=SHAPES[k]), top_init)
           for k in SHAPES for m in MIXERS]
    run_cells(e2b, seeds)

    # the partition sweep and the table width at the other rungs, top-down
    for r in down[1:]:
        init_r = init_file(INIT, r, "prive431", 16, "dosage", free_gpus()[0])
        rest = [(cell_name(m, "S3", p, "dosage", INIT, r), flags(m, "S3", p, "dosage", INIT, r), init_r)
                for p in PARTITIONS if p != "prive431" for m in MIXERS]
        rest += [(cell_name(m, "S3", "prive431", "dosage", INIT, r, e), flags(m, "S3", "prive431", "dosage", INIT, r, e),
                  init_file(INIT, r, "prive431", e, "dosage", free_gpus()[0])) for e in (4, 64) for m in MIXERS]
        run_cells(rest, seeds)

    for p in twins:                                          # the ridge checks and linear twins, before the tables
        if p.wait():
            raise RuntimeError("a ridge check or linear twin failed; see logs/lin_*_cpu.log")
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


def lin_main(phase, which, ds=None):
    """``lin gpu|cpu one <rung>``: one rung's phase; ``lin gpu|cpu rest``: the other rungs' in turn, top down."""
    for r in ([ds] if which == "one" else RUNGS[::-1][1:]):
        if lin(r, phase):
            raise SystemExit(1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["lin"]:
        lin_main(*sys.argv[2:])
    else:
        main()
