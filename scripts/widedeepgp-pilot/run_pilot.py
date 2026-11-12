#!/usr/bin/env python3
"""Run the WideDeepGP pilot: eight training schemes on the proxy dataset, two seeds each.

One run per GPU at a time. Whenever a run is to be started, the script counts the GPUs with
nothing running on them and starts it only if that leaves one of them free, so one GPU is
always left for other users (--gpus names a fixed set instead). --max-gpus caps the number of
simultaneous runs; a number written to <runs>/<experiment>/max_gpus changes the cap while the
pilot is running. Runs that already have a results.json are skipped, so the same command
resumes an interrupted pilot. Arms are queued seed by seed, so that after the first wave every
arm has one finished run.

    python scripts/widedeepgp-pilot/run_pilot.py                     # all arms
    python scripts/widedeepgp-pilot/run_pilot.py --arms default bf16 # some arms
    python scripts/widedeepgp-pilot/run_pilot.py --dry               # print the commands only

What the arms are and why is in methods.md next to this script.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEEPGP_REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
PY = {"fork": "/home/robert/envs/widedeepgp/bin/python",              # the published block (Vision-Mamba fork)
      "m3": "/mnt/sdg/robert/envs/widedeepgp-m3/bin/python"}          # upstream mamba_ssm with Mamba-3
DATA = "/mnt/sdg/robert/deepwidegp/data/datasets"
RUNS = "/mnt/sdg/robert/deepwidegp/runs"

# arm -> (environment, extra training flags). "default" is the planned configuration; every
# other arm changes one thing.
ARMS = {
    "default": ("fork", []),
    "target_only": ("fork", ["--loss_outputs", "target"]),
    "shuffled": ("fork", ["--sampler", "shuffled"]),
    "no_correction": ("fork", ["--no_sampling_correction"]),
    "gamma1": ("fork", ["--gamma", "1.0"]),
    "bf16": ("fork", ["--precision", "bf16"]),
    "annot": ("fork", ["--annot"]),
    "mamba3": ("m3", ["--mixer", "mamba3"]),
}
COMMON = ["--label", "ard30", "--snp_stats", "train", "--final_pool", "atten",
          "--d_model", "32", "--e_layers", "1", "--n_heads", "4", "--batch_size", "16", "--accumulate", "4",
          "--lr", "7e-4", "--l2", "1e-4", "--dropout", "0.2", "--eval_batch_size", "128", "--num_workers", "4",
          "--val_per_epoch", "2", "--stop_patience", "0"]


def free_gpus():
    """Indices of GPUs with no compute process and (almost) no memory in use."""
    q = lambda *a: subprocess.run(["nvidia-smi", *a, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                  check=True).stdout.strip().splitlines()
    busy = {line.split(",")[0].strip() for line in q("--query-compute-apps=gpu_uuid,pid") if line.strip()}
    out = []
    for line in q("--query-gpu=index,uuid,memory.used"):
        idx, uuid, used = [x.strip() for x in line.split(",")]
        if uuid not in busy and float(used) < 200:
            out.append(int(idx))
    return out


def command(arm, seed, a):
    env, extra = ARMS[arm]
    return [PY[env], "main_multilabel.py", "--data_dir", a.data, "--run_dir", a.runs, "--experiment", a.experiment,
            "--exp_name", f"{arm}_s{seed}", "--seed", str(seed), "--max_epoch_equiv", str(a.epoch_equiv),
            "--rd_mode", a.rd_mode, *COMMON, *extra]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--epoch-equiv", type=float, default=6.0, help="training budget per run, in epoch-equivalents")
    ap.add_argument("--experiment", default="pilot")
    ap.add_argument("--rd-mode", dest="rd_mode", default="hm3plus_proxy8",
                    help="dataset variant under <data>/ard30 (hm3plus_proxy8_sim: the simulated-trait control)")
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--gpus", nargs="+", type=int, default=None, help="a fixed set of GPU indices (default: free ones, leaving one free)")
    ap.add_argument("--max-gpus", type=int, default=99, help="at most this many runs at a time")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)

    jobs = [(arm, seed) for seed in a.seeds for arm in a.arms]
    exp_dir = Path(a.runs) / a.experiment
    todo = [(arm, s) for arm, s in jobs if not (exp_dir / f"{arm}_s{s}" / "results.json").exists()]
    print(f"{len(jobs)} runs, {len(jobs) - len(todo)} already finished", flush=True)
    if a.dry:
        for arm, s in todo:
            print(" ".join(command(arm, s, a)))
        return
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)

    def cap():
        try:
            return int((exp_dir / "max_gpus").read_text().strip())
        except (OSError, ValueError):
            return a.max_gpus

    def usable():
        """GPUs a new run may be started on now."""
        if a.gpus is not None:
            return [g for g in a.gpus if g not in running]
        free = [g for g in free_gpus() if g not in running]
        return free[:-1]                                 # leave one free GPU for other users

    running, failed, t0 = {}, [], time.time()           # gpu -> (process, arm, seed, log file, start time)
    while todo or running:
        for gpu in usable():
            if not todo or len(running) >= cap():
                break
            arm, s = todo.pop(0)
            log = open(exp_dir / "logs" / f"{arm}_s{s}.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            p = subprocess.Popen(command(arm, s, a), cwd=DEEPGP_REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = (p, arm, s, log, time.time())
            print(f"[{(time.time() - t0) / 60:6.1f} min] started {arm}_s{s} on GPU {gpu}", flush=True)
            time.sleep(45)                               # let it appear on the GPU before counting free GPUs again
        time.sleep(15)
        for gpu, (p, arm, s, log, t) in list(running.items()):
            if p.poll() is None:
                continue
            log.close()
            del running[gpu]
            res = exp_dir / f"{arm}_s{s}" / "results.json"
            if p.returncode == 0 and res.exists():
                r = json.load(open(res))
                print(f"[{(time.time() - t0) / 60:6.1f} min] finished {arm}_s{s}: best mean tuning AUROC "
                      f"{r['best_val_auroc_mean']:.4f}, {r['epoch_equivalents']:.2f} epoch-equivalents in "
                      f"{(time.time() - t) / 60:.1f} min", flush=True)
            else:
                failed.append(f"{arm}_s{s}")
                print(f"[{(time.time() - t0) / 60:6.1f} min] FAILED {arm}_s{s} (exit {p.returncode}); "
                      f"see {exp_dir / 'logs' / f'{arm}_s{s}.log'}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min; failed: {failed or 'none'}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
