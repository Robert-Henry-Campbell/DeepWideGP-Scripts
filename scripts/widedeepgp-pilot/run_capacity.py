#!/usr/bin/env python3
"""Largest batch, memory and step time of WideDeepGP at the real dataset's shape, for the
current sequence block and for Mamba-3, in 32-bit and bf16, with and without per-chromosome
checkpointing.

For each configuration the batch size is doubled from --start until a training step no longer
fits, then bisected; every trial is a separate process (capacity_probe.py). Trials are appended
to results/capacity_trials.csv, so an interrupted search resumes where it stopped.

    python scripts/widedeepgp-pilot/run_capacity.py                 # all configurations
    python scripts/widedeepgp-pilot/run_capacity.py --only mamba3   # those whose name contains "mamba3"

GPUs: the free ones minus one at launch (--gpus overrides), one configuration per GPU at a time.

A GPU looks idle between two trials. While run_pilot.py is also running, give this search the
highest-numbered GPU (--gpus 4): the pilot launcher always leaves the last free GPU alone, so
it never starts a run there. A trial that shares its GPU measures the wrong limit and speed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_pilot import DEEPGP_REPO, PY, free_gpus  # noqa: E402

TRIALS = HERE / "results" / "capacity_trials.csv"
FIELDS = ["config", "mixer", "precision", "grad_ckpt", "annot", "mode", "d_model", "e_layers", "batch", "ok", "error", "finite",
          "peak_gib", "reserved_gib", "step_s", "first_step_s", "samples_per_s", "n_parameters", "n_snps", "longest",
          "gpu", "gpu_total_gib"]

# name -> probe flags. All at model width 32, one layer, the full 1,258,200 SNPs.
CONFIGS = {
    "bimamba1_fp32": ["--mixer", "bimamba1", "--precision", "32"],
    "bimamba1_bf16": ["--mixer", "bimamba1", "--precision", "bf16"],
    "bimamba1_fp32_ckpt": ["--mixer", "bimamba1", "--precision", "32", "--grad_ckpt"],
    "bimamba1_bf16_ckpt": ["--mixer", "bimamba1", "--precision", "bf16", "--grad_ckpt"],
    "bimamba1_fp32_annot": ["--mixer", "bimamba1", "--precision", "32", "--annot"],
    "bimamba1_fp32_eval": ["--mixer", "bimamba1", "--precision", "32", "--mode", "eval"],
    "bimamba1_bf16_eval": ["--mixer", "bimamba1", "--precision", "bf16", "--mode", "eval"],
    "mamba3_fp32": ["--mixer", "mamba3", "--precision", "32"],
    "mamba3_bf16": ["--mixer", "mamba3", "--precision", "bf16"],
    "mamba3_fp32_ckpt": ["--mixer", "mamba3", "--precision", "32", "--grad_ckpt"],
    "mamba3_bf16_ckpt": ["--mixer", "mamba3", "--precision", "bf16", "--grad_ckpt"],
    "mamba3_fp32_eval": ["--mixer", "mamba3", "--precision", "32", "--mode", "eval"],
}
LOCK = threading.Lock()


def done_trials():
    if not TRIALS.exists():
        return {}
    with open(TRIALS) as f:
        return {(r["config"], int(r["batch"])): r["ok"] == "True" for r in csv.DictReader(f)}


def trial(name, batch, gpu, timeout):
    """Run one probe; returns True if the step fitted. The result row is appended to the table."""
    known = done_trials()
    if (name, batch) in known:
        return known[(name, batch)]
    flags = CONFIGS[name]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), DEEPGP_REPO=DEEPGP_REPO)
    py = PY["m3" if "mamba3" in flags else "fork"]
    cmd = [py, str(HERE / "capacity_probe.py"), *flags, "--batch", str(batch)]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=DEEPGP_REPO)
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
        if not lines:
            raise RuntimeError(f"{name} batch {batch}: no result (exit {p.returncode}): {p.stderr[-600:]}")
        r = json.loads(lines[-1][7:])
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{name} batch {batch}: no result within {timeout} s")
    r["config"] = name
    with LOCK:
        new = not TRIALS.exists()
        with open(TRIALS, "a", newline="") as f:
            w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(r)
        print(f"  {name:22} batch {batch:4d}: " + (f"fits, peak {r['peak_gib']:.1f} GiB, {r['step_s']:.2f} s/step"
                                                  if r["ok"] else (r.get("error") or "does not fit")), flush=True)
    return bool(r["ok"])


def search(name, gpu, start, cap, timeout):
    lo, hi, b = 0, None, start                       # lo: largest batch known to fit; hi: smallest known not to
    while hi is None and b <= cap:
        if trial(name, b, gpu, timeout):
            lo, b = b, b * 2
        else:
            hi = b
    if hi is None:
        return lo
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if trial(name, mid, gpu, timeout):
            lo = mid
        else:
            hi = mid
    return lo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="run the configurations whose name contains this text")
    ap.add_argument("--start", type=int, default=8)
    ap.add_argument("--cap", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=1800, help="seconds allowed per trial (the first Mamba-3 call compiles)")
    ap.add_argument("--gpus", nargs="+", type=int, default=None)
    a = ap.parse_args(argv)
    names = [n for n in CONFIGS if not a.only or a.only in n]
    if a.gpus is None:
        free = free_gpus()
        a.gpus = free[:-1]
        print(f"free GPUs {free}; using {a.gpus}", flush=True)
    if not a.gpus:
        sys.exit("no GPU to use: fewer than two are free")
    TRIALS.parent.mkdir(exist_ok=True)
    q, best, errors, t0 = queue.Queue(), {}, [], time.time()
    for n in names:
        q.put(n)

    def worker(gpu):
        while True:
            try:
                n = q.get_nowait()
            except queue.Empty:
                return
            try:
                best[n] = search(n, gpu, a.start, a.cap, a.timeout)
                print(f"{n}: largest batch {best[n]}  [{(time.time() - t0) / 60:.1f} min]", flush=True)
            except Exception as e:                   # keep the other configurations going
                errors.append(f"{n}: {e}")
                print(f"{n}: FAILED {e}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(json.dumps(best, indent=1))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
