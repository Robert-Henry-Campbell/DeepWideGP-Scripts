#!/usr/bin/env python3
"""One capacity trial of WideDeepGP at the real dataset's shape: does a training step at this
batch size fit on the GPU, how much memory does it take and how long?

Inputs are random numbers with the shapes of the real data (22 chromosomes with the SNP counts
of the genotype store, 12 covariates, 30 labels); no participant data is read. A trial is one
process, so a failed allocation leaves nothing behind. Prints one JSON line.

    CUDA_VISIBLE_DEVICES=<gpu> python capacity_probe.py --mixer bimamba1 --precision 32 --batch 18
    CUDA_VISIBLE_DEVICES=<gpu> <m3 python> capacity_probe.py --mixer mamba3 --precision bf16 --grad_ckpt --batch 64

--mode train  forward, loss, backward and optimiser step (the default)
--mode eval   forward only, without gradients (what validation and scoring do)
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time

import numpy as np

DEEPGP_REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
STORE = "/mnt/sdg/robert/deepwidegp/data/genostore/hm3plus"
N_LABELS, N_COV, N_ANNOT = 30, 12, 97


def sequence_lengths(store, fraction=None):
    """SNPs per chromosome in the store; ``fraction`` = (k, 8) takes the k-th eighth of each, as the proxy does."""
    out = []
    for c in range(1, 23):
        n = int(json.load(open(os.path.join(store, f"chr{c}.json")))["n_variants"])
        if fraction:
            k, parts = fraction
            n = (k + 1) * n // parts - k * n // parts
        out.append(n)
    return out


def build(a, seq_len):
    sys.path.insert(0, DEEPGP_REPO)
    from args_generator import args_initial
    from widedeepgp.model import WideDeepGP
    from widedeepgp.train import model_hparams

    argv = ["--label", "x", "--rd_mode", "x", "--dm", "snps_covs", "--snp_embed", "cov", "--final_pool", "atten",
            "--d_model", str(a.d_model), "--e_layers", str(a.e_layers), "--n_heads", "4", "--mixer", a.mixer,
            "--precision", a.precision] + (["--grad_ckpt"] if a.grad_ckpt else [])
    cfg = args_initial(argv)
    rng = np.random.default_rng(0)

    class Shape:                                     # what model_hparams reads off a data module
        multilabel, data_mode, n_cov = True, "snps_covs", N_COV
        label_names = [f"d{i}" for i in range(N_LABELS)]
        pos_chr = [np.arange(n) for n in seq_len]
        splits = {"train": {"y": (rng.random((5000, N_LABELS)) < 0.08).astype(np.float32)}}

    Shape.seq_len = seq_len
    annot_names = [f"a{i}" for i in range(N_ANNOT)] if a.annot else None
    annot = [rng.standard_normal((n, N_ANNOT), dtype=np.float32) for n in seq_len] if a.annot else None
    # the annotations' digest is passed where model_hparams records it; the code frozen for the
    # pilot (DEEPGP_REPO=runs/pilot_code_8f03ca6) predates that argument
    kw = {"annot": annot} if "annot" in inspect.signature(model_hparams).parameters else {}
    return WideDeepGP(model_hparams(cfg, Shape, None, annot_names, **kw), Shape.pos_chr, annot)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mixer", default="bimamba1")
    ap.add_argument("--precision", default="32")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--annot", action="store_true")
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--e_layers", type=int, default=1)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4, help="steps to run; the first is a warm-up and is not timed")
    ap.add_argument("--mode", default="train", choices=["train", "eval"])
    ap.add_argument("--proxy-eighth", type=int, default=None, help="use the k-th eighth of each chromosome instead")
    ap.add_argument("--store", default=STORE)
    a = ap.parse_args(argv)

    import torch
    seq_len = sequence_lengths(a.store, (a.proxy_eighth, 8) if a.proxy_eighth is not None else None)
    out = {"mixer": a.mixer, "precision": a.precision, "grad_ckpt": bool(a.grad_ckpt), "annot": bool(a.annot),
           "d_model": a.d_model, "e_layers": a.e_layers, "batch": a.batch, "mode": a.mode, "n_snps": int(sum(seq_len)),
           "longest": int(max(seq_len)), "gpu": torch.cuda.get_device_name(0),
           "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)}
    try:
        torch.manual_seed(0)
        model = build(a, seq_len).cuda()
        out["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
        g = torch.Generator(device="cuda").manual_seed(0)
        x = [torch.randn(a.batch, n, 1, device="cuda", generator=g) for n in seq_len]
        x.append(torch.randn(a.batch, N_COV, device="cuda", generator=g))
        y = (torch.rand(a.batch, N_LABELS, device="cuda", generator=g) < 0.08).float()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model.train(a.mode == "train")
        torch.cuda.reset_peak_memory_stats()
        times, finite = [], True
        for step in range(a.steps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            if a.mode == "train":
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss, _ = model.loss_fn(logits, y, step % N_LABELS)
                loss.backward()
                finite &= bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                                                              if p.grad is not None)
                opt.step()
            else:
                with torch.no_grad():
                    logits = model(x)
                finite &= bool(torch.isfinite(logits).all())
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        timed = times[1:] or times
        out.update(ok=True, finite=finite, peak_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 3),
                   reserved_gib=round(torch.cuda.max_memory_reserved() / 2 ** 30, 3),
                   first_step_s=round(times[0], 3), step_s=round(float(np.median(timed)), 4),
                   samples_per_s=round(a.batch / float(np.median(timed)), 3))
    except torch.cuda.OutOfMemoryError:
        out.update(ok=False, oom=True)
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            out.update(ok=False, oom=True)
        elif "illegal memory access" in msg:
            # seen with Mamba-3's kernels when batch x sequence length is very large: not a
            # memory limit of the GPU but an input size the kernel cannot index
            out.update(ok=False, oom=False, error="kernel: illegal memory access")
        else:
            raise
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
