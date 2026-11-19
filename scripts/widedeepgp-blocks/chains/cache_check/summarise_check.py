"""Summary of the first-layer cache check (aggregates only): cached re-score of the original best run, the cached
1-epoch run against the original's first validations, its streamed re-score, and wall time per step.
    python summarise_check.py <original run> <cached run> <check_rescore.json> <out.md>"""
import csv
import glob
import json
import os
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

orig, new, rescore_json, out_md = sys.argv[1:5]


def wall_per_step(run):
    """Median wall seconds per training step between logged train/loss events."""
    steps, walls = [], []
    for f in glob.glob(os.path.join(run, "tb", "**", "events*"), recursive=True):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if "train/loss" in ea.Tags()["scalars"]:
            ev = ea.Scalars("train/loss")
            steps += [e.step for e in ev]
            walls += [e.wall_time for e in ev]
    o = np.argsort(walls)
    s, w = np.diff(np.asarray(steps)[o]), np.diff(np.asarray(walls)[o])
    ok = s > 0
    return float(np.median(w[ok] / s[ok]))


ro, rn = (json.load(open(os.path.join(r, "results.json"))) for r in (orig, new))
cn = json.load(open(os.path.join(new, "config.json")))
rs = json.load(open(rescore_json))
by_step = {h["step"]: h["val_auroc_mean"] for h in ro["history"]}
pairs = [(h["step"], h["val_auroc_mean"], by_step.get(h["step"])) for h in rn["history"]]
with open(os.path.join(new, "scores_val.csv")) as f:
    streamed = {r["label"]: float(r["value"]) for r in csv.DictReader(f)
                if r["snapshot"] == "best" and r["metric"] == "auroc" and r["label"] != "MEAN"}
gap = max(abs(streamed[k] - v["auroc"]) for k, v in rn["val"].items())
w_orig, w_new = wall_per_step(orig), wall_per_step(new)
lines = [
    "# First-layer cache: real-data check (top rung, frozen fold-routed S3, no masking)",
    "",
    "| Check | Result |",
    "|---|---|",
    f"| Original best run re-scored from the cache (tuning AUROC, mean of 30) | {rs['cached_auroc_mean']:.4f} against "
    f"{rs['recorded_auroc_mean']:.4f} recorded; largest per-disease difference {rs['max_abs_diff_per_disease']:.1e} |",
    "| Cached 1-epoch run against the original, same steps (mean tuning AUROC) | " +
    "; ".join(f"step {s}: {a:.4f} vs {b:.4f}" if b is not None else f"step {s}: {a:.4f} vs n/a" for s, a, b in pairs) + " |",
    f"| Cached run's best snapshot re-scored by streaming | largest per-disease difference from its record {gap:.1e} |",
    f"| Building the cache (training + tuning sets) | {cn['first_layer_cache_seconds']:.0f} s in the run; "
    f"{rs['cache_build_seconds']:.0f} s in the re-score; {rs['cache_gib']:.2f} GiB |",
    f"| Wall time per training step (median) | {w_new:.3f} s cached against {w_orig:.3f} s streamed |",
    f"| One check on the tuning set | {rs['cached_tuning_pass_seconds']:.1f} s cached |",
    f"| Cached run, 1 epoch-equivalent | {rn['wall_seconds'] / 60:.1f} min wall (cache build included); original "
    f"10 epoch-equivalents {ro['wall_seconds'] / 3600:.2f} h |",
]
open(out_md, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
