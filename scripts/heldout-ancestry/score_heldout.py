#!/usr/bin/env python3
"""Score one model's finished runs on the held-out groups (methods.md section 3). Per run, in a shadow folder that
links the run's files (so its own records are never written): the tuning set again by the frozen code, whose
per-disease AUROCs must equal the run's recorded ones, then each group with per-person scores, which the model's
scorer saves beside the run's tuning-set file under data/. A record (heldout.json) per run; finished runs are
skipped; the first failure stops the model.

    python score_heldout.py --model kelemen --code <frozen code dir> [--rungs ...] [--device cuda] [--dry_run]
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("DEEPWIDEGP_PROJECT", "/mnt/sdg/robert/deepwidegp")
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(RUNS, "heldout_ancestry")
RUNGS = ["hm3plus_n13199", "hm3plus_n26287", "hm3plus_n53498", "hm3plus_n107743", "hm3plus"]
COHORTS = ["asian", "black"]
TOLERANCE = 1e-3          # per-disease tuning AUROC, scored again on a GPU (low precision is not bit-stable)
WDGP = "/home/robert/envs/widedeepgp/bin/python"
# model: its finished runs (a run is finished once its scorer wrote scores_val.csv), how its scorer is called
MODELS = {
    "kelemen": dict(runs="kelemen_mlp/ladder_prs_{rung}/*", cwd="KelemenMLP", cmd=["-m", "kmlp.score"], py=WDGP),
    "delphi": dict(runs="delphi/ladder_{rung}/*", cwd="Delphi", cmd=["-m", "dphi.score"], py=WDGP),
    "gln": dict(runs="gln/real_ladder_{rung}/*", cwd="GLN", cmd=["-m", "glnb.score"],
                py="/mnt/sdg/robert/envs/gln/bin/python"),
    "prsformer": dict(runs="prsformer/real_ladder/{rung}_[lp][ir]*", cwd="PRSformer", cmd=["-m", "prsf.score"],
                      py="/mnt/sdg/robert/envs/prsformer/bin/python"),
    "widedeepgp": dict(runs="widedeepgp_ladder/{rung}_s*", cwd="DeepWideGP", cmd=["score_snapshots.py"], py=WDGP,
                       one_split=True, extra=["--snapshots", "best", "--batch_size", "256"]),
}
# scorers that check each run's tuning set themselves (same rule, their own tolerance) and loop over its finished runs
SELF_CHECKING = {
    "classic": dict(cwd="DeepWideGP", cmd=["score_genome.py", "--all"], py=WDGP),
    "scprs": dict(cwd="scPRS", cmd=["-m", "scprs.score", "--all"], py="/mnt/sdg/robert/envs/scprs/bin/python",
                  pythonpath=["scPRS", "DeepWideGP"]),
}


def env(code, pythonpath=()):
    e = dict(os.environ, DEEPWIDEGP_PROJECT=ROOT, DEEPWIDEGP_REPO=os.path.join(code, "DeepWideGP"),
             KELEMEN_REPO=os.path.join(code, "KelemenMLP"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
             SYNTHETIC_DATA_DIR=os.path.join(ROOT, "data", "synthetic", "logs", "code_20261120"))
    e.pop("PYTHONPATH", None)
    # the frozen copy sits inside the umbrella repository, so git would name its commit: name them as the chains do
    e["SCPRS_CODE"] = e["GLN_CODE"] = ";".join(open(os.path.join(code, "COMMITS.txt")).read().split("\n")).strip(";")
    if pythonpath:
        e["PYTHONPATH"] = os.pathsep.join(os.path.join(code, p) for p in pythonpath)
    return e


def aurocs(path):
    """{disease: tuning AUROC} of a scores_<split>.csv: the model's rows, not the summary's or the covariates'."""
    return {r["label"]: float(r["value"]) for r in csv.DictReader(open(path))
            if r["metric"] == "auroc" and r["label"] != "MEAN" and r.get("snapshot", "") != "covariates_only"}


def shadow(run, model):
    """A folder of links to the run's files, minus what a scorer writes (scores_*, synthetic_*, test_access.log)."""
    d = os.path.join(OUT, "shadow", model, os.path.relpath(run, RUNS))
    os.makedirs(d, exist_ok=True)
    for name in os.listdir(run):
        if name.startswith(("scores_", "synthetic_")) or name == "test_access.log":
            continue
        link = os.path.join(d, name)
        if not os.path.lexists(link):
            os.symlink(os.path.join(run, name), link)
    return d


def call(m, code, shadow_dir, splits, device, save, log):
    groups = [[s] for s in splits] if m.get("one_split") else [splits]
    for sp in groups:
        argv = [m["py"], *m["cmd"], "--run", shadow_dir, *[a for s in sp for a in ("--split", s)], "--device", device,
                *m.get("extra", []), *(["--save_scores"] if save else [])]
        with open(log, "a") as f:
            f.write(f"$ {' '.join(argv)}\n")
            f.flush()
            if subprocess.run(argv, cwd=os.path.join(code, m["cwd"]), env=env(code), stdout=f, stderr=f).returncode:
                raise RuntimeError(f"the scorer failed ({' '.join(sp)}); see {log}")


def score_run(model, run, code, device):
    m = MODELS[model]
    d = shadow(run, model)
    done = os.path.join(d, "heldout.json")
    if os.path.exists(done):
        return "skipped"
    log = os.path.join(d, "heldout.log")
    t0 = time.time()
    call(m, code, d, ["val"], device, False, log)
    want, got = aurocs(os.path.join(run, "scores_val.csv")), aurocs(os.path.join(d, "scores_val.csv"))
    diff = max(abs(want[k] - got.get(k, float("nan"))) for k in want) if want else float("nan")
    if not want or set(got) != set(want) or not diff <= TOLERANCE:
        raise RuntimeError(f"{run}: the tuning set scored again gives other AUROCs (max difference {diff:.2e})")
    call(m, code, d, COHORTS, device, True, log)
    with open(done + ".tmp", "w") as f:
        json.dump({"model": model, "run": run, "code": open(os.path.join(code, "COMMITS.txt")).read().split("\n"),
                   "cohorts": COHORTS, "tuning_auroc_max_difference": diff, "device": device,
                   "seconds": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  f, indent=1)
    os.replace(done + ".tmp", done)
    return f"scored (tuning AUROC within {diff:.1e} of the record)"


def finished_runs(model, rungs):
    pats = [os.path.join(RUNS, MODELS[model]["runs"].format(rung=r)) for r in rungs]
    runs = [r for p in pats for r in sorted(glob.glob(p)) if os.path.exists(os.path.join(r, "scores_val.csv"))]
    if len(set(runs)) != len(runs):
        raise SystemExit(f"{model}: a run matches two rungs' patterns")
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS) + sorted(SELF_CHECKING))
    ap.add_argument("--code", required=True, help="a frozen copy (freeze.sh) holding the repos and COMMITS.txt")
    ap.add_argument("--rungs", nargs="+", default=RUNGS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry_run", action="store_true", help="list the finished runs and what would be done")
    a = ap.parse_args(argv)
    if a.model in SELF_CHECKING:
        m = SELF_CHECKING[a.model]
        argv = [m["py"], *m["cmd"], "--device", a.device]
        print(f"{a.model}: $ {' '.join(argv)}", flush=True)
        if a.dry_run:
            return 0
        return subprocess.run(argv, cwd=os.path.join(a.code, m["cwd"]), env=env(a.code, m.get("pythonpath", ()))).returncode
    runs = finished_runs(a.model, a.rungs)
    print(f"{a.model}: {len(runs)} finished runs", flush=True)
    for run in runs:
        if a.dry_run:
            done = os.path.exists(os.path.join(OUT, "shadow", a.model, os.path.relpath(run, RUNS), "heldout.json"))
            print(f"  {os.path.relpath(run, RUNS)}: {'done' if done else 'to score'}")
            continue
        print(f"  {os.path.relpath(run, RUNS)}: {score_run(a.model, run, a.code, a.device)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
