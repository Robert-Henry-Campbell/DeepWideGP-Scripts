#!/usr/bin/env python3
"""The overnight table: every run under runs/tweaks (and the searches' chosen trials) against the yardsticks.
    python summarise.py [--out results_table.md]
Columns: run, what changed, best tuning AUROC (mean of 30), at epoch-equivalents, gain over the base run, the streamed
re-score of the best snapshot (an independent check), I10 and E11, lift over the covariates alone and over the PGI the
first layer was built from (its training-row logistic fit with the covariates, pgi_yardstick.py), and the run's wall time.
Aggregates only."""
import argparse
import csv
import glob
import json
import os

ROOT = "/mnt/sdg/robert/deepwidegp/runs/tweaks"
BASE = 0.6735                       # blocks_diag/diag_S3_folds_frozen_nomask seed 1 (3-seed mean 0.6737)
FLOOR = 0.6627                      # covariates alone, training-split logistic fit scored on the tuning set
YARD = {"ridge": 0.6777, "twin": 0.6732}   # cross-fitted block ridge PRS + covariates; logistic ridge on the block channels


def read_results(d):
    p = os.path.join(d, "results.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        r = json.load(f)
    if r.get("tuning"):                                 # a search: its chosen trial
        t = os.path.join(d, r["best_trial_dir"])
        inner = read_results(t) or {}
        inner.update(search=True, best_params=r.get("best_params"), n_trials=sum(s["n_trials"] for s in r["stages"].values()),
                     search_seconds=r.get("train_seconds_all_trials"))
        return inner
    return r


def rescored(d):
    p = os.path.join(d, "scores_val_summary.csv")
    if not os.path.exists(p):
        return None, {}
    rows = list(csv.DictReader(open(p)))
    per = {r["label"]: float(r["auroc"]) for r in rows}
    return sum(per.values()) / len(per), per


EPOCHS = [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40, 50]


def trajectory(d):
    """{epoch-equivalent: mean AUROC} of a run's validations, keyed by the nearest value of EPOCHS (the last one at a key wins)."""
    p = os.path.join(d, "metrics.csv")
    if not os.path.exists(p):
        r = os.path.join(d, "results.json")
        if os.path.exists(r):
            rr = json.load(open(r))
            if rr.get("tuning"):
                p = os.path.join(d, rr["best_trial_dir"], "metrics.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    for row in csv.DictReader(open(p)):
        if row["label"] == "MEAN" and row["metric"] == "auroc":
            ee = float(row["epoch_equiv"])
            key = min(EPOCHS, key=lambda e: abs(e - ee))
            if abs(key - ee) <= 0.13:
                out[key] = float(row["value"])
    return out


def yardsticks():
    out = {}
    for p in glob.glob(os.path.join(ROOT, "yardsticks", "*_summary.json")):
        with open(p) as f:
            out[os.path.basename(p).split("_summary")[0]] = json.load(f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results_table.md"))
    a = ap.parse_args(argv)
    notes = {}
    for job in glob.glob(os.path.join(ROOT, "queue", "*", "*.sh")):
        with open(job) as f:
            lines = [l[2:].strip() for l in f if l.startswith("# ") and not l.startswith("# slots") and not l.startswith("# after")]
        name = os.path.basename(job)[:-3].split("_", 1)[1]
        notes[name] = " ".join(lines)
    yard = yardsticks()
    rows = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
        if not os.path.isdir(d) or os.path.basename(d).startswith(("code_", "queue", "worktree", "yardsticks", "tests_log")):
            continue
        r = read_results(d)
        name = os.path.basename(d)
        if r is None:
            rows.append({"run": name, "state": "running or failed"})
            continue
        auc = r["best_val_auroc_mean"]
        ee = r["best_samples_seen"] / 215872.0
        re_mean, per = rescored(d)
        cfg_path = os.path.join(d if not r.get("search") else os.path.join(d, json.load(open(os.path.join(d, "results.json")))["best_trial_dir"]), "config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        init = os.path.basename(cfg.get("model", {}).get("init_encoder") or "")
        pgi = None
        for key, y in yard.items():
            if key in init:
                pgi = y["mean_combined_auroc"]
        rows.append({"run": name, "state": "done" if os.path.exists(os.path.join(d, "scores_val_summary.csv")) or r.get("search") else "trained",
                     "auroc": auc, "ee": ee, "gain_base": auc - BASE, "rescored": re_mean, "I10": per.get("I10"), "E11": per.get("E11"),
                     "lift_cov": auc - FLOOR, "lift_pgi": None if pgi is None else auc - pgi, "pgi_yard": pgi,
                     "wall_min": (r.get("wall_seconds") or 0) / 60, "stopped_early": r.get("stopped_early"),
                     "params": r.get("best_params"), "note": notes.get(name, "")})
    def num(v, signed=False):
        return "" if v is None else (f"{v:+.4f}" if signed else f"{v:.4f}")

    with open(a.out, "w") as f:
        f.write("| run | what | tuning AUROC | at ee | vs base 0.6735 | re-scored | I10 | E11 | over covariates | over its PGI | min |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            if "auroc" not in r:
                f.write(f"| {r['run']} | {notes.get(r['run'], '')[:120]} | {r['state']} | | | | | | | | |\n")
                continue
            early = " (stopped early)" if r["stopped_early"] else ""
            verdict = "helped" if r["gain_base"] > 0.0005 else "hurt" if r["gain_base"] < -0.0005 else "no change"
            f.write(f"| {r['run']} | {r['note'][:120]} | {num(r['auroc'])}{early} | {r['ee']:.1f} | {num(r['gain_base'], True)} ({verdict}) | "
                    f"{num(r['rescored'])} | {num(r['I10'])} | {num(r['E11'])} | {num(r['lift_cov'], True)} | "
                    f"{num(r['lift_pgi'], True)} | {r['wall_min']:.0f} |\n")
            if r.get("params"):
                f.write(f"| | chosen settings: {json.dumps(r['params'])} | | | | | | | | | |\n")
        # a second table: the tuning AUROC by epoch-equivalent (Robert wants trajectories, not only the best pass)
        f.write("\n\nTuning AUROC (mean of 30) by epoch-equivalent, every validation rounded to the nearest quarter; "
                "base run: 1: 0.6664, 2: 0.6709, 3: 0.6714, 4: 0.6721, 5: 0.6724, 6: 0.6731, 7: 0.6730, 8: 0.6729, 9: 0.6733, 10: 0.6735\n\n")
        f.write("| run | " + " | ".join(str(e) for e in EPOCHS) + " |\n|---|" + "---|" * len(EPOCHS) + "\n")
        for r in rows:
            traj = trajectory(os.path.join(ROOT, r["run"]))
            if traj:
                f.write(f"| {r['run']} | " + " | ".join(f"{traj[e]:.4f}" if e in traj else "" for e in EPOCHS) + " |\n")
    with open(a.out.replace(".md", ".json"), "w") as f:
        json.dump({"rows": rows, "yardsticks": {"base": BASE, "covariates": FLOOR, **YARD, "pgi": yard}}, f, indent=1)
    print(open(a.out).read())


if __name__ == "__main__":
    main()
