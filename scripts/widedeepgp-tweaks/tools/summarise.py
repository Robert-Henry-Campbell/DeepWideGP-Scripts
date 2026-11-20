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


PGI_PROTOCOL = {"lassosum2": 0.6829, "ldpred2_auto": 0.6873}     # the PGI arm's combination with covariates, refit on the tuning set


def reference(name, cfg, per, yard, results):
    """What a run is read against: the base for full-rung ridge-table runs; its own frozen-table control for a run on a
    PGI table; the covariates alone for a one-disease classic run. Returns (value, label)."""
    labels = cfg.get("model", {}).get("label_names") or []
    init = os.path.basename(cfg.get("model", {}).get("init_encoder") or "")
    ds = os.path.basename(cfg.get("dataset_dir") or "")
    if len(labels) == 1:
        v = per.get("covariates_only")
        return v, "covariates alone"
    if "pgi" in init:
        method = "lassosum2" if "lassosum2" in init else "ldpred2_auto"
        ctrl = f"pgi_{'lasso' if method == 'lassosum2' else 'ldpred'}_frozen" + ("_c107" if "c107743" in ds else "")
        if name == ctrl:
            return None, "it is the control"
        r = results.get(ctrl)
        return (r["best_val_auroc_mean"] if r else None), f"its frozen-table control {ctrl}"
    return BASE, "the base 0.6735"


def pgi_yardstick(cfg, yard):
    """Lift over the naive PGI the table was built from: the PGI arm's protocol figure for a full-rung table, the
    training-row logistic fit for the half-rung tables (yardsticks/c107_*)."""
    init = os.path.basename(cfg.get("model", {}).get("init_encoder") or "")
    ds = os.path.basename(cfg.get("dataset_dir") or "")
    if "pgi" not in init:
        return None
    method = "lassosum2" if "lassosum2" in init else "ldpred2_auto"
    if "c107743" in ds:
        y = yard.get("c107_" + method)
        return y["mean_combined_auroc"] if y else None
    return PGI_PROTOCOL[method]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results_table.md"))
    a = ap.parse_args(argv)
    notes = {}
    for job in glob.glob(os.path.join(ROOT, "queue", "*", "*.sh")):
        with open(job) as f:
            lines = [l[2:].strip() for l in f if l.startswith("# ") and not l.startswith("# slots") and not l.startswith("# after")]
        name = os.path.basename(job)[:-3].split("_", 1)[1]
        notes[name] = " ".join(lines).replace("leak-free", "half-rung-weight").replace("Leak-free", "Half-rung-weight")
    for job, run in (("e11_classic", "classic_e11_r07_balanced"), ("e11_classic_r01", "classic_e11_r01_balanced")):
        if job in notes:                                 # these jobs' run folders are not named after the job
            notes[run] = notes[job]
    yard = yardsticks()
    results = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
        if os.path.isdir(d) and not os.path.basename(d).startswith(("code_", "queue", "worktree", "yardsticks", "tests_log")):
            r = read_results(d)
            if r:
                results[os.path.basename(d)] = r
    rows = []
    for name in sorted(results) + sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d)
                                           and os.path.basename(d) not in results and not os.path.basename(d).startswith(("code_", "queue", "worktree", "yardsticks", "tests_log"))):
        d = os.path.join(ROOT, name)
        r = results.get(name)
        if r is None:
            rows.append({"run": name, "state": "running or failed"})
            continue
        auc = r["best_val_auroc_mean"]
        cfg_dir = d if not r.get("search") else os.path.join(d, json.load(open(os.path.join(d, "results.json")))["best_trial_dir"])
        cfg = json.load(open(os.path.join(cfg_dir, "config.json"))) if os.path.exists(os.path.join(cfg_dir, "config.json")) else {}
        n_train = cfg.get("n_train", 215872)
        ee = r["best_samples_seen"] / float(n_train)
        re_mean, per = rescored(d)
        if os.path.exists(os.path.join(d, "scores_val_summary.csv")):
            cov = [float(x["auroc_covariates_only"]) for x in csv.DictReader(open(os.path.join(d, "scores_val_summary.csv")))]
            per["covariates_only"] = sum(cov) / len(cov)
        ref, ref_label = reference(name, cfg, per, yard, results)
        pgi = pgi_yardstick(cfg, yard)
        rows.append({"run": name, "state": "done" if os.path.exists(os.path.join(d, "scores_val_summary.csv")) or r.get("search") else "trained",
                     "auroc": auc, "ee": ee, "ref": ref, "ref_label": ref_label, "gain_base": None if ref is None else auc - ref,
                     "rescored": re_mean, "I10": per.get("I10"), "E11": per.get("E11"),
                     "lift_cov": None if per.get("covariates_only") is None else auc - per["covariates_only"],
                     "lift_pgi": None if pgi is None else auc - pgi, "pgi_yard": pgi,
                     "wall_min": (r.get("wall_seconds") or 0) / 60, "stopped_early": r.get("stopped_early"),
                     "params": r.get("best_params"), "note": notes.get(name, "")})
    def num(v, signed=False):
        return "" if v is None else (f"{v:+.4f}" if signed else f"{v:.4f}")

    with open(a.out, "w") as f:
        f.write("Tuning AUROC, mean of 30 diseases (one disease for the E11 classic runs), best validation pass. 'vs reference' is the "
                "base (0.6735) for runs on the ridge table at the full rung, the run's own frozen-table control for a run on a PGI table, "
                "and the covariates alone for a one-disease run; 'over its PGI' is the lift over the naive score the table was built from "
                "(the PGI arm's figure for the full-rung tables; the training-row logistic fit for the half-rung tables).\n\n"
                "| run | what | tuning AUROC | at ee | vs reference | reference | re-scored | I10 | E11 | over covariates | over its PGI | min |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            if "auroc" not in r:
                f.write(f"| {r['run']} | {notes.get(r['run'], '')[:120]} | {r['state']} | | | | | | | | | |\n")
                continue
            early = " (stopped early)" if r["stopped_early"] else ""
            g = r["gain_base"]
            verdict = "" if g is None else " (helped)" if g > 0.0005 else " (hurt)" if g < -0.0005 else " (no change)"
            f.write(f"| {r['run']} | {r['note'][:120]} | {num(r['auroc'])}{early} | {r['ee']:.1f} | {num(g, True)}{verdict} | {r['ref_label']} | "
                    f"{num(r['rescored'])} | {num(r['I10'])} | {num(r['E11'])} | {num(r['lift_cov'], True)} | "
                    f"{num(r['lift_pgi'], True)} | {r['wall_min']:.0f} |\n")
            if r.get("params"):
                f.write(f"| | chosen settings: {json.dumps(r['params'])} | | | | | | | | | | |\n")
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
