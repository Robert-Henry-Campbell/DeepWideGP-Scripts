#!/usr/bin/env python3
"""The E0 gate's read-out: every run's tuning AUROC per simulated trait beside the references (the ceiling,
the additive ceiling and GBLUP from the dataset's sim_log.json; the cross-fitted ridge PRS from the block
ridge's <dataset>/blocks/prs_val.npz; the block-linear comparators from <dataset>/scores/blocks_linear/*/
val_best.npz, shown, not judged), with the pass criteria of methods.md section "E0 gate". Writes
<out>/e0_results.md (aggregates only).

    python e0_summary.py --dataset <simulated dataset dir> --out <folder> <run dir> [<run dir> ...]

Exit status 0 when every mixer among the runs has a run passing every criterion, 1 otherwise (a missing
reference leaves its criteria undetermined, which is not a pass).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ridge_common import auroc

ADDITIVE_TOL = 0.01                 # additive traits: no further than this below the cross-fitted ridge PRS
GAP_SHARE = 0.5                     # planted traits: at least this share of (ceiling - additive ceiling) above the ridge
NULL_BAND = (0.40, 0.60)            # the null trait: a leakage check, not a decided criterion
ADDITIVE = ("add", "shared")


def scores(path, key):
    """{label: tuning AUROC} of a per-person score file (the block ridge's prs_val.npz, a comparator's val_best.npz)."""
    z = np.load(path)
    return pd.Series({str(lab): auroc(z[key][:, j], z["y"][:, j]) for j, lab in enumerate(z["labels"])})


def run_aurocs(run):
    """({trait: tuning AUROC}, mixer) of a run, or of a search's chosen trial."""
    d = Path(run)
    r = json.load(open(d / "results.json"))
    if r.get("val") is None and "best_trial_dir" in r:
        d = d / r["best_trial_dir"]
        r = json.load(open(d / "results.json"))
    mixer = json.load(open(d / "config.json"))["model"].get("mixer", "?") if (d / "config.json").exists() else "?"
    return {k: float(v["auroc"]) for k, v in r["val"].items()}, mixer


def verdict(kind, auroc, ridge, ceiling, additive_ceiling):
    """True, False, or None when the criterion cannot be applied."""
    if kind == "null":
        return None if auroc is None else NULL_BAND[0] <= auroc <= NULL_BAND[1]
    if auroc is None or pd.isna(ridge):
        return None
    if kind in ADDITIVE:
        return auroc >= ridge - ADDITIVE_TOL
    return auroc >= ridge + GAP_SHARE * (ceiling - additive_ceiling)


def markdown(df, fmt="{:.4f}"):
    cell = lambda v: fmt.format(v) if isinstance(v, float) and not pd.isna(v) else ("" if pd.isna(v) else str(v))
    lines = ["| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |", "|" + "---|" * (len(df.columns) + 1)]
    return "\n".join(lines + ["| " + " | ".join([str(i)] + [cell(v) for v in row]) + " |" for i, row in zip(df.index, df.to_numpy())])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    log = json.load(open(Path(a.dataset) / "sim_log.json"))
    if any("additive_ceiling_auroc_val" not in v for v in log["traits"].values()):
        raise SystemExit(f"{a.dataset}: sim_log.json has no additive ceiling; rebuild the dataset with the current simulate_traits.py")
    ref = pd.DataFrame({t: dict(kind=v["kind"], ceiling=v["ceiling_auroc_val"], additive_ceiling=v["additive_ceiling_auroc_val"],
                                gblup=v["gblup_auroc_val"]) for t, v in log["traits"].items()}).T
    ref.index.name = "trait"
    ridge = Path(a.dataset) / "blocks" / "prs_val.npz"
    ref["ridge"] = scores(ridge, "score").reindex(ref.index) if ridge.exists() else float("nan")
    for f in sorted((Path(a.dataset) / "scores" / "blocks_linear").glob("*/val_best.npz")):
        ref[f"twin_{f.parent.name}"] = scores(f, "logit_population").reindex(ref.index)
    table, verdicts, mixers = ref.copy(), {}, {}
    for run in a.runs:
        name = Path(run).name
        au, mixers[name] = run_aurocs(run)
        table[name] = pd.Series(au).reindex(ref.index)
        verdicts[name] = {t: verdict(r["kind"], au.get(t), r["ridge"], r["ceiling"], r["additive_ceiling"]) for t, r in ref.iterrows()}
    word = {True: "pass", False: "FAIL", None: "undetermined"}
    vt = pd.DataFrame({run: {"mixer": mixers[run], **{t: word[v] for t, v in vs.items()}} for run, vs in verdicts.items()}).T
    vt.index.name = "run"
    vt["all"] = ["pass" if all(v is True for v in vs.values()) else "FAIL" for vs in verdicts.values()]
    passed = {m for run, m in mixers.items() if vt.loc[run, "all"] == "pass"}
    ok = bool(mixers) and passed == set(mixers.values())
    text = "\n".join([
        "# E0 gate: tuning-set AUROC of the deep runs against the references", "",
        f"Dataset `{a.dataset}` (built {log['built']}, seed {log['seed']}, {log['n_snps']:,} SNPs; training rows {log['n']['train']:,}, "
        f"tuning rows {log['n']['val']:,}). Ceiling: the true genetic liability; additive ceiling: its best additive fit. GBLUP: ridge "
        f"through the training GRM. Ridge: the cross-fitted ridge PRS" + ("." if ridge.exists() else f" (MISSING: {ridge}; its criteria are undetermined)."), "",
        f"Criteria per run: additive traits (kinds {', '.join(ADDITIVE)}) no further than {ADDITIVE_TOL} below the ridge; planted traits at "
        f"least {GAP_SHARE:g} of (ceiling - additive ceiling) above the ridge; the null trait within [{NULL_BAND[0]}, {NULL_BAND[1]}] "
        "(leakage check). The gate passes when every mixer has a run that passes every criterion.", "",
        markdown(table), "", markdown(vt), "", f"**Gate: {'passed' if ok else 'not passed'}.**", ""])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "e0_results.md").write_text(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
