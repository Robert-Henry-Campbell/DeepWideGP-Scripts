#!/usr/bin/env python3
"""The E0 gate's read-out: every run's tuning AUROC per simulated trait beside the references (the ceiling
and GBLUP from the dataset's sim_log.json, the cross-fitted ridge PRS from <dataset>/blocks/ridge_summary.csv),
with the pass criteria of methods.md section "E0 gate". Writes <out>/e0_results.md (aggregates only).

    python e0_summary.py --dataset <simulated dataset dir> --out <folder> <run dir> [<run dir> ...]

Exit status 0 when every run passes every criterion, 1 otherwise (a missing reference fails too).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ADDITIVE_TOL = 0.01                 # additive traits: no further than this below the cross-fitted ridge PRS
GAP_SHARE = 0.5                     # planted traits: at least this share of (ceiling - ridge) above the ridge
NULL_BAND = (0.40, 0.60)
ADDITIVE = ("add", "shared")


def run_aurocs(run):
    """{trait: tuning AUROC} of a run, or of a search's chosen trial."""
    r = json.load(open(Path(run) / "results.json"))
    if r.get("val") is None and "best_trial_dir" in r:
        r = json.load(open(Path(run) / r["best_trial_dir"] / "results.json"))
    return {k: float(v["auroc"]) for k, v in r["val"].items()}


def verdict(kind, auroc, ridge, ceiling):
    """True, False, or None when the criterion cannot be applied."""
    if kind == "null":
        return None if auroc is None else NULL_BAND[0] <= auroc <= NULL_BAND[1]
    if auroc is None or pd.isna(ridge):
        return None
    if kind in ADDITIVE:
        return auroc >= ridge - ADDITIVE_TOL
    return auroc >= ridge + GAP_SHARE * (ceiling - ridge)


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
    ref = pd.DataFrame({t: dict(kind=v["kind"], ceiling=v["ceiling_auroc_val"], gblup=v["gblup_auroc_val"])
                        for t, v in log["traits"].items()}).T
    ref.index.name = "trait"
    ridge_csv = Path(a.dataset) / "blocks" / "ridge_summary.csv"
    ref["ridge"] = pd.read_csv(ridge_csv).set_index("label")["auroc_val"].reindex(ref.index) if ridge_csv.exists() else float("nan")
    table, verdicts = ref.copy(), {}
    for run in a.runs:
        name, au = Path(run).name, run_aurocs(run)
        table[name] = pd.Series(au).reindex(ref.index)
        verdicts[name] = {t: verdict(r["kind"], au.get(t), r["ridge"], r["ceiling"]) for t, r in ref.iterrows()}
    word = {True: "pass", False: "FAIL", None: "undetermined"}
    vt = pd.DataFrame({run: {t: word[v] for t, v in vs.items()} for run, vs in verdicts.items()}).T
    vt.index.name = "run"
    vt["all"] = ["pass" if all(v is True for v in vs.values()) else "FAIL" for vs in verdicts.values()]
    ok = bool(len(vt)) and all(vt["all"] == "pass")
    text = "\n".join([
        "# E0 gate: tuning-set AUROC of the deep runs against the references", "",
        f"Dataset `{a.dataset}` (built {log['built']}, seed {log['seed']}, {log['n_snps']:,} SNPs; training rows {log['n']['train']:,}, "
        f"tuning rows {log['n']['val']:,}). Ceiling: the true genetic liability. GBLUP: ridge through the training GRM. Ridge: the "
        f"cross-fitted ridge PRS" + ("." if ridge_csv.exists() else f" (MISSING: {ridge_csv}; its criteria are undetermined)."), "",
        f"Criteria per run: additive traits (kinds {', '.join(ADDITIVE)}) no further than {ADDITIVE_TOL} below the ridge; planted traits at "
        f"least {GAP_SHARE:g} of (ceiling - ridge) above the ridge; the null trait within [{NULL_BAND[0]}, {NULL_BAND[1]}].", "",
        markdown(table), "", markdown(vt), "", f"**Gate: {'passed' if ok else 'not passed'}.**", ""])
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "e0_results.md").write_text(text)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
