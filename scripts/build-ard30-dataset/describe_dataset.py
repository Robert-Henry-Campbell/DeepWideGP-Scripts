#!/usr/bin/env python3
"""Describe the built 30-disease dataset: cases by split, and what disease-balanced batches do
to it. Aggregate numbers only; nothing participant-level is printed or written, and no maximum
over participants (it would describe one person).

Reads   <data_dir>/datasets/ard30/hm3plus/{labels.json, build_log.json, samples_train.tsv}
Writes  results/ard30_cases_by_split.csv        one row per phenotype
        results/ard30_sampling_by_disease.csv   per phenotype: class weights, case share in batches
        results/ard30_sampling_summary.json     diseases per person, sampling rates, effective sample

The sampling rate (omega) is computed with the model repository's own function
(widedeepgp.labels.sampling_rate), so the numbers describe what training does.

    /home/robert/envs/widedeepgp/bin/python scripts/build-ard30-dataset/describe_dataset.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEEPGP_REPO = os.environ.get("DEEPGP_REPO", "/mnt/sdg/robert/deepwidegp/DeepWideGP")
BATCH_SIZES = (16, 18, 21, 36, 64)      # batch sizes for "how often does a shuffled batch hold no case"


def rotation_case_share(co, n_people):
    """Share of cases of disease j among the people drawn when disease i is the target.

    co[i, j] = people with both i and j (diagonal: cases). A batch balanced on i takes half its
    people from the cases of i and half from the non-cases of i, so the expected share of cases
    of j is  P(j | i) / 2 + P(j | not i) / 2.  Returns the [target i, disease j] matrix.
    """
    co = np.asarray(co, dtype=np.float64)
    n1 = np.diag(co)
    n0 = n_people - n1
    return 0.5 * co / n1[:, None] + 0.5 * (n1[None, :] - co) / n0[:, None]


def describe(y, names, co=None):
    """Sampling facts of a label matrix [people, diseases] of 0/1. Returns (per-disease table, summary)."""
    if DEEPGP_REPO not in sys.path:
        sys.path.insert(0, DEEPGP_REPO)
    from widedeepgp import labels as L

    y = np.asarray(y, dtype=np.float64)
    n_people, n = y.shape
    n1, n0 = L.case_counts(y)
    pi = n1 / n_people
    if co is None:
        co = y.T @ y
    share = rotation_case_share(co, n_people)             # [target, disease]
    over_rotation = share.mean(axis=0)
    # of the case examples of disease j drawn over a rotation, the part that arrives in other targets' batches
    from_other = 1.0 - (0.5 / n) / over_rotation
    per = pd.DataFrame({
        "label": names, "cases_train": n1.astype(int), "prevalence_train": pi,
        "class_weight_sqrt": L.class_weights(pi, 0.5, 20.0), "class_weight_parity": (1 - pi) / pi,
        "case_share_own_batch": 0.5,
        "case_share_other_batches_mean": (share.sum(axis=0) - 0.5) / (n - 1),
        "case_share_other_batches_max": np.where(np.eye(n, dtype=bool), -np.inf, share).max(axis=0),
        "case_share_over_rotation": over_rotation,
        "share_of_case_draws_from_other_batches": from_other,
        **{f"p_no_case_shuffled_batch_{b}": (1 - pi) ** b for b in BATCH_SIZES},
    })

    omega = L.sampling_rate(y)
    k = y.sum(axis=1)
    q = lambda a, p: float(np.percentile(a, p))
    none = k == 0
    summary = {
        "n_train": int(n_people), "n_diseases": int(n),
        "diseases_per_person": {"mean": float(k.mean()), "median": q(k, 50), "p90": q(k, 90), "p99": q(k, 99),
                                "share_none": float(none.mean()), "share_one": float((k == 1).mean()),
                                "share_five_or_more": float((k >= 5).mean()),
                                "share_twelve_or_more": float((k >= 12).mean())},
        "sum_of_prevalences": float(pi.sum()),
        "label_information_4pi(1-pi)_summed": float((4 * pi * (1 - pi)).sum()),
        "omega": {"mean": float(omega.mean()), "none_of_the_30": float(omega[none].mean()),
                  "min": float(omega.min()), "p50": q(omega, 50), "p90": q(omega, 90), "p99": q(omega, 99),
                  "mean_of_top_1pct": float(np.sort(omega)[-max(1, n_people // 100):].mean())},
        # the rotation draws person k with probability proportional to omega_k and the loss weights
        # them by 1 / omega_k: Kish effective share of the draws = 1 / (mean omega * mean 1/omega)
        "effective_sample_share_with_correction": float(1.0 / (omega.mean() * (1.0 / omega).mean())),
        "class_weight_sqrt_max": float(per["class_weight_sqrt"].max()),
        "class_weight_parity_max": float(per["class_weight_parity"].max()),
        "rarest": {"label": names[int(np.argmin(pi))], "prevalence": float(pi.min()),
                   "case_share_over_rotation": float(over_rotation[int(np.argmin(pi))])},
        "commonest": {"label": names[int(np.argmax(pi))], "prevalence": float(pi.max()),
                      "case_share_over_rotation": float(over_rotation[int(np.argmax(pi))])},
        "median_prevalence": float(np.median(pi)),
        "median_case_share_over_rotation": float(np.median(over_rotation)),
        "batches_one_rotation": int(n),
    }
    return per, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="/mnt/sdg/robert/deepwidegp/data/datasets/ard30/hm3plus")
    ap.add_argument("--out", default=str(HERE / "results"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)

    manifest = json.load(open(os.path.join(a.dataset, "labels.json")))
    log = json.load(open(os.path.join(a.dataset, "build_log.json")))
    names = [m["label"] for m in manifest]
    n_split = {g: log["groups"][g] for g in ("train", "val", "test", "wb_related")}
    cases = pd.DataFrame({
        "label": names, "icd10_codes": [";".join(m["icd10_codes"]) for m in manifest],
        "title": [m["title"] for m in manifest],
        **{f"cases_{g}": [m[f"cases_{g}"] for m in manifest] for g in n_split},
        **{f"prevalence_{g}": [m[f"cases_{g}"] / n_split[g] for m in manifest] for g in n_split},
    }).rename(columns={"cases_val": "cases_tune", "prevalence_val": "prevalence_tune"})
    cases.to_csv(os.path.join(a.out, "ard30_cases_by_split.csv"), index=False, float_format="%.6f")

    y = pd.read_csv(os.path.join(a.dataset, "samples_train.tsv"), sep="\t", usecols=names, dtype=np.uint8)[names]
    assert len(y) == n_split["train"] and (y.sum(axis=0).to_numpy() == cases["cases_train"].to_numpy()).all()
    per, summary = describe(y.to_numpy(), names, co=np.asarray(log["co_occurrence_train"], dtype=np.float64))
    summary["n_by_group"] = n_split
    per.to_csv(os.path.join(a.out, "ard30_sampling_by_disease.csv"), index=False, float_format="%.6f")
    with open(os.path.join(a.out, "ard30_sampling_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(per.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
