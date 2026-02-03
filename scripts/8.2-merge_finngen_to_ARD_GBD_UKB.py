#!/usr/bin/env python3
"""
Merge FinnGen endpoint matches onto the UKB casecount ARD table, filter to
matched FinnGen endpoints + UKB case threshold, and emit MVP-style artifacts:
filtered CSV, counts CSV, plot, and log. Also produces a cleaned final schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge FinnGen mappings into UKB ARD casecounts, filter, and write "
            "filtered CSV + counts + plot + log + cleaned schema."
        )
    )
    parser.add_argument(
        "--base-ard",
        default=(
            "data/3-identify-UKB-casecount-ARDs/extract_casecount_ukb2025/"
            "GBD_ARD_UKB_2025_ICD0_casecounts.csv"
        ),
        help="Base ARD+UKB casecount CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--finngen-map",
        default="data/8-integrate_finngen_UKB/UKB_ICD10_finngen.tsv",
        help="ICD10→FinnGen mapping TSV (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/8-integrate_finngen_UKB/2025_casecount",
        help="Directory for outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--output-prefix",
        default="FG_UKB",
        help="Prefix for output files (default: %(default)s).",
    )
    parser.add_argument(
        "--ukb-case-threshold",
        type=int,
        default=10000,
        help="Keep rows with UKB case counts above this threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting surviving rows by cause level.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")


def rename_ukb_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "phenotype" in df.columns and "UKB_phenotype" not in df.columns:
        rename_map["phenotype"] = "UKB_phenotype"
    if "case_count" in df.columns and "UKB_case_count" not in df.columns:
        rename_map["case_count"] = "UKB_case_count"
    return df.rename(columns=rename_map)


def load_finngen_map(path: Path) -> pd.DataFrame:
    fg = pd.read_csv(path, sep="\t")
    fg = fg[fg["endpoint_1"].notna() & (fg["endpoint_1"] != "")]
    fg = fg.rename(
        columns={
            "endpoint_1": "FG_endpoint_1",
            "endpoint_2": "FG_endpoint",
            "score": "FG_score",
            "matches_1": "FG_matches_1",
            "matches_2": "FG_matches_2",
            "regex_1": "FG_regex_1",
            "regex_2": "FG_regex_2",
            "info": "FG_info",
            "other_hits": "FG_other_hits",
        }
    )
    return fg


def ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df_out = df.copy()
    for c in cols:
        if c not in df_out.columns:
            df_out[c] = pd.NA
    return df_out


def plot_cause_level(
    df: pd.DataFrame,
    case_col: str,
    cause_col: str,
    plot_path: Path,
    threshold: int,
) -> None:
    df_plot = df.drop_duplicates(subset="ICD10_explo", keep="first")
    df_plot = df_plot.sort_values([cause_col, case_col], ascending=[True, False]).reset_index(drop=True)

    n = len(df_plot)
    x = np.arange(n)
    y = df_plot[case_col].values

    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
    ax.bar(x, y, width=0.8, edgecolor="black", linewidth=0.35)
    ax.axhline(threshold, color="red", linewidth=1)

    ax.set_ylabel("# of cases")
    ax.set_title("age-related disease UKB case threshold & FinnGen match")
    ax.set_xticks([])

    group_labels: List[str] = []
    group_positions: List[float] = []
    curr_group = None
    start_index = 0
    causes = df_plot[cause_col].values

    for i, c in enumerate(causes):
        if c != curr_group:
            if curr_group is not None:
                center = (start_index + i - 1) / 2
                group_positions.append(center)
                group_labels.append(curr_group)
            curr_group = c
            start_index = i

    if curr_group is not None:
        center = (start_index + n - 1) / 2
        group_positions.append(center)
        group_labels.append(curr_group)

    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()

    base_path = Path(args.base_ard).expanduser().resolve()
    fg_path = Path(args.finngen_map).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix
    log_path = output_dir / f"{prefix}_mapping_log.txt"
    plot_path = output_dir / f"{prefix}_surviving_cause_level2_plot.png"
    counts_csv_path = output_dir / f"{prefix}_surviving_cause_level2_counts_uniqueICD10_category.csv"
    filtered_csv_path = output_dir / f"{prefix}_mapped_ards_FinnGen_and_UKB_{args.ukb_case_threshold}case.csv"
    cleaned_csv_path = output_dir / f"{prefix}_final_cleaned_pheno_df.csv"

    log_lines: List[str] = [
        "=== FinnGen–UKB ARD Mapping Log ===",
        f"Base ARD input: {base_path}",
        f"FinnGen map input: {fg_path}",
        f"Output directory: {output_dir}",
        f"UKB case threshold: {args.ukb_case_threshold}",
        "",
    ]

    base_df = pd.read_csv(base_path)
    base_df = rename_ukb_columns(base_df)
    require_columns(base_df, ["ICD10_explo"], "Base ARD dataframe")

    fg_df = load_finngen_map(fg_path)
    require_columns(fg_df, ["FG_endpoint_1", "FG_endpoint"], "FinnGen mapping dataframe")

    log_lines.append(f"Initial base rows: {len(base_df)}")
    log_lines.append(f"Initial FinnGen map rows: {len(fg_df)}")
    log_lines.append("")

    merged = base_df.merge(
        fg_df,
        left_on="ICD10_explo",
        right_on="FG_endpoint_1",
        how="left",
        validate="many_to_one",
    )
    log_lines.append(f"Rows after ICD10 merge: {len(merged)}")

    if "UKB_case_count" in merged.columns:
        merged["UKB_case_count"] = pd.to_numeric(merged["UKB_case_count"], errors="coerce")

    filtered = merged.copy()
    before_case = len(filtered)
    if "UKB_case_count" in filtered.columns:
        filtered = filtered[filtered["UKB_case_count"] > args.ukb_case_threshold]
        log_lines.append(
            f"Rows after UKB case filter (UKB_case_count>{args.ukb_case_threshold}): "
            f"{len(filtered)} (dropped {before_case - len(filtered)} of {before_case})."
        )
    else:
        log_lines.append("UKB_case_count not found; skipping UKB case threshold filter.")

    before_fg = len(filtered)
    filtered["FG_score"] = pd.to_numeric(filtered.get("FG_score"), errors="coerce")
    fg_mask = (
        filtered["FG_endpoint"].notna()
        & (filtered["FG_endpoint"] != "NO MATCH")
        & (filtered["FG_score"] > 0)
    )
    filtered = filtered[fg_mask]
    log_lines.append(
        f"Rows after FinnGen match filter: {len(filtered)} "
        f"(dropped {before_fg - len(filtered)} of {before_fg})."
    )
    log_lines.append("")

    if "cause_level_2" in filtered.columns:
        cause_counts = (
            filtered.drop_duplicates(subset="ICD10_explo", keep="first")["cause_level_2"]
            .value_counts()
            .sort_index()
        )
        cause_counts.to_csv(counts_csv_path, header=["count"])
        log_lines.append("Surviving row counts by cause_level_2:")
        for k, v in cause_counts.items():
            log_lines.append(f"  {k}: {v}")
        log_lines.append(f"Saved counts to: {counts_csv_path}")
    else:
        log_lines.append("Column 'cause_level_2' not found — cannot compute category counts.")
    log_lines.append("")

    filtered.to_csv(filtered_csv_path, index=False)
    log_lines.append(f"Filtered dataset saved to: {filtered_csv_path}")

    if not args.no_plot and len(filtered) > 0 and "cause_level_2" in filtered.columns:
        plot_cause_level(filtered, "UKB_case_count", "cause_level_2", plot_path, args.ukb_case_threshold)
        log_lines.append(f"Plot saved to: {plot_path}")
    elif args.no_plot:
        log_lines.append("Plot skipped (--no-plot).")
    else:
        log_lines.append("Plot skipped (no rows or cause column missing).")

    final_cols = [
        "ICD10_category",
        "ICD10_y",
        "ICDO10_Phecode_mapped_flag",
        "ICD10_x",
        "ICD9",
        "cause_level_1",
        "cause_level_2",
        "cause_level_3",
        "cause_level_4",
        "UKB_case_count",
        "UKB_phenotype",
        "FG_endpoint_1",
        "FG_endpoint",
        "FG_score",
        "FG_matches_1",
        "FG_matches_2",
        "FG_regex_1",
        "FG_regex_2",
        "FG_info",
        "FG_other_hits",
    ]

    cleaned = filtered.copy()
    if "ICD10_category" not in cleaned.columns and "ICD10" in cleaned.columns:
        cleaned["ICD10_category"] = cleaned["ICD10"]
    if "ICD10_x" not in cleaned.columns and "ICD10_explo" in cleaned.columns:
        cleaned["ICD10_x"] = cleaned["ICD10_explo"]
    if "ICD10_y" not in cleaned.columns and "ICD10_explo" in cleaned.columns:
        cleaned["ICD10_y"] = cleaned["ICD10_explo"]

    cleaned = ensure_columns(cleaned, final_cols)
    cleaned = cleaned[final_cols]
    cleaned.to_csv(cleaned_csv_path, index=False)
    log_lines.append(f"Cleaned dataset saved to: {cleaned_csv_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"Done. Log written to {log_path}")


if __name__ == "__main__":
    main()
