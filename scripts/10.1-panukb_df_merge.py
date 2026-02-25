#!/usr/bin/env python3
"""
Merge Pan-UKB phenotype manifest metadata into GBD+UKB casecount rows using
the first 3 characters of:
- GBD dataframe column: ICD10_explo
- Pan-UKB dataframe column: description

Join behavior:
- Keep all rows from the GBD/UKB casecount dataframe (left join semantics).
- Pan-UKB rows without a matching 3-char key are discarded.
- Pan-UKB columns added to the output are prefixed with "panukb_".

Default output:
  /mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/data/10-panukb/GBD_ARD_UKB2025casecount_PANUKB.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge GBD_ARD_UKB_2025_ICD0_casecounts.csv with a Pan-UKB phenotype "
            "manifest using first-3-character string keys."
        )
    )
    parser.add_argument(
        "--gbd-input",
        default=(
            "/mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/data/"
            "3-identify-UKB-casecount-ARDs/extract_casecount_ukb2025/"
            "GBD_ARD_UKB_2025_ICD0_casecounts.csv"
        ),
        help="Path to the base GBD/UKB casecount CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--panukb-manifest",
        default=(
            "/mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/data/1-inputs/"
            "Pan-UK Biobank phenotype manifest - phenotype_manifest.csv"
        ),
        help=(
            "Path to Pan-UKB phenotype manifest (TSV/CSV; default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/data/10-panukb",
        help="Directory for merged output (default: %(default)s).",
    )
    parser.add_argument(
        "--output-name",
        default="GBD_ARD_UKB2025casecount_PANUKB.csv",
        help="Output filename (default: %(default)s).",
    )
    parser.add_argument(
        "--gbd-column",
        default="ICD10_explo",
        help="GBD dataframe join column (default: %(default)s).",
    )
    parser.add_argument(
        "--panukb-column",
        default="description",
        help="Pan-UKB dataframe join column (default: %(default)s).",
    )
    parser.add_argument(
        "--prefix",
        default="panukb",
        help="Prefix to add to Pan-UKB columns in merged output (default: %(default)s).",
    )
    parser.add_argument(
        "--key-len",
        type=int,
        default=3,
        help="Number of leading characters used to build join keys (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-all-panukb-per-key",
        action="store_true",
        help=(
            "Keep all Pan-UKB rows per 3-char key (can duplicate left rows). "
            "Default behavior keeps only first Pan-UKB row per key."
        ),
    )
    return parser.parse_args()


def detect_sep(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return "\t"
    return ","


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")


def to_key(series: pd.Series, key_len: int) -> pd.Series:
    # Treat entries as strings, strip surrounding whitespace, and take first N chars.
    s = series.astype("string").str.strip()
    s = s.where(s.notna() & (s != ""), pd.NA)
    return s.str[:key_len]


def normalize_ard_selected(series: pd.Series) -> pd.Series:
    """
    Normalize ARD_selected values to: 'True', 'False', or 'Unknown'.
    Handles booleans and common truthy/falsey strings.
    """
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series("Unknown", index=series.index, dtype="string")
    out[s.isin({"true", "t", "1", "yes", "y"})] = "True"
    out[s.isin({"false", "f", "0", "no", "n"})] = "False"
    return out


def load_dataframe(path: Path) -> pd.DataFrame:
    sep = detect_sep(path)
    return pd.read_csv(path, sep=sep)


def build_panukb_subset(
    panukb_df: pd.DataFrame,
    panukb_col: str,
    key_len: int,
    keep_all_per_key: bool,
    prefix: str,
) -> tuple[pd.DataFrame, int, int, int]:
    pan = panukb_df.copy()
    pan["_merge_key"] = to_key(pan[panukb_col], key_len)
    before_key_na = len(pan)
    pan = pan.dropna(subset=["_merge_key"])
    dropped_missing_key = before_key_na - len(pan)

    # Prefix all Pan-UKB columns (except merge key) to avoid collisions.
    rename_map = {c: f"{prefix}_{c}" for c in pan.columns if c != "_merge_key"}
    pan = pan.rename(columns=rename_map)

    dropped_duplicate_keys = 0
    if not keep_all_per_key:
        before_dedup = len(pan)
        pan = pan.drop_duplicates(subset=["_merge_key"], keep="first")
        dropped_duplicate_keys = before_dedup - len(pan)

    return pan, dropped_missing_key, dropped_duplicate_keys, dropped_missing_key + dropped_duplicate_keys


def choose_prefixed_key_col_name(prefix: str, panukb_col: str) -> str:
    return f"{prefix}_{panukb_col}"


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    gbd_path = Path(args.gbd_input).expanduser()
    if not gbd_path.is_absolute():
        gbd_path = gbd_path if gbd_path.exists() else project_root / gbd_path
    gbd_path = gbd_path.resolve()

    panukb_path = Path(args.panukb_manifest).expanduser()
    if not panukb_path.is_absolute():
        panukb_path = panukb_path if panukb_path.exists() else project_root / panukb_path
    panukb_path = panukb_path.resolve()

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = output_dir if output_dir.exists() else project_root / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name
    log_path = output_dir / f"{Path(args.output_name).stem}_merge_log.txt"

    print("=== Pan-UKB Merge Start ===")
    print(f"GBD input: {gbd_path}")
    print(f"Pan-UKB manifest input: {panukb_path}")
    print(f"Output CSV: {output_path}")
    print(f"Output log: {log_path}")
    print(f"Join columns: GBD[{args.gbd_column}] vs Pan-UKB[{args.panukb_column}]")
    print(f"Join key: first {args.key_len} characters")
    print(f"Pan-UKB column prefix in output: {args.prefix}_")
    print("")

    gbd_df = load_dataframe(gbd_path)
    panukb_df = load_dataframe(panukb_path)

    require_columns(gbd_df, [args.gbd_column], "GBD dataframe")
    require_columns(panukb_df, [args.panukb_column], "Pan-UKB dataframe")

    gbd_total_rows = len(gbd_df)
    panukb_total_rows = len(panukb_df)

    gbd_df["_merge_key"] = to_key(gbd_df[args.gbd_column], args.key_len)
    gbd_missing_key_rows = int(gbd_df["_merge_key"].isna().sum())

    (
        pan_subset,
        pan_dropped_missing_key,
        pan_dropped_duplicate_key,
        pan_dropped_for_unusable,
    ) = build_panukb_subset(
        panukb_df=panukb_df,
        panukb_col=args.panukb_column,
        key_len=args.key_len,
        keep_all_per_key=args.keep_all_panukb_per_key,
        prefix=args.prefix,
    )

    merged = gbd_df.merge(
        pan_subset,
        how="left",
        on="_merge_key",
    )

    prefixed_panukb_col = choose_prefixed_key_col_name(args.prefix, args.panukb_column)
    if prefixed_panukb_col not in merged.columns:
        prefixed_panukb_col = f"{args.prefix}_{args.panukb_column}"

    pan_key_set = set(pan_subset["_merge_key"].dropna().unique())
    gbd_mapped_mask = gbd_df["_merge_key"].notna() & gbd_df["_merge_key"].isin(pan_key_set)
    gbd_input_mapped_rows = int(gbd_mapped_mask.sum())
    gbd_input_unmatched_rows = int(len(gbd_df) - gbd_input_mapped_rows)

    merged_output_mapped_rows = int(merged[prefixed_panukb_col].notna().sum())
    merged_output_unmatched_rows = int(len(merged) - merged_output_mapped_rows)

    ard_breakdown_lines = []
    if "ARD_selected" in gbd_df.columns:
        gbd_ard_norm = normalize_ard_selected(gbd_df["ARD_selected"])
        gbd_diag = pd.DataFrame(
            {
                "_ard_flag": gbd_ard_norm,
                "_mapped": gbd_mapped_mask,
            }
        )
        for flag in ["True", "False", "Unknown"]:
            subset = gbd_diag[gbd_diag["_ard_flag"] == flag]
            if subset.empty:
                continue
            total = int(len(subset))
            mapped = int(subset["_mapped"].sum())
            unmatched = int(total - mapped)
            pct = (mapped / total * 100.0) if total > 0 else 0.0
            label = "ARD" if flag == "True" else "non-ARD" if flag == "False" else "unknown"
            ard_breakdown_lines.append(
                f"GBD rows ({label}; ARD_selected={flag}): total={total}, "
                f"mapped={mapped}, unmatched={unmatched}, mapped_pct={pct:.2f}%"
            )

    used_panukb_rows = 0
    dropped_panukb_rows_after_join = 0
    if prefixed_panukb_col in pan_subset.columns and "_merge_key" in pan_subset.columns:
        matched_keys = set(gbd_df.loc[gbd_mapped_mask, "_merge_key"].dropna().unique())
        if args.keep_all_panukb_per_key:
            used_panukb_rows = int(pan_subset["_merge_key"].isin(matched_keys).sum())
        else:
            used_panukb_rows = int(len(matched_keys))
        dropped_panukb_rows_after_join = int(len(pan_subset) - used_panukb_rows)

    merged = merged.drop(columns=["_merge_key"])

    merged.to_csv(output_path, index=False)

    log_lines = [
        "=== Pan-UKB Merge Log ===",
        f"GBD input: {gbd_path}",
        f"Pan-UKB manifest input: {panukb_path}",
        f"Output CSV: {output_path}",
        f"Join columns: GBD[{args.gbd_column}] vs Pan-UKB[{args.panukb_column}]",
        f"Join key length: {args.key_len}",
        f"Left join: yes (all GBD rows retained)",
        f"Pan-UKB keep all rows per key: {args.keep_all_panukb_per_key}",
        "",
        f"GBD rows (input): {gbd_total_rows}",
        f"GBD rows with missing/empty merge key: {gbd_missing_key_rows}",
        f"Pan-UKB rows (input): {panukb_total_rows}",
        f"Pan-UKB rows dropped pre-merge (missing/empty key): {pan_dropped_missing_key}",
        (
            "Pan-UKB rows dropped pre-merge (duplicate key collapse): "
            f"{pan_dropped_duplicate_key}"
        ),
        f"Pan-UKB rows dropped pre-merge (total): {pan_dropped_for_unusable}",
        f"Pan-UKB rows available for merge after preprocessing: {len(pan_subset)}",
        "",
        f"Merged rows (output): {len(merged)}",
        f"GBD input rows mapped to Pan-UKB: {gbd_input_mapped_rows}",
        f"GBD input rows unmatched (no Pan-UKB match): {gbd_input_unmatched_rows}",
        f"Merged output rows with Pan-UKB match: {merged_output_mapped_rows}",
        f"Merged output rows without Pan-UKB match: {merged_output_unmatched_rows}",
        f"Pan-UKB rows used in merge: {used_panukb_rows}",
        f"Pan-UKB rows dropped/discarded after merge (unmatched): {dropped_panukb_rows_after_join}",
        "",
        "GBD mapping breakdown by ARD_selected:",
    ]
    if ard_breakdown_lines:
        log_lines.extend(ard_breakdown_lines)
    else:
        log_lines.append("Column 'ARD_selected' not found; ARD/non-ARD breakdown unavailable.")

    log_text = "\n".join(log_lines)
    print(log_text)
    log_path.write_text(log_text + "\n", encoding="utf-8")

    print("")
    print("Merge complete.")


if __name__ == "__main__":
    main()
