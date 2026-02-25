#!/usr/bin/env python3
"""
Download Pan-UKB sumstats listed in merged GBD/UKB/Pan-UKB table.

Key behavior:
- Reads:
  /mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/data/10-panukb/
  GBD_ARD_UKB2025casecount_PANUKB.csv
- Uses URL column: panukb_aws_link
- Verifies downloads with: panukb_md5_hex
  - If checksum fails, retries download.
  - If existing local file checksum fails, re-downloads.
- Robust to blank link/checksum fields.
- Selection mode:
  - ARD_only (default): only rows with ARD_selected=True are selected.
  - all: include all rows.
- Writes downloads to:
  /mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/data/10-panukb/panukb_sumstats
- Writes logs/reports to:
  /mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/data/10-panukb
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_INPUT_CSV = (
    "/mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/data/10-panukb/"
    "GBD_ARD_UKB2025casecount_PANUKB.csv"
)
DEFAULT_OUTPUT_DIR = (
    "/mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/data/10-panukb/panukb_sumstats"
)
DEFAULT_LOG_DIR = "/mnt/sdg/robert/deepwidegp/DeepWideGP-Scripts/data/10-panukb"
DEFAULT_LOG_FILE = "10.2-panukb_download_ARDs_log.txt"
DEFAULT_REPORT_FILE = "10.2-panukb_download_ARDs_report.csv"


@dataclass
class Candidate:
    url: str
    filename: str
    expected_md5: str
    group: str
    source_rows: int
    has_md5_conflict: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Pan-UKB sumstats from panukb_aws_link and verify with "
            "panukb_md5_hex."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=DEFAULT_INPUT_CSV,
        help="Merged CSV containing panukb_aws_link and panukb_md5_hex.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where sumstat files are downloaded.",
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Directory for log and report outputs.",
    )
    parser.add_argument(
        "--selection-mode",
        default="ARD_only",
        choices=["ARD_only", "all"],
        help=(
            "Row selection mode: ARD_only keeps ARD_selected=True rows only; "
            "all keeps all rows."
        ),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max download attempts per file (default: 5).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="HTTP timeout in seconds per request (default: 120).",
    )
    parser.add_argument(
        "--backoff-base-seconds",
        type=float,
        default=2.0,
        help="Exponential retry backoff base seconds (default: 2.0).",
    )
    return parser.parse_args()


def normalize_ard_selected(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return "ARD"
    if text in {"false", "f", "0", "no", "n"}:
        return "non-ARD"
    return "unknown"


def is_selected(group: str, selection_mode: str) -> bool:
    if selection_mode == "all":
        return True
    return group == "ARD"


def normalize_md5(value: object) -> str:
    md5 = str(value).strip().lower()
    if not md5 or md5 == "nan":
        return ""
    return md5


def safe_filename_from_url(url: str) -> str:
    path = urllib.parse.urlsplit(url).path
    name = Path(path).name.strip()
    if name:
        return name
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()
    return f"download_{digest}.bin"


def uniquify_filename(base_name: str, url: str, used_names: Dict[str, str]) -> str:
    """
    Ensure filename uniqueness across different URLs that share the same basename.
    """
    if base_name not in used_names:
        used_names[base_name] = url
        return base_name
    if used_names[base_name] == url:
        return base_name

    p = Path(base_name)
    suffix = "".join(p.suffixes)
    stem = p.name[: -len(suffix)] if suffix else p.name
    digest = hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
    unique_name = f"{stem}__{digest}{suffix}"

    # Very unlikely collision; guard anyway.
    if unique_name in used_names and used_names[unique_name] != url:
        digest2 = hashlib.md5((url + "_2").encode("utf-8")).hexdigest()[:10]
        unique_name = f"{stem}__{digest2}{suffix}"

    used_names[unique_name] = url
    return unique_name


def md5_file(path: Path, chunk_size: int = 8_388_608) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_to_temp(url: str, temp_path: Path, timeout_seconds: int) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "panukb-downloader/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        with temp_path.open("wb") as out:
            while True:
                chunk = resp.read(8_388_608)
                if not chunk:
                    break
                out.write(chunk)


def require_columns(header: Iterable[str], required: Iterable[str]) -> None:
    header_set = set(header)
    missing = [c for c in required if c not in header_set]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")


def build_candidates(
    input_csv: Path,
    selection_mode: str,
) -> Tuple[List[Candidate], Dict[str, Dict[str, int]], Dict[str, int], List[str]]:
    required_cols = ["ARD_selected", "panukb_aws_link", "panukb_md5_hex"]
    grouped_rows: Dict[str, Dict[str, int]] = {
        "ARD": {
            "rows_total": 0,
            "rows_selected": 0,
            "rows_selected_blank_link": 0,
            "rows_selected_with_link": 0,
        },
        "non-ARD": {
            "rows_total": 0,
            "rows_selected": 0,
            "rows_selected_blank_link": 0,
            "rows_selected_with_link": 0,
        },
        "unknown": {
            "rows_total": 0,
            "rows_selected": 0,
            "rows_selected_blank_link": 0,
            "rows_selected_with_link": 0,
        },
    }
    overall = {
        "rows_total": 0,
        "rows_selected": 0,
        "rows_selected_blank_link": 0,
        "rows_selected_with_link": 0,
        "duplicate_selected_rows_same_url": 0,
        "urls_md5_conflict_count": 0,
    }
    warnings: List[str] = []

    by_url: Dict[str, Dict[str, object]] = {}

    with input_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        require_columns(reader.fieldnames or [], required_cols)

        for row in reader:
            overall["rows_total"] += 1
            group = normalize_ard_selected(row.get("ARD_selected", ""))
            grouped_rows[group]["rows_total"] += 1

            if not is_selected(group, selection_mode):
                continue

            grouped_rows[group]["rows_selected"] += 1
            overall["rows_selected"] += 1

            url = str(row.get("panukb_aws_link", "")).strip()
            md5 = normalize_md5(row.get("panukb_md5_hex", ""))
            if not url:
                grouped_rows[group]["rows_selected_blank_link"] += 1
                overall["rows_selected_blank_link"] += 1
                continue

            grouped_rows[group]["rows_selected_with_link"] += 1
            overall["rows_selected_with_link"] += 1

            if url not in by_url:
                by_url[url] = {
                    "source_rows": 0,
                    "groups": set(),
                    "md5_first_nonblank": md5,
                    "md5_set_nonblank": set([md5]) if md5 else set(),
                }
            else:
                overall["duplicate_selected_rows_same_url"] += 1

            entry = by_url[url]
            entry["source_rows"] = int(entry["source_rows"]) + 1
            cast_groups = entry["groups"]
            if isinstance(cast_groups, set):
                cast_groups.add(group)
            if md5:
                cast_md5 = entry["md5_set_nonblank"]
                if isinstance(cast_md5, set):
                    cast_md5.add(md5)
                if not entry["md5_first_nonblank"]:
                    entry["md5_first_nonblank"] = md5

    candidates: List[Candidate] = []
    used_names: Dict[str, str] = {}
    for url, info in by_url.items():
        groups = info["groups"] if isinstance(info["groups"], set) else set()
        group_label = sorted(groups)[0] if len(groups) == 1 else "mixed"
        md5_set = info["md5_set_nonblank"] if isinstance(info["md5_set_nonblank"], set) else set()
        has_conflict = len(md5_set) > 1
        if has_conflict:
            overall["urls_md5_conflict_count"] += 1
            warnings.append(
                f"URL has conflicting MD5 values in input rows; using first non-blank MD5: {url}"
            )
        expected_md5 = str(info["md5_first_nonblank"] or "").strip().lower()
        base_filename = safe_filename_from_url(url)
        filename = uniquify_filename(base_filename, url, used_names)
        candidates.append(
            Candidate(
                url=url,
                filename=filename,
                expected_md5=expected_md5,
                group=group_label,
                source_rows=int(info["source_rows"]),
                has_md5_conflict=has_conflict,
            )
        )

    return candidates, grouped_rows, overall, warnings


def init_outcome_counts() -> Dict[str, int]:
    return {
        "unique_urls": 0,
        "downloaded_ok": 0,
        "downloaded_ok_after_retry": 0,
        "skipped_existing_valid_md5": 0,
        "skipped_existing_no_md5": 0,
        "failed_download": 0,
        "failed_md5_mismatch": 0,
    }


def download_candidates(
    candidates: List[Candidate],
    output_dir: Path,
    max_retries: int,
    timeout_seconds: int,
    backoff_base_seconds: float,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, int]], Dict[str, int]]:
    group_stats: Dict[str, Dict[str, int]] = {
        "ARD": init_outcome_counts(),
        "non-ARD": init_outcome_counts(),
        "unknown": init_outcome_counts(),
        "mixed": init_outcome_counts(),
    }
    overall = {
        "retries_performed": 0,
        "md5_mismatch_events": 0,
        "network_error_events": 0,
    }
    report_rows: List[Dict[str, object]] = []

    for cand in candidates:
        if cand.group not in group_stats:
            group_stats[cand.group] = init_outcome_counts()
        stats = group_stats[cand.group]
        stats["unique_urls"] += 1

        dest = output_dir / cand.filename
        tmp = output_dir / f"{cand.filename}.part"

        status = ""
        error_text = ""
        attempts = 0
        used_md5 = bool(cand.expected_md5)
        actual_md5 = ""

        # Existing file check first.
        if dest.exists():
            if not used_md5:
                status = "skipped_existing_no_md5"
                stats["skipped_existing_no_md5"] += 1
            else:
                actual_md5 = md5_file(dest)
                if actual_md5 == cand.expected_md5:
                    status = "skipped_existing_valid_md5"
                    stats["skipped_existing_valid_md5"] += 1

        # Download (or re-download) if not already accepted.
        if not status:
            if tmp.exists():
                tmp.unlink()

            for attempt in range(1, max_retries + 1):
                attempts = attempt
                try:
                    download_to_temp(cand.url, tmp, timeout_seconds)
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    overall["network_error_events"] += 1
                    error_text = str(exc)
                    if attempt < max_retries:
                        overall["retries_performed"] += 1
                        time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                        continue
                    status = "failed_download"
                    stats["failed_download"] += 1
                    break

                # Download succeeded; now verify if md5 exists.
                if used_md5:
                    actual_md5 = md5_file(tmp)
                    if actual_md5 != cand.expected_md5:
                        overall["md5_mismatch_events"] += 1
                        error_text = (
                            f"MD5 mismatch: expected={cand.expected_md5} actual={actual_md5}"
                        )
                        if tmp.exists():
                            tmp.unlink()
                        if attempt < max_retries:
                            overall["retries_performed"] += 1
                            time.sleep(backoff_base_seconds * (2 ** (attempt - 1)))
                            continue
                        status = "failed_md5_mismatch"
                        stats["failed_md5_mismatch"] += 1
                        break

                # Success path (md5 passed, or md5 unavailable).
                tmp.replace(dest)
                status = "downloaded_ok"
                stats["downloaded_ok"] += 1
                if attempt > 1:
                    stats["downloaded_ok_after_retry"] += 1
                break

            if tmp.exists():
                tmp.unlink()

        report_rows.append(
            {
                "group": cand.group,
                "url": cand.url,
                "filename": cand.filename,
                "expected_md5": cand.expected_md5,
                "actual_md5": actual_md5,
                "used_md5_check": used_md5,
                "source_rows": cand.source_rows,
                "has_md5_conflict": cand.has_md5_conflict,
                "status": status,
                "attempts": attempts,
                "error": error_text,
            }
        )

    return report_rows, group_stats, overall


def write_report_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    cols = [
        "group",
        "url",
        "filename",
        "expected_md5",
        "actual_md5",
        "used_md5_check",
        "source_rows",
        "has_md5_conflict",
        "status",
        "attempts",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def log_lines_from_stats(
    args: argparse.Namespace,
    candidates: List[Candidate],
    grouped_rows: Dict[str, Dict[str, int]],
    pre_download_overall: Dict[str, int],
    warnings: List[str],
    group_outcomes: Dict[str, Dict[str, int]],
    download_overall: Dict[str, int],
    report_rows: List[Dict[str, object]],
    output_dir: Path,
    report_path: Path,
) -> List[str]:
    status_counts: Dict[str, int] = {}
    for row in report_rows:
        s = str(row.get("status", ""))
        status_counts[s] = status_counts.get(s, 0) + 1

    lines = [
        "=== Pan-UKB Sumstats Download Log ===",
        f"Input CSV: {Path(args.input_csv).resolve()}",
        f"Selection mode: {args.selection_mode}",
        f"Output download dir: {output_dir}",
        f"Report CSV: {report_path}",
        f"Max retries: {args.max_retries}",
        f"Timeout seconds: {args.timeout_seconds}",
        f"Backoff base seconds: {args.backoff_base_seconds}",
        "",
        "Row-level input summary (sorted by ARD class):",
        (
            "  ARD: total={rows_total}, selected={rows_selected}, selected_with_link={rows_selected_with_link}, "
            "selected_blank_link={rows_selected_blank_link}"
        ).format(**grouped_rows["ARD"]),
        (
            "  non-ARD: total={rows_total}, selected={rows_selected}, selected_with_link={rows_selected_with_link}, "
            "selected_blank_link={rows_selected_blank_link}"
        ).format(**grouped_rows["non-ARD"]),
        (
            "  unknown: total={rows_total}, selected={rows_selected}, selected_with_link={rows_selected_with_link}, "
            "selected_blank_link={rows_selected_blank_link}"
        ).format(**grouped_rows["unknown"]),
        "",
        f"Selected rows total: {pre_download_overall['rows_selected']}",
        f"Selected rows with blank link: {pre_download_overall['rows_selected_blank_link']}",
        f"Selected rows with link: {pre_download_overall['rows_selected_with_link']}",
        f"Unique URLs to process: {len(candidates)}",
        f"Duplicate selected rows collapsing to existing URL: {pre_download_overall['duplicate_selected_rows_same_url']}",
        f"URLs with conflicting non-blank MD5 across rows: {pre_download_overall['urls_md5_conflict_count']}",
        "",
        "URL-level outcome summary (sorted by ARD class):",
    ]

    for key in ["ARD", "non-ARD", "unknown", "mixed"]:
        st = group_outcomes.get(key, init_outcome_counts())
        lines.append(
            (
                f"  {key}: unique_urls={st['unique_urls']}, downloaded_ok={st['downloaded_ok']}, "
                f"downloaded_ok_after_retry={st['downloaded_ok_after_retry']}, "
                f"skipped_existing_valid_md5={st['skipped_existing_valid_md5']}, "
                f"skipped_existing_no_md5={st['skipped_existing_no_md5']}, "
                f"failed_download={st['failed_download']}, failed_md5_mismatch={st['failed_md5_mismatch']}"
            )
        )

    lines.extend(
        [
            "",
            "Overall technical counters:",
            f"  retries_performed={download_overall['retries_performed']}",
            f"  network_error_events={download_overall['network_error_events']}",
            f"  md5_mismatch_events={download_overall['md5_mismatch_events']}",
            "",
            "Final status counts:",
        ]
    )
    for k in sorted(status_counts):
        lines.append(f"  {k}: {status_counts[k]}")

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")

    return lines


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / DEFAULT_LOG_FILE
    report_path = log_dir / DEFAULT_REPORT_FILE

    candidates, grouped_rows, pre_download_overall, warnings = build_candidates(
        input_csv=input_csv,
        selection_mode=args.selection_mode,
    )

    report_rows, group_outcomes, download_overall = download_candidates(
        candidates=candidates,
        output_dir=output_dir,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
        backoff_base_seconds=args.backoff_base_seconds,
    )

    write_report_csv(report_path, report_rows)

    lines = log_lines_from_stats(
        args=args,
        candidates=candidates,
        grouped_rows=grouped_rows,
        pre_download_overall=pre_download_overall,
        warnings=warnings,
        group_outcomes=group_outcomes,
        download_overall=download_overall,
        report_rows=report_rows,
        output_dir=output_dir,
        report_path=report_path,
    )
    text = "\n".join(lines) + "\n"
    log_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
