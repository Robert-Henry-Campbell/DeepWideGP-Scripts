#!/usr/bin/env python3
"""
Download FinnGen sumstats for endpoints present in the master FG/UKB table.

Steps:
1) Read FinnGen manifest (phenocode + path_https/path_bucket).
2) Read master FG/UKB table and left join on FG_endpoint (master) -> phenocode (manifest).
3) From joined rows, build a unique download list (prefer path_https, fallback to path_bucket).
4) Download sequentially with retries + logging; skip existing files.
5) Write a download manifest JSON to the destination directory.

Dependencies:
  pip install requests tqdm pandas
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# Retry behaviour
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 2.0

# Shared HTTP session
session = requests.Session()
session.headers.update({"User-Agent": "finngen-sumstats-downloader/1.0"})


def setup_logging(dest_dir: str, verbose_console: bool = True) -> logging.Logger:
    os.makedirs(dest_dir, exist_ok=True)
    logger = logging.getLogger("finngen_downloader")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(dest_dir, f"finngen_download_{ts}.log")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if verbose_console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info("Logging initialised. Log file: %s", log_path)
    return logger


def _sleep_with_backoff(attempt: int, logger: Optional[logging.Logger], context: str) -> None:
    delay = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
    if logger:
        logger.info("%s: retrying in %.1f seconds (attempt %d)", context, delay, attempt)
    time.sleep(delay)


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {', '.join(missing)}")


def gs_to_https(gs_path: str) -> Optional[str]:
    if not isinstance(gs_path, str) or not gs_path.startswith("gs://"):
        return None
    # gs://bucket/path -> https://storage.googleapis.com/bucket/path
    stripped = gs_path[len("gs://") :]
    parts = stripped.split("/", 1)
    if not parts or not parts[0]:
        return None
    bucket = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    return f"https://storage.googleapis.com/{bucket}/{rest}"


def download_file_with_retries(
    url: str,
    tmp_path: str,
    logger: Optional[logging.Logger] = None,
) -> float:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if logger:
                logger.info("Starting download of %s to %s (attempt %d)", url, tmp_path, attempt)
            start_time = time.time()
            with session.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()

                total = r.headers.get("Content-Length")
                try:
                    total = int(total) if total is not None else None
                except Exception:
                    total = None

                filename = os.path.basename(tmp_path)
                with open(tmp_path, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=filename,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8_388_608):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))

            elapsed = time.time() - start_time
            if logger:
                size_bytes = os.path.getsize(tmp_path)
                logger.info(
                    "Completed download of %s (%d bytes) in %.1f seconds",
                    url,
                    size_bytes,
                    elapsed,
                )
            return elapsed
        except Exception as e:
            if logger:
                logger.error("Error downloading %s on attempt %d: %s", url, attempt, e)
            if attempt == MAX_RETRIES:
                raise
            _sleep_with_backoff(attempt, logger, "File download")

    raise RuntimeError(f"Failed to download {url}")


def process_one_file(
    url: str,
    dest_dir: str,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    fname = os.path.basename(url.split("?", 1)[0])
    dest_path = os.path.join(dest_dir, fname)
    tmp_path = dest_path + ".part"

    if logger:
        logger.info("Processing file %s (%s)", fname, url)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        size_bytes = os.path.getsize(dest_path)
        if logger:
            logger.info(
                "Existing file %s found; trusting existing file and skipping download.",
                dest_path,
            )
        return {
            "filename": fname,
            "url": url,
            "size_bytes": size_bytes,
            "elapsed_seconds": 0.0,
            "status": "skipped_existing_trusted",
        }

    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
            if logger:
                logger.info("Removed leftover partial file %s", tmp_path)
        except Exception as e:
            if logger:
                logger.error("Failed to remove leftover partial %s: %s", tmp_path, e)

    try:
        elapsed = download_file_with_retries(url, tmp_path, logger=logger)
    except Exception as e:
        if logger:
            logger.error("Failed to download %s after retries: %s", fname, e)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e2:
            if logger:
                logger.error("Failed to remove temporary file %s: %s", tmp_path, e2)
        return {
            "filename": fname,
            "url": url,
            "size_bytes": None,
            "elapsed_seconds": None,
            "status": "failed_download",
            "error": str(e),
        }

    try:
        os.replace(tmp_path, dest_path)
    except Exception as e:
        if logger:
            logger.error(
                "Failed to promote partial file %s to %s: %s", tmp_path, dest_path, e
            )
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e2:
            if logger:
                logger.error("Failed to remove temporary file %s: %s", tmp_path, e2)
        return {
            "filename": fname,
            "url": url,
            "size_bytes": None,
            "elapsed_seconds": elapsed,
            "status": "failed_promote",
            "error": str(e),
        }

    size_bytes = os.path.getsize(dest_path)
    if logger:
        logger.info("Downloaded %s (size=%d) without checksum verification.", fname, size_bytes)
    return {
        "filename": fname,
        "url": url,
        "size_bytes": size_bytes,
        "elapsed_seconds": elapsed,
        "status": "downloaded_no_checksum",
    }


def build_download_list(
    manifest_path: Path,
    master_path: Path,
    joined_out_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t")
    require_columns(manifest, ["phenocode", "path_https", "path_bucket"], "FinnGen manifest")

    master = pd.read_csv(master_path)
    require_columns(master, ["FG_endpoint"], "Master FG/UKB table")

    joined = master.merge(
        manifest,
        how="left",
        left_on="FG_endpoint",
        right_on="phenocode",
        validate="many_to_one",
    )

    if logger:
        logger.info("Master rows: %d", len(master))
        logger.info("Manifest rows: %d", len(manifest))
        logger.info("Joined rows: %d", len(joined))

    joined["download_url"] = joined["path_https"]
    missing_https = joined["download_url"].isna() & joined["path_bucket"].notna()
    if missing_https.any():
        joined.loc[missing_https, "download_url"] = joined.loc[missing_https, "path_bucket"].apply(gs_to_https)

    if joined_out_path is not None:
        joined_out_path.parent.mkdir(parents=True, exist_ok=True)
        joined.to_csv(joined_out_path, index=False)
        if logger:
            logger.info("Wrote joined master+manifest to %s", joined_out_path)

    downloads = (
        joined[["FG_endpoint", "phenocode", "download_url", "path_https", "path_bucket"]]
        .dropna(subset=["download_url"])
        .drop_duplicates(subset=["download_url"])
        .reset_index(drop=True)
    )
    if logger:
        logger.info("Unique downloadable files: %d", len(downloads))
    return downloads


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download FinnGen sumstats based on master FG/UKB table and FinnGen manifest."
    )
    parser.add_argument(
        "--manifest",
        default="data/9-finngen_download/finngen_R12_manifest.tsv",
        help="FinnGen manifest TSV (default: %(default)s).",
    )
    parser.add_argument(
        "--master",
        default="data/8-integrate_finngen_UKB/2025_casecount/FG_UKB_final_cleaned_pheno_df.csv",
        help="Master FG/UKB table (default: %(default)s).",
    )
    parser.add_argument(
        "--dest",
        default="data/9-finngen_download",
        help="Destination directory for downloads (default: %(default)s).",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable logging to console; only log to file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Optional limit on number of files to download (0 means no limit).",
    )

    args = parser.parse_args(argv)

    dest_dir = os.path.abspath(args.dest)
    logger = setup_logging(dest_dir, verbose_console=not args.no_console_log)
    logger.info("Starting FinnGen download run.")
    logger.info("Manifest: %s", args.manifest)
    logger.info("Master: %s", args.master)
    logger.info("Destination: %s", dest_dir)

    joined_out_path = Path(dest_dir) / "FG_UKB_final_cleaned_pheno_with_manifest.csv"
    downloads_df = build_download_list(
        Path(args.manifest),
        Path(args.master),
        joined_out_path=joined_out_path,
        logger=logger,
    )
    if args.max_files and args.max_files > 0:
        downloads_df = downloads_df.head(args.max_files)
        logger.info("Limiting downloads to first %d files", args.max_files)

    results: List[Dict] = []
    start_all = time.time()

    for _, row in downloads_df.iterrows():
        url = row["download_url"]
        res = process_one_file(url, dest_dir=dest_dir, logger=logger)
        res["FG_endpoint"] = row.get("FG_endpoint")
        res["phenocode"] = row.get("phenocode")
        res["path_https"] = row.get("path_https")
        res["path_bucket"] = row.get("path_bucket")
        results.append(res)

    end_all = time.time()

    manifest = {
        "manifest_path": os.path.abspath(args.manifest),
        "master_path": os.path.abspath(args.master),
        "dest_dir": dest_dir,
        "num_candidates": len(downloads_df),
        "downloads": results,
        "total_elapsed_seconds": end_all - start_all,
        "timestamp": time.time(),
    }

    manifest_path = os.path.join(dest_dir, "finngen_download_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    num_downloaded = sum(1 for r in results if r["status"].startswith("downloaded"))
    num_failed = sum(1 for r in results if r["status"].startswith("failed"))
    num_skipped = sum(1 for r in results if r["status"].startswith("skipped"))

    summary = {
        "dest_dir": dest_dir,
        "num_candidates": len(downloads_df),
        "num_downloaded": num_downloaded,
        "num_skipped": num_skipped,
        "num_failed": num_failed,
        "total_elapsed_seconds": manifest["total_elapsed_seconds"],
    }

    logger.info("Manifest written to %s", manifest_path)
    logger.info("Run summary: %s", json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
