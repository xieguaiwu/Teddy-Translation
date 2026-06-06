#!/usr/bin/env python3
"""
Download WMT news-commentary parallel data for SMT training.

Downloads and extracts the WMT news-commentary v16-v18 corpora
for Chinese-English machine translation. Falls back to available
versions if the exact version isn't accessible.

Usage:
    python download_wmt_data.py --output-dir ../data
"""

import argparse
import os
import sys
import urllib.request
import bz2
import gzip
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smt import utils

logger = utils.logger

# WMT news-commentary URLs
WMT_SOURCES = {
    "v16": {
        "url": "https://data.statmt.org/wmt16/translation-task/news-commentary-v16-en-zh.zip",
        "files": {
            "news-commentary-v16.en-zh.en": "train.en",
            "news-commentary-v16.en-zh.zh": "train.zh",
        },
    },
    "v15": {
        "url": "https://data.statmt.org/wmt15/translation-task/news-commentary-v15-en-zh.zip",
        "files": {
            "news-commentary-v15.en-zh.en": "train.en",
            "news-commentary-v15.en-zh.zh": "train.zh",
        },
    },
    "v14": {
        "url": "https://data.statmt.org/wmt14/translation-task/news-commentary-v14-en-zh.zip",
        "files": {
            "news-commentary-v14.en-zh.en": "train.en",
            "news-commentary-v14.en-zh.zh": "train.zh",
        },
    },
}


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with progress indicator."""
    try:
        logger.info(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest_path)
        logger.info(f"Saved to {dest_path} ({os.path.getsize(dest_path) / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_zip(zip_path: str, extract_dir: str) -> bool:
    """Extract a ZIP archive."""
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info(f"Extracted to {extract_dir}")
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download WMT news-commentary data")
    parser.add_argument("--output-dir", default="../data", help="Output directory")
    parser.add_argument("--version", default="v16", choices=["v14", "v15", "v16"],
                        help="WMT version (fallback: v15 → v14)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to download from preferred version, with fallbacks
    versions = [args.version] + [v for v in ["v16", "v15", "v14"] if v != args.version]
    downloaded = False

    for ver in versions:
        info = WMT_SOURCES[ver]
        zip_name = os.path.basename(info["url"])
        zip_path = output_dir / zip_name

        if zip_path.exists():
            logger.info(f"Found existing {zip_name}")
        else:
            if not download_file(info["url"], str(zip_path)):
                continue

        extract_zip(str(zip_path), str(output_dir))
        downloaded = True

        # Check if files exist
        for src_file, dst_name in info["files"].items():
            src_path = output_dir / src_file
            if src_path.exists():
                dst_path = output_dir / dst_name
                if not dst_path.exists():
                    shutil.copy(str(src_path), str(dst_path))
                    logger.info(f"Copied {src_file} → {dst_name}")
            else:
                logger.warning(f"Expected file not found: {src_file}")

        break

    # Also download monolingual data for LM training
    mono_urls = {
        "en": "https://data.statmt.org/news-crawl/en/news.2023.en.shuffled.gz",
    }

    for lang, url in mono_urls.items():
        mono_name = os.path.basename(url)
        mono_path = output_dir / mono_name
        if mono_path.exists():
            logger.info(f"Found existing {mono_name}")
        else:
            download_file(url, str(mono_path))

    if not downloaded:
        logger.error(
            "Failed to download WMT data. "
            "Try manually downloading from: https://data.statmt.org/wmt16/translation-task/"
        )
        sys.exit(1)

    # List downloaded files
    logger.info("\nDownloaded files:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size / 1024
        logger.info(f"  {f.name} ({size:.1f} KB)")


if __name__ == "__main__":
    utils.setup_logging()
    main()
