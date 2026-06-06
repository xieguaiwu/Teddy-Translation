#!/usr/bin/env python3
"""
Clean WMT parallel corpus by filtering non-English lines.
Uses langdetect to identify and remove non-English target sentences,
keeping the corresponding source lines aligned.

Usage:
    python scripts/clean_wmt.py --src data/wmt/wmt_word_zh.txt --tgt data/wmt/wmt_word_en.txt --out data/wmt/clean
"""

import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smt import utils


def is_english_fast(line: str) -> bool:
    """Fast pre-filter: check ASCII ratio. Falls back to langdetect."""
    tokens = line.split()
    if len(tokens) < 3:
        return False
    ascii_chars = sum(1 for c in line if ord(c) < 128)
    ratio = ascii_chars / max(len(line), 1)
    return ratio > 0.85  # >85% ASCII is a strong signal


def is_english_langdetect(line: str, detector) -> bool:
    """Use langdetect for accurate language detection."""
    try:
        return detector.detect(line) == 'en'
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Clean WMT parallel corpus")
    parser.add_argument("--src", required=True, help="Source (ZH) file")
    parser.add_argument("--tgt", required=True, help="Target (EN) file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--method", default="fast", choices=["fast", "langdetect", "both"],
                        help="Filtering method")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # Read files
    print(f"Reading {args.src}...")
    zh_lines = utils.read_lines(args.src)
    print(f"Reading {args.tgt}...")
    en_lines = utils.read_lines(args.tgt)

    assert len(zh_lines) == len(en_lines), f"Line count mismatch: {len(zh_lines)} vs {len(en_lines)}"
    total = len(zh_lines)

    # Filter
    kept_zh, kept_en = [], []
    removed_fast, removed_langdetect, removed_short = 0, 0, 0

    if args.method in ("langdetect", "both"):
        from langdetect import DetectorFactory, detect
        DetectorFactory.seed = 42
        print("Using langdetect (may be slow for large files)...")

    for i, (zh, en) in enumerate(zip(zh_lines, en_lines)):
        en = en.strip()
        zh = zh.strip()

        # Short line filter
        if len(en.split()) < 3:
            removed_short += 1
            continue

        # ASCII ratio fast filter
        if args.method in ("fast", "both"):
            if not is_english_fast(en):
                removed_fast += 1
                continue

        # Langdetect
        if args.method in ("langdetect", "both"):
            if not is_english_langdetect(en, detect):
                removed_langdetect += 1
                continue

        kept_zh.append(zh)
        kept_en.append(en)

        if (i + 1) % 50000 == 0:
            print(f"  Processed {i+1}/{total} ({time.time()-t0:.0f}s)")

    # Write
    zout = os.path.join(args.out, "zh.txt")
    eout = os.path.join(args.out, "en.txt")
    utils.write_lines(kept_zh, zout)
    utils.write_lines(kept_en, eout)

    # Stats
    stats = {
        "total": total,
        "kept": len(kept_zh),
        "removed_short": removed_short,
        "removed_fast": removed_fast,
        "removed_langdetect": removed_langdetect,
        "removed_total": total - len(kept_zh),
        "time_s": time.time() - t0,
    }
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nResults:")
    print(f"  Total: {total}")
    print(f"  Kept:  {len(kept_zh)} ({100*len(kept_zh)/total:.1f}%)")
    print(f"  Removed: {total - len(kept_zh)}")
    print(f"    - too short: {removed_short}")
    print(f"    - non-ASCII:  {removed_fast}")
    if args.method in ("langdetect", "both"):
        print(f"    - non-English: {removed_langdetect}")
    print(f"  Time: {stats['time_s']:.0f}s")
    print(f"  Output: {zout}, {eout}")


if __name__ == "__main__":
    main()
