#!/usr/bin/env python3
"""
collect_translations.py — Scan SMT and LLM translation output directories
and produce a unified index CSV mapping every translation file to its source.

Output: data/translation_index.csv
Columns: filepath, architecture, model, direction, genre, source_file, source_text
"""

import os
import re
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
TRANSLATIONS_DIR = PROJECT_ROOT / "translations"
SOURCE_TEXTS_DIR = PROJECT_ROOT / "data" / "source_texts"
OUT_CSV = PROJECT_ROOT / "data" / "translation_index.csv"

# SMT subdir → (source_subdir, direction)
SMT_SUBDIR_MAP = {
    "en_lit":  ("en_lit",  "en2zh"),
    "en_news": ("en_news", "en2zh"),
    "zh_lit":  ("zh_lit",  "zh2en"),
    "zh_news": ("zh_news", "zh2en"),
}

# LLM filename genre token → source subdir token
LLM_GENRE_MAP = {
    "literature": "lit",
    "news": "news",
}


def read_text(path: Path) -> str:
    """Read text content, stripping whitespace. Returns empty string on failure."""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def collect_smt() -> list[dict]:
    """Scan output/<variant>/<subdir>/text_NNN.txt files."""
    rows = []
    if not OUTPUT_DIR.is_dir():
        print(f"[WARN] SMT output directory not found: {OUTPUT_DIR}", file=sys.stderr)
        return rows

    for variant_dir in sorted(OUTPUT_DIR.iterdir()):
        if not variant_dir.is_dir():
            continue
        model = variant_dir.name  # e.g., smt, smt_fa, smt_sym, ...

        for subdir_name, (src_subdir, direction) in SMT_SUBDIR_MAP.items():
            subdir = variant_dir / subdir_name
            if not subdir.is_dir():
                continue

            for txt_file in sorted(subdir.glob("text_*.txt")):
                # text_NNN.txt → extract NNN
                match = re.match(r"text_(\d{3})\.txt", txt_file.name)
                if not match:
                    continue
                num = match.group(1)
                source_file = SOURCE_TEXTS_DIR / src_subdir / f"text_{num}.txt"
                source_text = read_text(source_file) if source_file.is_file() else ""

                rows.append({
                    "filepath": str(txt_file.resolve()),
                    "architecture": "smt",
                    "model": model,
                    "direction": direction,
                    "genre": src_subdir.split("_")[1],  # "lit" or "news"
                    "source_file": str(source_file.resolve()),
                    "source_text": source_text,
                })

    return rows


def collect_llm() -> list[dict]:
    """Scan translations/<model>/<src>_<genre>_NNN_<direction>.txt files."""
    rows = []
    if not TRANSLATIONS_DIR.is_dir():
        print(f"[WARN] LLM translations directory not found: {TRANSLATIONS_DIR}", file=sys.stderr)
        return rows

    LLM_FILE_RE = re.compile(
        r"^(en|zh)_(literature|news)_(\d{3})_(en2zh|zh2en)\.txt$"
    )

    for model_dir in sorted(TRANSLATIONS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for txt_file in sorted(model_dir.glob("*.txt")):
            match = LLM_FILE_RE.match(txt_file.name)
            if not match:
                continue

            src_lang, genre_token, num, direction = match.groups()
            src_genre = LLM_GENRE_MAP[genre_token]  # "lit" or "news"
            src_subdir = f"{src_lang}_{src_genre}"  # e.g., "en_lit"

            source_file = SOURCE_TEXTS_DIR / src_subdir / f"text_{num}.txt"
            source_text = read_text(source_file) if source_file.is_file() else ""

            rows.append({
                "filepath": str(txt_file.resolve()),
                "architecture": "llm",
                "model": model,
                "direction": direction,
                "genre": src_genre,
                "source_file": str(source_file.resolve()),
                "source_text": source_text,
            })

    return rows


def main():
    rows = collect_smt() + collect_llm()

    # Sort: architecture, model, direction, genre, source index
    rows.sort(key=lambda r: (
        r["architecture"],
        r["model"],
        r["direction"],
        r["genre"],
        r["source_file"],
    ))

    columns = ["filepath", "architecture", "model", "direction", "genre",
               "source_file", "source_text"]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {OUT_CSV}")

    # Summary stats
    smt_count = sum(1 for r in rows if r["architecture"] == "smt")
    llm_count = sum(1 for r in rows if r["architecture"] == "llm")
    models = sorted(set(r["model"] for r in rows))
    missing_sources = sum(1 for r in rows if not r["source_text"])

    print(f"      SMT: {smt_count} files across {sum(1 for m in models if any(r['model']==m and r['architecture']=='smt' for r in rows))} variants")
    print(f"      LLM: {llm_count} files across {sum(1 for m in models if any(r['model']==m and r['architecture']=='llm' for r in rows))} models")
    print(f"      Models: {', '.join(models)}")
    if missing_sources:
        print(f"      Missing source texts: {missing_sources}", file=sys.stderr)

    return rows


if __name__ == "__main__":
    rows = main()

    # Print first 20 lines of the output CSV for verification
    print("\n--- First 20 lines of CSV ---")
    with open(OUT_CSV, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 21:  # header + 20 rows
                break
            print(line.rstrip("\n"))
