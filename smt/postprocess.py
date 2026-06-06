#!/usr/bin/env python3
"""
P1.3: Post-processing pipeline — remove non-target-language tokens.

Cleans SMT translation output by filtering tokens that don't belong
to the target language. Handles two directions:

    EN output (from ZH→EN): removes non-ASCII tokens (Chinese chars, etc.)
    ZH output (from EN→ZH): removes non-CJK tokens (keeping punctuation)

Can be used as:
  1. A standalone script:  python3 -m smt.postprocess --lang en output/smt/
  2. A Python module:      from smt.postprocess import clean_text
  3. Integrated into batch translation pipeline

Usage:
    # Clean all translations in a directory
    python3 -m smt.postprocess --lang en output/smt/zh_news output/smt/zh_lit

    # Clean a single file
    python3 -m smt.postprocess --lang en output/smt/zh_news/text_001.txt

    # Clean and overwrite
    python3 -m smt.postprocess --lang zh --in-place output/smt/en_news/
"""

import sys, os, re, argparse
from pathlib import Path


# ── Allowed character ranges ──────────────────────────────────────────

# ASCII printable + common punctuation (for English output)
ASCII_ALLOWED = set(
    chr(c) for c in range(32, 127)  # printable ASCII
) | set("—–‘’“”…€£¥©®°")  # common non-ASCII punctuation

# CJK characters, Chinese punctuation, digits, and common symbols (for Chinese output)
# NOTE: does NOT include Latin letters (A-Z, a-z) — those are filtered out.
CJK_ALLOWED_RANGES = [
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
    (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
    (0x3000, 0x303F),   # CJK Symbols and Punctuation (includes 。、，！？)
    (0xFF00, 0xFFEF),   # Halfwidth/Fullwidth Forms (fullwidth digits, punctuation)
    (0x0030, 0x0039),   # ASCII digits 0-9
    (0x0020, 0x002F),   # ASCII space + basic punctuation !"#$%&'()*+,-./
    (0x003A, 0x0040),   # ASCII :;<=>?@
    (0x005B, 0x0060),   # ASCII [\]^_`
    (0x007B, 0x007E),   # ASCII {|}~
    (0x00A0, 0x00FF),   # Latin-1 Supplement (common symbols like °, ±, ×)
]


def is_cjk_allowed(char: str) -> bool:
    """Check if a character belongs to CJK-friendly ranges."""
    cp = ord(char)
    for lo, hi in CJK_ALLOWED_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def clean_text_en(text: str) -> str:
    """Clean English output: remove tokens with non-ASCII characters.

    Strategy: token-level filtering. A token is kept if >70% of its
    characters are ASCII. This preserves words like 'café' or 'naïve'
    while removing Chinese characters like '协议'.
    """
    tokens = text.split()
    cleaned = []
    for token in tokens:
        if not token:
            continue
        ascii_count = sum(1 for c in token if ord(c) < 128)
        ratio = ascii_count / len(token)
        if ratio > 0.7:
            cleaned.append(token)
    return ' '.join(cleaned)


def clean_text_zh(text: str) -> str:
    """Clean Chinese output: remove non-CJK characters.

    Strategy: character-level filtering (Chinese output has no spaces).
    Keeps characters that fall within CJK-allowed ranges (Chinese chars,
    punctuation, digits). Removes Latin letters and other non-CJK chars.
    Preserves the original character order.
    """
    # First, normalize: insert spaces between CJK/non-CJK boundaries
    # so that token-level cleaning can also work
    result = []
    prev_cjk = None
    for c in text:
        cur_cjk = is_cjk_allowed(c)
        if prev_cjk is not None and cur_cjk != prev_cjk:
            result.append(' ')
        result.append(c)
        prev_cjk = cur_cjk
    spaced = ''.join(result)

    # Token-level filter: keep tokens with >50% CJK-allowed chars
    tokens = spaced.split()
    cleaned = []
    for token in tokens:
        if not token:
            continue
        cjk_count = sum(1 for c in token if is_cjk_allowed(c))
        ratio = cjk_count / len(token)
        if ratio > 0.5:
            cleaned.append(token)

    # Re-join without spaces (Chinese convention)
    return ''.join(cleaned)


def clean_text(text: str, lang: str) -> str:
    """Clean text by target language."""
    if lang == "en":
        return clean_text_en(text)
    elif lang == "zh":
        return clean_text_zh(text)
    else:
        raise ValueError(f"Unknown language: {lang}")


def clean_file(filepath: str, lang: str, in_place: bool = False) -> tuple:
    """Clean a single translation file.

    Returns (original_tokens, cleaned_tokens) for reporting.
    """
    with open(filepath, encoding="utf-8") as f:
        original = f.read()

    cleaned = clean_text(original, lang)

    if in_place:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(cleaned + "\n")

    orig_tokens = len(original.split())
    clean_tokens = len(cleaned.split())
    return orig_tokens, clean_tokens


def clean_directory(dirpath: str, lang: str, in_place: bool = False) -> dict:
    """Clean all .txt files in a directory. Returns stats."""
    stats = {"files": 0, "orig_tokens": 0, "clean_tokens": 0}
    for f in sorted(Path(dirpath).glob("*.txt")):
        orig, clean = clean_file(str(f), lang, in_place)
        stats["files"] += 1
        stats["orig_tokens"] += orig
        stats["clean_tokens"] += clean
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Post-process SMT translations: remove non-target-language tokens")
    parser.add_argument("paths", nargs="+", help="Files or directories to clean")
    parser.add_argument("--lang", required=True, choices=["en", "zh"],
                        help="Target language of the translation")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite original files (default: print only)")
    args = parser.parse_args()

    total_stats = {"files": 0, "orig_tokens": 0, "clean_tokens": 0}

    for path in args.paths:
        if not os.path.exists(path):
            print(f"SKIP: {path} (not found)")
            continue

        if os.path.isdir(path):
            stats = clean_directory(path, args.lang, args.in_place)
        else:
            orig, clean = clean_file(path, args.lang, args.in_place)
            stats = {"files": 1, "orig_tokens": orig, "clean_tokens": clean}

        total_stats["files"] += stats["files"]
        total_stats["orig_tokens"] += stats["orig_tokens"]
        total_stats["clean_tokens"] += stats["clean_tokens"]

        removed = stats["orig_tokens"] - stats["clean_tokens"]
        pct = (removed / max(stats["orig_tokens"], 1)) * 100
        action = "Cleaned" if args.in_place else "Would clean"
        print(f"{action} {path}: {stats['files']} files, "
              f"{stats['orig_tokens']} → {stats['clean_tokens']} tokens "
              f"(-{removed}, {pct:.1f}%)")

    # ── Summary ───────────────────────────────────────────────────────
    total_removed = total_stats["orig_tokens"] - total_stats["clean_tokens"]
    total_pct = (total_removed / max(total_stats["orig_tokens"], 1)) * 100
    print(f"\nTotal: {total_stats['files']} files, "
          f"{total_stats['orig_tokens']} → {total_stats['clean_tokens']} tokens "
          f"(-{total_removed}, {total_pct:.1f}%)")


if __name__ == "__main__":
    main()
