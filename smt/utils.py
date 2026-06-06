"""
Utility functions for the SMT system.

Provides I/O helpers, sentence splitting, logging, and file operations
used across all modules in the package.
"""

import os
import re
import sys
import json
import logging
import pathlib
from typing import List, Optional, Iterator, TextIO, Dict, Any

logger = logging.getLogger("smt")


# ─── Logging ─────────────────────────────────────────────────────────


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root SMT logger with console (and optional file) output."""
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ─── File I/O ─────────────────────────────────────────────────────────


def read_lines(path: str, encoding: str = "utf-8") -> List[str]:
    """Read all non-empty lines from a text file (stripped)."""
    with open(path, encoding=encoding) as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(lines: List[str], path: str, encoding: str = "utf-8") -> None:
    """Write a list of strings as lines to a text file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        for line in lines:
            f.write(line + "\n")


def read_parallel(
    src_path: str, tgt_path: str, max_lines: Optional[int] = None,
    encoding: str = "utf-8",
) -> Iterator[tuple[str, str]]:
    """Yield (source, target) sentence pairs from parallel files.

    Args:
        src_path: Path to source-language file (one sentence per line).
        tgt_path: Path to target-language file (one sentence per line).
        max_lines: Maximum number of pairs to yield.
        encoding: File encoding.

    Yields:
        (source_sentence, target_sentence) tuples.
    """
    count = 0
    with open(src_path, encoding=encoding) as src_f, \
         open(tgt_path, encoding=encoding) as tgt_f:
        for src_line, tgt_line in zip(src_f, tgt_f):
            src = src_line.strip()
            tgt = tgt_line.strip()
            if not src or not tgt:
                continue
            yield src, tgt
            count += 1
            if max_lines and count >= max_lines:
                return


# ─── Sentence Splitting ──────────────────────────────────────────────


_SENT_SPLIT_EN = re.compile(r"(?<=[.?!])\s+(?=[A-Z\"'(])")
_SENT_SPLIT_ZH = re.compile(r"(?<=[。！？\n])\s*")


def split_sentences(text: str, lang: str = "en") -> List[str]:
    """Split text into sentences by language.

    Args:
        text: Raw text.
        lang: 'en' or 'zh'.

    Returns:
        List of sentence strings.
    """
    if lang == "zh":
        raw = _SENT_SPLIT_ZH.split(text)
    else:
        raw = _SENT_SPLIT_EN.split(text)
    return [s.strip() for s in raw if s.strip()]


# ─── Filesystem ──────────────────────────────────────────────────────


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist; return path."""
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def file_size(path: str) -> int:
    """Return file size in bytes, or 0 if not found."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# ─── Token counting ──────────────────────────────────────────────────


def count_tokens(tokens: List[str]) -> int:
    """Count tokens, excluding empty strings."""
    return sum(1 for t in tokens if t)


# ─── Configuration loading ───────────────────────────────────────────


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON config file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """Save data as pretty-printed JSON."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Timer ───────────────────────────────────────────────────────────


import time as _time


class Timer:
    """Simple context timer for logging elapsed time."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self.start = _time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = _time.perf_counter() - self.start
        if self.label:
            logger.info(f"{self.label}: {self.elapsed:.2f}s")
