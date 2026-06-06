#!/usr/bin/env python3
"""
fast_align integration for IBM2+HMM word alignment.

Replaces the custom IBM2 implementation with fast_align (Dyer et al. 2013),
a production-quality C++ aligner that supports:
- IBM Model 2 + HMM alignment (variational EM)
- Symmetrization via atools (grow-diag-final-and)
- Much faster and more accurate than our pure-Python IBM2

Interface:
    from smt.align_fast import fast_align_symmetrized
    alignments = fast_align_symmetrized(src_sents, tgt_sents, work_dir)

Usage as standalone:
    python3 smt/align_fast.py --src data/train.zh --tgt data/train.en --out alignments.txt
"""

import sys, os, subprocess, tempfile, shutil
from pathlib import Path
from typing import List, Tuple, Set

# Path to fast_align binaries (compiled locally at /tmp/fast_align/build/)
_FAST_ALIGN_BIN = "/tmp/fast_align/build/fast_align"
_ATOOLS_BIN = "/tmp/fast_align/build/atools"


def _find_binaries():
    """Locate fast_align and atools binaries."""
    # Try local build first
    if os.path.exists(_FAST_ALIGN_BIN) and os.path.exists(_ATOOLS_BIN):
        return _FAST_ALIGN_BIN, _ATOOLS_BIN
    # Try system path
    for name in ["fast_align", "atools"]:
        if shutil.which(name):
            pass  # found in PATH
    if shutil.which("fast_align") and shutil.which("atools"):
        return "fast_align", "atools"
    raise FileNotFoundError(
        "fast_align/atools not found. Build with:\n"
        "  git clone https://github.com/clab/fast_align\n"
        "  cd fast_align && mkdir build && cd build && cmake .. && make"
    )


def fast_align_symmetrized(
    src_sentences: List[List[str]],
    tgt_sentences: List[List[str]],
    work_dir: str = None,
    iterations: int = 5,
) -> List[Set[Tuple[int, int]]]:
    """Run fast_align in both directions and symmetrize with grow-diag-final-and.

    Args:
        src_sentences: Tokenized source sentences.
        tgt_sentences: Tokenized target sentences.
        work_dir: Working directory for temporary files.
        iterations: EM iterations for fast_align.

    Returns:
        List of sets of (src_pos, tgt_pos) alignment pairs.
    """
    fast_align_bin, atools_bin = _find_binaries()
    
    tmpdir = tempfile.mkdtemp(prefix="fast_align_", dir=work_dir)
    try:
        # Write parallel corpus in fast_align format: src ||| tgt
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for src, tgt in zip(src_sentences, tgt_sentences):
                f.write(" ".join(src) + " ||| " + " ".join(tgt) + "\n")

        # Forward alignment (src→tgt)
        fwd_path = os.path.join(tmpdir, "forward.align")
        cmd = f"{fast_align_bin} -i {corpus_path} -d -v -o -I {iterations}"
        result = subprocess.run(cmd, shell=True, check=True,
                               capture_output=True, text=True, cwd=tmpdir)
        with open(fwd_path, "w") as f:
            f.write(result.stdout)

        # Reverse alignment (tgt→src)  
        rev_path = os.path.join(tmpdir, "reverse.align")
        cmd = f"{fast_align_bin} -i {corpus_path} -d -v -o -r -I {iterations}"
        result = subprocess.run(cmd, shell=True, check=True,
                               capture_output=True, text=True, cwd=tmpdir)
        with open(rev_path, "w") as f:
            f.write(result.stdout)

        # Symmetrize
        sym_path = os.path.join(tmpdir, "sym.align")
        cmd = f"{atools_bin} -i {fwd_path} -j {rev_path} -c grow-diag-final-and"
        result = subprocess.run(cmd, shell=True, check=True,
                               capture_output=True, text=True, cwd=tmpdir)
        with open(sym_path, "w") as f:
            f.write(result.stdout)

        # Parse alignments
        alignments = []
        with open(sym_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    alignments.append(set())
                    continue
                pairs = set()
                for token in line.split():
                    if "-" in token:
                        parts = token.split("-")
                        if len(parts) == 2:
                            try:
                                i, j = int(parts[0]), int(parts[1])
                                pairs.add((i, j))
                            except ValueError:
                                pass
                alignments.append(pairs)

        return alignments

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def fast_align_forward(
    src_sentences: List[List[str]],
    tgt_sentences: List[List[str]],
    work_dir: str = None,
    iterations: int = 5,
) -> List[List[Tuple[int, int]]]:
    """Run fast_align in forward direction only (faster, for quick tests)."""
    fast_align_bin, _ = _find_binaries()
    tmpdir = tempfile.mkdtemp(prefix="fast_align_", dir=work_dir)
    try:
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for src, tgt in zip(src_sentences, tgt_sentences):
                f.write(" ".join(src) + " ||| " + " ".join(tgt) + "\n")

        out_path = os.path.join(tmpdir, "forward.align")
        subprocess.run(
            [fast_align_bin, "-i", corpus_path, "-d", "-v", "-o",
             "-I", str(iterations), ">", out_path],
            shell=True, check=True, capture_output=True, cwd=tmpdir,
        )

        alignments = []
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                pairs = []
                for token in line.strip().split():
                    if "-" in token:
                        parts = token.split("-")
                        if len(parts) == 2:
                            try:
                                pairs.append((int(parts[0]), int(parts[1])))
                            except ValueError:
                                pass
                alignments.append(pairs)
        return alignments
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─── Standalone CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from smt import utils

    parser = argparse.ArgumentParser(description="fast_align wrapper")
    parser.add_argument("--src", required=True, help="Source file")
    parser.add_argument("--tgt", required=True, help="Target file")
    parser.add_argument("--out", required=True, help="Output alignment file")
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    src = [s.split() for s in utils.read_lines(args.src)[:args.max_sentences]]
    tgt = [t.split() for t in utils.read_lines(args.tgt)[:args.max_sentences]]

    print(f"Running fast_align on {len(src)} sentences...")
    alignments = fast_align_symmetrized(src, tgt, iterations=args.iterations)

    with open(args.out, "w") as f:
        for al in alignments:
            f.write(" ".join(f"{i}-{j}" for i, j in sorted(al)) + "\n")
    print(f"Saved {len(alignments)} alignments to {args.out}")
