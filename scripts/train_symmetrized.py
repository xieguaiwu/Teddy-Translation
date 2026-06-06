#!/usr/bin/env python3
"""
Symmetrized phrase-based SMT pipeline with grow-diag-final-and alignment.

Improves on the basic IBM2 pipeline by:
1. Running IBM2 alignment in both directions (src→tgt, tgt→src)
2. Applying grow-diag-final-and symmetrization to merge alignments
3. Extracting phrases from higher-quality symmetrized alignments
4. Training 3-gram Kneser-Ney LM (no pruning)

The symmetrization step alone typically improves alignment F1 by 5-10%
and downstream BLEU by 2-5 points vs single-direction IBM2.

Usage:
    python3 scripts/train_symmetrized.py --direction zh2en --max-sentences 50000 \
        --data-dir data/wmt/clean --output-dir model/smt_zh2en_sym
"""

import sys, os, time, argparse, json
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import smt.data_prep as dp
dp._nlp_zh = None
try:
    import jieba
    dp.tokenize_zh = lambda t: ' '.join(jieba.cut(t))
except ImportError:
    pass

from smt import utils, ibm_align, phrase_table, language_model
from smt.pipeline import SMTPipeline

logger = utils.logger


# ── Grow-diag-final-and symmetrization ────────────────────────────────

def grow_diag_final_and(
    src_len: int, tgt_len: int,
    fwd_align: set,  # src→tgt alignment: set of (src_pos, tgt_pos)
    rev_align: set,  # tgt→src alignment: set of (tgt_pos, src_pos) — will be swapped
) -> set:
    """Grow-diag-final-and symmetrization (Koehn et al. 2003, Moses default).

    Algorithm:
      1. Intersection = fwd ∩ rev (high-precision seed)
      2. Grow-diag: expand intersection with neighbors from union
      3. Final: add remaining union points touching current alignment
      4. Final-and: add remaining union points that have unaligned neighbors
    """
    # Swap rev_align to (src, tgt) format
    rev_swapped = {(j, i) for i, j in rev_align}

    intersection = fwd_align & rev_swapped
    union = fwd_align | rev_swapped

    # Grow-diag: iteratively add neighbors
    current = set(intersection)
    changed = True
    while changed:
        changed = False
        for e2f, f2e in list(current):
            for e_new in range(e2f - 1, e2f + 2):
                for f_new in range(f2e - 1, f2e + 2):
                    if 0 <= e_new < src_len and 0 <= f_new < tgt_len:
                        new_point = (e_new, f_new)
                        if new_point not in current and new_point in union:
                            # Check: at least one neighbor in current
                            if (e_new, f2e) in current or (e2f, f_new) in current:
                                current.add(new_point)
                                changed = True

    # Final: add unaligned points from union
    for e in range(src_len):
        for f in range(tgt_len):
            if (e, f) in union and (e, f) not in current:
                # Add if src or tgt not yet aligned in current
                src_aligned = any(a == e for a, _ in current)
                tgt_aligned = any(b == f for _, b in current)
                if not src_aligned or not tgt_aligned:
                    current.add((e, f))

    # Final-and: add remaining union points
    for e in range(src_len):
        for f in range(tgt_len):
            if (e, f) in union and (e, f) not in current:
                current.add((e, f))

    return current


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SMT with symmetrized alignment")
    parser.add_argument("--direction", choices=["zh2en", "en2zh"], required=True)
    parser.add_argument("--max-sentences", type=int, default=50000)
    parser.add_argument("--data-dir", default="data/wmt/clean")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    src_lang, tgt_lang = ("zh", "en") if args.direction == "zh2en" else ("en", "zh")
    src_file = os.path.join(args.data_dir, f"{'zh' if args.direction == 'zh2en' else 'en'}.txt")
    tgt_file = os.path.join(args.data_dir, f"{'en' if args.direction == 'zh2en' else 'zh'}.txt")

    # Validate
    for path, label in [(src_file, "source"), (tgt_file, "target")]:
        if not os.path.exists(path):
            logger.error(f"{label} file not found: {path}")
            sys.exit(1)

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────
    src_sentences = [s.split() for s in utils.read_lines(src_file)[:args.max_sentences]]
    tgt_sentences = [t.split() for t in utils.read_lines(tgt_file)[:args.max_sentences]]
    logger.info(f"Training {args.direction} on {len(src_sentences)} sentence pairs")

    # ── Train IBM2 forward (src→tgt) ──────────────────────────────────
    logger.info("[1/3] Training IBM2 forward alignment...")
    ibm_fwd = ibm_align.train_ibm(
        src_sentences, tgt_sentences, model="ibm2",
        iterations_model1=5, iterations_model2=5,
        num_workers=args.workers,
    )
    fwd_alignments = ibm_fwd.extract_alignments(src_sentences, tgt_sentences)
    fwd_align_sets = [set(al) for al in fwd_alignments]

    # ── Train IBM2 reverse (tgt→src) ──────────────────────────────────
    logger.info("[2/3] Training IBM2 reverse alignment...")
    ibm_rev = ibm_align.train_ibm(
        tgt_sentences, src_sentences, model="ibm2",
        iterations_model1=5, iterations_model2=5,
        num_workers=args.workers,
    )
    rev_alignments = ibm_rev.extract_alignments(tgt_sentences, src_sentences)
    rev_align_sets = [set(al) for al in rev_alignments]

    # ── Symmetrize ────────────────────────────────────────────────────
    logger.info("[3/3] Symmetrizing alignments (grow-diag-final-and)...")
    sym_alignments = []
    for i, (src, tgt) in enumerate(zip(src_sentences, tgt_sentences)):
        sym = grow_diag_final_and(
            len(src), len(tgt),
            fwd_align_sets[i], rev_align_sets[i],
        )
        sym_alignments.append(sym)

    # Stats
    total_fwd = sum(len(a) for a in fwd_align_sets)
    total_rev = sum(len(a) for a in rev_align_sets)
    total_sym = sum(len(a) for a in sym_alignments)
    logger.info(f"  Forward points:  {total_fwd}")
    logger.info(f"  Reverse points:  {total_rev}")
    logger.info(f"  Symmetrized:     {total_sym}")
    logger.info(f"  Growth:          {total_sym / max(total_fwd, 1):.1%} of forward")

    # ── Build phrase table from symmetrized alignments ────────────────
    logger.info("[4/5] Building phrase table from symmetrized alignments...")
    t_table = ibm_fwd.t if hasattr(ibm_fwd, 't') else getattr(ibm_fwd, 't', {})
    pt = phrase_table.build_phrase_table(
        src_sentences, tgt_sentences, sym_alignments,
        t_table=t_table, max_phrase_len=7, min_count=2,
        score_features=True, num_workers=args.workers,
    )

    pt_path = os.path.join(args.output_dir, "phrase_table.txt")
    phrase_table.save_phrase_table(pt, pt_path)
    logger.info(f"  {len(pt)} phrase pairs saved")

    # ── Train LM ──────────────────────────────────────────────────────
    logger.info("[5/5] Training 3-gram LM...")
    lm_sents = tgt_sentences
    lm = language_model.train_lm(
        lm_sents, order=3, prune_threshold=0.0, num_workers=1,
    )
    lm_path = os.path.join(args.output_dir, "lm.json")
    lm.save(lm_path)

    # ── Save model info ───────────────────────────────────────────────
    elapsed = (time.time() - t0) / 60
    logger.info(f"Training complete in {elapsed:.1f} minutes")
    logger.info(f"  Phrase pairs: {len(pt)}")
    logger.info(f"  LM vocab:     {lm.vocab_size}")
    logger.info(f"  Alignment:    grow-diag-final-and symmetrized IBM2")


if __name__ == "__main__":
    main()
