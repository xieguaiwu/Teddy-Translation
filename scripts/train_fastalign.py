#!/usr/bin/env python3
"""
Train SMT with fast_align (IBM2+HMM) — replaces custom IBM2.

Pushes alignment quality significantly beyond IBM2+gdfa by using
fast_align's variational EM with HMM distortion model, which is
competitive with GIZA++ IBM4 for many language pairs.

Usage:
    python3 scripts/train_fastalign.py --direction zh2en --max-sentences 50000 \\
        --data-dir data/wmt/clean --output-dir model/smt_zh2en_fa
"""

import sys, os, time, argparse, json

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
from smt.align_fast import fast_align_symmetrized

logger = utils.logger


def main():
    parser = argparse.ArgumentParser(description="Train SMT with fast_align")
    parser.add_argument("--direction", choices=["zh2en", "en2zh"], required=True)
    parser.add_argument("--max-sentences", type=int, default=50000)
    parser.add_argument("--data-dir", default="data/wmt/clean")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fast-align-iterations", type=int, default=5)
    args = parser.parse_args()

    src_lang, tgt_lang = ("zh", "en") if args.direction == "zh2en" else ("en", "zh")
    src_file = os.path.join(args.data_dir, f"{'zh' if args.direction == 'zh2en' else 'en'}.txt")
    tgt_file = os.path.join(args.data_dir, f"{'en' if args.direction == 'zh2en' else 'zh'}.txt")

    for path, label in [(src_file, "source"), (tgt_file, "target")]:
        if not os.path.exists(path):
            logger.error(f"{label} file not found: {path}")
            sys.exit(1)

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    src_sentences = [s.split() for s in utils.read_lines(src_file)[:args.max_sentences]]
    tgt_sentences = [t.split() for t in utils.read_lines(tgt_file)[:args.max_sentences]]
    logger.info(f"Training {args.direction} on {len(src_sentences)} pairs (fast_align)")

    # ── Step 1: Train IBM2 for lexical scoring ──────────────────────
    logger.info("[1/4] Training IBM2 for lexical weighting...")
    t1 = time.time()
    ibm = ibm_align.train_ibm(
        src_sentences, tgt_sentences, model="ibm2",
        iterations_model1=3, iterations_model2=3, num_workers=1,
    )
    t_table = ibm.t if hasattr(ibm, 't') else getattr(ibm, 't', {})
    logger.info(f"  IBM2 trained ({time.time()-t1:.0f}s)")

    # ── Step 2: fast_align for superior alignments ───────────────────
    logger.info("[2/4] Running fast_align (HMM, symmetrized)...")
    t1 = time.time()
    sym_alignments = fast_align_symmetrized(
        src_sentences, tgt_sentences,
        iterations=args.fast_align_iterations,
    )
    total_pts = sum(len(a) for a in sym_alignments)
    logger.info(f"  {total_pts} alignment points ({time.time()-t1:.0f}s)")

    # ── Step 3: Phrase extraction with IBM2 lexical weights ──────────
    logger.info("[3/4] Building phrase table (fast_align alignments + IBM2 lexical)...")
    t1 = time.time()
    pt = phrase_table.build_phrase_table(
        src_sentences, tgt_sentences, sym_alignments,
        t_table=t_table, max_phrase_len=7, min_count=2,
        score_features=True, num_workers=1,
    )
    pt_path = os.path.join(args.output_dir, "phrase_table.txt")
    phrase_table.save_phrase_table(pt, pt_path)
    logger.info(f"  {len(pt)} phrase pairs ({time.time()-t1:.0f}s)")
    pt_path = os.path.join(args.output_dir, "phrase_table.txt")
    phrase_table.save_phrase_table(pt, pt_path)
    logger.info(f"  {len(pt)} phrase pairs ({time.time()-t1:.0f}s)")

    # ── Step 4: Language model ────────────────────────────────────────
    logger.info("[3/3] Training 3-gram LM...")
    t1 = time.time()
    lm = language_model.train_lm(
        tgt_sentences, order=3, prune_threshold=0.0, num_workers=1,
    )
    lm_path = os.path.join(args.output_dir, "lm.json")
    lm.save(lm_path)
    logger.info(f"  {lm.vocab_size} types, {os.path.getsize(lm_path)/1024/1024:.0f}MB ({time.time()-t1:.0f}s)")

    # ── Done ──────────────────────────────────────────────────────────
    elapsed = (time.time() - t0) / 60
    logger.info(f"Training complete in {elapsed:.1f} min")
    logger.info(f"  Phrase pairs: {len(pt)}")
    logger.info(f"  LM vocab:     {lm.vocab_size}")


if __name__ == "__main__":
    main()
