#!/usr/bin/env python3
"""
P1.1: Retrain SMT models with cleaned WMT data — LM pruning bypassed.

Key change from earlier attempts: sets prune_threshold=0 in the LM config
to avoid the O(n_grams) pruning bottleneck that stalled 50K+ training.

Supports both zh2en and en2zh directions. Designed for server execution
but works locally too.

Usage (Server 2):
    python3 scripts/retrain_v3.py --direction zh2en --max-sentences 50000 \
        --data-dir data/wmt/clean --output-dir model/smt_zh2en_v3

Usage (Server 1):
    python3 scripts/retrain_v3.py --direction en2zh --max-sentences 50000 \
        --data-dir data/wmt/clean --output-dir model/smt_en2zh_v3
"""

import sys, os, time, argparse, json, logging

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

import smt.data_prep as dp
dp._nlp_zh = None  # No spaCy on servers
try:
    import jieba
    dp.tokenize_zh = lambda t: ' '.join(jieba.cut(t))
except ImportError:
    pass  # jieba may not be installed; tokenization already done

from smt.pipeline import SMTPipeline
from smt import utils

logger = utils.logger



def parse_args():
    p = argparse.ArgumentParser(description="Retrain SMT v3 — no LM pruning")
    p.add_argument("--direction", choices=["zh2en", "en2zh"], required=True)
    p.add_argument("--max-sentences", type=int, default=50000)
    p.add_argument("--data-dir", default="data/wmt/clean")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--warm-start", default=None,
                   help="Path to existing model for warm-start IBM table")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skip-prep", action="store_true", default=True,
                   help="Use pre-tokenized data (default: True for WMT clean)")
    return p.parse_args()


def main():
    args = parse_args()
    src_lang, tgt_lang = ("zh", "en") if args.direction == "zh2en" else ("en", "zh")

    src_file = os.path.join(args.data_dir, f"{'zh' if args.direction == 'zh2en' else 'en'}.txt")
    tgt_file = os.path.join(args.data_dir, f"{'en' if args.direction == 'zh2en' else 'zh'}.txt")

    # ── Validate input files ──────────────────────────────────────────
    for path, label in [(src_file, "source"), (tgt_file, "target")]:
        if not os.path.exists(path):
            logger.error(f"{label} file not found: {path}")
            sys.exit(1)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        logger.info(f"{label}: {path} ({size_mb:.1f} MB)")

    # ── Count lines for reporting ─────────────────────────────────────
    total_lines = sum(1 for _ in open(src_file))
    actual = min(args.max_sentences, total_lines)
    logger.info(f"Training {args.direction} with {actual}/{total_lines} sentence pairs")

    # ── Create pipeline and override critical config ──────────────────
    t0 = time.time()
    p = SMTPipeline()
    # Set prune_threshold=0 to bypass the O(n_grams) pruning bottleneck.
    # Config uses nested dict access via __getitem__/__setitem__.
    p.config["language_model"] = {
        "order": 3,  # 3-gram standard for 50K-sentence SMT; 5-gram creates 300MB+ LMs
        "smoothing": "kneser_ney",
        "prune_threshold": 0.0,  # ← THE KEY FIX
    }
    p.config["parallel"] = {"enabled": True, "num_workers": args.workers}
    p.config["vocabulary"] = {"src_vocab_min_freq": 2, "tgt_vocab_min_freq": 2}

    try:
        result = p.train_python(
            src_file=src_file,
            tgt_file=tgt_file,
            output_dir=args.output_dir,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_sentences=args.max_sentences,
            num_workers=args.workers,
            skip_prep=args.skip_prep,
            warm_start_model=args.warm_start,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed_min = (time.time() - t0) / 60
    logger.info(f"Training complete in {elapsed_min:.1f} minutes")
    logger.info(f"Model saved to {args.output_dir}")

    # ── Quick sanity check ────────────────────────────────────────────
    pt_path = os.path.join(args.output_dir, "phrase_table.txt")
    lm_path = os.path.join(args.output_dir, "lm.json")
    info_path = os.path.join(args.output_dir, "model_info.json")

    for path in [pt_path, lm_path, info_path]:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            logger.info(f"  ✓ {os.path.basename(path)}: {size_kb:.1f} KB")
        else:
            logger.warning(f"  ✗ {os.path.basename(path)}: MISSING")

    # ── Print key metrics ─────────────────────────────────────────────
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        logger.info(f"  Phrase entries: {info.get('phrase_table_entries', '?')}")
        logger.info(f"  Vocabulary:     {info.get('vocab_size', '?')} types")
        logger.info(f"  LM order:       {info.get('lm_order', '?')}")
        logger.info(f"  Coverage:       {info.get('vocabulary_stats', {}).get('source', {}).get('coverage', '?'):.1%}")


if __name__ == "__main__":
    main()
