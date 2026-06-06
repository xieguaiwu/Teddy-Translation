#!/usr/bin/env python3
"""
P1.2: Enhance SMT language model with monolingual data from BOOKS.

Combines:
  1. Target-side sentences from the WMT parallel corpus (domain-specific)
  2. Monolingual sentences extracted from BOOKS (fluency, vocabulary breadth)

Then trains a new Kneser-Ney LM and replaces the existing LM in the model.

Usage:
    # Enhance ZH→EN model's English LM
    python3 scripts/enhance_lm.py \
        --model-dir model/smt_zh2en_v3 \
        --mono-file data/mono/en.txt \
        --wmt-target data/wmt/clean/en.txt \
        --output-dir model/smt_zh2en_v3_enhanced

    # Enhance EN→ZH model's Chinese LM
    python3 scripts/enhance_lm.py \
        --model-dir model/smt_en2zh_v3 \
        --mono-file data/mono/zh.txt \
        --wmt-target data/wmt/clean/zh.txt \
        --output-dir model/smt_en2zh_v3_enhanced \
        --tgt-lang zh
"""

import sys, os, argparse, shutil, time, json, random
from pathlib import Path

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

from smt import utils
from smt.language_model import KneserNeyLM, train_lm


def load_sentences(path: str, max_lines: int = None) -> list:
    """Load tokenized sentences from file, one per line."""
    lines = utils.read_lines(path)
    if max_lines:
        lines = lines[:max_lines]
    return [line.strip().split() for line in lines if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Enhance SMT LM with monolingual data")
    parser.add_argument("--model-dir", required=True, help="Path to existing SMT model")
    parser.add_argument("--mono-file", required=True, help="Monolingual text file (tokenized)")
    parser.add_argument("--wmt-target", required=True, help="WMT target-side text file")
    parser.add_argument("--output-dir", required=True, help="Output directory for enhanced model")
    parser.add_argument("--tgt-lang", default="en", choices=["en", "zh"],
                        help="Target language for tokenization")
    parser.add_argument("--max-mono", type=int, default=100000,
                        help="Max monolingual sentences to use")
    parser.add_argument("--mono-ratio", type=float, default=0.5,
                        help="Ratio of monolingual to parallel data (0-1)")
    parser.add_argument("--lm-order", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    t0 = time.time()

    # ── Load existing model ───────────────────────────────────────────
    info_path = os.path.join(args.model_dir, "model_info.json")
    with open(info_path) as f:
        model_info = json.load(f)
    logger = utils.logger
    logger.info(f"Enhancing model: {args.model_dir}")
    logger.info(f"  Original phrases: {model_info.get('phrase_table_entries', '?')}")
    logger.info(f"  Original LM vocab: {model_info.get('vocab_size', '?')}")

    # ── Load training data ────────────────────────────────────────────
    logger.info(f"Loading WMT target data: {args.wmt_target}")
    wmt_sents = load_sentences(args.wmt_target)
    logger.info(f"  {len(wmt_sents)} WMT sentences")

    logger.info(f"Loading monolingual data: {args.mono_file}")
    mono_sents = load_sentences(args.mono_file, args.max_mono)
    logger.info(f"  {len(mono_sents)} mono sentences")

    # ── Combine data ──────────────────────────────────────────────────
    n_mono = min(len(mono_sents), int(len(wmt_sents) * args.mono_ratio / (1 - args.mono_ratio)))
    mono_selected = random.sample(mono_sents, n_mono) if n_mono < len(mono_sents) else mono_sents
    combined = wmt_sents + mono_selected
    logger.info(f"Combined: {len(wmt_sents)} WMT + {len(mono_selected)} mono = {len(combined)} total")

    # ── Train enhanced LM ─────────────────────────────────────────────
    logger.info(f"Training {args.lm_order}-gram Kneser-Ney LM on {len(combined)} sentences...")
    # No pruning for speed (prune_threshold=0)
    lm = train_lm(combined, order=args.lm_order, prune_threshold=0.0, num_workers=1)
    logger.info(f"  Types: {lm.vocab_size}, {sum(len(d) for d in lm.counts.values())} n-grams")

    # ── Build enhanced model directory ────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy phrase table (unchanged)
    pt_src = os.path.join(args.model_dir, "phrase_table.txt")
    pt_dst = os.path.join(args.output_dir, "phrase_table.txt")
    shutil.copy2(pt_src, pt_dst)
    logger.info(f"Copied phrase table: {os.path.getsize(pt_dst)/1024:.0f} KB")

    # Save enhanced LM
    lm_path = os.path.join(args.output_dir, "lm.json")
    lm.save(lm_path)

    # Copy src_vocab.json if exists (unchanged)
    for fname in ["src_vocab.json", "tgt_vocab.json"]:
        src = os.path.join(args.model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))

    # ── Write updated model_info ──────────────────────────────────────
    model_info["lm_order"] = args.lm_order
    model_info["vocab_size"] = lm.vocab_size
    model_info["lm_enhanced"] = True
    model_info["mono_sentences"] = len(mono_selected)
    model_info["total_lm_sentences"] = len(combined)

    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)

    elapsed = (time.time() - t0) / 60
    logger.info(f"Enhanced model saved to {args.output_dir} ({elapsed:.1f} min)")
    logger.info(f"  LM entries: {lm.vocab_size} types, {sum(len(d) for d in lm.counts.values())} n-grams")


if __name__ == "__main__":
    main()
