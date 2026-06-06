#!/usr/bin/env python3
"""
Batch translation script for the cross-architecture experiment protocol.

Translates all 80 source texts (ZH→EN and EN→ZH) using the trained SMT model.
Corresponds to the protocol Section 3.1 — Week 4: Generation.

Directory structure expected:
    data/
    ├── source_texts/
    │   ├── zh_news/       # 20 Chinese news texts
    │   ├── en_news/       # 20 English news texts
    │   ├── zh_lit/        # 20 Chinese literary texts
    │   └── en_lit/        # 20 English literary texts
    └── references/        # Reference translations (for evaluation)

Usage:
    # Train + translate (if no model exists)
    python experiment_translate.py --train --src-zh data/train.zh --src-en data/train.en --model-dir model/python_smt

    # Translate only (with existing model)
    python experiment_translate.py --translate --model-dir model/python_smt

    # Evaluate
    python experiment_translate.py --evaluate
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smt import utils
from smt.pipeline import SMTPipeline

logger = utils.logger


def collect_source_files(source_base: str) -> Dict[str, List[str]]:
    """Collect source text files organized by language and genre.

    Args:
        source_base: Base directory containing zh_news, en_news, zh_lit, en_lit.

    Returns:
        Dict mapping group_name → [file_path, ...]
    """
    groups = {
        "zh2en_news": os.path.join(source_base, "zh_news"),
        "zh2en_lit": os.path.join(source_base, "zh_lit"),
        "en2zh_news": os.path.join(source_base, "en_news"),
        "en2zh_lit": os.path.join(source_base, "en_lit"),
    }

    result: Dict[str, List[str]] = {}
    for group, dirpath in groups.items():
        if os.path.isdir(dirpath):
            files = sorted([
                os.path.join(dirpath, f) for f in os.listdir(dirpath)
                if f.endswith(".txt") and os.path.isfile(os.path.join(dirpath, f))
            ])
            result[group] = files
            logger.info(f"  {group}: {len(files)} files")
        else:
            logger.warning(f"Directory not found: {dirpath}")
            result[group] = []

    return result


def translate_group(
    pipeline: SMTPipeline,
    files: List[str],
    output_dir: str,
    src_lang: str,
    model_dir: str,
) -> int:
    """Translate all files in a group.

    Args:
        pipeline: SMTPipeline instance.
        files: List of source file paths.
        output_dir: Output directory.
        src_lang: Source language ('zh' or 'en').
        model_dir: Path to trained model.

    Returns:
        Number of translated files.
    """
    utils.ensure_dir(output_dir)
    count = 0

    for src_path in files:
        filename = os.path.basename(src_path)
        out_path = os.path.join(output_dir, filename)

        try:
            # Read source
            with open(src_path, encoding="utf-8") as f:
                source_text = f.read().strip()

            # Split into sentences
            sentences = utils.split_sentences(source_text, lang=src_lang)

            # Write temp file (one sentence per line)
            tmp_src = os.path.join(output_dir, f"_tmp_{filename}")
            utils.write_lines(sentences, tmp_src)

            # Translate
            tmp_out = os.path.join(output_dir, f"_tmp_out_{filename}")
            result = pipeline.translate_python(
                input_file=tmp_src,
                output_file=tmp_out,
                src_lang=src_lang,
            )

            # Combine back to paragraph
            translated_sentences = utils.read_lines(tmp_out)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(" ".join(translated_sentences))

            # Cleanup
            for tmp in [tmp_src, tmp_out]:
                if os.path.exists(tmp):
                    os.remove(tmp)

            count += 1

        except Exception as e:
            logger.error(f"Failed to translate {filename}: {e}")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="SMT Batch Translation for Experiment Protocol"
    )
    parser.add_argument("--train", action="store_true",
                        help="Train SMT model before translating")
    parser.add_argument("--translate", action="store_true",
                        help="Translate source texts")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate translations against references")

    parser.add_argument("--src-zh", default="data/train.zh",
                        help="ZH training corpus")
    parser.add_argument("--src-en", default="data/train.en",
                        help="EN training corpus")
    parser.add_argument("--model-dir", default="model/python_smt",
                        help="Model directory")
    parser.add_argument("--source-base", default="data/source_texts",
                        help="Source texts base directory")
    parser.add_argument("--output-base", default="output/smt",
                        help="Output translations base directory")
    parser.add_argument("--ref-base", default="data/references",
                        help="Reference translations directory")
    parser.add_argument("--config", default="config.yaml",
                        help="Pipeline config file")
    parser.add_argument("--max-sentences", type=int, default=None,
                        help="Max training sentences")
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        logger.warning(f"Config not found: {args.config}, using defaults")
        args.config = None

    pipeline = SMTPipeline(config_path=args.config)

    # ─── Train ───────────────────────────────────────────────────
    if args.train:
        logger.info("=" * 60)
        logger.info("TRAINING PHASE")
        logger.info("=" * 60)

        if not os.path.exists(args.src_zh) or not os.path.exists(args.src_en):
            logger.error(
                f"Training data not found. Need:\n"
                f"  ZH: {args.src_zh}\n"
                f"  EN: {args.src_en}\n"
                f"Run scripts/download_wmt_data.py first."
            )
            sys.exit(1)

        pipeline.train_python(
            src_file=args.src_zh,
            tgt_file=args.src_en,
            output_dir=args.model_dir,
            src_lang="zh",
            tgt_lang="en",
            max_sentences=args.max_sentences,
        )

    # ─── Translate ────────────────────────────────────────────────
    if args.translate:
        logger.info("=" * 60)
        logger.info("TRANSLATION PHASE")
        logger.info("=" * 60)

        # Check model
        if not os.path.exists(os.path.join(args.model_dir, "phrase_table.txt")):
            logger.error(
                f"Model not found in {args.model_dir}. "
                f"Train first with --train"
            )
            sys.exit(1)

        pipeline.load_model(args.model_dir)

        # Collect source files
        groups = collect_source_files(args.source_base)

        # Direction mapping
        dir_map = {
            "zh2en_news": ("zh", os.path.join(args.output_base, "zh2en_news")),
            "zh2en_lit": ("zh", os.path.join(args.output_base, "zh2en_lit")),
            "en2zh_news": ("en", os.path.join(args.output_base, "en2zh_news")),
            "en2zh_lit": ("en", os.path.join(args.output_base, "en2zh_lit")),
        }

        total = 0
        for group, (src_lang, out_dir) in dir_map.items():
            files = groups.get(group, [])
            if not files:
                logger.warning(f"No files for {group}, skipping")
                continue

            logger.info(f"Translating {group} ({len(files)} files, {src_lang}→{tgt_str(src_lang)})")
            count = translate_group(
                pipeline, files, out_dir, src_lang, args.model_dir
            )
            total += count

        logger.info(f"Translation complete: {total} files translated")

    # ─── Evaluate ─────────────────────────────────────────────────
    if args.evaluate:
        logger.info("=" * 60)
        logger.info("EVALUATION PHASE")
        logger.info("=" * 60)

        # For each translation group, compare with references
        groups = [
            ("zh2en_news", "en"),
            ("zh2en_lit", "en"),
        ]

        for group_name, tgt_lang in groups:
            hyp_dir = os.path.join(args.output_base, group_name)
            ref_dir = os.path.join(args.ref_base, group_name)

            if not os.path.isdir(hyp_dir) or not os.path.isdir(ref_dir):
                logger.warning(f"Missing dirs for {group_name}, skipping")
                continue

            # Collect matching files
            hyps = sorted(os.listdir(hyp_dir))
            refs = sorted(os.listdir(ref_dir))

            # Align by filename
            for hyp_file in hyps:
                if hyp_file in refs:
                    hyp_path = os.path.join(hyp_dir, hyp_file)
                    ref_path = os.path.join(ref_dir, hyp_file)

                    result = pipeline.evaluate(hyp_path, ref_path)
                    bleu = result.get("bleu", {})
                    logger.info(
                        f"  {hyp_file}: BLEU = {bleu.get('bleu', 0.0):.2f}"
                    )

    logger.info("Done.")


def tgt_str(src_lang: str) -> str:
    return "en" if src_lang == "zh" else "zh"


if __name__ == "__main__":
    utils.setup_logging()
    main()
