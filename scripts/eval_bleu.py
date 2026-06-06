#!/usr/bin/env python3
"""
BLEU evaluation for SMT models using sacrebleu.
No spaCy dependency — uses jieba for Chinese tokenization via monkey-patch.

Usage:
  # Direct test files
  python3 scripts/eval_bleu.py \
      --model-dir model/smt_20k \
      --test-src data/generated_20k/test.zh \
      --test-ref data/generated_20k/test.en \
      --direction zh2en

  # EN→ZH direction
  python3 scripts/eval_bleu.py \
      --model-dir model/smt_20k_en2zh \
      --test-src data/generated_20k/test.en \
      --test-ref data/generated_20k/test.zh \
      --direction en2zh

  # Limit test set size
  python3 scripts/eval_bleu.py \
      --model-dir model/smt_20k \
      --test-src data/generated_20k/test.zh \
      --test-ref data/generated_20k/test.en \
      --direction zh2en \
      --max-sentences 50
"""
import sys, os, argparse, time

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Monkey-patch Chinese tokenization to use jieba (no spaCy) ─────────
import smt.data_prep as dp
import jieba
dp.tokenize_zh = lambda t: ' '.join(jieba.cut(t))

from smt.decoder import PhraseDecoder
from smt.phrase_table import load_phrase_table
from smt.language_model import KneserNeyLM

# ── Helpers ───────────────────────────────────────────────────────────

def detokenize(tokens, lang):
    """Detokenize token list for target language."""
    if lang == 'zh':
        return ''.join(tokens)
    else:
        return dp.detokenize_en(tokens)


def tokenize_text(text, lang):
    """Tokenize raw text for source language."""
    return dp.tokenize(text, lang=lang)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate SMT model BLEU using sacrebleu (no spaCy, uses jieba)')
    parser.add_argument('--model-dir', required=True,
                        help='Path to trained model directory')
    parser.add_argument('--direction', choices=['zh2en', 'en2zh'], required=True,
                        help='Translation direction')
    parser.add_argument('--test-src', default=None,
                        help='Path to test source file (one sentence per line)')
    parser.add_argument('--test-ref', default=None,
                        help='Path to test reference file (one sentence per line)')
    parser.add_argument('--max-sentences', type=int, default=None,
                        help='Limit number of test sentences')
    parser.add_argument('--beam-size', type=int, default=5,
                        help='Decoder beam size (default: 5)')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for hypotheses/ref output (default: <model-dir>/eval)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-sentence progress')
    parser.add_argument('--tokenize-bleu', default='intl',
                        choices=['intl', 'zh', '13a', 'none'],
                        help='sacrebleu tokenization (default: intl)')
    args = parser.parse_args()

    # ── Resolve language settings ───────────────────────────────────
    if args.direction == 'zh2en':
        src_lang, tgt_lang = 'zh', 'en'
    else:
        src_lang, tgt_lang = 'en', 'zh'

    # ── Load test data ──────────────────────────────────────────────
    if not args.test_src or not args.test_ref:
        print("ERROR: --test-src and --test-ref are required.", file=sys.stderr)
        print("Example: --test-src data/generated_20k/test.zh --test-ref data/generated_20k/test.en",
              file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.test_src):
        print(f"ERROR: Test source file not found: {args.test_src}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.test_ref):
        print(f"ERROR: Test reference file not found: {args.test_ref}", file=sys.stderr)
        sys.exit(1)

    with open(args.test_src, encoding='utf-8') as f:
        src_lines = [l.strip() for l in f if l.strip()]
    with open(args.test_ref, encoding='utf-8') as f:
        ref_lines = [l.strip() for l in f if l.strip()]

    n_total = min(len(src_lines), len(ref_lines))
    n = min(args.max_sentences, n_total) if args.max_sentences else n_total
    test_src = src_lines[:n]
    test_ref = ref_lines[:n]

    print(f"Test set: {n} sentences (from {n_total} available)")
    print(f"Direction: {args.direction}  |  Beam size: {args.beam_size}")
    print()

    # ── Load model ──────────────────────────────────────────────────
    pt_path = os.path.join(args.model_dir, 'phrase_table.txt')
    lm_path = os.path.join(args.model_dir, 'lm.json')

    if not os.path.exists(pt_path):
        print(f"ERROR: phrase_table.txt not found in {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(lm_path):
        print(f"ERROR: lm.json not found in {args.model_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {args.model_dir}...")
    pt = load_phrase_table(pt_path)
    lm = KneserNeyLM.load(lm_path)
    print(f"  Phrase table: {len(pt)} entries")
    print(f"  LM vocabulary: {lm.vocab_size} types, {lm.order}-gram")

    # ── Initialize decoder ──────────────────────────────────────────
    decoder_cfg = {
        'beam_size': args.beam_size,
        'stack_size': 50,
        'distortion_limit': 4,
        'lm_weight': 1.0,
        'translation_weight': 1.0,
        'distortion_weight': 0.3,
        'word_penalty': -0.5,
        'oov_strategy': 'copy',
        'future_cost_estimate': False,
    }
    dec = PhraseDecoder(pt, lm, config=decoder_cfg)

    # ── Translate test set ──────────────────────────────────────────
    print(f"\nTranslating {n} sentences...")
    t_start = time.perf_counter()
    hypotheses = []
    errors = 0

    for i, line in enumerate(test_src):
        if not args.quiet and (i + 1) % max(50, n // 10) == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  {i + 1}/{n}  ({elapsed:.1f}s)", flush=True)

        # Tokenize source
        src_tokens = tokenize_text(line, src_lang).split()
        if not src_tokens:
            hypotheses.append('')
            continue

        try:
            translation, score = dec.decode(src_tokens)
            text = detokenize(translation, tgt_lang)
            hypotheses.append(text)
        except Exception as e:
            print(f"  Error at sentence {i}: {e}", flush=True)
            hypotheses.append('')
            errors += 1

    t_total = time.perf_counter() - t_start
    print(f"  Done in {t_total:.1f}s ({t_total / max(n, 1):.3f}s/sentence)")
    if errors:
        print(f"  ({errors} errors)")

    # ── Save outputs ────────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join(args.model_dir, 'eval')
    os.makedirs(out_dir, exist_ok=True)

    hyp_file = os.path.join(out_dir, f'hyp_{args.direction}.txt')
    ref_file = os.path.join(out_dir, f'ref_{args.direction}.txt')

    with open(hyp_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(hypotheses))
    with open(ref_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_ref[:n]))

    # ── Compute BLEU ────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print("BLEU Evaluation (sacrebleu)")
    print(f"{'=' * 50}")

    try:
        import sacrebleu
        bleu = sacrebleu.corpus_bleu(
            hypotheses, [test_ref[:n]],
            tokenize=args.tokenize_bleu,
        )
        print(f"\n  BLEU score:        {bleu.score:.2f}")
        if bleu.precisions:
            p_str = " / ".join(f"{p:.2f}" for p in bleu.precisions)
            print(f"  Precisions (1-4):  {p_str}")
        else:
            print(f"  Precisions (1-4):  —")
        print(f"  Brevity penalty:   {bleu.bp:.4f}")
        print(f"  Length ratio:      {bleu.ratio:.4f}")
        print(f"  Hypothesis length: {bleu.sys_len}")
        print(f"  Reference length:  {bleu.ref_len}")

        # Also compute sentence-level average
        sent_bleus = []
        for hyp, ref in zip(hypotheses, test_ref[:n]):
            try:
                sb = sacrebleu.sentence_bleu(hyp, [ref], tokenize=args.tokenize_bleu)
                sent_bleus.append(sb.score)
            except Exception:
                sent_bleus.append(0.0)
        if sent_bleus:
            import statistics
            print(f"\n  Sentence BLEU mean:  {statistics.mean(sent_bleus):.2f}")
            print(f"  Sentence BLEU std:   {statistics.stdev(sent_bleus):.2f}" if len(sent_bleus) > 1
                  else f"  Sentence BLEU std:   —")
            print(f"  Sentence BLEU median: {statistics.median(sent_bleus):.2f}")

    except ImportError:
        print("\n  ERROR: sacrebleu not installed. Run: pip install sacrebleu")
        sys.exit(1)

    print(f"\nHypotheses saved to: {hyp_file}")
    print(f"References saved to:  {ref_file}")

    # Return score as exit-like summary
    print(f"\nRESULT: BLEU={bleu.score:.2f} dir={args.direction} model={os.path.basename(args.model_dir)}")


if __name__ == '__main__':
    main()
