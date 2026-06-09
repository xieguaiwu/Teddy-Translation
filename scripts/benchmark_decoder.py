#!/usr/bin/env python3
"""Benchmark the SMT decoder before and after optimizations.

Compares original decoder (decoder_original.py) vs optimized decoder (decoder.py)
on 3 test sentences of increasing length (5, 15, 25 tokens).

Usage:
    python3 scripts/benchmark_decoder.py [--runs N]
"""

import sys
import time
import logging
import importlib
import argparse
from typing import List, Tuple

# Suppress logging
logging.getLogger().setLevel(logging.ERROR)

sys.path.insert(0, '.')


def load_models():
    """Load phrase table and LM once for all tests."""
    from smt.phrase_table import load_phrase_table
    from smt.language_model import KneserNeyLM

    pt = load_phrase_table('models/zh2en_sym/phrase_table.txt')
    lm = KneserNeyLM.load('models/zh2en_sym/lm.pkl')
    return pt, lm


def benchmark_decoder(
    decoder_module,
    pt, lm,
    config: dict,
    sentences: List[Tuple[str, List[str]]],
    runs: int = 3,
    warmup: int = 1,
) -> List[Tuple[str, float, float]]:
    """Benchmark a decoder module on given sentences.

    Args:
        decoder_module: Module containing PhraseDecoder class.
        pt: Phrase table.
        lm: Language model.
        config: Decoder configuration.
        sentences: List of (name, token_list) pairs.
        runs: Number of timed runs.
        warmup: Number of warmup runs (not timed).

    Returns:
        List of (name, avg_time, stddev) tuples.
    """
    from smt.decoder_original import PhraseDecoder as OrigDecoder
    from smt.decoder import PhraseDecoder as OptDecoder

    if decoder_module.__name__.endswith('decoder_original'):
        DecoderCls = OrigDecoder
    else:
        DecoderCls = OptDecoder

    results = []
    for name, tokens in sentences:
        # Warmup
        for _ in range(warmup):
            decoder = DecoderCls(pt, lm, config=config)
            decoder.decode(tokens)

        times = []
        for _ in range(runs):
            decoder = DecoderCls(pt, lm, config=config)
            t0 = time.perf_counter()
            out, score = decoder.decode(tokens)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
        results.append((name, avg, std))

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark SMT decoder")
    parser.add_argument('--runs', type=int, default=3, help='Number of timed runs')
    parser.add_argument('--future-cost', action='store_true', default=False,
                        help='Enable future cost estimation')
    args = parser.parse_args()

    print("Loading models...", flush=True)
    pt, lm = load_models()
    print(f"  Phrase table: {len(pt)} entries")
    print(f"  LM vocab: {lm.vocab_size}")
    print()

    # Decoder configuration (matching the production config)
    config = {
        'beam_size': 3,
        'stack_size': 20,
        'max_phrase_len': 3,
        'distortion_limit': 2,
        'lm_weight': 1.0,
        'translation_weight': 1.0,
        'distortion_weight': 0.3,
        'word_penalty': -0.5,
        'future_cost_estimate': args.future_cost,
        'oov_strategy': 'copy',
    }

    # Test sentences
    test_sentences = [
        ("5 tokens", "我 是 学生 。 你好".split()),
        ("15 tokens", "今天 天气 非常 好 ， 我 想 出去 玩 。 但是 妈妈 说 不行 。".split()),
        ("25 tokens", "中国 政府 表示 将 继续 推动 经济 改革 ， 加强 与 其他 国家 的 合作 ， 共同 应对 全球 性 挑战 ， 促进 世界 和平 。".split()),
    ]

    # Verify token counts
    for name, tokens in test_sentences:
        print(f"  {name}: {len(tokens)} tokens: {' '.join(tokens[:8])}...")
    print()

    # Import both decoder modules
    import smt.decoder_original as orig_mod
    import smt.decoder as opt_mod

    print("=" * 70)
    print("Benchmarking ORIGINAL decoder...")
    print("=" * 70)
    orig_results = benchmark_decoder(orig_mod, pt, lm, config, test_sentences, runs=args.runs)
    for name, avg, std in orig_results:
        print(f"  {name:>10s}: {avg:.3f}s ± {std:.3f}s")

    print()
    print("=" * 70)
    print("Benchmarking OPTIMIZED decoder...")
    print("=" * 70)
    opt_results = benchmark_decoder(opt_mod, pt, lm, config, test_sentences, runs=args.runs)
    for name, avg, std in opt_results:
        print(f"  {name:>10s}: {avg:.3f}s ± {std:.3f}s")

    print()
    print("=" * 70)
    print("SPEED COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Sentence':>12s}  {'Original':>10s}  {'Optimized':>10s}  {'Speedup':>8s}  {'Improvement':>12s}")
    print("-" * 70)
    for (o_name, o_avg, o_std), (n_name, n_avg, n_std) in zip(orig_results, opt_results):
        speedup = o_avg / n_avg
        improvement = (1 - n_avg / o_avg) * 100
        print(f"{o_name:>12s}  {o_avg:>8.3f}s ±{o_std:.3f}  {n_avg:>8.3f}s ±{n_std:.3f}  {speedup:>7.2f}x  {improvement:>10.1f}%")
    print()

    # Geometric mean speedup
    import math
    geo_speedup = math.exp(sum(math.log(o_avg / n_avg) for (_, o_avg, _), (_, n_avg, _) in zip(orig_results, opt_results)) / len(orig_results))
    print(f"Geometric mean speedup: {geo_speedup:.2f}x")


if __name__ == '__main__':
    main()
