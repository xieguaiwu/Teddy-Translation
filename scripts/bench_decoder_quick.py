#!/usr/bin/env python3
"""Quick benchmark comparing original vs optimized SMT decoder."""
import sys, time, logging
sys.path.insert(0, '.')
logging.getLogger().setLevel(logging.ERROR)

from smt.phrase_table import load_phrase_table
from smt.language_model import KneserNeyLM

print("Loading models...", flush=True)
pt = load_phrase_table('models/zh2en_sym/phrase_table.txt')
lm = KneserNeyLM.load('models/zh2en_sym/lm.pkl')
print(f"  PT={len(pt)}, LM vocab={lm.vocab_size}", flush=True)

# Test sentences
tests = [
    ("5t",  "我 是 学生 。 你好".split()),
    ("15t", "今天 天气 非常 好 ， 我 想 出去 玩 。 但是 妈妈 说 不行 。".split()),
    ("25t", "中国 政府 表示 将 继续 推动 经济 改革 ， 加强 与 其他 国家 的 合作 ， 共同 应对 全球 性 挑战 ， 促进 世界 和平 。".split()),
]

# Minimal config for speed
CFG = {
    'beam_size': 2, 'stack_size': 10, 'max_phrase_len': 3,
    'distortion_limit': 1, 'lm_weight': 1.0, 'translation_weight': 1.0,
    'distortion_weight': 0.3, 'word_penalty': -0.5,
    'future_cost_estimate': False, 'oov_strategy': 'copy'
}

# Test original
print("\n--- ORIGINAL ---", flush=True)
import smt.decoder_original as orig_mod
for name, tokens in tests:
    decoder = orig_mod.PhraseDecoder(pt, lm, config=CFG)
    t0 = time.perf_counter()
    out, _ = decoder.decode(tokens)
    t = time.perf_counter() - t0
    print(f"  {name}: {t:.3f}s (len={len(tokens)})", flush=True)

# Test optimized
print("\n--- OPTIMIZED ---", flush=True)
import smt.decoder as opt_mod
for name, tokens in tests:
    decoder = opt_mod.PhraseDecoder(pt, lm, config=CFG)
    t0 = time.perf_counter()
    out, _ = decoder.decode(tokens)
    t = time.perf_counter() - t0
    print(f"  {name}: {t:.3f}s (len={len(tokens)})", flush=True)
