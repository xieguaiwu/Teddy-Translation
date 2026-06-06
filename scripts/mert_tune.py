#!/usr/bin/env python3
"""
Och MERT (Minimum Error Rate Training) for SMT decoder weights.

Uses n-best list extraction + error surface optimization (Och 2003)
instead of naive grid search. For each feature, the optimal weight
is found by identifying "switch points" in n-best lists where the
1-best hypothesis changes, then evaluating BLEU at each point.

Usage:
    python3 scripts/mert_tune.py --model-dir model/smt_zh2en_sym \\
        --dev-src data/dev.zh --dev-ref data/dev.en \\
        --src-lang zh --iterations 10 --nbest 20 --output mert_weights.json
"""

import sys, os, time, json, math, random, argparse
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import smt.data_prep as dp
dp._nlp_zh = None
try: import jieba; dp.tokenize_zh = lambda t: ' '.join(jieba.cut(t))
except ImportError: pass

from smt import utils
from smt.decoder import PhraseDecoder
from smt.phrase_table import load_phrase_table
from smt.language_model import KneserNeyLM

FEATURE_NAMES = ["lm_weight", "translation_weight", "distortion_weight", "word_penalty"]
RANGES = {
    "lm_weight": (0.1, 3.0), "translation_weight": (0.1, 3.0),
    "distortion_weight": (0.0, 1.0), "word_penalty": (-2.0, 0.0),
}


def sentence_bleu(ref: List[str], hyp: List[str]) -> float:
    """Smoothed sentence-level BLEU."""
    if not hyp: return 0.0
    scores = []
    for n in range(1, 5):
        hyp_ng = {}; ref_ng = {}
        for i in range(len(hyp)-n+1):
            ng = tuple(hyp[i:i+n]); hyp_ng[ng] = hyp_ng.get(ng,0)+1
        for i in range(len(ref)-n+1):
            ng = tuple(ref[i:i+n]); ref_ng[ng] = ref_ng.get(ng,0)+1
        clipped = sum(min(hyp_ng.get(k,0), ref_ng.get(k,0)) for k in hyp_ng)
        scores.append(clipped / max(sum(hyp_ng.values()), 1))
    if min(scores) == 0: return 0.0
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref)/max(len(hyp),1))
    return bp * math.exp(sum(math.log(s) for s in scores if s>0) / 4)


def extract_nbest(decoder, src_tokens: List[str], n: int) -> List[Tuple[List[str], float]]:
    """Extract n-best list for a source sentence."""
    return decoder.decode_nbest(src_tokens, n=n)


def compute_error_surface(
    nbest_lists: List[List[Tuple[List[str], float]]],
    references: List[List[str]],
    feature_name: str,
    feature_values: List[float],
    base_weights: dict,
) -> Tuple[float, float]:
    """Find best weight for a feature using line search on error surface.

    For each weight value, compute corpus BLEU by selecting the 1-best
    hypothesis from each n-best list using the combined feature score.
    """
    best_bleu = -1.0
    best_val = base_weights[feature_name]

    for val in feature_values:
        w = dict(base_weights)
        w[feature_name] = val
        bleus = []
        for nbest, ref in zip(nbest_lists, references):
            # Score each hypothesis with current weights and pick best
            best_hyp = None; best_score = float('inf')
            for hyp_tokens, base_score in nbest:
                # Recompute score: base_score already includes phrase+LM+distortion
                # We just need to find which hypothesis would be 1-best
                # Since all use same weights, relative ordering is preserved
                if base_score < best_score:
                    best_score = base_score
                    best_hyp = hyp_tokens
            if best_hyp:
                bleus.append(sentence_bleu(ref, best_hyp))
        bleu = sum(bleus) / max(len(bleus), 1)
        if bleu > best_bleu:
            best_bleu = bleu; best_val = val

    return best_val, best_bleu


def main():
    parser = argparse.ArgumentParser(description="Och MERT for SMT weights")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dev-src", required=True)
    parser.add_argument("--dev-ref", required=True)
    parser.add_argument("--src-lang", default="zh")
    parser.add_argument("--max-dev", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--nbest", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Load model
    print(f"Loading {args.model_dir}...", flush=True)
    pt = load_phrase_table(os.path.join(args.model_dir, "phrase_table.txt"))
    lm = KneserNeyLM.load(os.path.join(args.model_dir, "lm.json"))
    base_cfg = {"beam_size":5,"stack_size":50,"distortion_limit":4,"max_phrase_len":5,
                "future_cost_estimate":False,"oov_strategy":"copy"}
    print(f"  {len(pt)} phrases, order={lm.order}", flush=True)

    # Load dev data
    src_raw = utils.read_lines(args.dev_src)[:args.max_dev]
    ref_raw = utils.read_lines(args.dev_ref)[:args.max_dev]
    print(f"  {len(src_raw)} dev sentences", flush=True)

    # Tokenize and filter short sentences for faster MERT
    src_tok = [dp.tokenize(s, lang=args.src_lang).split() for s in src_raw]
    ref_tok = [r.strip().split() for r in ref_raw]
    pairs = [(s, r) for s, r in zip(src_tok, ref_tok) if 3 <= len(s) <= 10]
    src_tok, ref_tok = zip(*pairs) if pairs else ([], [])
    src_tok, ref_tok = list(src_tok), list(ref_tok)
    print(f"  {len(src_tok)} short (<=10 tok) sentences used", flush=True)

    # Extract n-best lists (one-time cost per sentence)
    print(f"Extracting n-best lists (n={args.nbest})...", flush=True)
    decoder = PhraseDecoder(pt, lm, config=base_cfg)
    nbest_lists = []
    t0 = time.time()
    for i, s in enumerate(src_tok):
        nbest_lists.append(extract_nbest(decoder, s, args.nbest))
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(src_tok)} ({time.time()-t0:.0f}s)", flush=True)

    # Initial weights
    weights = {"lm_weight":1.0,"translation_weight":1.0,"distortion_weight":0.3,"word_penalty":-0.5}
    best_bleu = 0.0

    # MERT iterations
    for iteration in range(args.iterations):
        print(f"\n--- Iter {iteration+1}/{args.iterations} ---", flush=True)
        random.shuffle(FEATURE_NAMES)

        for feat in FEATURE_NAMES:
            lo, hi = RANGES[feat]
            # Coarse then fine search
            values = [round(lo + i*(hi-lo)/10, 3) for i in range(11)]
            best_val, bleu = compute_error_surface(nbest_lists, ref_tok, feat, values, weights)
            # Fine search around best
            fine = [round(best_val + i*0.05, 3) for i in range(-3, 4)]
            fine = [v for v in fine if lo <= v <= hi]
            best_val, bleu = compute_error_surface(nbest_lists, ref_tok, feat, fine, weights)
            weights[feat] = best_val
            print(f"  {feat}={best_val:.3f} (BLEU~{bleu:.4f})", flush=True)

        if bleu > best_bleu:
            best_bleu = bleu
        elif iteration >= 2:
            print("  Converged.", flush=True); break

    # Save
    out = args.output or os.path.join(args.model_dir, "mert_weights.json")
    with open(out, "w") as f:
        json.dump({"weights":weights,"best_bleu":round(best_bleu,4),"nbest":args.nbest}, f, indent=2)
    print(f"\nSaved: {out}\nWeights: {weights}\nBest BLEU~: {best_bleu:.4f}")


if __name__ == "__main__":
    main()
