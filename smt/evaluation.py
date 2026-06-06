"""
Translation quality evaluation.

Provides BLEU scoring via sacrebleu and other MT evaluation metrics
as specified in the experimental protocol.
"""

import math
import re
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

from . import utils

logger = utils.logger

# ─── BLEU with sacrebleu ────────────────────────────────────────────


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    tokenize: str = "intl",
) -> Dict[str, float]:
    """Compute corpus-level BLEU using sacrebleu.

    Args:
        hypotheses: List of translated sentences (detokenized/text).
        references: List of reference sentences (detokenized/text).
        tokenize: Tokenization method for BLEU:
            - "intl": International tokenization (default)
            - "zh": Chinese tokenization
            - "13a": Moses tokenizer

    Returns:
        Dict with keys: "bleu", "precisions", "brevity_penalty",
        "ratio", "hyp_len", "ref_len".
    """
    try:
        import sacrebleu

        # Map our tokenize parameter to sacrebleu's
        tok_map = {
            "intl": "intl",
            "zh": "zh",
            "13a": "13a",
        }
        tok = tok_map.get(tokenize, "intl")

        bleu = sacrebleu.corpus_bleu(
            hypotheses, [references],
            tokenize=tok,
        )
        return {
            "bleu": bleu.score,
            "precisions": bleu.precisions,
            "brevity_penalty": bleu.bp,
            "ratio": bleu.ratio,
            "hyp_len": bleu.sys_len,
            "ref_len": bleu.ref_len,
        }
    except ImportError:
        logger.warning("sacrebleu not installed; using fallback BLEU")
        return _bleu_fallback(hypotheses, references, tokenize)


def _bleu_fallback(
    hypotheses: List[str],
    references: List[str],
    tokenize: str = "intl",
) -> Dict[str, float]:
    """Simple BLEU implementation as fallback when sacrebleu unavailable."""
    # Tokenize
    def tokenize_en(text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def tokenize_zh(text: str) -> List[str]:
        # Simple character-based for Chinese
        return list(text.replace(" ", ""))

    if tokenize == "zh":
        tok_fn = tokenize_zh
    else:
        tok_fn = tokenize_en

    # Count n-grams
    max_n = 4
    total_ngram_counts: List[float] = [0.0] * max_n
    total_clipped: List[float] = [0.0] * max_n
    total_hyp_len = 0
    total_ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_tokens = tok_fn(hyp)
        ref_tokens = tok_fn(ref)

        total_hyp_len += len(hyp_tokens)
        total_ref_len += len(ref_tokens)

        ref_ngrams: List[Counter] = [Counter() for _ in range(max_n)]
        for n in range(1, max_n + 1):
            for i in range(len(ref_tokens) - n + 1):
                ng = tuple(ref_tokens[i:i + n])
                ref_ngrams[n - 1][ng] += 1

        hyp_ngrams: List[Counter] = [Counter() for _ in range(max_n)]
        for n in range(1, max_n + 1):
            for i in range(len(hyp_tokens) - n + 1):
                ng = tuple(hyp_tokens[i:i + n])
                hyp_ngrams[n - 1][ng] += 1
                total_ngram_counts[n - 1] += 1

        for n in range(1, max_n + 1):
            for ng, count in hyp_ngrams[n - 1].items():
                clipped = min(count, ref_ngrams[n - 1].get(ng, 0))
                total_clipped[n - 1] += clipped

    # Precision
    precisions: List[float] = []
    for n in range(max_n):
        if total_ngram_counts[n] > 0:
            precisions.append(total_clipped[n] / total_ngram_counts[n])
        else:
            precisions.append(0.0)

    # Brevity penalty
    bp = min(1.0, math.exp(1 - total_ref_len / max(total_hyp_len, 1)))

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        bleu = 0.0
    else:
        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / max_n) * 100

    return {
        "bleu": bleu,
        "precisions": precisions,
        "brevity_penalty": bp,
        "ratio": total_hyp_len / max(total_ref_len, 1),
        "hyp_len": total_hyp_len,
        "ref_len": total_ref_len,
    }


# ─── Sentence-level metrics ──────────────────────────────────────────


def sentence_lengths(sentences: List[List[str]]) -> Dict[str, float]:
    """Compute sentence length statistics.

    Args:
        sentences: List of tokenized sentences.

    Returns:
        Dict with mean, std, min, max, median sentence lengths.
    """
    lengths = [len(s) for s in sentences if s]
    if not lengths:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    import statistics
    return {
        "mean": statistics.mean(lengths),
        "std": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "min": float(min(lengths)),
        "max": float(max(lengths)),
        "median": float(statistics.median(lengths)),
    }


def translation_quality_report(
    hypotheses: List[str],
    references: List[str],
    src_sentences: Optional[List[str]] = None,
    tokenize: str = "intl",
) -> Dict:
    """Full translation quality report.

    Args:
        hypotheses: Translated sentences.
        references: Reference translations.
        src_sentences: Original source sentences (optional).
        tokenize: BLEU tokenization method.

    Returns:
        Dict with BLEU score and sentence-level statistics.
    """
    report = {
        "bleu": compute_bleu(hypotheses, references, tokenize=tokenize),
        "num_sentences": len(hypotheses),
    }

    # Sentence lengths
    hyp_tokenized = [s.split() for s in hypotheses]
    ref_tokenized = [s.split() for s in references]
    report["hypothesis_lengths"] = sentence_lengths(hyp_tokenized)
    report["reference_lengths"] = sentence_lengths(ref_tokenized)

    return report


# ─── Format for report ───────────────────────────────────────────────


def format_bleu_report(bleu_result: Dict[str, float]) -> str:
    """Format BLEU results for display."""
    lines = [
        f"BLEU = {bleu_result['bleu']:.2f}",
        f"Brevity Penalty = {bleu_result['brevity_penalty']:.4f}",
        f"Ratio = {bleu_result['ratio']:.4f}",
        f"Hypothesis Length = {bleu_result['hyp_len']}",
        f"Reference Length = {bleu_result['ref_len']}",
    ]
    precisions = bleu_result.get("precisions", [])
    if precisions:
        p_str = "/".join(f"{p:.4f}" for p in precisions)
        lines.append(f"Precisions (1-4) = {p_str}")
    return "\n".join(lines)
