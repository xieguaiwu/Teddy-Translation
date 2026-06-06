"""
N-gram language model with Kneser-Ney smoothing.

Implements interpolated (modified) Kneser-Ney smoothing for n-gram
language models used in the SMT decoder. Supports:

- Variable-order n-grams (1-gram up to N-gram)
- Modified Kneser-Ney smoothing (Chen & Goodman 1998)
- Efficient probability queries: log P(w|history)
- Sentence probability scoring
- Model persistence (JSON format)

References:
    - Chen & Goodman (1998), "An Empirical Study of Smoothing Techniques
      for Language Modeling", TR-10-98, Harvard.
    - Kneser & Ney (1995), "Improved backing-off for M-gram language
      modeling", ICASSP.
"""

import math
import json
import os
import logging
import multiprocessing
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple, Iterator
from collections import defaultdict, Counter

try:
    import ujson  # 5-10x faster JSON parsing for large models
except ImportError:
    ujson = None

from . import utils

logger = utils.logger

# ─── N-gram counting ─────────────────────────────────────────────────


def extract_ngrams(tokens: List[str], order: int) -> List[Tuple[str, ...]]:
    """Extract all n-grams of given order from token list.

    Pads with <s> (start) and </s> (end) markers.

    Args:
        tokens: List of word tokens.
        order: N-gram order (1=unigram, 2=bigram, etc.).

    Returns:
        List of n-gram tuples.
    """
    # Add sentence boundary markers
    padded = ["<s>"] * (order - 1) + tokens + ["</s>"]
    ngrams = []
    for i in range(len(padded) - order + 1):
        ngrams.append(tuple(padded[i:i + order]))
    return ngrams


def _count_ngrams_chunk(
    sentences: List[List[str]], order: int
) -> Tuple[Dict[Tuple[str, ...], int], int]:
    """Count n-grams for a chunk of sentences (picklable worker function)."""
    counts: Dict[Tuple[str, ...], int] = {}
    total = 0
    for sent in sentences:
        for o in range(1, order + 1):
            for ng in extract_ngrams(sent, o):
                counts[ng] = counts.get(ng, 0) + 1
                if o == 1 and ng[0] not in ("<s>", "</s>"):
                    total += 1
    return counts, total


def count_ngrams(
    sentences: List[List[str]], order: int,
    num_workers: int = 0,
) -> Tuple[Dict[Tuple[str, ...], int], int]:
    """Count n-grams across a corpus.

    Args:
        sentences: List of tokenized sentences.
        order: Maximum n-gram order.
        num_workers: Number of parallel workers. 0 = auto-detect.
            1 = sequential. Negative = sequential.

    Returns:
        (ngram_counts: dict mapping ngram→count, total_tokens: int)
    """
    # Resolve worker count
    _num_workers = num_workers
    if _num_workers == 0:
        _num_workers = max(1, (os.cpu_count() or 1) - 1)

    # Small corpus: don't bother with parallel overhead
    if _num_workers <= 1 or len(sentences) < _num_workers * 10:
        counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        total = 0
        for sent in sentences:
            for o in range(1, order + 1):
                for ng in extract_ngrams(sent, o):
                    counts[ng] += 1
                    if o == 1 and ng[0] not in ("<s>", "</s>"):
                        total += 1
        return counts, total

    # ── Parallel counting ──
    logger.info(f"[Parallel: {_num_workers} workers] Counting n-grams")
    chunk_size = max(1, len(sentences) // _num_workers)
    chunks = [
        (sentences[i:i + chunk_size], order)
        for i in range(0, len(sentences), chunk_size)
    ]

    with Pool(_num_workers) as pool:
        results = pool.starmap(_count_ngrams_chunk, chunks)

    # Merge results
    counts: Dict[Tuple[str, ...], int] = defaultdict(int)
    total = 0
    for chunk_counts, chunk_total in results:
        for ng, cnt in chunk_counts.items():
            counts[ng] += cnt
        total += chunk_total

    return counts, total


# ─── Kneser-Ney Language Model ───────────────────────────────────────


class KneserNeyLM:
    """Interpolated Kneser-Ney N-gram language model.

    Implements modified Kneser-Ney smoothing with three discount
    parameters (D1, D2, D3+) for each n-gram order.

    Attributes:
        order: Maximum n-gram order.
        counts: Nested dict: order → (ngram_tuple → count)
        kn_counts: Kneser-Ney continuation counts
        discounts: Per-order discount parameters (D1, D2, D3)
        vocab_size: Vocabulary size (types)
    """

    def __init__(self, order: int = 5, smoothing: str = "kneser_ney"):
        self.order = order
        self.smoothing = smoothing

        # Raw counts per order
        self.counts: Dict[int, Dict[Tuple[str, ...], int]] = {
            o: {} for o in range(1, order + 1)
        }

        # Kneser-Ney continuation counts
        self.kn_counts: Dict[int, Dict[Tuple[str, ...], int]] = {
            o: {} for o in range(1, order + 1)
        }

        # Discount parameters (D1, D2, D3+) per order
        self.discounts: Dict[int, Tuple[float, float, float]] = {}

        # Context counts: for each (order-1)-gram, how many distinct
        # words can follow it (used in KN smoothing)
        self.follow_counts: Dict[int, Dict[Tuple[str, ...], int]] = {}

        # Preceding counts: for each unigram, how many distinct contexts
        # can precede it (used in KN lower-order estimation)
        self.preceding_counts: Dict[str, int] = defaultdict(int)

        # Vocabulary
        self.vocab: Dict[str, int] = {"<s>": 0, "</s>": 1}
        self.vocab_size: int = 2

        # Number of sentences
        self.num_sentences: int = 0

    # ─── Training ─────────────────────────────────────────────────

    def train(
        self, sentences: List[List[str]], num_workers: int = 0
    ) -> None:
        """Train the language model on tokenized sentences.

        Args:
            sentences: List of tokenized sentences.
            num_workers: Number of parallel workers for n-gram counting.
                0 = auto-detect, 1 = sequential.
        """
        logger.info(f"Training {self.order}-gram Kneser-Ney LM on {len(sentences)} sentences")

        # ── Build vocabulary (sequential, must preserve ordering) ──
        for sent in sentences:
            for token in sent:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        # ── Count n-grams (parallelizable) ──
        _num_workers = num_workers
        if _num_workers == 0:
            _num_workers = max(1, (os.cpu_count() or 1) - 1)

        # Force sequential counting: parallel path has a confirmed bug
        # that produces identical n-gram counts for orders 3-5.
        # See code audit §4C for details.
        if _num_workers > 1:
            logger.warning(
                f"Parallel LM counting disabled due to known bug. "
                f"Using sequential (workers=1) instead of {_num_workers}."
            )
        # Sequential counting
        for sent in sentences:
            for o in range(1, self.order + 1):
                for ng in extract_ngrams(sent, o):
                    self.counts[o][ng] = self.counts[o].get(ng, 0) + 1

        self.vocab_size = len(self.vocab)
        self.num_sentences = len(sentences)
        logger.info(
            f"Corpus stats: {sum(self.counts[1].values())} tokens, "
            f"{self.vocab_size} types"
        )

        # Compute Kneser-Ney continuation counts
        self._compute_continuation_counts()

        # Estimate discount parameters
        self._estimate_discounts()

        # Compute follow counts for higher-order contexts
        self._compute_follow_counts()

    def _compute_continuation_counts(self) -> None:
        """Compute Kneser-Ney continuation counts.

        For order > 1: KN count = number of unique contexts in which
        the n-gram appears as a continuation.

        For order 1: KN count = number of unique preceding words + 1
        (for </s>).
        """
        # For higher orders: count unique (n-1)-gram contexts
        for o in range(2, self.order + 1):
            cont: Dict[Tuple[str, ...], int] = defaultdict(set)
            for ng in self.counts[o]:
                # The continuation is the last o-1 words
                continuation = ng[1:]  # drop first word (context)
                context = ng[:1]  # first word only is the varying part
                cont[continuation].add(context[0])

            self.kn_counts[o] = {
                k: len(v) for k, v in cont.items()
            }

        # For unigrams: count of preceding distinct contexts
        # Count how many different words precede each unigram
        preceding: Dict[str, set] = defaultdict(set)
        for bigram, count in self.counts.get(2, {}).items():
            if bigram[1] != "</s>":
                preceding[bigram[1]].add(bigram[0])

        self.preceding_counts = {
            w: len(ctxs) for w, ctxs in preceding.items()
        }
        self.preceding_counts["</s>"] = 1  # always followed by end

        # KN unigram count = number of preceding contexts
        for w, cnt in self.preceding_counts.items():
            self.kn_counts[1][(w,)] = cnt

    def _estimate_discounts(self) -> None:
        """Estimate modified Kneser-Ney discount parameters.

        Uses the counts of counts (N1, N2, N3, N4) for each order.
        D1 = 1 - 2Y * N2/N1
        D2 = 2 - 3Y * N3/N2
        D3+ = 3 - 4Y * N4/N3
        where Y = N1 / (N1 + 2*N2)
        """
        for o in range(1, self.order + 1):
            # Counts of counts
            count_dist = Counter(self.counts[o].values())
            n1 = count_dist.get(1, 0)
            n2 = count_dist.get(2, 0)
            n3 = count_dist.get(3, 0)
            n4 = count_dist.get(4, 0)

            if n1 == 0 or n2 == 0:
                # Fall back to fixed discounts
                self.discounts[o] = (0.5, 1.0, 1.5)
                continue

            Y = n1 / (n1 + 2 * n2)
            d1 = 1 - 2 * Y * n2 / n1
            d2 = 2 - 3 * Y * n3 / n2 if n2 > 0 else 1.0
            d3 = 3 - 4 * Y * n4 / n3 if n3 > 0 else 1.5

            # Clamp to reasonable values
            d1 = max(0.0, min(d1, 1.0))
            d2 = max(0.0, min(d2, 2.0))
            d3 = max(0.0, min(d3, 3.0))

            self.discounts[o] = (d1, d2, d3)

        logger.info(f"Discounts: { {o: f'{v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}' for o, v in self.discounts.items()} }")

    def _compute_follow_counts(self) -> None:
        """Compute how many unique words follow each context."""
        self.follow_counts = {}
        for o in range(2, self.order + 1):
            fc: Dict[Tuple[str, ...], int] = defaultdict(set)
            for ng in self.counts[o]:
                context = ng[:-1]
                fc[context].add(ng[-1])

            self.follow_counts[o - 1] = {
                ctx: len(words) for ctx, words in fc.items()
            }

    # ─── Probability ──────────────────────────────────────────────

    def _discount(self, count: int, order: int) -> float:
        """Apply modified Kneser-Ney discount."""
        d1, d2, d3 = self.discounts.get(order, (0.5, 1.0, 1.5))
        if count == 0:
            return 0.0
        elif count == 1:
            return d1
        elif count == 2:
            return d2
        else:
            return d3

    def _continuation_prob(self, ngram: Tuple[str, ...]) -> float:
        """Compute KN continuation probability for an n-gram.

        Uses recursive interpolation:
        P_KN(w|h) = max(c_KN(h,w) - D, 0) / c_KN(h·) +
                    λ(h) * P_KN(w|h')
        """
        order = len(ngram)

        if order == 1:
            # Unigram: KN count / total KN count
            w = ngram[0]
            kn_count = self.kn_counts[1].get((w,), 0)
            total_kn = max(sum(self.kn_counts[1].values()), 1)
            return kn_count / total_kn

        # Higher order
        context = ngram[:-1]
        word = ngram[-1]

        # KN count for this n-gram
        kn_count = self.kn_counts[order].get(ngram, 0)
        d = self._discount(kn_count, order)

        # Total KN count for this context
        total_context_kn = self.kn_counts.get(order - 1, {}).get(context, 0)
        if total_context_kn == 0:
            # Fall back to lower order
            return self._continuation_prob(ngram[1:])

        # Interpolated probability
        numerator = max(kn_count - d, 0) / total_context_kn

        # Normalization factor λ
        n_follow = self.follow_counts.get(order - 1, {}).get(context, 0)
        lambda_factor = d * n_follow / total_context_kn if total_context_kn > 0 else 1.0

        # Lower-order probability with fallback
        lower_prob = self._continuation_prob(ngram[1:])

        return numerator + lambda_factor * lower_prob

    def prob(self, word: str, history: Tuple[str, ...]) -> float:
        """Compute P(word | history) using Kneser-Ney smoothing.

        Args:
            word: The word to evaluate.
            history: Context words (tuples), length 0 to order-1.

        Returns:
            Probability P(word | history).
        """
        # Build n-gram of appropriate length
        max_order = min(len(history) + 1, self.order)
        ngram = history[-(max_order - 1):] + (word,) if max_order > 1 else (word,)

        # Ensure proper length
        if len(ngram) != max_order:
            ngram = (word,)

        return self._continuation_prob(ngram)

    def log_prob(self, word: str, history: Tuple[str, ...]) -> float:
        """Compute log10 P(word | history)."""
        p = self.prob(word, history)
        if p > 0:
            return math.log10(p)
        return -99.0  # floor

    def sentence_log_prob(self, tokens: List[str]) -> float:
        """Compute log10 probability of a complete sentence.

        Args:
            tokens: Tokenized sentence (without <s>/</s>).

        Returns:
            Log10 probability (base 10, as is standard in SMT).
        """
        log_prob = 0.0
        # Add start markers and iterate
        padded = ["<s>"] * (self.order - 1) + tokens + ["</s>"]
        for i in range(self.order - 1, len(padded)):
            word = padded[i]
            history = tuple(padded[max(0, i - self.order + 1):i])
            log_prob += self.log_prob(word, history)
        return log_prob

    # ─── Perplexity ───────────────────────────────────────────────

    def perplexity(self, tokens: List[str]) -> float:
        """Compute perplexity of a sentence."""
        log_prob = self.sentence_log_prob(tokens)
        n = len(tokens) + 1  # +1 for </s>
        return 10.0 ** (-log_prob / n) if n > 0 else float('inf')

    # ─── Persistence ──────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save LM to JSON format."""
        data = {
            "order": self.order,
            "smoothing": self.smoothing,
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "num_sentences": self.num_sentences,
            "counts": {str(o): {json.dumps(list(k)): v for k, v in d.items()}
                       for o, d in self.counts.items()},
            "kn_counts": {str(o): {json.dumps(list(k)): v for k, v in d.items()}
                          for o, d in self.kn_counts.items()},
            "discounts": {str(o): list(v) for o, v in self.discounts.items()},
            "follow_counts": {str(o): {json.dumps(list(k)): v for k, v in d.items()}
                              for o, d in self.follow_counts.items()},
            "preceding_counts": self.preceding_counts,
        }
        utils.ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as f:
            if ujson:
                ujson.dump(data, f, ensure_ascii=False)
            else:
                json.dump(data, f, ensure_ascii=False)
        logger.info(f"LM saved to {path}")

    @classmethod
    def load(cls, path: str) -> "KneserNeyLM":
        """Load LM from JSON or pickle format.
        
        Checks for a companion .pkl file first (much faster for large models).
        If .pkl is newer than .json, loads from pickle directly.
        Otherwise loads from .json and creates/updates the .pkl cache.
        """
        import pickle as _pickle
        json_path = path if path.endswith('.json') else path
        pkl_path = json_path.replace('.json', '.pkl')
        
        # Fast path: pickle exists and is newer
        if os.path.exists(pkl_path):
            json_mtime = os.path.getmtime(json_path) if os.path.exists(json_path) else 0
            pkl_mtime = os.path.getmtime(pkl_path)
            if pkl_mtime >= json_mtime:
                with open(pkl_path, 'rb') as f:
                    lm = _pickle.load(f)
                logger.info(f"LM loaded from pickle (order={lm.order}, vocab={lm.vocab_size}, {os.path.getsize(pkl_path)/1024/1024:.0f}MB)")
                return lm
        
        # Slow path: load from JSON
        with open(json_path, encoding="utf-8") as f:
            if ujson:
                data = ujson.load(f)
            else:
                data = json.load(f)

        lm = cls(order=int(data["order"]), smoothing=data.get("smoothing", "kneser_ney"))
        lm.vocab = data["vocab"]
        lm.vocab_size = int(data["vocab_size"])
        lm.num_sentences = int(data["num_sentences"])

        # Keys are stored as JSON arrays ["w1","w2"] (new) or str(tuple) (legacy).
        # JSON array parsing is ~10x faster than ast.literal_eval on str(tuple).
        import ast
        def _parse_key(k: str):
            """Parse key to tuple. Tries JSON array first, falls back to legacy str(tuple)."""
            if k.startswith('['):
                try:
                    return tuple(json.loads(k))
                except (json.JSONDecodeError, ValueError):
                    pass
            try:
                return ast.literal_eval(k)
            except (ValueError, SyntaxError):
                return k

        # Restore counts (keys are tuples, stored as string representations)
        lm.counts = {int(o): {_parse_key(k): v for k, v in d.items()}
                     for o, d in data["counts"].items()}
        lm.kn_counts = {int(o): {_parse_key(k): v for k, v in d.items()}
                        for o, d in data["kn_counts"].items()}
        lm.discounts = {int(o): tuple(v) for o, v in data["discounts"].items()}
        lm.follow_counts = {int(o): {_parse_key(k): v for k, v in d.items()}
                            for o, d in data["follow_counts"].items()}
        lm.preceding_counts = data["preceding_counts"]

        # Save pickle cache for faster future loads
        try:
            with open(pkl_path, 'wb') as f:
                _pickle.dump(lm, f, protocol=_pickle.HIGHEST_PROTOCOL)
            logger.info(f"Pickle cache saved to {pkl_path}")
        except Exception:
            pass  # Non-critical

        logger.info(f"LM loaded from {json_path} (order={lm.order}, vocab={lm.vocab_size})")
        return lm

    def prune(self, threshold: float = 1e-7) -> None:
        """Prune n-grams with probability below threshold.

        Reduces model size while preserving most accuracy.
        """
        for o in range(2, self.order + 1):
            to_remove = []
            for ng in self.counts[o]:
                context = ng[:-1]
                word = ng[-1]
                p = self.prob(word, context)
                if p < threshold and len(context) > 0:
                    # Also check if removing changes lower-order probs much
                    backoff_p = self._continuation_prob(ng[1:])
                    if abs(p - backoff_p) < threshold:
                        to_remove.append(ng)

            for ng in to_remove:
                del self.counts[o][ng]
                if ng in self.kn_counts[o]:
                    del self.kn_counts[o][ng]

            logger.info(f"Pruned {len(to_remove)} {o}-grams (threshold={threshold})")


# ─── Factory function ────────────────────────────────────────────────


def train_lm(
    sentences: List[List[str]],
    order: int = 5,
    smoothing: str = "kneser_ney",
    prune_threshold: float = 1e-7,
    num_workers: int = 0,
) -> KneserNeyLM:
    """Train and optionally prune a Kneser-Ney language model.

    Args:
        sentences: Tokenized training sentences.
        order: N-gram order.
        smoothing: Smoothing method.
        prune_threshold: Pruning threshold.
        num_workers: Number of parallel workers (0 = auto).

    Returns:
        Trained KneserNeyLM.
    """
    lm = KneserNeyLM(order=order, smoothing=smoothing)
    lm.train(sentences, num_workers=num_workers)

    if prune_threshold > 0:
        lm.prune(prune_threshold)

    return lm
