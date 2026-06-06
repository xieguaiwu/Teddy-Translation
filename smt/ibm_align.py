"""
IBM Model 1 & 2 word alignment using Expectation-Maximization.

Implements the standard IBM alignment models for statistical machine
translation. Model 1 learns lexical translation probabilities P(f|e)
with uniform alignment prior. Model 2 adds absolute distortion
probabilities P(j|i,l,m).

References:
    - Brown et al. (1993), "The Mathematics of Statistical Machine
      Translation: Parameter Estimation", Computational Linguistics.
    - Koehn (2010), "Statistical Machine Translation", Cambridge.
"""

import math
import os
import logging
import multiprocessing
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from . import utils

logger = utils.logger

# Type aliases
Table = Dict[str, Dict[str, float]]        # P(target|source)
Distortion = Dict[Tuple[int, int, int, int], float]  # P(j|i,l,m)


# ─── Data structures ─────────────────────────────────────────────────


def extract_vocab(sentences: List[List[str]]) -> Dict[str, int]:
    """Build a word-to-index vocabulary from tokenized sentences."""
    vocab: Dict[str, int] = {"NULL": 0}
    idx = 1
    for sent in sentences:
        for word in sent:
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def parallel_vocab(
    src_sentences: List[List[str]],
    tgt_sentences: List[List[str]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build source and target vocabularies."""
    return extract_vocab(src_sentences), extract_vocab(tgt_sentences)


# ─── Multiprocessing helpers for parallel E-step ─────────────────────

# Global references for worker processes (set before forking)
_IBM1_WORKER_T: Optional[Dict[str, Dict[str, float]]] = None
_IBM2_WORKER_T: Optional[Dict[str, Dict[str, float]]] = None
_IBM2_WORKER_A: Optional[Dict[Tuple[int, int, int, int], float]] = None
_IBM2_WORKER_SRC_VOCAB_SIZE: int = 1


def _ibm1_e_step_sentence(
    src_sent: List[str], tgt_sent: List[str]
) -> Tuple[Dict[Tuple[str, str], float], float]:
    """E-step for one sentence pair in IBM1 (picklable, uses global table).

    Computes expected counts and log-likelihood contribution for a single
    sentence pair. Reads from the module-level _IBM1_WORKER_T which
    should be set before spawning worker processes.
    """
    t = _IBM1_WORKER_T

    def _lookup(e: str, f: str) -> float:
        inner = t.get(e, {})
        if isinstance(inner, dict):
            return inner.get(f, 0.0)
        return 0.0

    m = len(src_sent)
    l = len(tgt_sent)
    counts: Dict[Tuple[str, str], float] = {}
    ll = 0.0

    for j, f_j in enumerate(src_sent):
        denom = 0.0
        for i in range(l + 1):
            denom += _lookup(tgt_sent[i - 1], f_j) if i > 0 else _lookup("NULL", f_j)

        if denom == 0:
            continue

        for i, e_i in enumerate(tgt_sent):
            delta = _lookup(e_i, f_j) / denom
            counts[(e_i, f_j)] = counts.get((e_i, f_j), 0.0) + delta

        delta_null = _lookup("NULL", f_j) / denom
        counts[("NULL", f_j)] = counts.get(("NULL", f_j), 0.0) + delta_null

        ll += math.log(denom)

    return counts, ll


def _ibm2_e_step_sentence(
    src_sent: List[str], tgt_sent: List[str]
) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[int, int, int, int], float], float]:
    """E-step for one sentence pair in IBM2 (picklable, uses globals).

    Returns (t_counts, a_counts, log_likelihood).
    """
    t = _IBM2_WORKER_T
    a = _IBM2_WORKER_A
    src_vocab_size = _IBM2_WORKER_SRC_VOCAB_SIZE

    def _lookup_t(e: str, f: str) -> float:
        inner = t.get(e, {})
        if isinstance(inner, dict):
            return inner.get(f, 1.0 / max(src_vocab_size, 1))
        return 1.0 / max(src_vocab_size, 1)

    def _get_d(j: int, i: int, l: int, m: int) -> float:
        return a.get((j, i, l, m), 1.0 / max(m, 1))

    m = len(src_sent)
    l = len(tgt_sent)
    t_counts: Dict[Tuple[str, str], float] = {}
    a_counts: Dict[Tuple[int, int, int, int], float] = {}
    ll = 0.0

    for j, f_j in enumerate(src_sent):
        denom = 0.0
        for i in range(l + 1):
            e_i = tgt_sent[i - 1] if i > 0 else "NULL"
            p = _lookup_t(e_i, f_j)
            dist = _get_d(j, i, l, m)
            denom += p * dist

        if denom == 0:
            continue

        for i in range(l + 1):
            e_i = tgt_sent[i - 1] if i > 0 else "NULL"
            p = _lookup_t(e_i, f_j)
            dist = _get_d(j, i, l, m)
            delta = (p * dist) / denom
            t_counts[(e_i, f_j)] = t_counts.get((e_i, f_j), 0.0) + delta
            a_counts[(j, i, l, m)] = a_counts.get((j, i, l, m), 0.0) + delta

        ll += math.log(denom)

    return t_counts, a_counts, ll


# ─── IBM Model 1 ─────────────────────────────────────────────────────


class IBM1:
    """IBM Model 1: lexical translation with uniform alignment.

    P(f|e) is learned via EM from sentence-aligned parallel data.
    Assumes all alignments are equally likely a priori.
    """

    def __init__(self, null_prob: float = 0.08):
        self.t: Table = defaultdict(lambda: defaultdict(float))  # t(f|e)
        self.null_prob = null_prob
        self._src_vocab: Set[str] = set()
        self._tgt_vocab: Set[str] = set()
        self._num_sentences: int = 0

    @property
    def num_params(self) -> int:
        """Number of non-zero translation parameters."""
        return sum(1 for e in self.t for f in self.t[e] if self.t[e][f] > 0)

    def initialize(self, src_sentences: List[List[str]], tgt_sentences: List[List[str]]) -> None:
        """Initialize t(f|e) uniformly from the parallel corpus vocabulary."""
        # Collect vocabularies
        for sent in src_sentences:
            self._src_vocab.update(sent)
        for sent in tgt_sentences:
            self._tgt_vocab.update(sent)
        self._num_sentences = len(src_sentences)

        # SPARSE initialization: only store entries for co-occurring pairs.
        # The dense |E|×|F| loop creates billions of entries for word-level
        # tokenization (e.g. 30K EN × 50K ZH = 1.5B entries = 100+GB).
        # Sparse init only creates ~100K entries for realistic sentence pairs.
        src_vocab_size = len(self._src_vocab)
        uniform = 1.0 / src_vocab_size if src_vocab_size > 0 else 1.0
        self._uniform = uniform  # fallback for unseen pairs
        cooccur = set()
        for src, tgt in zip(src_sentences, tgt_sentences):
            for e in tgt:
                for f in src:
                    cooccur.add((e, f))
            # Also add NULL alignments
            for f in src:
                cooccur.add(("NULL", f))
        for e, f in cooccur:
            self.t[e][f] = uniform

        logger.info(
            f"IBM1 initialized: |E|={len(self._tgt_vocab)}, "
            f"|F|={len(self._src_vocab)}, "
            f"pairs={self._num_sentences}, sparsity={len(cooccur)}/"
            f"{len(self._tgt_vocab)*len(self._src_vocab)}"
        )

    def _expectation_step(
        self, src_sent: List[str], tgt_sent: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """E-step: compute expected counts for one sentence pair.

        Args:
            src_sent: Source sentence (list of tokens).
            tgt_sent: Target sentence (list of tokens).

        Returns:
            Dict[(e, f), count]: Expected counts.
        """
        m = len(src_sent)
        l = len(tgt_sent)
        counts: Dict[Tuple[str, str], float] = defaultdict(float)

        for j, f_j in enumerate(src_sent):
            # Normalization factor: sum over all target positions i=0..l
            denom = sum(
                self.t[tgt_sent[i - 1]][f_j] if i > 0 else self.t["NULL"][f_j]
                for i in range(l + 1)
            )
            if denom == 0:
                continue

            for i, e_i in enumerate(tgt_sent):
                delta = self.t[e_i][f_j] / denom
                counts[(e_i, f_j)] += delta

            # NULL word
            delta_null = self.t["NULL"][f_j] / denom
            counts[("NULL", f_j)] += delta_null

        return counts

    def _maximization_step(
        self, expected_counts: Dict[Tuple[str, str], float]
    ) -> None:
        """M-step: re-estimate t(f|e) from expected counts."""
        # Sum over f for each e (normalization denominator)
        norm: Dict[str, float] = defaultdict(float)
        for (e, f), count in expected_counts.items():
            norm[e] += count

        # MLE
        for (e, f), count in expected_counts.items():
            if norm[e] > 0:
                self.t[e][f] = count / norm[e]

    def train(
        self,
        src_sentences: List[List[str]],
        tgt_sentences: List[List[str]],
        iterations: int = 5,
        parallel: bool = False,
        num_workers: int = 0,
    ) -> List[float]:
        """Run EM training for the specified number of iterations.

        Args:
            src_sentences: List of source sentences (token lists).
            tgt_sentences: List of target sentences (token lists).
            iterations: Number of EM iterations.
            parallel: If True, parallelize the E-step across sentence pairs.
            num_workers: Number of worker processes (0 = auto: cpu_count-1).
                Only used when parallel=True.

        Returns:
            List of log-likelihood values per iteration.
        """
        self.initialize(src_sentences, tgt_sentences)
        log_likelihoods: List[float] = []

        # Resolve worker count
        if parallel:
            if num_workers <= 0:
                num_workers = max(1, (os.cpu_count() or 1) - 1)
            logger.info(f"[Parallel: {num_workers} workers] IBM1 E-step")

        for it in range(iterations):
            total_counts: Dict[Tuple[str, str], float] = defaultdict(float)
            ll = 0.0

            if parallel and num_workers > 1:
                # ── Parallel E-step ──
                global _IBM1_WORKER_T
                _IBM1_WORKER_T = self.t  # set before fork, inherited by workers

                valid_pairs = [
                    (s, t) for s, t in zip(src_sentences, tgt_sentences)
                    if s and t
                ]

                try:
                    ctx = multiprocessing.get_context('fork')
                except ValueError:
                    logger.warning(
                        "Fork start method not available; falling back to sequential E-step"
                    )
                    ctx = None

                if ctx is not None:
                    with ctx.Pool(num_workers) as pool:
                        results = pool.starmap(_ibm1_e_step_sentence, valid_pairs)

                    for sent_counts, sent_ll in results:
                        for pair, count in sent_counts.items():
                            total_counts[pair] += count
                        ll += sent_ll
                else:
                    # Fallback: sequential
                    for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
                        if not src_sent or not tgt_sent:
                            continue
                        sent_counts = self._expectation_step(src_sent, tgt_sent)
                        for pair, count in sent_counts.items():
                            total_counts[pair] += count
                        m, l = len(src_sent), len(tgt_sent)
                        sent_ll = 0.0
                        for j, f_j in enumerate(src_sent):
                            sum_align = sum(
                                self.t[tgt_sent[i - 1]][f_j] if i > 0 else self.t["NULL"][f_j]
                                for i in range(l + 1)
                            )
                            if sum_align > 0:
                                sent_ll += math.log(sum_align)
                        ll += sent_ll
            else:
                # ── Sequential E-step ──
                for sent_idx, (src_sent, tgt_sent) in enumerate(zip(src_sentences, tgt_sentences)):
                    if not src_sent or not tgt_sent:
                        continue

                    # E-step for this sentence pair
                    sent_counts = self._expectation_step(src_sent, tgt_sent)

                    # Accumulate counts
                    for pair, count in sent_counts.items():
                        total_counts[pair] += count

                    # Log-likelihood contribution
                    m = len(src_sent)
                    l = len(tgt_sent)
                    sent_ll = 0.0
                    for j, f_j in enumerate(src_sent):
                        sum_align = sum(
                            self.t[tgt_sent[i - 1]][f_j] if i > 0 else self.t["NULL"][f_j]
                            for i in range(l + 1)
                        )
                        if sum_align > 0:
                            sent_ll += math.log(sum_align)
                    ll += sent_ll

            # M-step
            self._maximization_step(total_counts)

            log_likelihoods.append(ll)
            logger.info(
                f"IBM1 iter {it + 1}/{iterations}: "
                f"log-likelihood = {ll:.1f}, "
                f"params = {self.num_params}"
            )

        return log_likelihoods

    def align(self, src_sent: List[str], tgt_sent: List[str]) -> List[int]:
        """Find the best alignment (Viterbi) for a sentence pair.

        For IBM1: align each source word to the target word with
        highest t(f_j|e_i). Uses stored probability or uniform fallback.
        """
        uniform = getattr(self, '_uniform', 1e-10)
        alignment: List[int] = []
        for j, f_j in enumerate(src_sent):
            best_i = 0
            empty = self.t.get("NULL", {})
            best_p = empty.get(f_j, uniform)
            for i, e_i in enumerate(tgt_sent):
                inner = self.t.get(e_i, {})
                p = inner.get(f_j, uniform)
                if p > best_p:
                    best_p = p
                    best_i = i + 1  # 1-indexed target position
            alignment.append(best_i)
        return alignment

    def save(self, path: str) -> None:
        """Save translation table to text file (Moses-style)."""
        lines: List[str] = []
        for e in self.t:
            for f, p in self.t[e].items():
                if p > 0:
                    lines.append(f"{e} {f} {p:.10f}")
        utils.write_lines(lines, path)
        logger.info(f"IBM1 table saved to {path} ({len(lines)} entries)")

    def load(self, path: str) -> None:
        """Load translation table from Moses-style text file."""
        self.t.clear()
        lines = utils.read_lines(path)
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                e, f = parts[0], parts[1]
                p = float(parts[2])
                self.t[e][f] = p
        logger.info(f"IBM1 table loaded from {path} ({len(lines)} entries)")


# ─── IBM Model 2 ─────────────────────────────────────────────────────


class IBM2:
    """IBM Model 2: lexical translation + absolute distortion.

    Adds a distortion model P(j|i,l,m) that captures the tendency
    of words at position i in the target to align to position j
    in the source, given sentence lengths l (target) and m (source).
    """

    def __init__(self, null_prob: float = 0.08):
        self.t: Table = defaultdict(lambda: defaultdict(float))  # t(f|e)
        # a(j|i,l,m): distortion — prob source pos j aligns to target pos i
        self.a: Dict[Tuple[int, int, int, int], float] = {}
        self.null_prob = null_prob
        self._src_vocab: Set[str] = set()
        self._tgt_vocab: Set[str] = set()
        self._num_sentences: int = 0

    @property
    def num_params_t(self) -> int:
        return sum(1 for e in self.t for f in self.t[e] if self.t[e][f] > 0)

    @property
    def num_params_a(self) -> int:
        return len(self.a)

    def initialize(self, src_sentences: List[List[str]], tgt_sentences: List[List[str]]) -> None:
        """Initialize with IBM1-style uniform t(f|e) and uniform a(j|i,l,m).
        Uses sparse initialization to avoid |E|×|F| memory explosion."""
        for sent in src_sentences:
            self._src_vocab.update(sent)
        for sent in tgt_sentences:
            self._tgt_vocab.update(sent)
        self._num_sentences = len(src_sentences)

        # Sparse uniform t(f|e)
        src_vocab_size = len(self._src_vocab)
        uniform_t = 1.0 / src_vocab_size if src_vocab_size > 0 else 1.0
        self._uniform = uniform_t
        cooccur = set()
        for src, tgt in zip(src_sentences, tgt_sentences):
            for e in tgt:
                for f in src:
                    cooccur.add((e, f))
            for f in src:
                cooccur.add(("NULL", f))
        for e, f in cooccur:
            self.t[e][f] = uniform_t

        logger.info(
            f"IBM2 initialized: |E|={len(self._tgt_vocab)}, "
            f"|F|={len(self._src_vocab)}, pairs={self._num_sentences}, "
            f"sparsity={len(cooccur)}/{len(self._tgt_vocab)*len(self._src_vocab)}"
        )

    def _get_distortion(
        self, j: int, i: int, l: int, m: int
    ) -> float:
        """Get distortion probability a(j|i,l,m)."""
        key = (j, i, l, m)
        return self.a.get(key, 1.0 / max(m, 1))

    def _expectation_step(
        self, src_sent: List[str], tgt_sent: List[str]
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[int, int, int, int], float]]:
        """E-step for one sentence pair."""
        m = len(src_sent)   # source length
        l = len(tgt_sent)   # target length
        t_counts: Dict[Tuple[str, str], float] = defaultdict(float)
        a_counts: Dict[Tuple[int, int, int, int], float] = defaultdict(float)

        for j, f_j in enumerate(src_sent):
            # Normalization: sum over all target positions i=0..l
            denom = 0.0
            for i in range(l + 1):
                e_i = tgt_sent[i - 1] if i > 0 else "NULL"
                p = self.t.get(e_i, {}).get(f_j, 1.0 / max(len(self._src_vocab), 1))
                dist = self._get_distortion(j, i, l, m)
                denom += p * dist

            if denom == 0:
                continue

            for i in range(l + 1):
                e_i = tgt_sent[i - 1] if i > 0 else "NULL"
                p = self.t.get(e_i, {}).get(f_j, 1.0 / max(len(self._src_vocab), 1))
                dist = self._get_distortion(j, i, l, m)
                delta = (p * dist) / denom

                t_counts[(e_i, f_j)] += delta
                a_counts[(j, i, l, m)] += delta

        return t_counts, a_counts

    def _maximization_step(
        self,
        t_counts: Dict[Tuple[str, str], float],
        a_counts: Dict[Tuple[int, int, int, int], float],
    ) -> None:
        """M-step: re-estimate both t(f|e) and a(j|i,l,m)."""
        # Re-estimate t(f|e)
        t_norm: Dict[str, float] = defaultdict(float)
        for (e, f), count in t_counts.items():
            t_norm[e] += count
        for (e, f), count in t_counts.items():
            if t_norm[e] > 0:
                self.t[e][f] = count / t_norm[e]

        # Re-estimate a(j|i,l,m) — group by (i,l,m) for normalization
        a_norm: Dict[Tuple[int, int, int], float] = defaultdict(float)
        for (j, i, l, m), count in a_counts.items():
            a_norm[(i, l, m)] += count
        for (j, i, l, m), count in a_counts.items():
            norm_key = (i, l, m)
            if a_norm[norm_key] > 0:
                self.a[(j, i, l, m)] = count / a_norm[norm_key]

    def train(
        self,
        src_sentences: List[List[str]],
        tgt_sentences: List[List[str]],
        iterations: int = 5,
        parallel: bool = False,
        num_workers: int = 0,
    ) -> List[float]:
        """Run EM training.

        Args:
            src_sentences: Source sentences (token lists).
            tgt_sentences: Target sentences (token lists).
            iterations: Number of EM iterations.
            parallel: If True, parallelize the E-step across sentence pairs.
            num_workers: Number of worker processes (0 = auto: cpu_count-1).

        Returns:
            Log-likelihoods per iteration.
        """
        self.initialize(src_sentences, tgt_sentences)
        log_likelihoods: List[float] = []

        # Resolve worker count
        if parallel:
            if num_workers <= 0:
                num_workers = max(1, (os.cpu_count() or 1) - 1)
            logger.info(f"[Parallel: {num_workers} workers] IBM2 E-step")

        for it in range(iterations):
            total_t_counts: Dict[Tuple[str, str], float] = defaultdict(float)
            total_a_counts: Dict[Tuple[int, int, int, int], float] = defaultdict(float)
            ll = 0.0

            if parallel and num_workers > 1:
                # ── Parallel E-step ──
                global _IBM2_WORKER_T, _IBM2_WORKER_A, _IBM2_WORKER_SRC_VOCAB_SIZE
                _IBM2_WORKER_T = self.t
                _IBM2_WORKER_A = self.a
                _IBM2_WORKER_SRC_VOCAB_SIZE = len(self._src_vocab)

                valid_pairs = [
                    (s, t) for s, t in zip(src_sentences, tgt_sentences)
                    if s and t
                ]

                try:
                    ctx = multiprocessing.get_context('fork')
                except ValueError:
                    logger.warning(
                        "Fork start method not available; falling back to sequential E-step"
                    )
                    ctx = None

                if ctx is not None:
                    with ctx.Pool(num_workers) as pool:
                        results = pool.starmap(_ibm2_e_step_sentence, valid_pairs)

                    for t_c, a_c, sent_ll in results:
                        for k, v in t_c.items():
                            total_t_counts[k] += v
                        for k, v in a_c.items():
                            total_a_counts[k] += v
                        ll += sent_ll
                else:
                    # Fallback: sequential
                    for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
                        if not src_sent or not tgt_sent:
                            continue
                        m, l = len(src_sent), len(tgt_sent)
                        t_c, a_c = self._expectation_step(src_sent, tgt_sent)
                        for k, v in t_c.items():
                            total_t_counts[k] += v
                        for k, v in a_c.items():
                            total_a_counts[k] += v
                        sent_ll = 0.0
                        for j, f_j in enumerate(src_sent):
                            sum_align = 0.0
                            for i in range(l + 1):
                                e_i = tgt_sent[i - 1] if i > 0 else "NULL"
                                p = self.t.get(e_i, {}).get(f_j, 0.0)
                                dist = self._get_distortion(j, i, l, m)
                                sum_align += p * dist
                            if sum_align > 0:
                                sent_ll += math.log(sum_align)
                        ll += sent_ll
            else:
                # ── Sequential E-step ──
                for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
                    if not src_sent or not tgt_sent:
                        continue

                    m = len(src_sent)
                    l = len(tgt_sent)

                    t_c, a_c = self._expectation_step(src_sent, tgt_sent)
                    for k, v in t_c.items():
                        total_t_counts[k] += v
                    for k, v in a_c.items():
                        total_a_counts[k] += v

                    # Log-likelihood
                    sent_ll = 0.0
                    for j, f_j in enumerate(src_sent):
                        sum_align = 0.0
                        for i in range(l + 1):
                            e_i = tgt_sent[i - 1] if i > 0 else "NULL"
                            p = self.t.get(e_i, {}).get(f_j, 0.0)
                            dist = self._get_distortion(j, i, l, m)
                            sum_align += p * dist
                        if sum_align > 0:
                            sent_ll += math.log(sum_align)
                    ll += sent_ll

            # M-step
            self._maximization_step(total_t_counts, total_a_counts)

            log_likelihoods.append(ll)
            logger.info(
                f"IBM2 iter {it + 1}/{iterations}: "
                f"log-likelihood = {ll:.1f}, "
                f"t_params = {self.num_params_t}, "
                f"a_params = {self.num_params_a}"
            )

        return log_likelihoods

    def align(self, src_sent: List[str], tgt_sent: List[str]) -> List[int]:
        """Viterbi alignment using both t(f|e) and a(j|i,l,m).

        Args:
            src_sent: Source tokens.
            tgt_sent: Target tokens.

        Returns:
            List of target indices (1-indexed, 0=NULL) for each source token.
        """
        m = len(src_sent)
        l = len(tgt_sent)
        alignment: List[int] = []

        for j, f_j in enumerate(src_sent):
            best_i = 0
            best_p = self.t["NULL"][f_j] * self._get_distortion(j, 0, l, m)
            for i, e_i in enumerate(tgt_sent):
                p = self.t[e_i][f_j] * self._get_distortion(j, i + 1, l, m)
                if p > best_p:
                    best_p = p
                    best_i = i + 1
            alignment.append(best_i)

        return alignment

    def extract_alignments(
        self, src_sentences: List[List[str]], tgt_sentences: List[List[str]]
    ) -> List[List[Tuple[int, int]]]:
        """Extract symmetrized word alignments for a corpus.

        Returns alignments as (src_pos, tgt_pos) pairs (0-indexed).
        Uses Model 2 alignments (intersection with Model 1 as fallback).
        """
        alignments: List[List[Tuple[int, int]]] = []
        for src_sent, tgt_sent in zip(src_sentences, tgt_sentences):
            src_align = self.align(src_sent, tgt_sent)
            # src_align[j] = target position (1-indexed, 0=null)
            pairs = []
            for j, tgt_idx in enumerate(src_align):
                if tgt_idx > 0:
                    pairs.append((j, tgt_idx - 1))
            alignments.append(pairs)
        return alignments

    def save(self, path_t: str, path_a: Optional[str] = None) -> None:
        """Save translation and distortion tables."""
        lines_t = [f"{e} {f} {p:.10f}" for e in self.t for f, p in self.t[e].items() if p > 0]
        utils.write_lines(lines_t, path_t)

        if path_a:
            lines_a = [
                f"{j} {i} {l} {m} {p:.10f}"
                for (j, i, l, m), p in self.a.items()
            ]
            utils.write_lines(lines_a, path_a)
            logger.info(f"Distortion table saved to {path_a} ({len(lines_a)} entries)")

        logger.info(f"Translation table saved to {path_t} ({len(lines_t)} entries)")

    def load(self, path_t: str) -> None:
        """Load translation table from Moses-style text file."""
        self.t.clear()
        lines = utils.read_lines(path_t)
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                e, f = parts[0], parts[1]
                p = float(parts[2])
                self.t[e][f] = p
        logger.info(f"IBM2 table loaded from {path_t} ({len(lines)} entries)")


# ─── Training convenience ────────────────────────────────────────────


def train_ibm(
    src_sentences: List[List[str]],
    tgt_sentences: List[List[str]],
    model: str = "ibm2",
    iterations_model1: int = 5,
    iterations_model2: int = 5,
    null_prob: float = 0.08,
    num_workers: int = 0,
    warm_t_table=None,
) -> IBM2:
    """Train IBM alignment models (Model 1 then Model 2).

    Standard practice: use IBM1 parameters to initialize IBM2.
    If warm_t_table is provided, skip IBM1 and use the table
    to initialize IBM2 directly (warm-start).

    Args:
        src_sentences: Source sentences (token lists).
        tgt_sentences: Target sentences (token lists).
        model: "ibm1" or "ibm2".
        iterations_model1: EM iterations for Model 1.
        iterations_model2: EM iterations for Model 2.
        null_prob: NULL insertion probability.
        num_workers: Number of multiprocessing workers.
        warm_t_table: Pre-trained t(f|e) table for warm-starting IBM2.

    Returns:
        Trained IBM2 model.
    """
    use_parallel = num_workers >= 0
    if use_parallel and num_workers == 0:
        num_workers = max(1, (os.cpu_count() or 1) - 1)

    if warm_t_table is not None:
        # Warm-start: skip IBM1, use pre-trained table for IBM2
        logger.info(f"[Warm-start] Skipping IBM1, using pre-trained t-table")
        ibm2 = IBM2(null_prob=null_prob)
        ibm2._src_vocab = set(w for sent in src_sentences for w in sent)
        ibm2._tgt_vocab = set(w for sent in tgt_sentences for w in sent)
        ibm2.t = warm_t_table  # Use the pre-trained table directly
        ibm2.train(
            src_sentences, tgt_sentences,
            iterations=iterations_model2,
            parallel=use_parallel,
            num_workers=num_workers,
        )
        return ibm2

    if model == "ibm1":
        ibm1 = IBM1(null_prob=null_prob)
        ibm1.train(
            src_sentences, tgt_sentences,
            iterations=iterations_model1,
            parallel=use_parallel,
            num_workers=num_workers,
        )
        return ibm1  # type: ignore

    # Train IBM1 first for initialization
    ibm1 = IBM1(null_prob=null_prob)
    ibm1.train(
        src_sentences, tgt_sentences,
        iterations=iterations_model1,
        parallel=use_parallel,
        num_workers=num_workers,
    )

    # Transfer parameters to IBM2
    ibm2 = IBM2(null_prob=null_prob)
    ibm2._src_vocab = set(src_sentences[0] if src_sentences else [])
    ibm2._tgt_vocab = set(tgt_sentences[0] if tgt_sentences else [])
    ibm2.t = ibm1.t

    # Train IBM2
    ibm2.train(
        src_sentences, tgt_sentences,
        iterations=iterations_model2,
        parallel=use_parallel,
        num_workers=num_workers,
    )

    return ibm2
