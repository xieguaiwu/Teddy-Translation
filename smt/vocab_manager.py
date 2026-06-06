"""
Vocabulary management for SMT system.

Provides vocabulary building, frequency tracking, coverage analysis,
and serialization for efficient OOV handling in machine translation.
Supports both Chinese and English token-level vocabularies.

Usage:
    mgr = VocabularyManager.build_from_corpus(sentences, min_freq=2)
    mgr.save("vocab.json")
    mgr.coverage_report(test_sentences)
"""

import json
import os
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

from . import utils

logger = utils.logger


class VocabularyManager:
    """Manages vocabulary with frequency tracking and coverage analysis.

    Attributes:
        word_freq: Counter mapping word → frequency.
        word_to_id: Dict mapping word → integer ID.
        id_to_word: Dict mapping integer ID → word.
        total_tokens: Total token count seen during building.
        total_types: Number of unique words.
        min_freq: Minimum frequency threshold.
        max_size: Maximum vocabulary size (truncates least frequent).
    """

    def __init__(self) -> None:
        """Initialize an empty vocabulary manager."""
        self.word_freq: Counter = Counter()
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.total_tokens: int = 0
        self.total_types: int = 0
        self.min_freq: int = 1
        self.max_size: Optional[int] = None
        self._special_tokens: Set[str] = {"<unk>", "<s>", "</s>", "<pad>"}

    # ─── Construction ────────────────────────────────────────────────

    @classmethod
    def build_from_corpus(
        cls,
        sentences: List[List[str]],
        min_freq: int = 1,
        max_size: Optional[int] = None,
    ) -> "VocabularyManager":
        """Build vocabulary from a list of tokenized sentences.

        Args:
            sentences: List of tokenized sentences (each a list of str).
            min_freq: Minimum word frequency to include in vocabulary.
            max_size: Truncate vocabulary to this many most-frequent words.

        Returns:
            Configured VocabularyManager instance.
        """
        mgr = cls()
        mgr.min_freq = max(min_freq, 1)
        mgr.max_size = max_size

        # Count all tokens
        word_counter: Counter = Counter()
        for sent in sentences:
            word_counter.update(sent)

        mgr.total_tokens = sum(word_counter.values())

        # Filter by min_freq
        filtered = {
            word: count
            for word, count in word_counter.items()
            if count >= mgr.min_freq
        }

        # Truncate by max_size (keep most frequent)
        if max_size is not None and len(filtered) > max_size:
            most_common = Counter(filtered).most_common(max_size)
            filtered = dict(most_common)

        mgr.word_freq = Counter(filtered)
        mgr.total_types = len(mgr.word_freq)

        # Build id mappings
        for idx, word in enumerate(sorted(mgr.word_freq.keys())):
            mgr.word_to_id[word] = idx
            mgr.id_to_word[idx] = word

        # Add special token ids (always last)
        offset = len(mgr.id_to_word)
        for tok in mgr._special_tokens:
            if tok not in mgr.word_to_id:
                mgr.word_to_id[tok] = offset
                mgr.id_to_word[offset] = tok
                offset += 1

        logger.info(
            f"Vocabulary built: {mgr.total_types} types "
            f"(from {mgr.total_tokens} tokens, min_freq={mgr.min_freq}"
            f"{f', max_size={max_size}' if max_size else ''})"
        )

        return mgr

    # ─── Persistence ─────────────────────────────────────────────────

    def save(self, path: str) -> str:
        """Save vocabulary to a JSON file.

        Args:
            path: Output file path.

        Returns:
            Path to saved file.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data: Dict = {
            "word_freq": dict(self.word_freq.most_common()),
            "word_to_id": self.word_to_id,
            "total_tokens": self.total_tokens,
            "total_types": self.total_types,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Vocabulary saved to {path} ({self.total_types} types)")
        return path

    @classmethod
    def load(cls, path: str) -> "VocabularyManager":
        """Load vocabulary from a JSON file.

        Args:
            path: Input file path.

        Returns:
            Loaded VocabularyManager instance.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocabulary file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        mgr = cls()
        mgr.total_tokens = data.get("total_tokens", 0)
        mgr.total_types = data.get("total_types", 0)
        mgr.min_freq = data.get("min_freq", 1)
        mgr.max_size = data.get("max_size")
        mgr.word_freq = Counter(data.get("word_freq", {}))
        mgr.word_to_id = data.get("word_to_id", {})
        mgr.id_to_word = {v: k for k, v in mgr.word_to_id.items()}

        logger.info(
            f"Vocabulary loaded from {path} ({mgr.total_types} types, "
            f"{mgr.total_tokens} tokens)"
        )
        return mgr

    # ─── Query ───────────────────────────────────────────────────────

    def contains(self, word: str) -> bool:
        """Check if a word is in the vocabulary.

        Args:
            word: Token to check.

        Returns:
            True if word exists in vocabulary.
        """
        return word in self.word_to_id

    def get_freq(self, word: str) -> int:
        """Get frequency of a word.

        Args:
            word: Token to look up.

        Returns:
            Frequency count (0 if not in vocabulary).
        """
        return self.word_freq.get(word, 0)

    def get_id(self, word: str) -> int:
        """Get integer ID for a word.

        Args:
            word: Token to look up.

        Returns:
            Integer ID, or the <unk> ID if unknown.
        """
        if word in self.word_to_id:
            return self.word_to_id[word]
        # Return <unk> id; default to 0 if no special tokens added
        unk_id = self.word_to_id.get("<unk>")
        return unk_id if unk_id is not None else 0

    def lookup(self, word_id: int) -> str:
        """Get word for an integer ID.

        Args:
            word_id: ID to look up.

        Returns:
            Word string or '<unk>' if ID not found.
        """
        return self.id_to_word.get(word_id, "<unk>")

    def get_top_n(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get the top-N most frequent words.

        Args:
            n: Number of top words to return.

        Returns:
            List of (word, frequency) tuples.
        """
        return self.word_freq.most_common(n)

    @property
    def size(self) -> int:
        """Number of unique vocabulary types."""
        return self.total_types

    @property
    def special_tokens(self) -> Set[str]:
        """Set of special tokens."""
        return self._special_tokens

    # ─── Coverage Analysis ───────────────────────────────────────────

    def coverage_report(self, sentences: List[List[str]]) -> Dict[str, float]:
        """Compute OOV statistics over a set of sentences.

        Args:
            sentences: Tokenized sentences to analyze.

        Returns:
            Dict with:
                coverage: fraction of tokens in vocab (0.0 - 1.0)
                oov_rate: fraction of tokens NOT in vocab
                known_tokens: count of known tokens
                unknown_tokens: count of OOV tokens
                total_tokens: total tokens in input
                known_types: count of unique known types
                unknown_types: count of unique OOV types
                sent_coverage: fraction of sentences fully covered
        """
        total: int = 0
        known: int = 0
        unknown: int = 0
        unknown_types: Set[str] = set()
        known_types: Set[str] = set()
        fully_covered: int = 0

        for sent in sentences:
            if not sent:
                continue
            sent_known = sum(1 for tok in sent if self.contains(tok))
            total += len(sent)
            known += sent_known
            unknown += len(sent) - sent_known

            for tok in sent:
                if self.contains(tok):
                    known_types.add(tok)
                else:
                    unknown_types.add(tok)

            if sent_known == len(sent):
                fully_covered += 1

        coverage = known / total if total > 0 else 1.0
        oov_rate = 1.0 - coverage
        sent_coverage = fully_covered / len(sentences) if sentences else 1.0

        report: Dict[str, float] = {
            "coverage": coverage,
            "oov_rate": oov_rate,
            "known_tokens": known,
            "unknown_tokens": unknown,
            "total_tokens": total,
            "known_types": len(known_types),
            "unknown_types": len(unknown_types),
            "sent_coverage": sent_coverage,
        }

        logger.info(
            f"Coverage: {coverage:.2%} ({known}/{total} tokens), "
            f"OOV rate: {oov_rate:.2%}, "
            f"types: {len(known_types)} known / {len(unknown_types)} unknown"
        )

        return report

    def get_stats(self) -> Dict:
        """Get comprehensive vocabulary statistics.

        Returns:
            Dict with:
                total_types, total_tokens, min_freq, max_size,
                hapax_legomena: count of words occurring exactly once,
                freq_distribution: distribution bucket counts,
                mean_freq, median_freq, top_10, bottom_10.
        """
        freqs = list(self.word_freq.values())

        if not freqs:
            return {
                "total_types": 0,
                "total_tokens": 0,
                "min_freq": self.min_freq,
                "max_size": self.max_size,
            }

        freq_dist: Dict[str, int] = {}
        for f in freqs:
            if f == 1:
                bucket = "1"
            elif f <= 5:
                bucket = "2-5"
            elif f <= 10:
                bucket = "6-10"
            elif f <= 50:
                bucket = "11-50"
            elif f <= 100:
                bucket = "51-100"
            elif f <= 500:
                bucket = "101-500"
            else:
                bucket = "501+"
            freq_dist[bucket] = freq_dist.get(bucket, 0) + 1

        sorted_freqs = sorted(freqs)
        median_idx = len(sorted_freqs) // 2

        return {
            "total_types": self.total_types,
            "total_tokens": self.total_tokens,
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "hapax_legomena": sum(1 for f in freqs if f == 1),
            "freq_distribution": freq_dist,
            "mean_freq": sum(freqs) / len(freqs),
            "median_freq": sorted_freqs[median_idx] if sorted_freqs else 0.0,
            "min_freq_value": sorted_freqs[0] if sorted_freqs else 0,
            "max_freq_value": sorted_freqs[-1] if sorted_freqs else 0,
            "top_10": self.get_top_n(10),
        }

    # ─── Special token helpers ───────────────────────────────────────

    @property
    def unk_id(self) -> int:
        """Integer ID for <unk> token."""
        return self.word_to_id.get("<unk>", 0)

    @property
    def bos_id(self) -> int:
        """Integer ID for <s> token."""
        return self.word_to_id.get("<s>", 0)

    @property
    def eos_id(self) -> int:
        """Integer ID for </s> token."""
        return self.word_to_id.get("</s>", 0)
