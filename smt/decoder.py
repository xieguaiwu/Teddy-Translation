"""
Phrase-based beam search decoder for SMT.

Implements the standard SMT decoder with:
- Beam search with hypothesis stacks
- Future cost estimation for pruning
- Distortion (reordering) with limit
- Language model integration
- Phrase table lookup
- Word penalty

References:
    - Koehn (2004), "Pharaoh: a beam search decoder for phrase-based
      statistical machine translation", AMTA.
    - Koehn (2010), "Statistical Machine Translation", Ch. 6-7.
"""

import math
import heapq
from typing import Dict, List, Optional, Set, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass, field

from . import utils
from typing import Literal

logger = utils.logger

# OOV strategy type
OOVStrategy = Literal["copy", "drop", "unk"]

# ─── Types ───────────────────────────────────────────────────────────


@dataclass(order=True)
class Hypothesis:
    """A partial translation hypothesis in the beam search.

    Hypotheses are ordered by score (negative for max-heap via min-heap).
    """
    score: float = field(compare=True)      # Negative log-prob (lower = better)
    target_tokens: List[str] = field(compare=False, default_factory=list)
    source_covered: Set[int] = field(compare=False, default_factory=set)
    lm_history: Tuple[str, ...] = field(compare=False, default_factory=tuple)
    # Coverage bitmap as sorted tuple for hashability
    coverage_key: Tuple[int, ...] = field(compare=False, default_factory=tuple)
    num_phrases: int = field(compare=False, default=0)
    # Last source position (for distortion cost)
    last_src_pos: int = field(compare=False, default=-1)

    def __hash__(self):
        return hash((self.coverage_key, tuple(self.target_tokens[-4:])))


# ─── Phrase-based Decoder ────────────────────────────────────────────


class PhraseDecoder:
    """Beam search decoder for phrase-based SMT.

    Decodes source sentences using a phrase table and language model.
    Supports multiple decoding features with configurable weights.
    """

    def __init__(
        self,
        phrase_table: Dict[Tuple[str, str], List[Dict[str, float]]],
        lm: object,  # KneserNeyLM
        config: Optional[Dict] = None,
        oov_strategy: OOVStrategy = "copy",
    ):
        """
        Args:
            phrase_table: Dict mapping (src_key, tgt_key) → features list.
            lm: Language model (must have log_prob(word, history) and order).
            config: Decoder configuration dict.
            oov_strategy: How to handle out-of-vocabulary words.
                'copy' — pass source word through unchanged (default).
                'drop' — skip the unknown word entirely.
                'unk'  — replace with <unk> token.
        """
        if oov_strategy not in ("copy", "drop", "unk"):
            raise ValueError(
                f"Invalid oov_strategy: {oov_strategy!r}. "
                f"Must be one of: 'copy', 'drop', 'unk'"
            )
        self.oov_strategy: OOVStrategy = oov_strategy
        self.phrase_table = phrase_table
        self.lm = lm

        # Build source-side index for fast phrase lookup
        self.src_index: Dict[str, List[Tuple[str, List[Dict[str, float]]]]] = defaultdict(list)
        for (src_key, tgt_key), features_list in phrase_table.items():
            self.src_index[src_key].append((tgt_key, features_list))

        # Configuration with defaults
        self.beam_size = 10
        self.stack_size = 100
        self.max_phrase_len = 7
        self.distortion_limit = 6
        self.lm_weight = 1.0
        self.translation_weight = 1.0
        self.distortion_weight = 0.3
        self.word_penalty = -0.5  # Moses default
        self.use_future_cost = True

        if config:
            self.beam_size = config.get("beam_size", self.beam_size)
            self.stack_size = config.get("stack_size", self.stack_size)
            self.max_phrase_len = config.get("max_phrase_len", self.max_phrase_len)
            self.distortion_limit = config.get("distortion_limit", self.distortion_limit)
            self.lm_weight = config.get("lm_weight", self.lm_weight)
            self.translation_weight = config.get("translation_weight", self.translation_weight)
            self.distortion_weight = config.get("distortion_weight", self.distortion_weight)
            self.word_penalty = config.get("word_penalty", self.word_penalty)
            self.use_future_cost = config.get("future_cost_estimate", self.use_future_cost)
            # OOV strategy from config overrides constructor parameter
            if "oov_strategy" in config:
                strat = config["oov_strategy"]
                if strat in ("copy", "drop", "unk"):
                    self.oov_strategy = strat

        self.lm_order = getattr(lm, "order", 5)

        # Future cost cache
        self._future_cost_cache: Dict[Tuple[int, int], float] = {}

    # ─── Future Cost Estimation ───────────────────────────────────────

    def _estimate_future_cost(self, source_tokens: List[str], start: int, end: int) -> float:
        """Estimate the minimum cost to translate source[start:end].

        Uses the phrase table to find the cheapest translation for each
        source word/phrase, ignoring LM and reordering costs.

        This is an admissible heuristic (underestimates) which ensures
        optimality can still be achieved with pruning.
        """
        key = (start, end)
        if key in self._future_cost_cache:
            return self._future_cost_cache[key]

        if start >= end:
            return 0.0

        # Greedy: find cheapest single-phrase translation for each segment
        total_cost = 0.0
        pos = start
        while pos < end:
            best_cost = float('inf')
            # Try phrases starting at pos
            for plen in range(1, min(self.max_phrase_len, end - pos) + 1):
                src_phrase = " ".join(source_tokens[pos:pos + plen])
                translations = self.src_index.get(src_phrase, [])
                for tgt_key, features_list in translations:
                    features = features_list[0]  # Best occurrence
                    # Cost = -log(phi) (translation score only, no LM)
                    cost = -features.get("log_phi_f_e", 0) * self.translation_weight
                    if cost < best_cost:
                        best_cost = cost

            if best_cost == float('inf'):
                # Unknown phrase: use heuristic cost
                best_cost = 5.0 * len(source_tokens[pos:pos + 1])

            total_cost += best_cost
            pos += 1

        self._future_cost_cache[key] = total_cost
        return total_cost

    # ─── Translation Option Extraction ───────────────────────────────

    def _extract_options(
        self,
        source_tokens: List[str],
        covered: Set[int],
    ) -> List[Tuple[range, str, str, Dict[str, float]]]:
        """Extract all applicable translation options for uncovered spans.

        Args:
            source_tokens: Source sentence tokens.
            covered: Set of covered source positions.

        Returns:
            List of (source_span, src_phrase, tgt_phrase, features).
        """
        options: List[Tuple[range, str, str, Dict[str, float]]] = []
        src_len = len(source_tokens)

        for start in range(src_len):
            if start in covered:
                continue

            for plen in range(1, min(self.max_phrase_len, src_len - start) + 1):
                # Check all positions in range are uncovered
                positions = range(start, start + plen)
                if any(p in covered for p in positions):
                    continue

                src_phrase = " ".join(source_tokens[start:start + plen])
                translations = self.src_index.get(src_phrase, [])

                for tgt_key, features_list in translations:
                    # Use the best (first) feature set
                    features = features_list[0]
                    options.append((positions, src_phrase, tgt_key, features))

                # ── OOV Handling Strategies ──────────────────────────
                if not translations and plen == 1:
                    if self.oov_strategy == "drop":
                        # Strategy 2 (drop): skip the unknown word entirely.
                        # Do not add a translation option at all.
                        continue
                    elif self.oov_strategy == "unk":
                        # Strategy 3 (unk): replace with <unk> token.
                        features = {
                            "log_phi_f_e": math.log(0.01),
                            "log_phi_e_f": math.log(0.01),
                            "phrase_penalty": -1.0,
                            "tgt_len": 1.0,
                        }
                        options.append((positions, src_phrase, "<unk>", features))
                    else:
                        # Strategy 1 (copy, default): pass source word through unchanged.
                        features = {
                            "log_phi_f_e": math.log(0.01),
                            "log_phi_e_f": math.log(0.01),
                            "phrase_penalty": -1.0,
                            "tgt_len": 1.0,
                        }
                        options.append((positions, src_phrase, src_phrase, features))

        return options

    # ─── LM Scoring ─────────────────────────────────────────────────

    def _score_lm(self, target_tokens: List[str], new_words: List[str]) -> float:
        """Score new target words with the language model.

        Computes log10 P(new_words | target_tokens_context) incrementally,
        summed over all words in new_words accounting for LM history.

        Args:
            target_tokens: Previously translated target tokens.
            new_words: Newly added target words.

        Returns:
            Log10 probability summed over new words.
        """
        score = 0.0
        # Build the full history context
        context = target_tokens[-(self.lm_order - 1):] if target_tokens else []

        for word in new_words:
            history = tuple(context[-(self.lm_order - 1):]) if context else tuple()
            score += self.lm.log_prob(word, history)
            context.append(word)

        return score

    # ─── Hypothesis Expansion ────────────────────────────────────────

    def _apply_option(
        self,
        hypothesis: Hypothesis,
        option: Tuple[range, str, str, Dict[str, float]],
        source_len: int,
    ) -> Optional[Hypothesis]:
        """Apply a translation option to a hypothesis, creating a new one.

        Args:
            hypothesis: Current hypothesis.
            option: (positions, src_phrase, tgt_phrase, features).
            source_len: Total source length (for coverage key).

        Returns:
            New hypothesis or None if invalid.
        """
        positions, src_phrase, tgt_phrase, features = option
        new_covered = hypothesis.source_covered | set(positions)
        new_tgt_words = tgt_phrase.split()

        # Translation cost
        trans_cost = -features.get("log_phi_f_e", 0) * self.translation_weight

        # Phrase penalty
        phrase_cost = -features.get("phrase_penalty", 0)

        # Word penalty
        wp_cost = self.word_penalty * len(new_tgt_words)

        # Distortion cost (reordering)
        dist_cost = 0.0
        if hypothesis.last_src_pos >= 0:
            # Jump distance: difference between new start and last covered end
            jump = abs(positions[0] - hypothesis.last_src_pos - 1)
            if jump > 0:
                dist_cost = self.distortion_weight * jump

        # LM cost
        lm_cost = -self._score_lm(hypothesis.target_tokens, new_tgt_words) * self.lm_weight

        # Total score (additive, lower is better)
        new_score = (hypothesis.score + trans_cost + lm_cost +
                     dist_cost + phrase_cost + wp_cost)

        # Coverage key for hypothesis comparison
        coverage_key = tuple(sorted(new_covered))

        # LM history for future scoring
        lm_history = tuple(
            (hypothesis.target_tokens + new_tgt_words)[-(self.lm_order - 1):]
        ) if self.lm_order > 1 else tuple()

        return Hypothesis(
            score=new_score,
            target_tokens=hypothesis.target_tokens + new_tgt_words,
            source_covered=new_covered,
            lm_history=lm_history,
            coverage_key=coverage_key,
            num_phrases=hypothesis.num_phrases + 1,
            last_src_pos=positions[-1],
        )

    # ─── Pruning ────────────────────────────────────────────────────

    def _prune_stack(
        self, stack: List[Hypothesis], stack_size: int
    ) -> List[Hypothesis]:
        """Keep only the top `stack_size` hypotheses.

        Uses beam search pruning + histogram pruning.
        """
        # Recombination: keep best hypothesis per (coverage, lm_context) key.
        # Including LM context (last order-1 words) prevents merging hypotheses
        # with different target histories, which improves search quality.
        lm_ctx_len = max(0, getattr(self.lm, 'order', 3) - 1)
        best_per_coverage: Dict[Tuple, Hypothesis] = {}
        for h in stack:
            key = (h.coverage_key, tuple(h.target_tokens[-lm_ctx_len:]) if lm_ctx_len > 0 else ())
            if key not in best_per_coverage or h.score < best_per_coverage[key].score:
                best_per_coverage[key] = h

        # Sort by score and keep top
        sorted_hyps = sorted(best_per_coverage.values(), key=lambda h: h.score)
        return sorted_hyps[:stack_size]

    # ─── Main Decode ────────────────────────────────────────────────

    def decode(self, source_tokens: List[str]) -> Tuple[List[str], float]:
        """Translate a tokenized source sentence.

        Args:
            source_tokens: Tokenized source sentence.

        Returns:
            (translated_tokens, score): Best translation and its score.
        """
        if not source_tokens:
            return [], 0.0

        src_len = len(source_tokens)
        self._future_cost_cache.clear()

        # Initialize stacks: one per number of covered source words
        # (actually we use coverage-based pruning instead)
        stacks: Dict[int, List[Hypothesis]] = defaultdict(list)

        # Initial hypothesis
        initial = Hypothesis(
            score=0.0,
            target_tokens=[],
            source_covered=set(),
            coverage_key=tuple(),
            num_phrases=0,
            last_src_pos=-1,
        )
        stacks[0].append(initial)

        # Main decoding loop
        completed: List[Hypothesis] = []

        for covered_count in range(src_len):
            current_stack = stacks.get(covered_count, [])
            if not current_stack:
                continue

            # Prune current stack
            current_stack = self._prune_stack(current_stack, self.stack_size)

            for hypothesis in current_stack:
                # Check if complete
                if len(hypothesis.source_covered) == src_len:
                    completed.append(hypothesis)
                    continue

                # Extract options
                options = self._extract_options(source_tokens, hypothesis.source_covered)

                for option in options:
                    new_h = self._apply_option(hypothesis, option, src_len)
                    if new_h is None:
                        continue

                    new_covered = len(new_h.source_covered)
                    target_stack = stacks[new_covered]

                    # Future cost estimation
                    if self.use_future_cost:
                        # Find uncovered segments
                        uncovered = sorted(set(range(src_len)) - new_h.source_covered)
                        if uncovered:
                            future_cost = 0.0
                            # Group uncovered into contiguous segments
                            seg_start = uncovered[0]
                            for i in range(1, len(uncovered)):
                                if uncovered[i] != uncovered[i - 1] + 1:
                                    future_cost += self._estimate_future_cost(
                                        source_tokens, seg_start, uncovered[i - 1] + 1
                                    )
                                    seg_start = uncovered[i]
                            future_cost += self._estimate_future_cost(
                                source_tokens, seg_start, uncovered[-1] + 1
                            )

                            new_h.score += future_cost

                    target_stack.append(new_h)

            # Beam pruning across the current stack size
            if len(stacks[covered_count]) > self.stack_size * 2:
                stacks[covered_count] = self._prune_stack(
                    stacks[covered_count], self.stack_size
                )

        # Gather all completed hypotheses
        for h in stacks.get(src_len, []):
            completed.append(h)

        # Also check hypotheses at any stack that cover all words
        for covered, stack in stacks.items():
            if covered > 0:
                for h in stack:
                    if len(h.source_covered) == src_len:
                        completed.append(h)

        if not completed:
            # No complete hypothesis found; take the best partial
            best_partial = None
            best_score = float('inf')
            for stack in stacks.values():
                for h in stack:
                    # Estimate completion cost
                    uncovered = sorted(set(range(src_len)) - h.source_covered)
                    if uncovered:
                        completion_cost = self._estimate_future_cost(
                            source_tokens, uncovered[0], uncovered[-1] + 1
                        )
                        total = h.score + completion_cost
                        if total < best_score:
                            best_score = total
                            best_partial = h

            if best_partial is not None:
                completed.append(best_partial)

        if not completed:
            return source_tokens.copy(), 999.0

        # Sort by score and return best
        completed.sort(key=lambda h: h.score)
        best = completed[0]

        return best.target_tokens, best.score

    def decode_nbest(
        self, source_tokens: List[str], n: int = 10
    ) -> List[Tuple[List[str], float]]:
        """Return top-N translation hypotheses with scores.

        Used by MERT tuning to extract n-best lists for error surface
        computation. The underlying beam search is the same as decode(),
        but all completed hypotheses are collected and returned.

        Args:
            source_tokens: Tokenized source sentence.
            n: Number of top hypotheses to return.

        Returns:
            List of (translated_tokens, score) tuples, sorted best-first.
        """
        if not source_tokens:
            return [([], 0.0)]

        src_len = len(source_tokens)
        self._future_cost_cache.clear()

        stacks: Dict[int, List[Hypothesis]] = defaultdict(list)
        initial = Hypothesis(score=0.0, target_tokens=[], source_covered=set(),
                            coverage_key=tuple(), num_phrases=0, last_src_pos=-1)
        stacks[0].append(initial)

        completed: List[Hypothesis] = []

        for covered_count in range(src_len):
            current_stack = stacks.get(covered_count, [])
            if not current_stack:
                continue
            current_stack = self._prune_stack(current_stack, self.stack_size)

            for hypothesis in current_stack:
                if len(hypothesis.source_covered) == src_len:
                    completed.append(hypothesis)
                    continue
                options = self._extract_options(source_tokens, hypothesis.source_covered)
                for option in options:
                    new_h = self._apply_option(hypothesis, option, src_len)
                    if new_h is None:
                        continue
                    new_covered = len(new_h.source_covered)
                    stacks[new_covered].append(new_h)

            if len(stacks[covered_count]) > self.stack_size * 2:
                stacks[covered_count] = self._prune_stack(stacks[covered_count], self.stack_size)

        for h in stacks.get(src_len, []):
            completed.append(h)
        for stack in stacks.values():
            for h in stack:
                if len(h.source_covered) == src_len:
                    completed.append(h)

        if not completed:
            return [(source_tokens.copy(), 999.0)]

        # Deduplicate by target tokens
        seen = set()
        unique = []
        completed.sort(key=lambda h: h.score)
        for h in completed:
            key = tuple(h.target_tokens)
            if key not in seen:
                seen.add(key)
                unique.append(h)

        return [(h.target_tokens, h.score) for h in unique[:n]]


# ─── Batch Translation ───────────────────────────────────────────────


def batch_translate(
    decoder: PhraseDecoder,
    src_sentences: List[List[str]],
    beam_size: int = 10,
    verbose: bool = True,
) -> List[Tuple[List[str], float]]:
    """Translate a batch of tokenized source sentences.

    Args:
        decoder: Initialized PhraseDecoder instance.
        src_sentences: List of tokenized source sentences.
        beam_size: Beam search width.
        verbose: Log progress.

    Returns:
        List of (translated_tokens, score) tuples.
    """
    results: List[Tuple[List[str], float]] = []
    prev_beam = decoder.beam_size
    decoder.beam_size = beam_size

    for idx, src in enumerate(src_sentences):
        if verbose and (idx + 1) % 10 == 0:
            logger.info(f"Translating sentence {idx + 1}/{len(src_sentences)}")

        translation, score = decoder.decode(src)
        results.append((translation, score))

    decoder.beam_size = prev_beam
    return results
