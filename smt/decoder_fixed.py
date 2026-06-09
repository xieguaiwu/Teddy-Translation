"""
Phrase-based beam search decoder for SMT.

Implements the standard SMT decoder with:
- Beam search with hypothesis stacks
- Future cost estimation for pruning
- Distortion (reordering) with limit
- Language model integration
- Phrase table lookup
- Word penalty

Optimizations applied:
    P1: Precomputed subphrase strings (avoids repeated " ".join)
    P2: Integer coverage bitmap (replaces Set[int] operations)
    P4: LM log_prob cache (avoids redundant LM queries)
    P6: Pre-filled future cost table (O(1) lookup during beam search)

References:
    - Koehn (2004), "Pharaoh: a beam search decoder for phrase-based
      statistical machine translation", AMTA.
    - Koehn (2010), "Statistical Machine Translation", Ch. 6-7.
"""

import math
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from dataclasses import dataclass, field

from . import utils
from typing import Literal

logger = utils.logger

# OOV strategy type
OOVStrategy = Literal["copy", "drop", "unk"]


def _bit_count(n: int) -> int:
    """Count set bits in integer (portable)."""
    return bin(n).count("1")


def _span_mask(start: int, length: int) -> int:
    """Return bitmask for positions [start, start+length)."""
    return ((1 << length) - 1) << start


def _uncovered_positions(covered: int, src_len: int) -> List[int]:
    """Return sorted list of uncovered source positions from bitmap."""
    return [i for i in range(src_len) if not ((covered >> i) & 1)]


# ─── Types ───────────────────────────────────────────────────────────


@dataclass(order=True)
class Hypothesis:
    """A partial translation hypothesis in the beam search.

    Hypotheses are ordered by score (negative for max-heap via min-heap).

    P2 optimization: source_covered is an integer bitmask instead of Set[int].
    coverage_key is removed (bitmap serves as the recombination key directly).
    """
    score: float = field(compare=True)      # Negative log-prob (lower = better)
    target_tokens: List[str] = field(compare=False, default_factory=list)
    source_covered: int = field(compare=False, default=0)  # P2: bitmask, not Set[int]
    lm_history: Tuple[str, ...] = field(compare=False, default_factory=tuple)
    num_phrases: int = field(compare=False, default=0)
    last_src_pos: int = field(compare=False, default=-1)

    def __hash__(self):
        # P2: use bitmap directly instead of coverage_key
        return hash((self.source_covered, tuple(self.target_tokens[-4:])))

    @property
    def covered_count(self) -> int:
        """Number of covered source positions (P2: bit_count)."""
        return _bit_count(self.source_covered)


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

        # P1: Per-sentence subphrase lookup table (built in _setup_sentence)
        self._subphrases: List[List[Optional[str]]] = []

        # P4: LM log_prob cache
        self._lm_cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        self._lm_cache_max = 10000

        # P6: Future cost table (pre-filled per sentence)
        self._future_cost_table: Dict[Tuple[int, int], float] = {}

        # Store current source tokens for _extract_options access
        self._source_tokens: List[str] = []

    # ─── P4: LM cache ────────────────────────────────────────────────

    def _cached_log_prob(self, word: str, history: Tuple[str, ...]) -> float:
        """P4: Cached LM log_prob lookup to avoid redundant LM queries."""
        key = (word, history)
        val = self._lm_cache.get(key)
        if val is not None:
            return val
        val = self.lm.log_prob(word, history)
        if len(self._lm_cache) < self._lm_cache_max:
            self._lm_cache[key] = val
        return val

    # ─── P1: Subphrase Precomputation ────────────────────────────────

    def _build_subphrases(self, source_tokens: List[str]) -> None:
        """P1: Precompute all subphrase strings for the current sentence.

        Builds a 2D array _subphrases[start][plen] = joined source phrase string.
        For single-token phrases (plen=1), stores source_tokens[start] directly.
        """
        src_len = len(source_tokens)
        self._subphrases = []
        for start in range(src_len):
            max_plen = min(self.max_phrase_len, src_len - start)
            row = [None] * (max_plen + 1)  # index 0 unused
            for plen in range(1, max_plen + 1):
                if plen == 1:
                    row[plen] = source_tokens[start]
                else:
                    row[plen] = " ".join(source_tokens[start:start + plen])
            self._subphrases.append(row)

    def _setup_sentence(self, source_tokens: List[str]) -> None:
        """Prepare per-sentence data structures before decoding."""
        self._source_tokens = source_tokens
        self._build_subphrases(source_tokens)
        if self.use_future_cost:
            self._prefill_future_cost()

    # ─── P6: Pre-filled Future Cost ──────────────────────────────────

    def _prefill_future_cost(self) -> None:
        """P6: Pre-compute future cost for all (start, end) span pairs.

        After this, _estimate_future_cost becomes an O(1) dict lookup.
        """
        src_len = len(self._source_tokens)
        self._future_cost_table.clear()

        # Dynamic programming: compute from shorter spans to longer ones
        for start in range(src_len):
            for end in range(start + 1, src_len + 1):
                self._future_cost_table[(start, end)] = self._compute_future_cost(start, end)

    def _compute_future_cost(self, start: int, end: int) -> float:
        """P1+P6: Compute future cost for a span using precomputed subphrases.

        Finds the cheapest single-phrase translation for each source position,
        ignoring LM and reordering costs. This is an admissible heuristic.
        """
        if start >= end:
            return 0.0

        total_cost = 0.0
        pos = start
        while pos < end:
            best_cost = float('inf')
            max_plen_at_pos = min(self.max_phrase_len, end - pos)
            row = self._subphrases[pos]

            for plen in range(1, max_plen_at_pos + 1):
                src_phrase = row[plen]  # P1: precomputed string
                translations = self.src_index.get(src_phrase, [])
                for _tgt_key, features_list in translations:
                    features = features_list[0]  # Best occurrence
                    cost = -features.get("log_phi_f_e", 0) * self.translation_weight
                    if cost < best_cost:
                        best_cost = cost

            if best_cost == float('inf'):
                # Unknown phrase: use heuristic per-token cost
                best_cost = 5.0  # single token

            total_cost += best_cost
            pos += 1

        return total_cost

    def _estimate_future_cost(self, start: int, end: int) -> float:
        """P6: O(1) future cost lookup from pre-filled table."""
        return self._future_cost_table.get((start, end), 0.0)

    # ─── Translation Option Extraction ───────────────────────────────

    def _extract_options(
        self,
        source_tokens: List[str],
        covered: int,  # P2: int bitmask instead of Set[int]
    ) -> List[Tuple[range, str, str, Dict[str, float]]]:
        """Extract all applicable translation options for uncovered spans.

        P1: Uses precomputed _subphrases instead of " ".join().
        P2: covered is an int bitmask; span checks use single bitwise AND.

        Args:
            source_tokens: Source sentence tokens (for OOV fallback only).
            covered: Integer bitmask of covered source positions.

        Returns:
            List of (source_span, src_phrase, tgt_phrase, features).
        """
        options: List[Tuple[range, str, str, Dict[str, float]]] = []
        src_len = len(source_tokens)

        for start in range(src_len):
            if (covered >> start) & 1:  # P2: fast bit test
                continue

            row = self._subphrases[start]
            max_plen_at_start = min(self.max_phrase_len, src_len - start)

            for plen in range(1, max_plen_at_start + 1):
                # P2: Check all positions in range are uncovered with single bitwise AND
                if covered & _span_mask(start, plen):
                    continue

                positions = range(start, start + plen)
                src_phrase = row[plen]  # P1: precomputed
                translations = self.src_index.get(src_phrase, [])

                for tgt_key, features_list in translations:
                    # Use the best (first) feature set
                    features = features_list[0]
                    options.append((positions, src_phrase, tgt_key, features))

                # ── OOV Handling Strategies ──────────────────────────
                if not translations and plen == 1:
                    if self.oov_strategy == "drop":
                        # Strategy 2 (drop): skip the unknown word entirely.
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

        P4: Uses _cached_log_prob for redundant query elimination.

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
            score += self._cached_log_prob(word, history)  # P4: cached
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

        P2: Uses integer bitmask for coverage operations.

        Args:
            hypothesis: Current hypothesis.
            option: (positions, src_phrase, tgt_phrase, features).
            source_len: Total source length (used for full-coverage check).

        Returns:
            New hypothesis or None if invalid.
        """
        positions, src_phrase, tgt_phrase, features = option
        new_tgt_words = tgt_phrase.split()

        # P2: bitmask union (single bitwise OR)
        # positions is always range(start, start+plen), so stop-start = plen
        span_mask = _span_mask(positions.start, positions.stop - positions.start)
        new_covered = hypothesis.source_covered | span_mask

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

        # LM cost (P4: uses _cached_log_prob internally)
        lm_cost = -self._score_lm(hypothesis.target_tokens, new_tgt_words) * self.lm_weight

        # Total score (additive, lower is better)
        new_score = (hypothesis.score + trans_cost + lm_cost +
                     dist_cost + phrase_cost + wp_cost)

        # LM history for future scoring
        lm_history = tuple(
            (hypothesis.target_tokens + new_tgt_words)[-(self.lm_order - 1):]
        ) if self.lm_order > 1 else tuple()

        return Hypothesis(
            score=new_score,
            target_tokens=hypothesis.target_tokens + new_tgt_words,
            source_covered=new_covered,  # P2: int
            lm_history=lm_history,
            num_phrases=hypothesis.num_phrases + 1,
            last_src_pos=positions[-1],
        )

    # ─── Pruning ────────────────────────────────────────────────────

    def _prune_stack(
        self, stack: List[Hypothesis], stack_size: int
    ) -> List[Hypothesis]:
        """Keep only the top `stack_size` hypotheses.

        P2: Uses integer bitmap directly as recombination key (no coverage_key needed).

        Uses beam search pruning + histogram pruning.
        """
        # Recombination: keep best hypothesis per (coverage, lm_context) key.
        lm_ctx_len = max(0, getattr(self.lm, 'order', 3) - 1)
        best_per_coverage: Dict[Tuple, Hypothesis] = {}
        for h in stack:
            # P2: bitmap is directly hashable and comparable
            key = (h.source_covered, tuple(h.target_tokens[-lm_ctx_len:]) if lm_ctx_len > 0 else ())
            if key not in best_per_coverage or h.score < best_per_coverage[key].score:
                best_per_coverage[key] = h

        # Sort by score and keep top
        sorted_hyps = sorted(best_per_coverage.values(), key=lambda h: h.score)
        return sorted_hyps[:stack_size]

    # ─── Main Decode ────────────────────────────────────────────────

    def decode(self, source_tokens: List[str]) -> Tuple[List[str], float]:
        """Translate a tokenized source sentence.

        P1+P2+P4+P6 optimizations applied throughout.

        Args:
            source_tokens: Tokenized source sentence.

        Returns:
            (translated_tokens, score): Best translation and its score.
        """
        if not source_tokens:
            return [], 0.0

        src_len = len(source_tokens)
        full_mask = (1 << src_len) - 1  # P2: mask for all positions covered

        # P1 + P6: Setup per-sentence data structures
        self._setup_sentence(source_tokens)

        # Initialize stacks: one per number of covered source words
        stacks: Dict[int, List[Hypothesis]] = defaultdict(list)

        # Initial hypothesis (P2: source_covered=0)
        initial = Hypothesis(
            score=0.0,
            target_tokens=[],
            source_covered=0,
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
                # P2: Check if complete using bitmask equality
                if hypothesis.source_covered == full_mask:
                    completed.append(hypothesis)
                    continue

                # Extract options (P2: covered is int)
                options = self._extract_options(source_tokens, hypothesis.source_covered)

                for option in options:
                    new_h = self._apply_option(hypothesis, option, src_len)
                    if new_h is None:
                        continue

                    # P2: count covered positions with bit_count
                    new_covered_count = _bit_count(new_h.source_covered)
                    target_stack = stacks[new_covered_count]

                    # Future cost estimation (P6: O(1) lookup)
                    if self.use_future_cost:
                        future_cost = 0.0
                        seg_start = -1
                        for i in range(src_len):
                            if not ((new_h.source_covered >> i) & 1):
                                if seg_start == -1:
                                    seg_start = i
                            else:
                                if seg_start != -1:
                                    future_cost += self._estimate_future_cost(seg_start, i)
                                    seg_start = -1
                        if seg_start != -1:
                            future_cost += self._estimate_future_cost(seg_start, src_len)
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

        # Also check hypotheses at any stack that cover all words (P2: bitmask check)
        for stack in stacks.values():
            for h in stack:
                if h.source_covered == full_mask:
                    completed.append(h)

        if not completed:
            # No complete hypothesis found; take the best partial
            best_partial = None
            best_score = float('inf')
            for stack in stacks.values():
                for h in stack:
                    # Estimate completion cost (P6: O(1))
                    uncovered = _uncovered_positions(h.source_covered, src_len)
                    if uncovered:
                        completion_cost = self._estimate_future_cost(
                            uncovered[0], uncovered[-1] + 1
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

        P1+P2+P4+P6 optimizations applied.

        Args:
            source_tokens: Tokenized source sentence.
            n: Number of top hypotheses to return.

        Returns:
            List of (translated_tokens, score) tuples, sorted best-first.
        """
        if not source_tokens:
            return [([], 0.0)]

        src_len = len(source_tokens)
        full_mask = (1 << src_len) - 1  # P2

        # P1 + P6: Setup per-sentence data structures
        self._setup_sentence(source_tokens)

        stacks: Dict[int, List[Hypothesis]] = defaultdict(list)
        initial = Hypothesis(score=0.0, target_tokens=[], source_covered=0,
                            num_phrases=0, last_src_pos=-1)
        stacks[0].append(initial)

        completed: List[Hypothesis] = []

        for covered_count in range(src_len):
            current_stack = stacks.get(covered_count, [])
            if not current_stack:
                continue
            current_stack = self._prune_stack(current_stack, self.stack_size)

            for hypothesis in current_stack:
                # P2: bitmask full-coverage check
                if hypothesis.source_covered == full_mask:
                    completed.append(hypothesis)
                    continue
                options = self._extract_options(source_tokens, hypothesis.source_covered)
                for option in options:
                    new_h = self._apply_option(hypothesis, option, src_len)
                    if new_h is None:
                        continue
                    # P2: bit_count
                    new_covered_count = _bit_count(new_h.source_covered)
                    stacks[new_covered_count].append(new_h)

            if len(stacks[covered_count]) > self.stack_size * 2:
                stacks[covered_count] = self._prune_stack(stacks[covered_count], self.stack_size)

        for h in stacks.get(src_len, []):
            completed.append(h)
        for stack in stacks.values():
            for h in stack:
                # P2: bitmask check
                if h.source_covered == full_mask:
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
