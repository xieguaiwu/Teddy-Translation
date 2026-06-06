"""
Phrase extraction and scoring for phrase-based SMT.

Implements the standard phrase extraction algorithm that:
1. Extracts all phrase pairs consistent with word alignments
2. Computes standard feature scores:
   - φ(f|e): Phrase translation probability (source→target)
   - φ(e|f): Phrase translation probability (target→source)
   - lex(f|e): Lexical weighting (source→target)
   - lex(e|f): Lexical weighting (target→source)
   - Phrase penalty (constant)

References:
    - Koehn, Och, Marcu (2003), "Statistical Phrase-Based Translation", NAACL.
"""

import math
import os
import logging
import multiprocessing
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple, Iterator
from collections import defaultdict, Counter

from . import utils
from . import ibm_align

logger = utils.logger

# ─── Types ───────────────────────────────────────────────────────────

# A phrase pair: source_phrase, target_phrase, source_start, source_end,
#                target_start, target_end, alignment_points
PhrasePair = Tuple[List[str], List[str], int, int, int, int, Set[Tuple[int, int]]]

# Phrase table entry: feature dict (can include probabilities, penalties)
PhraseFeatures = Dict[str, float]

# Full phrase table: (src_phrase_key, tgt_phrase_key) → features
PhraseTable = Dict[Tuple[str, str], List[PhraseFeatures]]


# ─── Alignment Data Structure ────────────────────────────────────────


def _key(phrase: List[str]) -> str:
    """Convert a token list to a hashable string key."""
    return " ".join(phrase)


# ─── Phrase Extraction ──────────────────────────────────────────────


def extract_phrases(
    src_sent: List[str],
    tgt_sent: List[str],
    alignment: Set[Tuple[int, int]],
    max_phrase_len: int = 7,
) -> List[PhrasePair]:
    """Extract all phrase pairs consistent with word alignment.

    Implements the standard phrase extraction algorithm (Koehn et al. 2003):
    For each source phrase [j_start, j_end], find the minimal target span
    [i_start, i_end] that covers all aligned target words. If all words
    in the target span align consistently to the source span, add the pair.

    Args:
        src_sent: Source sentence tokens.
        tgt_sent: Target sentence tokens.
        alignment: Set of (src_pos, tgt_pos) pairs (0-indexed).
        max_phrase_len: Maximum phrase length in tokens.

    Returns:
        List of PhrasePair tuples.
    """
    # Build alignment index: for each source position, which target positions?
    src_to_tgt: Dict[int, Set[int]] = defaultdict(set)
    tgt_to_src: Dict[int, Set[int]] = defaultdict(set)
    for s, t in alignment:
        src_to_tgt[s].add(t)
        tgt_to_src[t].add(s)

    m = len(src_sent)
    n = len(tgt_sent)
    phrase_pairs: List[PhrasePair] = []

    # Iterate over source phrase spans
    for j_start in range(m):
        for j_end in range(j_start, min(m, j_start + max_phrase_len)):
            src_phrase = src_sent[j_start:j_end + 1]

            # Find target span covering all aligned words
            tgt_set: Set[int] = set()
            for j in range(j_start, j_end + 1):
                tgt_set.update(src_to_tgt.get(j, set()))

            if not tgt_set:
                # Source words unaligned: can align to empty target phrase
                # Standard: skip unaligned source
                continue

            i_start = min(tgt_set)
            i_end = max(tgt_set)

            # Check if target span is consistent
            # All words in target span must align only to words in source span
            consistent = True
            for i in range(i_start, i_end + 1):
                src_points = tgt_to_src.get(i, set())
                if not src_points.issubset(set(range(j_start, j_end + 1))):
                    consistent = False
                    break

            if not consistent:
                continue

            # Target span length check
            if i_end - i_start + 1 > max_phrase_len:
                continue

            tgt_phrase = tgt_sent[i_start:i_end + 1]

            phrase_pairs.append((
                src_phrase, tgt_phrase,
                j_start, j_end, i_start, i_end,
                {(s, t) for s in range(j_start, j_end + 1)
                 for t in src_to_tgt.get(s, set())
                 if i_start <= t <= i_end}
            ))

    return phrase_pairs


# ─── Lexical Weighting ───────────────────────────────────────────────


_LEX_EPSILON = 1e-7  # was 1e-10, caused log-space underflow for multi-word OOV phrases


def _safe_log(value: float, eps: float = _LEX_EPSILON) -> float:
    """Log with safe handling of zero/small values."""
    return math.log(max(value, eps))


def _lookup_prob(t_table: ibm_align.Table, e_word: str, f_word: str) -> float:
    """Look up t(f|e) from translation table with fallback."""
    inner = t_table.get(e_word, {})
    if isinstance(inner, dict):
        prob = inner.get(f_word, _LEX_EPSILON)
    else:
        prob = _LEX_EPSILON
    return prob if prob > 0 else _LEX_EPSILON


def lexical_weight(
    src_phrase: List[str],
    tgt_phrase: List[str],
    alignment: Set[Tuple[int, int]],
    t_table,
    direction: str = "f_given_e",
) -> float:
    """Compute lexical weight for a phrase pair.

    lex(f|e) = ∏_{j=1}^{m} 1/|{i: (j,i) ∈ A}| *
                Σ_{(j,i)∈A} w(f_j|e_i)

    where w(f_j|e_i) is the IBM Model 1 translation probability.

    Args:
        src_phrase: Source phrase tokens (f).
        tgt_phrase: Target phrase tokens (e).
        alignment: Set of (src_pos, tgt_pos) pairs (relative to phrase).
        t_table: Lexical translation table t(f|e).
        direction: "f_given_e" (lex(f|e)) or "e_given_f" (lex(e|f)).

    Returns:
        Lexical weight (raw probability, 0-1).
    """
    EPS = _LEX_EPSILON

    if direction == "f_given_e":
        # Compute lex(f|e)
        total = 0.0
        for j, f_j in enumerate(src_phrase):
            alignments = {(s, t) for s, t in alignment if s == j}
            if not alignments:
                prob = _lookup_prob(t_table, "NULL", f_j)
                total += _safe_log(prob, EPS)
            else:
                sum_log = 0.0
                for _, t in alignments:
                    e_i = tgt_phrase[t] if t < len(tgt_phrase) else "NULL"
                    prob = _lookup_prob(t_table, e_i, f_j)
                    sum_log += _safe_log(prob, EPS)
                total += sum_log / len(alignments)
        return max(0.0, math.exp(total)) if total > float('-inf') else 0.0
    else:
        # Compute lex(e|f)
        total = 0.0
        for i, e_i in enumerate(tgt_phrase):
            alignments = {(s, t) for s, t in alignment if t == i}
            if not alignments:
                prob = _lookup_prob(t_table, "NULL", e_i)
                total += _safe_log(prob, EPS)
            else:
                sum_log = 0.0
                for s, _ in alignments:
                    f_j = src_phrase[s] if s < len(src_phrase) else "NULL"
                    prob = _lookup_prob(t_table, e_i, f_j)
                    sum_log += _safe_log(prob, EPS)
                total += sum_log / len(alignments)
        return max(0.0, math.exp(total)) if total > float('-inf') else 0.0


# ─── Phrase Table Construction ───────────────────────────────────────


def build_phrase_table(
    src_sentences: List[List[str]],
    tgt_sentences: List[List[str]],
    alignments: List[Set[Tuple[int, int]]],
    t_table,
    max_phrase_len: int = 7,
    min_count: int = 2,
    score_features: bool = True,
    src_vocab_min_freq: Optional[int] = None,
    tgt_vocab_min_freq: Optional[int] = None,
    num_workers: int = 0,
) -> PhraseTable:
    """Build a phrase table from aligned parallel corpus.

    Args:
        src_sentences: Source sentences (token lists).
        tgt_sentences: Target sentences (token lists).
        alignments: Word alignments [(src_pos, tgt_pos), ...] per sentence.
        t_table: IBM Model 1/2 lexical translation table.
        max_phrase_len: Max phrase length in tokens.
        min_count: Minimum count to include a phrase pair.
        score_features: Whether to compute full feature set.
        src_vocab_min_freq: If set, exclude phrases containing source words
            that appear fewer than this many times in the corpus.
        tgt_vocab_min_freq: If set, exclude phrases containing target words
            that appear fewer than this many times in the corpus.
        num_workers: Number of parallel workers for phrase extraction.
            0 = auto-detect (cpu_count - 1), 1 = sequential.

    Returns:
        PhraseTable: { (src_key, tgt_key): [feature_dict, ...] }
    """
    # Step 0: Build vocabulary frequency counters if filtering requested
    src_word_freq: Dict[str, int] = {}
    tgt_word_freq: Dict[str, int] = {}

    if src_vocab_min_freq is not None or tgt_vocab_min_freq is not None:
        src_counter: Counter = Counter()
        tgt_counter: Counter = Counter()
        for src_sent in src_sentences:
            src_counter.update(src_sent)
        for tgt_sent in tgt_sentences:
            tgt_counter.update(tgt_sent)
        src_word_freq = dict(src_counter)
        tgt_word_freq = dict(tgt_counter)
        logger.info(
            f"Vocabulary frequency filters: src types={len(src_word_freq)}, "
            f"tgt types={len(tgt_word_freq)}"
        )

    # Resolve worker count
    _num_workers = num_workers
    if _num_workers == 0:
        _num_workers = max(1, (os.cpu_count() or 1) - 1)

    # Step 1: Collect all phrase pairs with counts
    # Memory-safe: only stores count + first example (not all occurrences)
    # Full storage of all examples caused OOM at 128K unique pairs (~30GB)
    raw_pairs: Dict[Tuple[str, str], Tuple[int, List[str], List[str], Set]] = {}

    total_extracted = 0

    if _num_workers > 1:
        # ── Parallel phrase extraction ──
        logger.info(f"[Parallel: {_num_workers} workers] Extracting phrases")
        extract_args = [
            (src_sent, tgt_sent, al, max_phrase_len)
            for src_sent, tgt_sent, al in zip(src_sentences, tgt_sentences, alignments)
        ]

        with Pool(_num_workers) as pool:
            all_phrases = pool.starmap(extract_phrases, extract_args)

        for phrases in all_phrases:
            total_extracted += len(phrases)
            for src_ph, tgt_ph, js, je, is_, ie, al_points in phrases:
                key = (_key(src_ph), _key(tgt_ph))
                if key in raw_pairs:
                    raw_pairs[key][0] += 1
                else:
                    raw_pairs[key] = [1, src_ph, tgt_ph, al_points]
    else:
        # ── Sequential phrase extraction ──
        for sent_idx, (src_sent, tgt_sent, al) in enumerate(
            zip(src_sentences, tgt_sentences, alignments)
        ):
            phrases = extract_phrases(src_sent, tgt_sent, al, max_phrase_len)
            total_extracted += len(phrases)
            for src_ph, tgt_ph, js, je, is_, ie, al_points in phrases:
                key = (_key(src_ph), _key(tgt_ph))
                if key in raw_pairs:
                    raw_pairs[key][0] += 1
                else:
                    raw_pairs[key] = [1, src_ph, tgt_ph, al_points]

    logger.info(
        f"Extracted {total_extracted} phrase candidates, "
        f"{len(raw_pairs)} unique pairs"
    )

    # Step 2: Filter by count, frequency, and compute features
    table: PhraseTable = defaultdict(list)

    # Pre-compute denominator sums (O(N) instead of O(N²) per pair)
    tgt_denom = defaultdict(int)
    src_denom = defaultdict(int)
    for (s_key, t_key), (cnt, _, _, _) in raw_pairs.items():
        tgt_denom[t_key] += cnt
        src_denom[s_key] += cnt

    filtered_count = 0
    freq_filtered = 0
    for (src_key, tgt_key), (count, src_ph, tgt_ph, al_points) in raw_pairs.items():
        if count < min_count:
            continue

        # ── Vocabulary frequency filtering ───────────────────────────
        if src_vocab_min_freq is not None:
            src_words = src_key.split()
            if any(src_word_freq.get(w, 0) < src_vocab_min_freq for w in src_words):
                freq_filtered += 1
                continue

        if tgt_vocab_min_freq is not None:
            tgt_words = tgt_key.split()
            if any(tgt_word_freq.get(w, 0) < tgt_vocab_min_freq for w in tgt_words):
                freq_filtered += 1
                continue

        filtered_count += 1

        if score_features:
            # Phrase translation probability φ(f|e)
            phi_f_given_e = count / tgt_denom.get(tgt_key, 1)

            # Phrase translation probability φ(e|f) (inverse)
            total_src = src_denom.get(src_key, 1)
            phi_e_given_f = count / total_src if total_src > 0 else count

            # Lexical weighting (using first occurrence only)
            lex_f_given_e = lexical_weight(
                src_ph, tgt_ph, al_points, t_table, "f_given_e"
            )
            lex_e_given_f = lexical_weight(
                src_ph, tgt_ph, al_points, t_table, "e_given_f"
            )

            features: PhraseFeatures = {
                "count": float(count),
                "log_phi_f_e": math.log(phi_f_given_e + 1e-20),
                "log_phi_e_f": math.log(phi_e_given_f + 1e-20),
                "log_lex_f_e": math.log(lex_f_given_e + 1e-20),
                "log_lex_e_f": math.log(lex_e_given_f + 1e-20),
                "phrase_penalty": -1.0,  # Moses: exp(1) per phrase in output
                # Source and target lengths for future cost estimation
                "src_len": float(len(src_key.split())),
                "tgt_len": float(len(tgt_key.split())),
            }
        else:
            features = {
                "count": float(count),
                "phrase_penalty": -1.0,
                "src_len": float(len(src_key.split())),
                "tgt_len": float(len(tgt_key.split())),
            }

        table[(src_key, tgt_key)].append(features)

    log_parts = [f"min_count={min_count}"]
    if src_vocab_min_freq is not None:
        log_parts.append(f"src_vocab_min_freq={src_vocab_min_freq}")
    if tgt_vocab_min_freq is not None:
        log_parts.append(f"tgt_vocab_min_freq={tgt_vocab_min_freq}")
    if freq_filtered > 0:
        log_parts.append(f"freq_filtered={freq_filtered}")

    logger.info(
        f"Phrase table built: {filtered_count} entries "
        f"(after {', '.join(log_parts)} filter)"
    )

    return table


# ─── Phrase Table I/O ────────────────────────────────────────────────


def save_phrase_table(table: PhraseTable, path: str) -> None:
    """Save phrase table to text file.

    Format (Moses-style tab-separated):
    src_phrase ||| tgt_phrase ||| count log_phi_f_e log_phi_e_f log_lex_f_e log_lex_e_f phrase_penalty
    """
    utils.ensure_dir(os.path.dirname(path) or ".")
    lines: List[str] = []
    for (src_key, tgt_key), features_list in table.items():
        for features in features_list:
            feat_str = " ".join(
                f"{k}={v:.6f}" for k, v in features.items()
            )
            lines.append(f"{src_key} ||| {tgt_key} ||| {feat_str}")
    utils.write_lines(lines, path)
    logger.info(f"Phrase table saved to {path} ({len(lines)} entries)")


def load_phrase_table(path: str) -> PhraseTable:
    """Load phrase table from text file."""
    table: PhraseTable = defaultdict(list)
    lines = utils.read_lines(path)
    for line in lines:
        parts = line.split(" ||| ")
        if len(parts) >= 3:
            src_key, tgt_key = parts[0].strip(), parts[1].strip()
            feat_str = parts[2].strip()
            features: PhraseFeatures = {}
            for item in feat_str.split():
                if "=" in item:
                    k, v = item.split("=", 1)
                    features[k] = float(v)
            table[(src_key, tgt_key)].append(features)
    logger.info(f"Phrase table loaded from {path} ({len(lines)} entries)")
    return table


# ─── Lookup utility ──────────────────────────────────────────────────


def lookup_phrase(
    table: PhraseTable,
    src_phrase: str,
) -> List[Tuple[str, PhraseFeatures]]:
    """Look up all target-side translations for a source phrase.

    Args:
        table: Phrase table.
        src_phrase: Source phrase string.

    Returns:
        List of (target_phrase, features) sorted by count descending.
    """
    results: List[Tuple[str, PhraseFeatures]] = []
    for (src_key, tgt_key), features_list in table.items():
        if src_key == src_phrase:
            for features in features_list:
                results.append((tgt_key, features))

    # Sort by count descending
    results.sort(key=lambda x: x[1].get("count", 0), reverse=True)
    return results


import os  # noqa: E402 — used above in save_phrase_table
