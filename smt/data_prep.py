"""
Data preparation for SMT training.

Handles tokenization (sentence splitting + word segmentation),
truecasing, and corpus cleaning/filtering for both Chinese and English.
Mirrors the Moses preprocessing pipeline: tokenize → truecase → clean.
"""

import re
import os
from typing import List, Optional, Tuple, Dict, Callable

import spacy

from . import utils

logger = utils.logger

# ─── SpaCy NLP singletons (lazy-loaded) ─────────────────────────────

_nlp_en: Optional[Callable] = None
_nlp_zh: Optional[Callable] = None


def _get_nlp(lang: str):
    """Get spaCy language model (lazy-loaded singleton)."""
    global _nlp_en, _nlp_zh
    if lang == "en":
        if _nlp_en is None:
            _nlp_en = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
        return _nlp_en
    elif lang == "zh":
        if _nlp_zh is None:
            _nlp_zh = spacy.load("zh_core_web_sm", disable=["ner", "textcat"])
        return _nlp_zh
    else:
        raise ValueError(f"Unsupported language: {lang}")


# ─── English Tokenization ────────────────────────────────────────────

# Moses-style English tokenization without external dependency
_EN_CONTRACTIONS = {
    r"'s\\b": " 's",
    r"'t\\b": " 't",
    r"'re\\b": " 're",
    r"'ve\\b": " 've",
    r"'ll\\b": " 'll",
    r"'d\\b": " 'd",
    r"'m\\b": " 'm",
    r"n't\\b": " n't",
    r"\\bcannot\\b": "can not",
    r"\\bcan't\\b": "ca n't",
}

_EN_PUNCT_PATTERN = re.compile(r"([,;:.?!\"()\-\{\}\[\]])")


def tokenize_en(text: str) -> str:
    """Moses-style tokenizer for English.

    Separates punctuation from words and handles contractions.
    Returns space-separated tokens.
    """
    # Handle contractions first
    for pattern, replacement in _EN_CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)

    # Separate punctuation
    text = _EN_PUNCT_PATTERN.sub(r" \1 ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


_truecase_model: Optional[Dict[str, str]] = None


def train_truecaser(lines: List[str], output_path: Optional[str] = None) -> Dict[str, str]:
    """Train a truecasing model from tokenized text.

    Maps each case-insensitive word to its most common casing form.

    Args:
        lines: List of tokenized sentences.
        output_path: Optional path to save truecaser model.

    Returns:
        Dict: {lowercase_word: most_common_case_form}
    """
    freq: Dict[str, Dict[str, int]] = {}
    for line in lines:
        for token in line.split():
            lower = token.lower()
            if lower not in freq:
                freq[lower] = {}
            freq[lower][token] = freq[lower].get(token, 0) + 1

    model = {}
    for lower, forms in freq.items():
        model[lower] = max(forms, key=forms.get)

    if output_path:
        utils.save_json(model, output_path)
        logger.info(f"Truecaser model saved to {output_path} ({len(model)} entries)")

    return model


def load_truecaser(path: str) -> Dict[str, str]:
    """Load a pre-trained truecasing model from JSON."""
    return utils.load_json(path)


def truecase(line: str, model: Dict[str, str]) -> str:
    """Apply truecasing to a tokenized sentence.

    Leaves the first word lowercased (truecasing standard: first word
    is left as-is since sentence case is handled elsewhere).

    Args:
        line: Tokenized sentence.
        model: {lowercase: most_common_case} mapping.

    Returns:
        Truecased tokenized sentence.
    """
    tokens = line.split()
    result = []
    for i, token in enumerate(tokens):
        lower = token.lower()
        if lower in model and i > 0:
            result.append(model[lower])
        else:
            result.append(token)
    return " ".join(result)


# ─── Chinese Tokenization ────────────────────────────────────────────


def tokenize_zh(text: str) -> str:
    """Chinese word segmentation using spaCy.

    Args:
        text: Raw Chinese text.

    Returns:
        Space-separated segmented text.
    """
    nlp = _get_nlp("zh")
    doc = nlp(text)
    return " ".join([tok.text for tok in doc])


# ─── Language-agnostic tokenization ──────────────────────────────────


def tokenize(text: str, lang: str = "en") -> str:
    """Tokenize text according to language.

    Args:
        text: Input sentence.
        lang: 'en' or 'zh'.

    Returns:
        Tokenized sentence (space-separated tokens).
    """
    text = text.strip()
    if not text:
        return ""
    if lang == "zh":
        return tokenize_zh(text)
    else:
        return tokenize_en(text)


def detokenize_en(tokens: List[str]) -> str:
    """Simple English detokenization (heuristic)."""
    text = " ".join(tokens)
    text = re.sub(r"\s+'s\b", "'s", text)
    text = re.sub(r"\s+'t\b", "n't", text)
    text = re.sub(r"\s+'re\b", "'re", text)
    text = re.sub(r"\s+'ve\b", "'ve", text)
    text = re.sub(r"\s+'ll\b", "'ll", text)
    text = re.sub(r"\s+'d\b", "'d", text)
    text = re.sub(r"\s+'m\b", "'m", text)
    text = re.sub(r"\s+'em\b", "'em", text)
    text = re.sub(r"\s+n't\b", "n't", text)
    text = re.sub(r"\s+([,;:.?!\"()\-\{\}\[\]])", r"\1", text)
    text = re.sub(r"([\"\(\[\{-])\s+", r"\1", text)
    return text.strip()


def detokenize_zh(tokens: List[str]) -> str:
    """Simple Chinese detokenization: join tokens directly without spaces.

    Chinese text is already segmented into words; detokenization just
    concatenates all tokens with no separators.

    Args:
        tokens: List of Chinese word tokens.

    Returns:
        Detokenized Chinese text.
    """
    return "".join(tokens)


# ─── Corpus Cleaning ─────────────────────────────────────────────────


def clean_sentence_pair(
    src_tokens: List[str],
    tgt_tokens: List[str],
    max_src_len: int = 80,
    max_tgt_len: int = 100,
    ratio: float = 9.0,
) -> Tuple[bool, List[str], List[str]]:
    """Filter and clean a sentence pair.

    Rules (Moses cleaning standard):
    - Both sides must have at least 1 token
    - Neither side exceeds max token count
    - Length ratio src/tgt <= ratio and tgt/src <= ratio
    - No empty segments

    Args:
        src_tokens: Source token list.
        tgt_tokens: Target token list.
        max_src_len: Max source tokens.
        max_tgt_len: Max target tokens.
        ratio: Max length ratio (src/tgt or tgt/src).

    Returns:
        (keep, src_tokens, tgt_tokens): Whether to keep, cleaned tokens.
    """
    if not src_tokens or not tgt_tokens:
        return False, src_tokens, tgt_tokens

    if len(src_tokens) > max_src_len or len(tgt_tokens) > max_tgt_len:
        return False, src_tokens, tgt_tokens

    src_len = len(src_tokens)
    tgt_len = len(tgt_tokens)
    if src_len / max(tgt_len, 1) > ratio or tgt_len / max(src_len, 1) > ratio:
        return False, src_tokens, tgt_tokens

    return True, src_tokens, tgt_tokens


def clean_corpus(
    src_sentences: List[str],
    tgt_sentences: List[str],
    src_lang: str = "zh",
    tgt_lang: str = "en",
    max_src_len: int = 80,
    max_tgt_len: int = 100,
) -> Tuple[List[str], List[str]]:
    """Tokenize, clean, and filter a parallel corpus.

    Args:
        src_sentences: Source sentences (raw).
        tgt_sentences: Target sentences (raw).
        src_lang: Source language.
        tgt_lang: Target language.
        max_src_len: Max source tokens.
        max_tgt_len: Max target tokens.

    Returns:
        (cleaned_src, cleaned_tgt): Filtered, tokenized sentences.
    """
    src_out: List[str] = []
    tgt_out: List[str] = []
    discarded = 0

    for src_raw, tgt_raw in zip(src_sentences, tgt_sentences):
        src_tok = tokenize(src_raw, lang=src_lang)
        tgt_tok = tokenize(tgt_raw, lang=tgt_lang)

        keep, src_toks, tgt_toks = clean_sentence_pair(
            src_tok.split(), tgt_tok.split(),
            max_src_len=max_src_len, max_tgt_len=max_tgt_len,
        )
        if keep:
            src_out.append(" ".join(src_toks))
            tgt_out.append(" ".join(tgt_toks))
        else:
            discarded += 1

    if discarded:
        logger.info(f"Cleaning discarded {discarded} sentence pairs")

    return src_out, tgt_out


# ─── Full Preparation Pipeline ───────────────────────────────────────


def prepare_corpus(
    src_raw: str,
    tgt_raw: str,
    src_out: str,
    tgt_out: str,
    src_lang: str = "zh",
    tgt_lang: str = "en",
    truecaser_model: Optional[str] = None,
    max_src_len: int = 80,
    max_tgt_len: int = 100,
    max_pairs: Optional[int] = None,
) -> Tuple[int, int]:
    """End-to-end corpus preparation: read → tokenize → truecase → clean → write.

    Args:
        src_raw: Raw source file path.
        tgt_raw: Raw target file path.
        src_out: Cleaned source output path.
        tgt_out: Cleaned target output path.
        src_lang: Source language.
        tgt_lang: Target language.
        truecaser_model: Path to truecaser JSON (if None, skip truecasing).
        max_src_len: Max source tokens.
        max_tgt_len: Max target tokens.
        max_pairs: Max sentence pairs to process.

    Returns:
        (num_kept, num_discarded)
    """
    logger.info(f"Preparing corpus: {src_raw} ↔ {tgt_raw}")

    src_sentences = utils.read_lines(src_raw)[:max_pairs] if max_pairs else utils.read_lines(src_raw)
    tgt_sentences = utils.read_lines(tgt_raw)[:max_pairs] if max_pairs else utils.read_lines(tgt_raw)

    # Tokenize + clean
    src_clean, tgt_clean = clean_corpus(
        src_sentences, tgt_sentences,
        src_lang=src_lang, tgt_lang=tgt_lang,
        max_src_len=max_src_len, max_tgt_len=max_tgt_len,
    )

    # Truecasing (English side only for zh→en)
    if truecaser_model and os.path.exists(truecaser_model):
        tc_model = load_truecaser(truecaser_model)
        if tgt_lang == "en":
            tgt_clean = [truecase(line, tc_model) for line in tgt_clean]
        if src_lang == "en":
            src_clean = [truecase(line, tc_model) for line in src_clean]

    # Write
    utils.write_lines(src_clean, src_out)
    utils.write_lines(tgt_clean, tgt_out)

    kept = len(src_clean)
    discarded = len(src_sentences) - kept
    logger.info(f"Corpus prepared: {kept} kept, {discarded} discarded")
    return kept, discarded


def train_truecaser_from_corpus(
    corpus_path: str,
    output_path: str,
    lang: str = "en",
) -> Dict[str, str]:
    """Train a truecasing model from a tokenized corpus file."""
    lines = utils.read_lines(corpus_path)
    model = train_truecaser(lines, output_path=output_path)
    return model
