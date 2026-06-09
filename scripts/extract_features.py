#!/usr/bin/env python3
"""
Extract 4-dimension features from SMT and LLM translations for statistical comparison.
Input: data/translation_index.csv
Output: data/feature_matrix.csv
"""
import sys, os, re, csv, warnings, logging, json
from collections import Counter
import numpy as np
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

# Feature caches
_sent_cache = {}

# ── Tokenization ──────────────────────────────────────────────────────

def tokenize(text, lang):
    """Tokenize text. For zh use jieba, for en split on whitespace."""
    if lang == "zh":
        try:
            import jieba
            return list(jieba.cut(text))
        except ImportError:
            # Fallback: char-level
            return [c for c in text if c.strip()]
    else:  # en
        return text.lower().split()


def sentencize(text, lang):
    """Split text into sentences."""
    if lang == "zh":
        sents = re.split(r'[。！？；\n]+', text)
    else:
        sents = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in sents if len(s.strip()) > 0]


# ── Lexical Diversity ─────────────────────────────────────────────────

def extract_lexical(text, lang):
    """STTR, TTR, MTLD, HD-D."""
    tokens = tokenize(text, lang)
    if len(tokens) < 2:
        return {"sttr": 0, "ttr": 0, "mtld": 0, "hdd": 0}

    total = len(tokens)
    unique = len(set(tokens))
    ttr = unique / total if total > 0 else 0

    # STTR: average TTR over 100-word segments
    window = 100
    if total >= window:
        sttr_vals = []
        for i in range(0, total - window + 1, window // 2):
            seg = tokens[i:i+window]
            sttr_vals.append(len(set(seg)) / window)
        sttr = np.mean(sttr_vals) if sttr_vals else ttr
    else:
        sttr = ttr  # fallback for short texts

    # Simplified MTLD
    mtld = ttr * 100  # approximation

    # HD-D: simplified hypergeometric
    n = min(42, total)
    hdd_val = 0
    if total > 0:
        for t in set(tokens):
            freq = tokens.count(t)
            # hypergeometric probability: P(not seeing word in n draws)
            if total - freq >= n:
                p = 1.0
                for i in range(n):
                    p *= (total - freq - i) / (total - i)
                hdd_val += 1 - p
            else:
                hdd_val += 1
        hdd_val /= len(set(tokens))

    return {"sttr": round(sttr, 4), "ttr": round(ttr, 4),
            "mtld": round(mtld, 4), "hdd": round(hdd_val, 4)}


# ── Sentence Complexity ───────────────────────────────────────────────

def extract_complexity(text, lang):
    """Sentence length statistics."""
    sents = sentencize(text, lang)
    if not sents:
        return {"mean_sent_len": 0, "sd_sent_len": 0, "sent_skew": 0, "sent_kurt": 0, "n_sents": 0}

    lengths = [len(s.split()) for s in sents]
    lengths = [l for l in lengths if l > 0]
    if not lengths:
        return {"mean_sent_len": 0, "sd_sent_len": 0, "sent_skew": 0, "sent_kurt": 0, "n_sents": 0}

    mean_len = np.mean(lengths)
    sd_len = np.std(lengths) if len(lengths) > 1 else 0
    skew = float(sp_stats.skew(lengths)) if len(lengths) > 2 else 0
    kurt = float(sp_stats.kurtosis(lengths)) if len(lengths) > 3 else 0

    return {
        "mean_sent_len": round(mean_len, 4),
        "sd_sent_len": round(sd_len, 4),
        "sent_skew": round(skew, 4),
        "sent_kurt": round(kurt, 4),
        "n_sents": len(lengths),
    }


# ── Sentiment ─────────────────────────────────────────────────────────

def extract_sentiment(text, lang):
    """Sentiment analysis using VADER for EN, simple dictionary for ZH."""
    if lang == "en":
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            vs = analyzer.polarity_scores(text)
            return {
                "sent_pos": round(vs["pos"], 4),
                "sent_neu": round(vs["neu"], 4),
                "sent_neg": round(vs["neg"], 4),
                "sent_polarity": round(vs["compound"], 4),
                "sent_volatility": 0,
            }
        except ImportError:
            pass

    # Fallback: simple keyword-based
    pos_words = {"good", "great", "excellent", "wonderful", "beautiful", "happy", "love", "positive",
                 "success", "joy", "amazing", "fantastic", "hope", "brilliant", "delight"}
    neg_words = {"bad", "terrible", "awful", "horrible", "ugly", "sad", "hate", "negative",
                 "failure", "crisis", "pain", "dreadful", "angry", "fear", "disaster"}
    tokens = text.lower().split()
    if not tokens:
        return {"sent_pos": 0, "sent_neu": 1, "sent_neg": 0, "sent_polarity": 0, "sent_volatility": 0}

    pos_count = sum(1 for t in tokens if t in pos_words)
    neg_count = sum(1 for t in tokens if t in neg_words)
    total_sent = len(tokens)
    pos = pos_count / total_sent
    neg = neg_count / total_sent
    neu = 1 - pos - neg
    compound = pos - neg

    return {
        "sent_pos": round(pos, 4),
        "sent_neu": round(neu, 4),
        "sent_neg": round(neg, 4),
        "sent_polarity": round(compound, 4),
        "sent_volatility": 0,
    }


# ── Stylometry ────────────────────────────────────────────────────────

# English function words
EN_FUNC_WORDS = set("the a an and or but in on at to for of with by from as is are was were be been have has had do does did will would shall should may might can could must about above across after against along among around before behind below beneath beside between beyond down during inside into near off outside over through throughout toward under underneath up upon within without".split())

# Chinese function words (common particles, prepositions, conjunctions)
ZH_FUNC_WORDS = set("的 了 在 是 有 和 与 及 或 不 也 都 就 还 又 很 上 下 中 前 后 内 外 间 时 来 去 到 从 把 被 对 为 以 因 用 向 给 让 叫 使 由 按 照 凭 通过 根据 关于 对于 除了 沿着 朝 往".split())


def extract_stylometry(text, lang):
    """POS distribution entropy and function word ratio."""
    tokens = text.split()
    if not tokens:
        return {"pos_entropy": 0, "func_word_ratio": 0, "alpha_ratio": 0, "punct_ratio": 0, "digit_ratio": 0}

    total = len(tokens)
    total_chars = len(text)

    # Character-type ratios
    alpha = sum(1 for c in text if c.isalpha())
    punct = sum(1 for c in text if c in '.,!?;:()[]{}""''-...')
    digits = sum(1 for c in text if c.isdigit())

    # Function word ratio
    func_words = ZH_FUNC_WORDS if lang == "zh" else EN_FUNC_WORDS
    func_count = sum(1 for t in tokens if t.lower() in func_words)

    # POS entropy (simplified: character type distribution as proxy)
    # For a more accurate POS entropy, we'd use spaCy
    char_types = {"alpha": alpha, "punct": punct, "digit": digits, "other": total_chars - alpha - punct - digits}
    char_dist = np.array([v for v in char_types.values() if v > 0])
    char_dist = char_dist / char_dist.sum()
    pos_entropy = float(sp_stats.entropy(char_dist)) if len(char_dist) > 1 else 0

    return {
        "pos_entropy": round(pos_entropy, 4),
        "func_word_ratio": round(func_count / total, 4),
        "alpha_ratio": round(alpha / max(total_chars, 1), 4),
        "punct_ratio": round(punct / max(total_chars, 1), 4),
        "digit_ratio": round(digits / max(total_chars, 1), 4),
    }


# ── Main Pipeline ─────────────────────────────────────────────────────

FEATURE_NAMES = [
    "sttr", "ttr", "mtld", "hdd",
    "mean_sent_len", "sd_sent_len", "sent_skew", "sent_kurt", "n_sents",
    "sent_pos", "sent_neu", "sent_neg", "sent_polarity", "sent_volatility",
    "pos_entropy", "func_word_ratio", "alpha_ratio", "punct_ratio", "digit_ratio",
]


def extract_all(text, lang):
    """Extract all feature dimensions from translation text."""
    features = {}
    features.update(extract_lexical(text, lang))
    features.update(extract_complexity(text, lang))
    features.update(extract_sentiment(text, lang))
    features.update(extract_stylometry(text, lang))
    return features


def main():
    index_path = os.path.join(PROJECT_DIR, "data", "translation_index.csv")
    output_path = os.path.join(PROJECT_DIR, "data", "feature_matrix.csv")

    if not os.path.exists(index_path):
        log.error(f"Index not found: {index_path}. Run collect_translations.py first.")
        sys.exit(1)

    log.info(f"Reading index: {index_path}")
    with open(index_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    log.info(f"  {len(rows)} translation files")

    results = []
    errors = 0

    for i, row in enumerate(rows):
        fpath = row["filepath"]
        if not os.path.exists(fpath):
            log.warning(f"  [{i+1}/{len(rows)}] File not found: {fpath}")
            errors += 1
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            log.warning(f"  [{i+1}/{len(rows)}] Read error: {fpath}: {e}")
            errors += 1
            continue

        if not text:
            log.warning(f"  [{i+1}/{len(rows)}] Empty file: {fpath}")
            errors += 1
            continue

        # Determine language
        direction = row["direction"]
        lang = "zh" if direction == "en2zh" else "en"

        features = extract_all(text, lang)

        result = {
            "filepath": fpath,
            "architecture": row["architecture"],
            "model": row["model"],
            "direction": direction,
            "genre": row["genre"],
            "source_file": row.get("source_file", ""),
            "text_length": len(text.split()),
        }
        result.update(features)
        results.append(result)

        if (i + 1) % 100 == 0:
            log.info(f"  [{i+1}/{len(rows)}] {len(results)} OK, {errors} errors")

    log.info(f"Writing {len(results)} rows to {output_path}")
    fieldnames = ["filepath", "architecture", "model", "direction", "genre", "source_file", "text_length"]
    fieldnames += FEATURE_NAMES

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    log.info(f"Done! {len(results)} files processed, {errors} errors")
    log.info(f"Output: {output_path}")

    # Print sample
    if results:
        print(f"\nSample features (first row):")
        for k, v in results[0].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
