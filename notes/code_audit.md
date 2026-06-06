# SMT Implementation Code Audit

**Date:** 2026-06-06
**Scope:** `smt_model/smt/` — decoder.py, ibm_align.py, phrase_table.py, language_model.py, pipeline.py
**Models analyzed:** smt_20k (20K templates), smt_zh2en (10K real), smt_zh2en_v3 (200K incremental)

---

## 1. decoder.py — Why is EN→ZH 2000× slower than ZH→EN?

**No single root cause.** The slowdown is combinatorial—multiple factors compound in the EN→ZH direction.

### 1A. Option-explosion from English function words

**File:** `decoder.py:180-230` — `_extract_options()`

For each uncovered source position, the method tries every phrase length (1..7) and collects ALL phrase-table entries. English has many high-frequency function words that pair with dozens of Chinese translations:

- `"the"`, `"a"`, `"of"`, `"in"`, `"to"`, `"is"`, `"that"`, `"and"` each appear in hundreds of phrase-table entries
- Each entry → one new hypothesis in the beam
- For a 20-token EN sentence, **one hypothesis expansion generates O(uncovered_positions × max_phrase_len × avg_translations_per_phrase) options**
- With avg translations per phrase = 5–50+ for common words, a single expansion can produce **hundreds to thousands of candidates**

In contrast, ZH function words are less frequent (Chinese relies more on word order than function words), so ZH→EN has far fewer options per expansion.

### 1B. Hypothesis recombination ignores LM state

**File:** `decoder.py:276-285` — `_prune_stack()`

```python
best_per_coverage: Dict[Tuple[int, ...], Hypothesis] = {}
for h in stack:
    key = h.coverage_key
    if key not in best_per_coverage or h.score < best_per_coverage[key].score:
        best_per_coverage[key] = h
```

Recombination key is **only coverage**—two hypotheses covering identical source positions but with different target-word histories are merged. The LM state (`lm_history`) is ignored. When `EN→ZH` produces more diverse target-side continuations (Chinese has larger vocab with richer combinatorial patterns), this causes more surviving hypotheses that later diverge in quality.

### 1C. Future-cost estimation per hypothesis

**File:** `decoder.py:97-128` and `decoder.py:333-343`

Every new hypothesis triggers future-cost computation that iterates over uncovered source segments and calls `_estimate_future_cost()`, which does its own phrase-table lookups. For EN (longer source sentences), more uncovered segments exist, and this overhead multiplies per hypothesis.

### 1D. Recursive LM scoring

**File:** `decoder.py:242-253` → `language_model.py:280-310`

`_score_lm()` calls `log_prob()` → `prob()` → `_continuation_prob()` **recursively** from order 5 down to order 1. Each level performs dict lookups (`kn_counts[order]`, `follow_counts[order-1]`). With 1.1M 5-grams and 400K 4-grams in the EN→ZH target LM, these lookups are **O(5 × dict_size)** per new word.

**Compound effect:**
- Longer EN source → more decoding iterations
- More translation options per EN phrase → larger beam width usage
- Recursive LM scoring hits 1M+ entry dicts→ scales poorly
- Weak recombination pruning allows more hypotheses to survive
- Fixed future-cost overhead for each hypothesis

### 1E. OOV amplification

**File:** `decoder.py:207-228`

When an English source word has no phrase-table entry:
- OOV strategy `"copy"` creates one option per uncovered source word
- Each OOV-triggered hypothesis has poor translation score
- The beam fills with junk hypotheses, crowding out legitimate partial translations

---

## 2. ibm_align.py — Is IBM2 sufficient?

### 2A. What IBM2 captures vs what it misses

**IBM2** = `P(f|e)` lexical translation + absolute distortion `a(j|i,l,m)`.

**Missing from IBM3/4/5:**

| Model | Parameter | Why critical for ZH↔EN |
|-------|-----------|----------------------|
| IBM3 | **Fertility** `p(φ|e)` | Chinese `人工智能` (1 token) ↔ EN `artificial intelligence` (2 tokens). IBM2 forces 1:1 alignment, so one English word aligns to `NULL` → phrase extraction misses valid pairs. |
| IBM4 | **Relative distortion** | Chinese usually SVO but allows topicalization: `这本书 我 读过` (lit. "this book I have read"). IBM2's absolute `P(j|i,l,m)` can't model this, but IBM4's relative `P(j - j_{i-1} | ...)` can. |
| IBM5 | **Fertile + relative** | Combines both; needed for systematic NULL generation and long-distance reordering. |

### 2B. Viterbi alignment is greedy

**File:** `ibm_align.py:300-315` (IBM1) and `ibm_align.py:383-398` (IBM2)

```python
for j, f_j in enumerate(src_sent):
    best_i = 0
    best_p = self.t["NULL"][f_j] * self._get_distortion(j, 0, l, m)
    for i, e_i in enumerate(tgt_sent):
        p = self.t[e_i][f_j] * self._get_distortion(j, i + 1, l, m)
        if p > best_p:
            best_p = p
            best_i = i + 1
    alignment.append(best_i)
```

Each source word independently picks its best target word. There is **no constraint** that each target word should align to a reasonable number of source words. This produces:
- One-to-many alignments (correct for fertility, but IBM2 can't model fertility)
- **No many-to-one enforcement** — multiple source words all aligning to the same target word is allowed
- **No NULL alignment optimization** — `NULL` is only "best" when `t["NULL"][f_j]` is high, which rarely happens after training

### 2C. Parallel E-step uses global variables

**File:** `ibm_align.py:34-82`

```python
_IBM1_WORKER_T: Optional[Dict[str, Dict[str, float]]] = None
```

Uses module-level globals for multiprocessing (`fork` context). This works on Linux but:
- **Not thread-safe** (if ever used with threads)
- `fork` on modern Python 3.8+ macOS may trigger `[NSForwarding warning]`
- The global `_IBM1_WORKER_T = self.t` is set before `Pool()` — works only because of fork semantics

### 2D. Sparse initialization is correct

**File:** `ibm_align.py:120-135`

```python
cooccur = set()
for src, tgt in zip(src_sentences, tgt_sentences):
    for e in tgt:
        for f in src:
            cooccur.add((e, f))
```

Only stores entries for co-occurring word pairs, avoiding the `|E|×|F|` explosion. This is the right approach for real data where ∼30K EN × 50K ZH would require 1.5B entries.

---

## 3. phrase_table.py — Score computation

### 3A. φ(f|e) and φ(e|f) — correct

**File:** `phrase_table.py:310-316`

```python
phi_f_given_e = count / tgt_denom.get(tgt_key, 1)
phi_e_given_f = count / src_denom.get(src_key, 1)
```

Denominators are pre-computed per unique phrase string across the whole corpus. This is the standard MLE.

**Bug:** `tgt_denom` and `src_denom` are computed from `raw_pairs` which only contains pairs with `count >= min_count`... actually no — `raw_pairs` contains ALL pairs (count is stored, filtering happens later). The denominators are computed from the full set. This is correct.

### 3B. Lexical weighting — only first occurrence

**File:** `phrase_table.py:328-338`

```python
lex_f_given_e = lexical_weight(
    src_ph, tgt_ph, al_points, t_table, "f_given_e"
)
```

The `al_points` come from the **first occurrence** of the phrase pair. If a phrase pair appears in multiple sentences with different internal alignments, only one is used for lexical weighting. Moses collects all occurrences and averages.

**Impact:** Lexical weights are biased toward the first sentence where the phrase pair was encountered, not the full distribution.

### 3C. `_LEX_EPSILON = 1e-10` causes underflow

**File:** `phrase_table.py:218`

```python
_LEX_EPSILON = 1e-10
```

Used as fallback for unseen `t(f|e)` word pairs. In log space:

```python
# phrase_table.py:264
total += _safe_log(prob, EPS)  # log(1e-10) = -23.0
```

For a 5-word phrase with all OOV word pairs: `5 × log(1e-10) = -115` → `exp(-115)` → underflow to 0.0. The `max(0.0, math.exp(total))` clamp at `phrase_table.py:268` returns 0.0, which means "impossible" translation, even though the word-pair might be plausible. Moses uses `1e-7` or `1e-9` and also smooths the LM contributions.

**Fix:** Increase to `1e-7` or add a small mass to the lexical weight.

### 3D. Missing features vs Moses

**File:** `phrase_table.py:340-351`

Moses phrase table has 5 required features + `exp(1)` phrase penalty:

| Feature | Python SMT | Moses |
|---------|-----------|-------|
| φ(f\|e) | ✅ `log_phi_f_e` | ✅ |
| φ(e\|f) | ✅ `log_phi_e_f` | ✅ |
| lex(f\|e) | ✅ `log_lex_f_e` | ✅ |
| lex(e\|f) | ✅ `log_lex_e_f` | ✅ |
| Phrase penalty | ✅ `-1.0` fixed | ✅ `exp(1)` per phrase |
| Word penalty | ❌ (in decoder) | ✅ per-option |
| Unknown word penalty | ❌ | ✅ |

---

## 4. language_model.py — 111s load time

### 4A. Bottleneck confirmed: ast.literal_eval for 3.5M keys

**File:** `language_model.py:393-405`

```python
def _parse_key(k: str):
    try:
        return ast.literal_eval(k)
    except (ValueError, SyntaxError):
        return k
```

**Why it's slow:**

The JSON format stores n-gram tuples as Python-string representations:
```json
{"counts": {"3": {"(\"<s>\", \"<s>\", \"Fiscal\")": 2, ...}}}
```

Loading `smt_zh2en_v3/lm.json` (302 MB, 3,496,165 n-grams):

| Operation | Measured time |
|-----------|--------------|
| `json.load()` 302 MB | ~10–15s |
| `ast.literal_eval()` on 3.5M keys | **~44s** (12.5 μs/key) |
| Dict reconstruction & assignment | ~10–15s |
| `pickle.dump()` 250 MB cache | ~15–25s |
| **Total first load** | **~85–110s** |

Benchmark confirms: 12.5 μs per `ast.literal_eval()` call × 3.5M = 43.8s.

**Root cause:** Serializing tuple keys as `str(tuple)` / deserializing with `ast.literal_eval` is orders of magnitude slower than using JSON arrays:

```python
# Slow (current)
"('word1', 'word2')" → ast.literal_eval → ('word1', 'word2')

# Fast alternative
["word1", "word2"] → tuple(json_array) → ('word1', 'word2')
```

Alternative: **Use JSON arrays instead of stringified tuples** for keys would eliminate the parsing bottleneck entirely.

### 4B. Pickle cache mitigates but doesn't solve

**File:** `language_model.py:350-365`

```python
# Fast path: pickle exists and is newer
if os.path.exists(pkl_path):
    json_mtime = os.path.getmtime(json_path) if os.path.exists(json_path) else 0
    pkl_mtime = os.path.getmtime(pkl_path)
    if pkl_mtime >= json_mtime:
        with open(pkl_path, 'rb') as f:
            lm = _pickle.load(f)
        ...
        return lm
```

Pickle loads are **~3–5× faster** (~20–30s for 250 MB). But the first load is always slow, and if the JSON is regenerated or the pickle is missing, the user waits 111s.

### 4C. Suspicious identical n-gram counts for orders 3–5

**File:** Trained models `model/smt_zh2en/lm.json` and `model/smt_20k/lm.json`

`smt_zh2en` (10K real sentences):
```
Order 1: 20,793 n-grams
Order 2: 125,619 n-grams
Order 3: 1,634 n-grams   ← identical
Order 4: 1,634 n-grams   ← identical
Order 5: 1,634 n-grams   ← identical
```

`smt_20k` (20K template sentences):
```
Order 1: 507 n-grams
Order 2: 13,285 n-grams
Order 3: 195 n-grams     ← identical
Order 4: 195 n-grams     ← identical
Order 5: 195 n-grams     ← identical
```

**This is a counting bug.** For a corpus with avg 24 tokens/sentence, each sentence produces `len(tokens)+1 = 25` n-grams per order. With 10K sentences, that's 250K 3-grams. Getting only **1,634 distinct** across the entire corpus, and having exactly the same number for orders 3, 4, and 5, is impossible without a bug.

**Hypothesis:** The bug is in the parallel `count_ngrams` path where `_count_ngrams_chunk` returns a flat dict and the parent distributes by `len(ng)`. The identical counts suggest only padding n-grams (`<s>...`) are being counted for orders ≥3.

**Contrast:** The `smt_zh2en_v3` model (trained with `--workers 1` sequential) has correct distribution:
```
Order 3: 852,072 n-grams
Order 4: 1,065,311 n-grams
Order 5: 1,130,784 n-grams
```

This confirms the parallel counting path has a latent bug.

### 4D. Discount estimation with low counts

**File:** `language_model.py:237-258`

```python
Y = n1 / (n1 + 2 * n2)
d1 = 1 - 2 * Y * n2 / n1
d2 = 2 - 3 * Y * n3 / n2 if n2 > 0 else 1.0
d3 = 3 - 4 * Y * n4 / n3 if n3 > 0 else 1.5
```

When `n1` (singleton count) is 0, falls back to fixed discounts `(0.5, 1.0, 1.5)`. For the buggy models (orders 3–5 all identical), `n1=0` for many orders because there are so few n-grams. The fixed discounts mean poor probability estimates for higher-order n-grams.

---

## 5. pipeline.py train_python — Missing vs Moses

### 5A. Missing pipeline steps

| Standard Moses Step | Python SMT | Impact |
|--------------------|------------|--------|
| Tokenization | ✅ `data_prep.tokenize()` | |
| Truecasing | ℹ️ Configurable but `train_python` passes `truecaser_path=None` | Missing truecasing reduces BLEU by ~0.5–1 point on EN |
| Corpus cleaning | ✅ `data_prep.clean_corpus()` | |
| IBM alignment | ✅ `ibm_align.train_ibm()` | IBM2 only, see §2 |
| **Lexicalized reordering** | ❌ **Not implemented** | Moses trains 3 reordering models (monotone, swap, discontinuous). Without it, German SOV/V2 and Chinese topicalization are poorly handled. |
| Phrase extraction | ✅ `phrase_table.build_phrase_table()` | See §3 |
| **Lexical weight from all occurrences** | ❌ Only first occurrence | |
| Language model | ✅ `language_model.train_lm()` | |
| **MERT tuning** | ❌ **Not implemented** | Feature weights are hardcoded (`config.py:104-112`). Moses runs Minimum Error Rate Training to optimize BLEU. Without it, decoder weights are not data-driven. |
| **Log-linear weights optimization** | ❌ **No tuning** | See above |
| **Unknown word handling** | ❌ **Copy-only** | Moses replaces UNK through GIZA++ alignment. Python SMT only copies source word unchanged. |

**File references:**
- `pipeline.py:160-200` — `train_python()` — no reordering model
- `pipeline.py:165` — `self.prepare_data(src_raw=src_file, tgt_raw=tgt_file, ..., truecaser_path=None)` — no truecasing
- `decoder.py:68-76` — hardcoded weights `lm_weight=1.0, translation_weight=1.0, distortion_weight=0.3`

### 5B. Warm-starting loses alignment tables

**File:** `train_200k.py:50-60` and `pipeline.py:185-195`

The incremental training calls `train_python` with `warm_start_model=dir` for each batch. But examining `model/smt_zh2en_v3/`:
```
Files present:    phrase_table.txt (5.6MB), lm.json (302MB), lm.pkl (250MB)
Files MISSING:    lex.table, distortion.table, model_info.json, src_vocab.json, tgt_vocab.json
```

The warm-start path at `pipeline.py:185` loads `lex.table` from the previous batch, trains IBM2 on current batch, and saves the new tables. **But only the final batch's directory is retained** (the `train_200k.py` script symlinks `smt_zh2en_v3 → smt_zh2en_v3_b20`). The earlier batch directories are deleted.

**Result:** The final model has no `lex.table` or `distortion.table`, meaning it cannot re-align or re-extract phrases. It can only decode from the final batch's phrase table and LM.

### 5C. Missing `model_info.json` in v3 model

Confirmed: `model/smt_zh2en_v3/` has no `model_info.json`. The final batch directory was output by `train_python`, which writes `model_info.json` at `pipeline.py:265`—but the file is absent. Possible cause: `train_python` was interrupted or the file was deleted.

### 5D. Truecasing bypass

**File:** `pipeline.py:165`

```python
prep_out = self.prepare_data(
    src_raw=src_file, tgt_raw=tgt_file,
    output_prefix=os.path.join(output_dir, "train"),
    src_lang=src_lang, tgt_lang=tgt_lang,
    truecaser_path=None,  # ← No truecaser!
)
```

And later at `pipeline.py:195` — the `train_python` method accepts `skip_prep=True` but has no truecasing fallback. For `train_200k.py` which uses `skip_prep=True`, the input data is expected to be pre-tokenized but there's no truecasing step applied.

### 5E. decoupled `load_model` fails silently

**File:** `pipeline.py:290-303`

```python
def load_model(self, model_dir):
    pt_path = os.path.join(model_dir, "phrase_table.txt")
    lm_path = os.path.join(model_dir, "lm.json")
    if not os.path.exists(pt_path) or not os.path.exists(lm_path):
        raise FileNotFoundError(...)
    self._pt = phrase_table.load_phrase_table(pt_path)
    self._lm = language_model.KneserNeyLM.load(lm_path)
```

No `lex.table` or `distortion.table` load required, so the missing alignment tables in v3 model don't prevent loading. But `translate_python` or `translate_sentence` silently uses a model with incomplete training artifacts.

---

## Summary of Critical Issues

| # | Severity | Component | Issue | File:Line |
|---|----------|-----------|-------|-----------|
| 1 | **High** | LM load | `ast.literal_eval` on 3.5M keys takes ~44s | `language_model.py:397` |
| 2 | **High** | LM training | Identical n-gram counts for orders 3–5 (parallel counting bug) | `language_model.py:163-185` |
| 3 | **High** | Pipeline | No lexicalized reordering model → poor on ZH topicalization | Missing from entire codebase |
| 4 | **High** | Pipeline | No MERT tuning → hardcoded suboptimal feature weights | Missing from entire codebase |
| 5 | **Medium** | Decoder | Hypothesis recombination ignores LM state → beam search inefficiency | `decoder.py:276-285` |
| 6 | **Medium** | Decoder | OOV amplification fills beam with junk hypotheses | `decoder.py:207-228` |
| 7 | **Medium** | Phrase table | Lexical weighting uses only first occurrence of each phrase pair | `phrase_table.py:328-338` |
| 8 | **Medium** | Phrase table | `_LEX_EPSILON=1e-10` causes underflow in log-space | `phrase_table.py:218` |
| 9 | **Medium** | IBM alignment | No fertility model → poor alignment for 1:many/many:1 | `ibm_align.py` (IBM2 only) |
| 10 | **Low** | IBM alignment | Greedy Viterbi with no symmetry constraints | `ibm_align.py:383-398` |
| 11 | **Low** | Pipeline | Warm-starting loses alignment tables in v3 model | `train_200k.py` / `pipeline.py:185` |
| 12 | **Low** | Pipeline | No truecaser used in default `train_python` | `pipeline.py:165` |
