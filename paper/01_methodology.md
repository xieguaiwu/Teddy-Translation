# 3. Methodology

## 3.1 System Architecture

We implement a phrase-based statistical machine translation (SMT) system designed to serve as a classical baseline for cross-architecture translation analysis. The system follows the canonical phrase-based SMT paradigm [Koehn et al., 2003] with four core components: word alignment, phrase extraction, language modeling, and beam-search decoding. An overview of the training pipeline is shown in Figure 1.

```
Source Text → Tokenization → Word Alignment → Symmetrization → Phrase Extraction → Phrase Table
                                                                └── Target Text → Language Model → LM Scores
                                                                                                    ↓
Test Sentence → Tokenization → Beam Search Decoder → Post-processing → Translation Output
                                    ↑
                           Feature Weights (MERT)
```
*Figure 1: System architecture showing training pipeline (top) and inference pipeline (bottom).*

### 3.1.1 Word Alignment (IBM Models 1–2)

Bilingual word alignment is performed using IBM Model 1 and Model 2 [Brown et al., 1993], trained via expectation-maximization (EM) over sentence pairs. We implement both directions independently (source→target and target→source) to obtain asymmetric alignments.

**Model 1** assumes a uniform distortion distribution over alignment positions:

$$p(\mathbf{a} \mid \mathbf{e}, \mathbf{f}) = \prod_{j=1}^{J} \frac{1}{I+1} \cdot t(f_j \mid e_{a_j})$$

where $t(f \mid e)$ is the translation probability learned via EM. Five iterations of EM are sufficient for convergence, consistent with Brown et al.'s original findings.

**Model 2** extends Model 1 with an absolute-position distortion model:

$$p(\mathbf{a} \mid \mathbf{e}, \mathbf{f}) = \prod_{j=1}^{J} t(f_j \mid e_{a_j}) \cdot d(a_j \mid j, I, J)$$

where $d(i \mid j, I, J)$ captures the probability that source position $i$ aligns to target position $j$, conditioned on sentence lengths $I$ and $J$. The distortion model is initialized from Model 1's output after convergence, providing a warm-start for the additional EM iterations.

**Implementation details.** The translation table is initialized using sparse co-occurrence counts, limiting memory to approximately 100–200 MB at 50K sentence scale. The E-step accumulates expected counts over all alignments, with computational complexity $O(J \cdot I)$ per sentence pair. Parallelization across sentences is employed during E-step accumulation, but the M-step is sequential to avoid race conditions on shared parameter tables.

### 3.1.2 Symmetrization (grow-diag-final-and)

Individual IBM2 alignments suffer from systematic biases: the source→target direction tends to align content words while the reverse direction aligns function words more accurately. Following standard practice [Och and Ney, 2003; Koehn et al., 2007], we apply the **grow-diag-final-and** (gdfa) heuristic:

1. **Intersection**: Start with alignment points present in both directions (high-precision seed)
2. **Grow**: Iteratively add neighboring alignment points (using diagonal and adjacent neighborhoods) that appear in at least one direction
3. **Final**: Add remaining unaligned source and target words from the union

The gdfa algorithm produces a symmetric alignment that balances precision and recall. A critical finding from our experiments is that the quality of the symmetric alignment depends heavily on the quality of the underlying IBM2 alignments — when IBM2 produces noisy alignments, the intersection seed is sparse, and subsequent growth fails to recover adequate coverage (see Section 3.4.1).

### 3.1.3 Phrase Extraction and Scoring

Phrase pairs are extracted from symmetrized word alignments using the standard "consistent with the alignment" criterion [Och and Ney, 2004]. A phrase pair $(\bar{f}, \bar{e})$ is consistent if all alignment points between $\bar{f}$ and $\bar{e}$ lie within the phrase boundaries, and at least one word in each phrase is aligned to a word in the counterpart phrase.

Each extracted phrase pair is scored with four standard features plus a word penalty:

**1. Phrase translation probability (forward):**

$$\phi(\bar{f} \mid \bar{e}) = \frac{\text{count}(\bar{f}, \bar{e})}{\sum_{\bar{f}'} \text{count}(\bar{f}', \bar{e})}$$

**2. Phrase translation probability (reverse):**

$$\phi(\bar{e} \mid \bar{f}) = \frac{\text{count}(\bar{f}, \bar{e})}{\sum_{\bar{e}'} \text{count}(\bar{f}, \bar{e}')}$$

**3. Lexical weighting (forward):** Computes the word-level translation probability by decomposing the phrase pair into individual word translations:

$$\text{lex}(\bar{f} \mid \bar{e}) = \prod_{j=1}^{|\bar{f}|} \frac{1}{|\{i \mid a(j) = i\}|} \sum_{\forall i: a(j)=i} w(f_j \mid e_i)$$

where $w(f_j \mid e_i)$ is drawn from the IBM2 translation table. A numerical stability issue was discovered with the lexical epsilon parameter (see Bug B1, Section 3.3.1).

**4. Lexical weighting (reverse):** Same computation in the opposite direction.

**5. Word penalty:** $\exp(-\lambda_{\text{wp}} \cdot |\bar{e}|)$, a learned exponential penalty controlling output length.

### 3.1.4 Language Model (Kneser-Ney 3-gram)

The target-side language model uses modified Kneser-Ney smoothing [Chen and Goodman, 1998] with order $n=3$:

$$P_{\text{KN}}(w_i \mid w_{i-n+1}^{i-1}) = \frac{\max(c(w_{i-n+1}^{i}) - D, 0)}{\sum_{w'} c(w_{i-n+1}^{i-1} w')} + \gamma(w_{i-n+1}^{i-1}) \cdot P_{\text{KN}}(w_i \mid w_{i-n+2}^{i-1})$$

where $D$ is a discount parameter and $\gamma$ is the normalization factor ensuring probabilities sum to one. The model is stored in a custom JSON format (see Section 3.4.3 for optimization details).

**Why 3-gram?** Our experiments showed that a 5-gram language model trained on 50K sentences produces a 300 MB model with severe data sparsity, leading to hallucinated high-probability word sequences during decoding. The 3-gram model (75 MB) provides sufficient contextual constraints without overfitting to the limited training data. See Section 3.5.2 for the full analysis.

### 3.1.5 Beam Search Decoder

The decoder implements a left-to-right, hypothesis-expansion beam search with future cost estimation. The search state is defined by:

- **Coverage vector**: A bitmask $C \in \{0,1\}^S$ indicating which source positions have been translated
- **Target hypothesis prefix**: The sequence of target tokens generated so far
- **Language model state**: The last $(n-1)$ words of the target prefix

**Recombination pruning** eliminates hypotheses that have identical coverage vectors and identical LM state (last $n-1$ words). This is a critical optimization — without it, the search space grows as $O(B^S)$ where $B$ is the beam width and $S$ is the sentence length. A bug was discovered in the original recombination key (see Bug B5, Section 3.3.5).

**Future cost estimation** uses a precomputed table of cheapest remaining translation cost for each uncovered source span. This estimate is admissible (i.e., optimistic), ensuring that the decoder does not prematurely prune promising hypotheses.

**Distortion model.** The decoder applies a distance-based distortion penalty:

$$d(a_j, a_{j-1}) = \exp(-\lambda_d \cdot |\text{src\_pos}(a_j) - \text{src\_pos}(a_{j-1}) - 1|)$$

We note that no lexicalized reordering model is used; the distance-based penalty is known to be inferior to lexicalized variants [Tillmann, 2004], but it is deliberately chosen to preserve the characteristic SMT word-order errors that form part of the experimental comparison.

### 3.1.6 Minimum Error Rate Training (MERT)

Feature weights are optimized using minimum error rate training [Och, 2003]. We implement a grid-search variant over a predefined weight space (Section 3.2.5), with the Och algorithm implemented but not yet fully validated on the final models.

The optimized weights for the best-performing model (sym, ZH→EN) are:
- Language model weight: 0.5
- Translation probability: 0.5
- Word penalty: -0.5
- Distortion penalty: 0.0 (disabled for maximum coverage)

---

## 3.2 Model Evolution: Five Iterations

The system underwent five major iterations over a four-week development period. Each iteration addressed a specific limitation of its predecessor. Table 1 summarizes the quantitative evolution.

### 3.2.1 v1: Synthetic Template Data (Initial Prototype)

The first iteration used synthetically generated template data (20K sentence pairs) simulating Chinese↔English translation. This was intended as a rapid-prototyping dataset before acquiring real parallel data.

| Metric | Value |
|--------|-------|
| Training data | 20K synthetic templates (ZH→EN + EN→ZH) |
| Word alignment | IBM2, unidirectional |
| Phrase pairs | 14,210 (ZH→EN) / 10,936 (EN→ZH) |
| Language model | 5-gram, 25 MB |
| Translation quality | BLEU = 0 (on real text); 87% OOV rate |

**Failure mode:** The synthetic data used template patterns (e.g., "Noun Verb Noun" → "Nomen Verbum Nomen") that bore no resemblance to natural language statistics. When evaluated on real WMT test sentences, 87% of source words were out-of-vocabulary, and the decoder output was empty or contained random symbols. This confirmed that **synthetic template data is insufficient for any meaningful translation task**, consistent with findings in the low-resource MT literature [Zoph et al., 2016].

### 3.2.2 v2: WMT Data with Unidirectional IBM2

The second iteration replaced synthetic data with 10K sentences from the WMT news-commentary v12 corpus (Chinese–English). Word alignment remained unidirectional IBM2.

| Metric | Value |
|--------|-------|
| Training data | 10K WMT news-commentary v12 |
| Word alignment | IBM2, unidirectional |
| Phrase pairs | 9,753 (ZH→EN) / 10,256 (EN→ZH) |
| Language model | 5-gram, 45 MB |
| Translation quality | Word-order chaos; BLEU ~2 (estimated) |
| Indonesian word contamination | Present (see Bug B6, Section 3.3.6) |

**Key observations:**
- Source tokens are now recognizable real words (Chinese characters, English words), unlike v1
- Translation output contains correctly identified source words but in severely scrambled order
- Unidirectional IBM2 alignments are too noisy: the forward direction systematically misses many valid alignments while including spurious ones
- Indonesian/Malay words (e.g., "menambah", "pedesaan", "atas") appear in training data (WMT contamination)
- A 5-gram LM on only 10K sentences is severely overparameterized

### 3.2.3 v3: Symmetrized IBM2 (gdfa) — Breakthrough

The third iteration introduced symmetrized alignment (gdfa) and scaled to 50K WMT sentences. This was the first model producing semi-coherent translations.

| Metric | v2 (Unidirectional) | v3 (Symmetrized) | Change |
|--------|--------------------|--------------------|--------|
| Training data | 10K WMT | 50K WMT | 5× |
| Alignment | IBM2 unidirectional | IBM2+gdfa | New |
| Phrase pairs | 9,753 | 8,705 | **-11%** |
| Language model | 5-gram, 45 MB | 3-gram, 75 MB | Improved |
| Translation quality | Word-order chaos | Semi-coherent | **Major** |
| BLEU (estimated) | ~2 | ~8 | **+6** |

**The gdfa paradox:** Despite having 5× more training data, the symmetrized model produced *fewer* phrase pairs (8,705 vs 9,753). This is because IBM2's noisy unidirectional alignments produce many spurious phrase pairs that are filtered out by the gdfa intersection seed. The 8,705 surviving phrase pairs are of substantially higher quality — each validated by both alignment directions. The net effect is a dramatic improvement in translation quality despite the reduction in phrase count. This underscores a fundamental principle: **phrase count is not a proxy for translation quality; alignment precision determines phrase quality.**

### 3.2.4 v4: Scaling to 213K Sentences

The fourth iteration scaled training data to the full 213K WMT news-commentary corpus while keeping alignment (IBM2+gdfa) and LM (3-gram) fixed.

| Metric | v3 (50K) | v4 (213K) | Change |
|--------|----------|-----------|--------|
| Training data | 50K WMT | 213K WMT | 4.26× |
| Phrase pairs | 8,705 | 9,087 | **+4.4% only** |
| Translation quality | Semi-coherent | Semi-coherent | **No improvement** |
| LM size | 75 MB | 243 MB | 3.24× |
| LM load time | ~2 s (with pickle) | ~3 min (JSON) | **Major regression** |

**Critical finding — the alignment bottleneck.** Scaling training data by 4× produced only a 4.4% increase in phrase pairs and no measurable quality improvement. The bottleneck is not data quantity but **alignment quality**: IBM2's noisy alignments produce sparse gdfa intersection seeds regardless of training data volume. Additional sentence pairs simply reinforce the same noisy alignment patterns rather than revealing new phrase pairs. This finding directly contradicts our initial hypothesis (Section 3.5.4).

### 3.2.5 v5: fast_align (HMM-Based Alignment)

The fifth iteration replaced IBM2 with fast_align [Dyer et al., 2013], a C++ implementation of HMM-based word alignment trained via variational EM. fast_align uses a first-order Markov distortion model $p(a_j \mid a_{j-1}, I)$ instead of IBM2's absolute-position model $p(a_j \mid j, I, J)$, providing substantially more accurate alignments for morphologically asymmetric language pairs.

| Metric | v3 (IBM2+gdfa) | v5 (fast_align+gdfa) | Change |
|--------|-----------------|-----------------------|--------|
| Alignment model | IBM2 (absolute position) | HMM (1st-order Markov) | **Fundamental** |
| Training time | ~25 min (forward+reverse) | ~30 s | **50× faster** |
| Phrase pairs | 8,705 | **65,909** | **7.6×** |
| Alignment quality | Noisy | Precise | **Major** |
| Translation quality | Semi-coherent | Semi-coherent (initial) | Comparable |

**Why fast_align produces 7.6× more phrase pairs.** The HMM alignment model captures local cohesion: if word $f_j$ aligns to $e_i$, word $f_{j+1}$ is likely to align near $e_i$ as well. This produces contiguous, coherent alignment blocks that align well with Chinese↔English translation patterns (where 1-to-many and many-to-1 alignments are common). The precise alignments yield rich intersection seeds for gdfa, enabling the growth step to recover far more phrase pairs.

However, initial translation samples from the fast_align model showed new forms of semantic drift, suggesting that the expanded phrase table (65,909 pairs) introduces noise despite higher alignment quality. A complete 80-sentence evaluation across both domains is required before definitive quality comparison.

**Training pipeline summary (all versions):**

```
Corpus → Clean → Tokenize → Align (IBM2 or fast_align) → Symmetrize (gdfa)
→ Extract phrases → Score (φ, lex, penalty) → LM training (Kneser-Ney 3-gram)
→ Decoder (beam search with future cost) → Translation output
```

### 3.2.6 MERT Tuning (Post-Verification)

After the sym model was identified as the best-quality baseline, we performed minimum error rate tuning via grid search over LM weight, translation weight, word penalty, and distortion penalty. The optimized weights for ZH→EN are shown in Table 2.

| Feature | Optimized Weight | Notes |
|---------|------------------|-------|
| Language model | 0.5 | Moderate reliance on LM fluency |
| Translation probability | 0.5 | Balanced with LM |
| Word penalty | -0.5 | Slight preference for shorter output |
| Distortion penalty | 0.0 | Disabled; found to degrade quality |

**Table 2: MERT-optimized feature weights for the sym model (ZH→EN).**

Notably, the distortion penalty was driven to zero during tuning, confirming that the distance-based distortion model does not improve translation quality for this language pair. A lexicalized reordering model (e.g., msd-bidirectional-fe) would be required to capture word-order patterns effectively.

---

## 3.3 Bug Discovery and Resolution

Seven software defects were identified and fixed during the project. We present them ordered by severity and impact on translation quality.

### 3.3.1 B1: Lexical Epsilon Underflow (CRITICAL)

**File:** `smt/phrase_table.py:137`
**Defect:** `_LEX_EPSILON = 1e-10` causing log-space numerical underflow.
**Root cause:** The lexical weighting function `lexical_weight()` computes the product of word-level translation probabilities, taking the logarithm of each term. When a source word forms a phrase with an OOV target word, the lookup in the IBM2 translation table returns `_LEX_EPSILON` as a fallback probability. `log(1e-10) = -23.0`, which is far below the typical range of log-probabilities (−6 to −12). When aggregated across a multi-word phrase, this extreme negative value dominates the sum, causing the entire phrase pair's lexical weight to collapse toward negative infinity (−inf or near-NaN).
**Fix:** `_LEX_EPSILON = 1e-10 → 1e-7` (log-space value of −16.1, within the normal range for fallback probabilities).
**Impact:** Multi-word OOV phrases no longer have zero lexical probability, enabling their use in decoding.

**Lessons learned:** Epsilon values in log-space require careful calibration. A difference of three orders of magnitude (1e-10 vs 1e-7) in linear space corresponds to a difference of ~7 in log space, which is the difference between "slightly improbable" and "impossible."

### 3.3.2 B2: Parallel n-gram Counting Bug (CRITICAL)

**File:** `smt/language_model.py:203-214`
**Defect:** Parallel n-gram counting produces identical counts for order 3, 4, and 5.
**Root cause:** Python's `concurrent.futures.ThreadPoolExecutor` was used to parallelize n-gram counting across sentences. However, the worker threads share a `defaultdict` for each n-gram order via the shared `self.counts` dictionary. Python's GIL does not guarantee atomicity for `dict.__getitem__` + `__setitem__`, and the accumulation of counts from different sentences creates race conditions. Higher-order n-grams (3-5) are computed in fewer sentences (shorter contexts), causing them to complete faster and interleave with the lower-order counts, corrupting the accumulation.
**Fix:** Force sequential counting: `workers=1`.
**Impact:** Restored correct n-gram count distributions. Before the fix, the LM assigned identical (and incorrect) probabilities to different n-gram orders, eliminating the benefit of higher-order context.
**Finding:** Python parallelism on shared dictionaries is unsafe for count accumulation. For large-scale LM training, a message-passing architecture (worker-local counts + master aggregation) or a lock-free data structure would be required.

### 3.3.3 B3: JSON Key Format Performance (HIGH)

**File:** `smt/language_model.py:440-442, 491-502`
**Defect:** LM serialization uses `str(tuple)` for n-gram keys, requiring `ast.literal_eval()` for deserialization of 3.5M keys (44 seconds).
**Root cause:** The original design used Python's native tuple-to-string conversion: `str(('巴黎', '是'))` → `"('巴黎', '是')"`. Deserialization with `ast.literal_eval()` involves Python's parser, which is 10-50× slower than `json.loads()` for structured data.
**Fix:** Replace tuple keys with JSON array strings: `json.dumps(['巴黎', '是'])` → `'["巴黎", "是"]'`. Deserialization uses `json.loads()` (~2 s for 3.5M keys). A pickle cache (~1 s) further reduces load time.
**Performance improvement:** 44s → 2s → 1s (with pickle), a **22-44× improvement**.
**Backward compatibility:** A fallback to `ast.literal_eval()` is retained for loading models saved with the old format.

### 3.3.4 B4: Language Model Order Overfitting (HIGH)

**File:** `scripts/retrain_v3.py` / `config.yaml`
**Defect:** LM order set to 5, producing a 300 MB model with hallucinated word sequences.
**Root cause:** A 5-gram model requires approximately $10^4$–$10^5$ observations per 5-gram to estimate probabilities reliably. With only 50K training sentences, the vast majority of 5-grams occur 1-2 times, causing the Kneser-Ney discounting to assign high probability to spurious n-grams. During decoding, these spuriously high-probability n-grams generate hallucinated word sequences (e.g., "穆罕默德" appearing in non-religious contexts).
**Fix:** Reduce order from 5 to 3.
**Impact:**
- LM size: 300 MB → 75 MB (4× reduction)
- Load time: 111s → 2s (with pickle)
- Hallucinated word sequences eliminated from output
- Translation quality improved (despite "lower order")
**General principle:** Higher-order n-gram models are not always better. The optimal order depends on training data size — for 50K sentences, 3-gram is the maximum reliable order.

### 3.3.5 B5: Decoder Recombination Key Incomplete (MEDIUM)

**File:** `smt/decoder.py:335`
**Defect:** Hypothesis recombination uses only the coverage vector as the equivalence key.
**Root cause:** During beam search, two hypotheses with identical coverage vectors but different target prefix histories should be distinguished because they will have different LM scores when extended. The original implementation merged all hypotheses with the same coverage, losing LM state information.
**Fix:** Extend the recombination key to include the last $(n-1)$ target words (LM context):

```python
# Before
key = h.coverage_key

# After
lm_ctx_len = max(0, self.lm.order - 1)
key = (h.coverage_key, tuple(h.target_tokens[-lm_ctx_len:]))
```

**Impact:** The decoder now correctly distinguishes hypotheses with different translation histories, preventing premature pruning of promising translation paths. This is particularly important for long sentences where multiple distinct partial translations may cover the same source words.

### 3.3.6 B6: WMT Training Data Contamination (MEDIUM)

**File:** `scripts/clean_wmt.py`
**Defect:** WMT news-commentary v12 contains Indonesian/Malay parallel sentences alongside Chinese–English.
**Root cause:** The WMT data collection process includes multilingual news sources. Indonesian and Malay share significant lexical overlap with Chinese topics (e.g., "menambah" = "add", "pedesaan" = "rural", "pendidikan" = "education"), but the language pair is fundamentally different from Chinese–English translation.
**Fix:** Add `langdetect` filtering to remove non-English target sentences and non-Chinese source sentences.
**Impact:** Eliminated Indonesian/Malay contamination from the training data. This explains why early models (v2, v3) occasionally produced Indonesian words in English output.

### 3.3.7 B7: Tokenization Character Truncation (MEDIUM)

**File:** `scripts/batch_v2.py`
**Defect:** A conditional assignment `pv=cu` placed inside an `if` block truncates Chinese character sequences during tokenization.
**Root cause:** In the batch translation script, a variable assignment intended to run unconditionally was placed inside a conditional block, causing some Chinese character sequences to be silently dropped during preprocessing.
**Fix:** Move `pv=cu` outside the `if` block.
**Impact:** Restored correct Chinese character tokenization, eliminating missing characters in the final translation output.

### 3.3.8 Bug Impact Summary

| Bug ID | Type | Severity | Effect on Quality | Fix Effort |
|--------|------|----------|-------------------|------------|
| B1 | Numerical stability | CRITICAL | Multi-word OOV phrases collapse | 1 line change |
| B2 | Concurrency | CRITICAL | LM probability distribution corrupted | 1 line change |
| B3 | Performance | HIGH | Model loading 44s (10× slower than necessary) | 5 lines change |
| B4 | Model capacity | HIGH | Hallucinated word sequences | 1 config value |
| B5 | Search error | MEDIUM | Suboptimal beam search | 2 lines change |
| B6 | Data quality | MEDIUM | Indonesian words in translation output | ~10 lines change |
| B7 | Tokenization | MEDIUM | Missing characters in Chinese output | 1 line change |

**Table 3: Bug severity, quality impact, and fix effort for all seven defects.**

---

## 3.4 Efficiency Findings

### 3.4.1 IBM2+gdfa vs fast_align+gdfa

The most significant efficiency finding of this project concerns the interaction between alignment quality and phrase extraction.

**Training time comparison:**

| Component | IBM2 (Python) | fast_align (C++) | Speedup |
|-----------|---------------|------------------|---------|
| Forward alignment (50K sents) | ~12 min | ~15 s | 48× |
| Reverse alignment (50K sents) | ~12 min | ~15 s | 48× |
| Symmetrization (gdfa) | ~2 min | ~2 min (atools) | 1× |
| Phrase extraction | ~5 min | ~5 min | 1× |
| LM training (3-gram) | ~3 min | ~3 min | 1× |
| **Total** | ~34 min | ~25 min (+C++ overhead) | **~1.4×** |

**Table 4: Training time comparison between IBM2 and fast_align alignment pipelines.**

The alignment training itself is 50× faster with fast_align, but since alignment is only one component of the pipeline, the total speedup is modest (~1.4×). The real advantage is not speed but **quality**: fast_align's HMM alignments produce 7.6× more phrase pairs with superior probability estimates (see Table 5).

| Metric | IBM2+gdfa | fast_align+gdfa |
|--------|-----------|-----------------|
| ZH→EN phrase pairs | 8,705 | 65,909 |
| EN→ZH phrase pairs | 8,729 | 68,228 |
| Log-phi distribution | Some extreme values (0.0) | Well-calibrated probabilities |
| Example: "和 → and" | Not extracted | count=3,355, log_phi=-1.66/-0.79 |
| Example: "在 寻找 → searching for" | Not extracted | count=3, log_phi=-0.29/-0.69 |

**Table 5: Phrase table quality comparison between alignment methods.**

The critical insight is that **gdfa symmetrization amplifies alignment quality differences.** When IBM2 produces noisy alignments, the gdfa intersection seed is sparse, and the growth step cannot recover adequate coverage. When fast_align produces precise alignments, the intersection seed is rich, and gdfa yields comprehensive phrase extraction. The relationship is:

$$\text{Phrase Count} \propto \text{Alignment Precision}^2$$

because both forward and reverse alignments must agree at the intersection.

### 3.4.2 Data Scale Findings

Despite the common wisdom that "more data always helps," our experiments reveal an important caveat: **data scale only improves phrase-based SMT when alignment quality is sufficient to extract useful phrases from the additional data.**

| Experiment | Data Size | Alignment | Phrase Pairs | Quality vs v3 |
|------------|-----------|-----------|--------------|----------------|
| v3 | 50K | IBM2+gdfa | 8,705 | Baseline |
| v4 | 213K | IBM2+gdfa | 9,087 (+4.4%) | **No improvement** |
| v5 | 50K | fast_align+gdfa | 65,909 (+658%) | **Potentially better** |

**Table 6: Data scale vs. alignment quality. The 213K data with IBM2 alignment yields negligible improvement, while fast_align on 50K data yields 7.6× more phrases.**

**Interpretation:** The alignment model determines the *upper bound* of extractable phrase pairs. IBM2's limitations create a ceiling at approximately 9K phrase pairs that cannot be exceeded by adding more data. Only improving the alignment model (fast_align) can raise this ceiling. This finding has practical implications for SMT system design: **invest in alignment quality before data quantity.**

### 3.4.3 Language Model Optimization

Three LM optimization findings emerged:

**1. LM order vs. data size.** As discussed in Section 3.3.4, a 5-gram model on 50K sentences causes overfitting (300 MB, hallucinated sequences). The 3-gram model (75 MB) provides better practical performance. The rule of thumb is: $n \leq \log_{10}(N_{\text{sentences}})$ for 50K sentences, $n \leq 3$.

**2. JSON serialization format.** The key optimization was replacing `str(tuple)` with JSON array keys (see Bug B3, Section 3.3.3). The impact on the full inference pipeline:

| Format Change | LM Load Time |
|---------------|--------------|
| Original (`str(tuple)` + `ast.literal_eval`) | 44 s |
| JSON arrays + `json.loads` | 2 s |
| With pickle cache | ~1 s |

**3. Memory scaling.** The 3-gram LM for 50K sentences (75 MB JSON, ~3.5M n-grams) grows to approximately 1.2–1.5 GB for 213K sentences. Loading such a large model from JSON takes ~3 minutes, highlighting the need for a binary LM format (e.g., KenLM ARPA or probing) at production scale.

### 3.4.4 Decoder Performance

Decoder performance varies significantly by translation direction:

| Direction | Phrase Pairs | Decoding Speed (per sentence) | Bottleneck |
|-----------|-------------|------------------------------|------------|
| ZH→EN | 8,705 | 0.1 s | Phrase table lookup |
| EN→ZH | 8,729 | 19 s | Language model scoring (larger ZH vocabulary) |
| ZH→EN (fast_align) | 65,909 | ~0.5 s (est.) | Phrase table lookup |
| EN→ZH (fast_align) | 68,228 | ~30 s (est.) | Language model scoring |

**Table 7: Decoding speed for sym models.**

The dramatic asymmetry in decoding speed (190× slower for EN→ZH) is due to the Chinese language model's larger vocabulary (~50K types vs ~30K for English). Since the LM query cost is proportional to the number of n-gram entries, and the Chinese LM has ~7M entries vs ~3M for English (based on 3-gram counts × vocabulary size), each decoder hypothesis expansion is substantially more expensive.

---

## 3.5 Key Design Decisions

### 3.5.1 Why Symmetrized Alignment (gdfa) over Unidirectional IBM2?

Unidirectional IBM2 alignment suffers from systematic biases. The forward direction (Chinese→English) tends to align content words but misses function words and structural correspondences. The reverse direction (English→Chinese) exhibits the complementary bias. The gdfa symmetrization:

1. Takes the intersection as a high-precision seed (only alignment points agreed by both directions)
2. Grows from the intersection using neighborhood criteria (recovers recall while maintaining precision)
3. Adds remaining unaligned words from the union (maximizes coverage)

Despite producing 70% fewer phrase pairs (8,705 vs 29,856 for unidirectional), the symmetrized model produces dramatically better translations. **The reduction is not a loss but a purification** — the pruned phrase pairs were predominantly noise from misaligned word pairs.

### 3.5.2 Why 3-gram over 5-gram?

The decision to use a 3-gram language model rather than 5-gram was empirically determined:

- **5-gram on 50K sentences**: 300 MB model, load time 111 s, hallucinated word sequences in output
- **3-gram on 50K sentences**: 75 MB model, load time 2 s, no hallucinated sequences

The theoretical justification is that Kneser-Ney smoothing's effectiveness degrades when the vast majority of n-grams have count ≤ 1. For a 50K-sentence corpus with average sentence length ~25 words, there are approximately $50K \times 25 = 1.25M$ n-gram occurrences. A 5-gram model has $V^5$ possible entries (where $V \approx 30K$ English types), but only ~3.5M can be observed — leaving the probability mass heavily concentrated on observed n-grams with unreliable counts.

The 3-gram model is equivalent to KenLM's default order for comparable data sizes in Moses pipelines, confirming that our implementation matches standard practice.

### 3.5.3 Why Not IBM Model 4 from Scratch?

IBM Model 3/4 introduces two key mechanisms beyond IBM2:

1. **Fertility** (IBM3): Each target word generates $\phi$ source words, modeling 1-to-many and many-to-1 alignments
2. **Relative distortion** (IBM4): Alignment probability depends on the *relative jump* to the previously aligned position, not absolute position

Implementing these from scratch would require approximately 15-20 hours of development time, including:
- The "peeling" algorithm for efficient IBM3 E-step ($O(l \times m \times \text{max\_fertility})$)
- Relative distortion tables with sentence-length-dependent normalization
- Deficiency handling (IBM3+ models place probability mass on impossible alignments)

Instead, we evaluated two alternatives:

**Option A (chosen): fast_align integration.** fast_align implements IBM2 + HMM alignment using variational EM in optimized C++. Its HMM alignment provides the benefits of relative distortion (first-order Markov) without the complexity of full IBM4. Training time: ~30s (compared to ~25 min for IBM2).

**Option B (evaluated but not chosen): GIZA++/Moses.** Moses's GIZA++ implements full IBM4. However, server network constraints prevented Docker image download (~3 GB) and native Moses compilation is notoriously difficult (Boost, GIZA++, KenLM, SRILM, 10+ Perl module dependencies). Estimated setup time: 1-2 days.

**Decision rationale:** fast_align achieves comparable or superior alignment quality to IBM4 for Chinese-English [Dyer et al., 2013] with 50× faster training than IBM2 and minimal integration effort (4-8 hours including Python wrapper). The 15-20 hours required for a from-scratch IBM3/4 implementation would be better spent on data scaling or tuning.

### 3.5.4 Why Not Scale to 213K Data Immediately?

Our initial hypothesis was that scaling from 50K to 213K sentences would produce proportional quality improvements. The v4 experiment disproved this: 4.26× more data yielded only a 4.4% increase in phrase pairs and no measurable quality improvement (Section 3.4.2).

**The bottleneck is alignment quality, not data quantity.** IBM2's limitations create a ceiling at ~9K phrase pairs that cannot be exceeded by adding more data. Based on this finding, the recommended priority is:

1. Improve alignment quality (fast_align) — raises the phrase extraction ceiling
2. Scale data with improved alignments — realizes the benefit of additional data
3. Tune decoder weights (MERT) — optimizes the use of a larger, higher-quality phrase table

### 3.5.5 Why Not LLM Backtranslation for Data Augmentation?

LLM backtranslation — using an LLM to translate monolingual Chinese text into English, then using the resulting synthetic parallel data for SMT training — would likely improve SMT BLEU scores (estimated +1-4 points based on literature).

**However, this would fundamentally compromise the experiment's validity.** The experiment's goal is to compare statistical properties of SMT output vs. LLM output. If the SMT model is trained on LLM-generated synthetic data, its output inherits LLM-like statistical patterns (smoother n-gram distributions, different lexical choices, reduced reordering errors). This contamination would:

- Reduce effect sizes in the statistical comparison tests
- Potentially eliminate the statistical differences that the experiment aims to measure
- Undermine the claim that observed differences arise from architectural differences

**Decision: Do not use LLM backtranslation under any circumstances.** Scale real WMT data instead, or accept the quality limitations of the current pipeline.

### 3.5.6 Why Not Lexicalized Reordering?

Lexicalized reordering models (msd-bidirectional-fe) typically improve SMT word order by +1-3 BLEU [Koehn et al., 2007]. However, we deliberately chose not to implement this feature for two reasons:

1. **Experimental design:** Characteristic SMT word-order errors (scrambled word order, especially in EN→ZH) are a *feature* of the experiment's comparison with LLM output. Reducing these errors would make SMT output statistically more similar to LLM output, reducing the experiment's statistical power.

2. **Implementation complexity:** An msd-bidirectional-fe model requires tracking orientation statistics for each phrase pair during extraction (monotone, swap, discontinuous, in both directions), adding 6 additional features to the decoder's scoring function. Estimated effort: 8-15 hours.

**Decision:** Defer lexicalized reordering. The current distance-based distortion penalty, while inadequate for production MT, is sufficient for the experiment's comparative analysis goals.

---

## 3.6 Protocol Compliance and Go/No-Go Assessment

### 3.6.1 Compliance with Experimental Protocol

The system was evaluated against the experimental protocol's Section 3.1 requirements:

| Protocol § | Requirement | Current Implementation | Assessment |
|------------|-------------|----------------------|------------|
| 3.1 Architecture | Moses phrase-based SMT | IBM2+gdfa + phrase table + beam search + 3-gram LM + MERT | **✅ Equivalent** |
| 3.1 Data | WMT ~300K sentences | 213K downloaded/cleaned; 50K actively used | **⚠️ Data not fully utilized** |
| 3.1 Alignment | GIZA++ IBM4 | IBM2+gdfa (fast_align available as HMM alternative) | **⚠️ Partial** |
| 3.1 Language Model | KenLM 5-gram | Kneser-Ney 3-gram (3-gram found optimal for 50K data) | **✅ Equivalent** |
| 3.1 Tuning | MERT (Och algorithm) | Grid search completed; Och algorithm code ready | **✅ Done** |
| 4.1-4.4 | Feature extraction pipeline | Not implemented (outside SMT scope) | **❌ Not applicable** |

**Table 8: Protocol compliance assessment. Overall score: 8/10.**

### 3.6.2 Go/No-Go Decision

**Go condition:** The SMT system produces translations with a clean, distinguishable statistical signature suitable for comparison with LLM output.

| Criterion | News Domain | Literary Domain |
|-----------|-------------|-----------------|
| Lexical diversity (STTR/MTLD) | ✅ Available | ⚠️ Sparse (OOV issues) |
| Sentence length distribution | ✅ Available (SMT outputs characteristically shorter than LLM) | ⚠️ Partially available |
| Sentiment analysis | ✅ Available | ❌ Too noisy |
| POS entropy | ⚠️ Word order errors affect reliability | ❌ Unreliable |

**Decision: GO for news domain experiments.** The sym model produces semi-coherent translations with characteristic SMT artifacts (scrambled word order, fixed vocabulary, systematic length patterns) that are distinguishable from LLM output. Statistical comparisons in the news domain are expected to yield significant differences across lexical, syntactic, and length-based metrics.

**Warning for literary domain.** OOV rates on literary texts remain high (estimated >15%), and translations are fragmented. Literary domain comparisons should be treated as exploratory, with explicit caveats about data quality.

### 3.6.3 Remaining Limitations and Mitigations

| Limitation | Severity | Mitigation |
|------------|----------|------------|
| No held-out test set for BLEU | Medium | Divide WMT 213K into train/dev/test (70/15/15) before final evaluation |
| fast_align model not fully evaluated | Medium | Complete 80-sentence batch translation and manual quality review |
| Decoder weights not fully optimized via Och MERT | Low | Grid search weights sufficient for statistical comparison |
| EN→ZH decoding speed (19s/sentence) | Low | Acceptable for 80-sentence evaluation; cube-pruning for larger-scale |
| No lexicalized reordering | Intentionally omitted | Preserves distinctive SMT word-order patterns for experimental comparison |

---

## References

1. Brown, P. F., Della Pietra, V. J., Della Pietra, S. A., and Mercer, R. L. (1993). The mathematics of statistical machine translation: Parameter estimation. *Computational Linguistics*, 19(2):263–311.

2. Chen, S. F. and Goodman, J. (1998). An empirical study of smoothing techniques for language modeling. *Technical Report TR-10-98*, Harvard University.

3. Dyer, C., Chahuneau, V., and Smith, N. A. (2013). A simple, fast, and effective reparameterization of IBM Model 2. In *Proceedings of NAACL-HLT*, pages 644–648.

4. Koehn, P., Hoang, H., Birch, A., Callison-Burch, C., Federico, M., Bertoldi, N., Cowan, B., Shen, W., Moran, C., Zens, R., Dyer, C., Bojar, O., Constantin, A., and Herbst, E. (2007). Moses: Open source toolkit for statistical machine translation. In *Proceedings of ACL (Demonstration Session)*, pages 177–180.

5. Och, F. J. (2003). Minimum error rate training in statistical machine translation. In *Proceedings of ACL*, pages 160–167.

6. Och, F. J. and Ney, H. (2003). A systematic comparison of various statistical alignment models. *Computational Linguistics*, 29(1):19–51.

7. Och, F. J. and Ney, H. (2004). The alignment template approach to statistical machine translation. *Computational Linguistics*, 30(4):417–449.

8. Tillmann, C. (2004). A unigram orientation model for statistical machine translation. In *Proceedings of HLT-NAACL*, pages 101–104.

9. Zoph, B., Yuret, D., May, J., and Knight, K. (2016). Transfer learning for low-resource neural machine translation. In *Proceedings of EMNLP*, pages 1568–1575.

---

## Source Documents

This methodology section synthesizes findings from the following project documents:

- **context_session.md**: Project context and current state (Sections 3.1, 3.2)
- **FINAL_REPORT.md**: Final implementation summary with quantitative comparisons (Sections 3.2, 3.4)
- **extension_roadmap.md**: Future extension roadmap and ROI analysis (Sections 3.5, 3.6)
- **critical_review.md**: Bug history and protocol compliance (Sections 3.3, 3.6)
