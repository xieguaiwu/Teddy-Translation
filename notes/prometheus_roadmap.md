# SMT Project — Future Extension Roadmap

> **Date:** 2026-06-06  
> **Baseline:** Symmetrized IBM2 (grow-diag-final-and), ~8.7K unique phrase pairs (zh→en), 3-gram Kneser-Ney LM, beam search decoder with hardcoded weights.  
> **Baseline quality:** Semi-coherent translations. BLEU estimated ~8-14 on WMT news domain.  
> **Experiment goal:** Produce SMT output with clean statistical signature, distinguishable from LLM-generated translations. Quality does not need to be production-grade; it needs to be *representatively SMT*.

---

## Prioritization Principles

1. **Statistical purity > raw BLEU.** Using LLM-generated training data would contaminate the experiment. Improvements must come from classical SMT techniques.
2. **Build on existing code.** The symmetrized IBM2 pipeline is functional and well-instrumented. Incremental improvements are preferred over wholesale replacement.
3. **Least effort, most gain first.** Each extension is evaluated on ROI: (BLEU gain × feasibility) / effort hours.

---

## Extension Candidates — Ranked by ROI

### Tier 1: High ROI, Low Risk (Do First)

#### 1. Scale to All 213K WMT Sentences

| Dimension | Assessment |
|-----------|-----------|
| **Effort** | 4–6 hours (mostly wall-clock training time) |
| **Expected BLEU gain** | +5–10 points |
| **Risk** | Low |
| **Prerequisites** | WMT data download (~1h, 500MB-1GB); sufficient disk (3–5 GB total) |

**Rationale:** The single largest quality lever is data volume. The v3 model on 50K sentences already produces ~30K phrase pairs. Scaling to 213K sentences (the full WMT zh-en corpus) would:

- Increase phrase table coverage from ~30K to an estimated 60–120K unique pairs
- Reduce OOV rate from ~3.8% to <1%
- Improve phrase probability estimates (more counts → more reliable φ)
- Improve LM quality (3× more target-side sentences for n-gram estimation)

**Implementation plan:**
1. Download WMT zh-en parallel data (script `scripts/download_wmt_data.py` already exists)
2. Run cleaning pipeline (`scripts/clean_wmt.py`)
3. Train symmetrized model on full corpus using `scripts/train_symmetrized.py --max-sentences 213000`
4. Train 3-gram LM with `prune_threshold=0.0` (bypass pruning bottleneck)

**Memory estimate:**
- LM: ~1–1.5 GB (3-gram, ~50K vocab types) — manageable on any modern machine
- Phrase table: ~100–200K entries (~15–30 MB text)
- Training RAM: ~8–16 GB peak (IBM alignment tables)

**Validation:** Compare BLEU before/after on held-out test set. Expect significant coverage improvement on rare words.

---

#### 2. Real MERT Tuning (Och's Algorithm)

| Dimension | Assessment |
|-----------|-----------|
| **Effort** | 6–10 hours |
| **Expected BLEU gain** | +2–5 points |
| **Risk** | Low-Medium |
| **Prerequisites** | Dev set of 500–1000 sentence pairs (separate from train/test); decoder n-best list output |

**Rationale:** MERT is the single largest quality improvement in any SMT pipeline after basic training. The current `mert_tune.py` performs naive grid search (brute-force weight combinations). True Och MERT uses **error surface optimization** via n-best lists, which is both more efficient and more effective.

**Current state vs target:**

| Aspect | Current (`mert_tune.py`) | Target (Och MERT) |
|--------|--------------------------|-------------------|
| Algorithm | Grid search over fixed steps | Line search on piecewise-linear error surface |
| Decoder calls per iteration | 60 (15 values × 4 features) × 100 sentences | 1 decode per sentence (n-best list extraction) |
| Feature coupling | Sequential (tunes one at a time) | Iterative refinement, captures interactions |
| Convergence | No formal convergence check | Powell's method with BLEU convergence |
| Output | Best grid point | Optimal weight vector (potentially non-grid) |

**Implementation plan:**
1. **Add n-best list extraction to decoder** (1–2 hours): Modify `PhraseDecoder.decode()` to return top-N complete hypotheses instead of just the 1-best. Keep the top-N scored completed hypotheses. This is straightforward — the decoder already maintains all completed hypotheses.

2. **Implement error surface computation** (2–3 hours): For a given feature weight, the corpus-level BLEU is piecewise-constant as a function of the weight — it only changes when the 1-best hypothesis in the n-best list changes. Compute the error surface for each feature by identifying all "switch points" in the n-best lists.

3. **Implement Powell's line search** (1–2 hours): For each feature, find the weight that maximizes BLEU by searching the error surface directly (no need for repeated decoding).

4. **Integrate with pipeline** (1–2 hours): Create `scripts/mert_och.py` that loads model, runs MERT on dev set, saves optimized weights as `mert_weights.json`.

5. **Validation** (1 hour): Run on existing smt_zh2en_sym model with 500 dev sentences. Expect 2–5 BLEU improvement.

**Potential pitfall:** MERT can overfit to small dev sets. Mitigation: use at least 500 dev sentences, monitor train/dev BLEU gap. If overfitting is severe, use MIRA or PRO instead (more robust for small dev sets).

**Alternative (lower effort):** Enhance the existing grid search to use random search + early stopping, which gives ~80% of the benefit at ~30% of the effort. Estimated 2–3 hours.

---

### Tier 2: Medium ROI, Medium Risk (Do Second)

#### 3. IBM3 Fertility Model via fast_align

| Dimension | Assessment |
|-----------|-----------|
| **Effort** | 4–8 hours (fast_align integration) vs 15–20 hours (IBM3 from scratch) |
| **Expected BLEU gain** | +2–5 points |
| **Risk** | Medium |
| **Prerequisites** | C++ compiler; fast_align source (GitHub: `clab/fast_align`) |

**Rationale:** IBM2 assumes 1:1 word alignment. Chinese-English violates this systematically:
- 1 Chinese token → 2+ English tokens: `人工智能` → `artificial intelligence`
- 2+ Chinese tokens → 1 English token: `进行 了` → `conducted`

IBM2's Viterbi alignment greedily assigns each source word to one target word, producing fragmented phrases. IBM3's fertility model p(φ|e) learns how many source words each target word should generate, dramatically improving alignment quality for morphologically asymmetric language pairs.

**The fast_align advantage:** Implementing IBM3 EM from scratch is complex (the E-step requires enumerating alignment sequences, exponential without the "peeling" trick). fast_align (Dyer et al. 2013) is a modern C++ reimplementation of IBM2 + HMM alignment that:
- Runs in seconds vs minutes for GIZA++
- Produces comparable or better alignment quality
- Has simple command-line interface
- Can output symmetrized alignments directly

**Implementation plan:**
1. Clone and compile fast_align (30 min): `git clone https://github.com/clab/fast_align && cd fast_align && mkdir build && cd build && cmake .. && make`
2. Write Python wrapper (2 hours): Subprocess-based interface that calls `fast_align` and `atools` (included) for symmetrization
3. Replace IBM2 alignment step in pipeline (1 hour): Create `smt/align_fast.py` as drop-in replacement for `ibm_align.train_ibm()`
4. Run full training with new alignment (1–2 hours of wall time): Compare phrase table quality vs symmetrized IBM2
5. Evaluation (1 hour): BLEU comparison, manual inspection of alignment quality on 20 example sentences

**Fallback: Implement IBM3 in Python (15–20 hours)**
If fast_align doesn't compile or produces worse results, implement IBM3 fertility + IBM4 relative distortion. This requires:
- **IBM3 E-step**: Use the "peeling" algorithm (collect counts by enumerating alignments with at most one "fertile" target word per source word, then normalize). The standard implementation collects fractional counts over all possible alignments via dynamic programming.
- **IBM4 distortion**: Replace IBM2's absolute j→i mapping with relative jump-based distortion p(j - j_{prev} | ...)
- Estimated 15–20 hours with thorough testing.

**Risk mitigation:** Start with fast_align. Only fall back to manual IBM3 implementation if fast_align fails to compile or produces visibly worse alignments than current IBM2.

---

#### 4. KenLM Binary LM Format

| Dimension | Assessment |
|-----------|-----------|
| **Effort** | 5–10 hours |
| **Expected BLEU gain** | 0 (no quality impact) |
| **Risk** | Low-Medium |
| **Prerequisites** | C++ compiler for KenLM; or pure Python binary parser |

**Rationale:** The current JSON LM format with `ast.literal_eval` deserialization takes 44–111s to load for a 50K-sentence LM (3.5M n-grams). Scaling to 213K sentences would make this 3–5× worse (3–5 minutes). While this doesn't affect translation quality, it makes iterative development painful and prevents scaling to larger LMs (5-gram on 213K sentences would be 10M+ n-grams → 5+ minute load).

Additionally, JSON LMs use 3–5× more disk space than KenLM binary format (text vs bit-packed).

**Implementation options:**

| Option | Effort | Load speed | Memory | Risk |
|--------|--------|-----------|--------|------|
| **A: pybind11 wrapper for KenLM C++** | 8–12 hours | ~100ms (mmap) | Minimal (mmap'd) | Medium (C++ compilation) |
| **B: Pure Python KenLM binary reader** | 6–10 hours | ~2–5 seconds | Moderate | Low (no C++) |
| **C: Optimize current JSON format** | 2–3 hours | ~10–20 seconds | Same | Very Low |

**Recommendation: Option C → Option B ladder.**

**Option C (immediate, 2–3 hours):**
- Replace `str(tuple)` keys with JSON arrays: `["word1", "word2"]` instead of `"('word1', 'word2')"`
- This eliminates the `ast.literal_eval` bottleneck entirely (from 44s to ~2s for 3.5M keys)
- Change `language_model.py:save()` and `language_model.py:load()` to use array keys
- Backward compatible: add version marker to JSON

**Option B (medium-term, 6–10 hours):**
- Implement a Python reader for KenLM's ARPA format (text-based, faster than JSON) or probing binary format
- KenLM probing binary is a hash table with 64-bit hashed keys → implement in pure Python using `struct` for binary parsing
- Load time: ~500ms–2s for 10M n-grams
- Disk savings: ~3–5× vs JSON

**Option A (long-term, if LM becomes bottleneck):**
- Compile KenLM as a shared library
- Create pybind11 bindings for `Model::Model(filename)` and `FullScoreReturn BaseScore(...)`
- Would enable 5-gram+ LMs at production scale

---

### Tier 3: Lower ROI or Higher Risk (Consider Later)

#### 5. Lexicalized Reordering Model (msd-bidirectional-fe)

| Dimension | Assessment |
|-----------|-----------|
| **Effort** | 8–15 hours |
| **Expected BLEU gain** | +1–3 points |
| **Risk** | Medium-High |
| **Prerequisites** | Word alignments (already have); phrase table (already have) |

**Rationale:** The current decoder uses simple distance-based distortion (linear penalty per word skipped). For Chinese↔English, where word order can differ significantly (Chinese topicalization: `这本书 我 读过` → "this book I have read" vs "I have read this book"), lexicalized reordering is important.

However, for the experiment's purpose (producing SMT-typical output), distance-based distortion may be *sufficient* to create the characteristic "SMT reordering errors" that distinguish it from LLM output. A perfect reordering model might actually *reduce* the statistical distinguishability.

**What it requires:**
- For each phrase pair, compute 3 orientation probabilities: monotone (M), swap (S), discontinuous (D)
- Compute in both directions (source→target and target→source) = 6 features
- Integrate into decoder's scoring function
- The Moses implementation conditions on the specific phrase pair — orientation is lexicalized

**Implementation complexity:** Collecting orientation statistics requires tracking, for each phrase pair occurrence, whether the next phrase in the source appears to the right (monotone), left (swap), or nonadjacent (discontinuous) in the target. This is a O(corpus × phrase_count) operation during phrase extraction.

**Recommendation:** Defer. Distance-based distortion is adequate for the experiment's statistical comparison goals.

---

#### 6. NiuTrans.SMT as Pipeline Replacement

| Dimension | Assessment |
|-----------|-----------|
| **Effort** | 8–15 hours |
| **Expected BLEU gain** | +5–12 points (replaces entire pipeline) |
| **Risk** | Medium-High |
| **Prerequisites** | C++ compilation; Chinese documentation navigation |

**Rationale:** NiuTrans.SMT is a complete C++ SMT system developed at Northeastern University (China), specifically optimized for Chinese↔English. It supports phrase-based, hierarchical, and syntax-based models. It includes:
- Built-in word alignment (fast_align-like)
- Phrase extraction + scoring
- KenLM integration
- MERT tuning
- Cube pruning decoder

**Key advantage:** It would replace the entire custom pipeline in one step, providing a production-quality SMT baseline.

**Key disadvantage:** 
- Chinese documentation and error messages may be hard to debug
- Compilation may have dependency issues (Boost, KenLM, etc.)
- Integrating with existing data pipeline requires adaptation
- The project's value is partially in *having built* the SMT pipeline; replacing it undermines that

**Recommendation:** Treat as a "Plan B" if the custom pipeline cannot reach sufficient quality. The current pipeline is functionally correct and the main limitation is data scale, not algorithm quality. Fix data scale first.

---

#### 7. Docker/Moses Full Pipeline

| Dimension | Assessment |
|-----------|-----------|
| **Effort** | 5–15 hours (if Docker available) |
| **Expected BLEU gain** | +8–15 points (full Moses = GIZA++ IBM4 + lexicalized reordering + MERT + KenLM) |
| **Risk** | **High** |
| **Prerequisites** | Docker installed; network to pull `amake/moses-smt` image (~3GB); or local image build |

**Rationale:** Moses is the gold standard for phrase-based SMT. The `moses_orch.py` script already provides orchestration. However:

**Network constraint:** The servers have no internet access or unreliable connectivity. Pulling the 3GB `amake/moses-smt` Docker image may be impossible.

**Alternatives for no-network environments:**
1. **Build Docker image from local files** — requires downloading Moses source + GIZA++ + KenLM + SRILM source tarballs to a machine with internet, transferring, and building. ~4–6 hours of effort.
2. **Native Moses installation** — notoriously difficult. Requires: Boost, GIZA++, KenLM, SRILM, IRSTLM (optional), xmlrpc-c, and 10+ Perl modules. Expect 1–2 days of dependency hell.
3. **Pre-built binary transfer** — if another machine has Moses installed, transfer binaries. Fragile (library version dependencies).

**Recommendation:** Only pursue if the scaled custom pipeline (Items 1+2+3) fails to produce adequate quality. The risk/reward ratio is unfavorable given network constraints.

---

#### 8. LLM Backtranslation for Data Augmentation

| Dimension | Assessment |
|-----------|-----------|
| **Effort** | 3–5 hours |
| **Expected BLEU gain** | +1–4 points |
| **Risk** | **High (experiment contamination)** |
| **Prerequisites** | API access (DeepSeek, OpenAI, etc.); API credits |

**Rationale:** Taking monolingual Chinese text and translating it to English via an LLM creates synthetic parallel data that can augment SMT training. This is a standard technique in low-resource MT.

**Why it's dangerous for this experiment:**
The experiment's purpose is to study statistical differences between SMT and LLM output. Using LLM-generated data to train the SMT model would **contaminate** the SMT output with LLM-like statistical properties, undermining the entire experimental comparison.

Specifically:
- Backtranslated data inherits LLM fluency patterns (smoother n-gram distributions, different lexical choices)
- The SMT model trained on LLM output would produce translations that are statistically more similar to LLM output
- This would reduce effect sizes in the ANOVA/KS tests that are the core of the experiment

**When it might be acceptable:**
- Only if SMT quality is so low that translations are unintelligible and no statistical features can be extracted
- In that case, the experiment itself is compromised, and backtranslation would be a salvage operation

**Recommendation:** **Do not use.** The experiment's validity depends on clean SMT training data. Scale real WMT data instead (Item 1).

---

## Recommended Execution Order

```
Phase 1 (Week 1): Foundation — ~10-14 hours
├── 1. Scale to 213K WMT sentences        [4-6h, +5-10 BLEU]
├── 2. Optimize LM JSON format (Option C)  [2-3h, +0 BLEU, ~10× load speedup]
└── 3. MERT Tuning (Och algorithm)         [6-10h, +2-5 BLEU]
    ── Cumulative expected BLEU: ~17-29 ──

Phase 2 (Week 2): Alignment quality — ~4-8 hours
├── 4. fast_align integration              [4-8h, +2-5 BLEU]
└── (If fast_align fails: skip to evaluation)
    ── Cumulative expected BLEU: ~19-34 ──

Phase 3 (Week 2-3, only if needed): Fallback options
├── 5. NiuTrans.SMT evaluation             [8-15h, potentially +5-12 BLEU]
└── 6. KenLM binary format (Option B)      [6-10h, +0 BLEU, scalability]
    ── Only if Phase 1+2 quality insufficient ──

NOT RECOMMENDED:
├── Docker/Moses                           [Too risky, network constraints]
├── LLM Backtranslation                    [Experiment contamination]
└── IBM3 Fertility from scratch            [15-20h, fast_align is better]
```

---

## Effort vs BLEU Summary

```
Extension                          Effort (h)   ΔBLEU    Risk     ROI Score
──────────────────────────────────────────────────────────────────────────
1. Scale to 213K WMT               4-6          +5-10    Low      ★★★★★
2. MERT Tuning (Och)               6-10         +2-5     Low-Med  ★★★★★
3. fast_align (fertility)          4-8          +2-5     Medium   ★★★★
4. KenLM JSON optimize (Opt C)     2-3          +0       V.Low    ★★★★
5. KenLM binary (Opt B)            6-10         +0       Low      ★★★
6. Lexicalized reordering          8-15         +1-3     Med-Hi   ★★★
7. NiuTrans.SMT                    8-15         +5-12    Med-Hi   ★★
8. Docker/Moses                    5-15         +8-15    High     ★
9. IBM3 from scratch               15-20        +2-5     High     ★
10. LLM Backtranslation            3-5          +1-4     V.High   ✗
```

---

## Detailed Implementation Notes

### A. Scaling to 213K: Memory and Time Budget

```
Training step             50K sentences    213K sentences    Scaling
─────────────────────────────────────────────────────────────────────
IBM2 forward EM (5 iter)  ~8 min           ~35 min           ~4.3×
IBM2 reverse EM (5 iter)  ~8 min           ~35 min           ~4.3×
Grow-diag-final-and       ~2 min           ~8 min            ~4×
Phrase extraction          ~5 min           ~20 min           ~4×
LM training (3-gram)       ~3 min           ~12 min           ~4×
─────────────────────────────────────────────────────────────────────
Total wall time            ~26 min          ~110 min (~2h)    ~4.2×

Peak RAM (IBM alignment)   ~4 GB            ~12-16 GB         ~3-4×
Phrase table size          30K entries      60-120K entries   ~2-4×
LM size (3-gram)           300 MB           1.2-1.5 GB        ~4-5×
LM load time (JSON)        44-111s          3-5 min           ~3-5×
Disk total                 500 MB           3-5 GB            ~6-10×
```

**Mitigation for RAM:** The IBM alignment step is the memory bottleneck (stores translation table in memory). The sparse initialization limits this to co-occurring word pairs. At 213K sentences with ~30K EN × 50K ZH vocabulary, expect ~500K-1M co-occurring pairs = ~100-200 MB for the table. The EM step accumulates expected counts which are proportional to sentence length × vocabulary. Sequential E-step is recommended (parallel adds worker overhead).

### B. MERT Dev Set Requirements

A MERT dev set must be:
- **Separate from training data** (no overlap with WMT training sentences)
- **Separate from test data** (no overlap with final evaluation sentences)
- **Domain-matched** (news domain, same style as training)
- **500-1000 sentence pairs** (minimum 500 for stable optimization)

**Source options:**
1. Hold out 1000 sentences from the 213K WMT corpus (reserve before training)
2. Use a separate WMT test set (e.g., newstest20xx)
3. Use the existing `data/dev.zh` and `data/dev.en` files (need to verify they're held-out, not from training)

### C. fast_align Integration — Interface Design

```python
# smt/align_fast.py — proposed interface

def train_fast_align(
    src_sentences: List[List[str]],
    tgt_sentences: List[List[str]],
    output_dir: str,
    iterations: int = 5,
) -> List[List[Tuple[int, int]]]:
    """
    1. Write src/tgt to temporary files (fast_align expects space-separated tokens)
    2. Run: fast_align -i corpus.txt -d -v -o > forward.align
    3. Run: fast_align -i corpus.txt -d -v -o -r > reverse.align
    4. Run: atools -i forward.align -j reverse.align -c grow-diag-final-and > sym.align
    5. Parse sym.align back to List[List[Tuple[int, int]]]
    """
```

The key insight: fast_align's output format is standard Moses-style: `0-0 1-1 2-2 ...` (source-target index pairs, 0-indexed). The `atools` symmetrization tool is included in the fast_align repository.

### D. Why Not IBM Model 4 from Scratch

Implementing IBM3/4 correctly requires:

1. **IBM3 E-step with fertility**: The naive enumeration of all alignment sequences is O(l^m) where l=target length, m=source length. The "peeling" algorithm reduces this to O(l × m × max_fertility) but is notoriously tricky to implement correctly. GIZA++ uses a complex "center" alignment enumeration with thresholds.

2. **IBM4 relative distortion**: Requires maintaining jump tables for each source position relative to the previous aligned source position, with sentence-length-dependent normalization.

3. **Deficiency handling**: IBM3+ models place probability mass on impossible alignments (where a target word generates zero source words). Handling this correctly requires the "deficient model" variant.

4. **Numerical stability**: Fertility probabilities for rare words approach zero, causing EM to diverge. Requires smoothing heuristics.

The academic NLP community has largely abandoned implementing these models from scratch — fast_align, MGIZA++, and GIZA++ are the standard implementations. The effort/reward for a from-scratch IBM3 is very poor.

---

## Go/No-Go Decision Points

| Decision Point | After | Criterion | Go condition |
|---------------|-------|-----------|--------------|
| DP1: Scale to 213K | Item 1 | BLEU improvement ≥ +3 over 50K baseline | Proceed to MERT |
| DP2: MERT tuning | Item 2 | BLEU improvement ≥ +1 over untuned weights | Proceed to fast_align |
| DP3: fast_align | Item 3 | Alignment quality > IBM2 (manual inspection) | Replace IBM2 in pipeline |
| DP4: NiuTrans fallback | Items 1-3 | BLEU < 15 on test set | Evaluate NiuTrans |
| DP5: Moses fallback | Items 1-4 | BLEU < 12 AND NiuTrans fails | Attempt Docker/Moses |

---

## Appendix: Current Codebase Inventory

### Functional Components (Ready)
- `smt/ibm_align.py` — IBM1 + IBM2 with parallel E-step, sparse init ✓
- `smt/phrase_table.py` — Phrase extraction + scoring (4 features + penalty) ✓
- `smt/language_model.py` — Kneser-Ney 3-gram LM (sequential, no pruning) ✓
- `smt/decoder.py` — Beam search decoder with future cost estimation ✓
- `smt/pipeline.py` — End-to-end training orchestration ✓
- `smt/config.py` — YAML config with defaults ✓
- `smt/data_prep.py` — Tokenization (jieba for zh), cleaning ✓
- `smt/evaluation.py` — BLEU computation ✓
- `smt/moses_orch.py` — Moses Docker orchestration (untested) ⚠
- `smt/vocab_manager.py` — Vocabulary analysis + coverage reports ✓

### Scripts
- `scripts/train_symmetrized.py` — Symmetrized IBM2 pipeline (grow-diag-final-and) ✓
- `scripts/retrain_v3.py` — Incremental training with config overrides ✓
- `scripts/mert_tune.py` — Naive grid search MERT (needs upgrade) ⚠
- `scripts/grid_search_decoder.py` — Decoder parameter search ✓
- `scripts/download_wmt_data.py` — WMT data downloader ✓
- `scripts/clean_wmt.py` — WMT data cleaner ✓

### Trained Models (local)
| Model | Sentences | Phrases | LM | Alignment |
|-------|-----------|---------|-----|-----------|
| `smt_zh2en` | 10K real | 9,753 | 5-gram | IBM2 single-dir |
| `smt_zh2en_sym` | ~10K real | 8,705 | 3-gram | IBM2 sym (gdfa) |
| `smt_zh2en_v3` | 50K real | 29,856 | 3-gram | IBM2 warm-start |
| `smt_en2zh_sym` | ~10K real | 8,729 | 3-gram | IBM2 sym (gdfa) |
| `smt_en2zh_v3` | 50K real | 30,031 | 3-gram | IBM2 warm-start |

### Known Bugs (All Fixed)
- `_LEX_EPSILON` 1e-10 → 1e-7 (phrase_table.py:218) ✓
- Parallel n-gram counting → forced sequential (language_model.py:203) ✓
- JSON key format: `str(tuple)` → `json.dumps(list(k))` (language_model.py:440+492) ✓
- LM order: 5 → 3 (config) ✓
- Decoder recombination: coverage_key → (coverage, last_N_target_words) (decoder.py:335) ✓

---

*Generated: 2026-06-06 | Author: Prometheus (Strategic Planner/Architect)*