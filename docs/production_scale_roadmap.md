# Teddy → Production-Scale Translation System: Phased Roadmap

> **Author:** Prometheus (Strategic Planner / Architect)  
> **Date:** 2026-06-07  
> **Baseline:** Pure Python phrase-based SMT, WMT 50K zh↔en, fast_align+gdfa alignment, 3-gram Kneser-Ney LM, beam search decoder, BLEU ≈ 8  
> **Target:** Production-grade translation quality comparable to Google/Microsoft Translate (BLEU 40+ on WMT news)

---

## Executive Summary

The Teddy SMT project is a fully functional research-scale SMT system (~10,400 lines of Python). It implements the complete phrase-based SMT pipeline correctly but at tiny scale. Production systems (Google Translate ~2015 pre-NMT era, modern Microsoft Translator) achieve BLEU 35–50+ through: (1) **data scale** (100M+ sentence pairs vs 50K), (2) **model sophistication** (IBM4/GIZA++, 5-gram KenLM, lexicalized reordering, hierarchical phrases), (3) **engineering** (distributed training, optimized inference), and (4) ultimately **neural architectures** (Transformer, BPE, back-translation, domain adaptation).

This roadmap maps the journey in three phases of ascending ambition and resource requirements.

### Key Numbers at a Glance

| Metric | Current (Teddy) | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|:-------|:---------------:|:--------------:|:--------------:|:--------------:|
| BLEU (ZH→EN, WMT news) | ~8 | 25–30 | 32–38 | 42–50 |
| Training data | 50K pairs | 5–10M pairs | 10–50M pairs | 100M+ pairs |
| GPU hours (training) | 0 (CPU only) | 0–10 | 50–200 | 500–5,000 |
| Inference latency | 0.6s/sentence | 0.1s/sentence | 0.5s/sentence | 50ms/sentence |
| Disk footprint | ~200 MB | ~5–15 GB | ~20–50 GB | ~10–50 GB |
| Engineering effort | – | 2–3 person-months | 4–8 person-months | 8–24 person-months |

---

## Phase 1: Production-Grade SMT (BLEU 8 → 25–30)

**Philosophy:** Squeeze every drop of quality from the phrase-based paradigm before going neural. This phase keeps the classical SMT architecture but upgrades every component to state-of-the-art levels.

**Timeline:** 2–3 months (1–2 engineers)  
**GPU requirement:** None (all CPU-bound, 32–64 core machine recommended)  
**Data requirement:** 5–10M parallel sentence pairs (WMT + OPUS + UN Parallel Corpus)

### 1.1 Massive Data Scaling (50K → 5M+ Sentence Pairs)

**Expected BLEU gain:** +8–12 points  
**Core strategy:** The single most impactful lever. Phrase table coverage, alignment quality, and LM fluency all scale monotonically with data volume.

**Data sources (Chinese→English):**
| Source | Size | Domain | Quality | Access |
|:-------|:-----|:-------|:--------|:-------|
| WMT news-commentary | 213K | News | High | Free (WMT) |
| UN Parallel Corpus v1.0 | 15M | Government | Medium | Free (OPUS) |
| OPUS (OpenSubtitles, EUbookshop, etc.) | 10M+ | Mixed | Variable | Free |
| CWMT (China Workshop on MT) | 2–5M | News/Web | High | Registration |
| Back-translated monolingual data | 5–20M | Synthetic | Medium | Requires NMT engine |

**Implementation:**
```bash
# Step 1: Download from OPUS
opus_read -d UN -s zh -t en -m 5000000 -w un_zh.txt un_en.txt
opus_read -d OpenSubtitles -s zh -t en -m 3000000 -w sub_zh.txt sub_en.txt

# Step 2: Cleaning pipeline
#   - Language detection (remove non-zh/non-en)
#   - Length ratio filter (0.5–2.0)
#   - Deduplication (Bloom filter or MinHash)
#   - HTML/boilerplate removal
#   - Script normalization (full-width → half-width)

# Step 3: Train with fast_align on full corpus
python scripts/train_fastalign.py --direction zh2en --max-sentences 5000000
```

**Scaling bottlenecks and solutions:**
| Problem | Solution |
|:--------|:---------|
| IBM2 EM on 5M sentences = O(n) per iter | Use MGIZA++ (multi-threaded GIZA++) or parallelized fast_align |
| Phrase table > 2M entries | Disk-based phrase table with prefix-indexed lookup |
| LM training on 5M+ sentences | KenLM C++ (O(n) single pass, probing binary format) |
| Memory > 64GB for alignment | Split-shard alignment: train on N shards, merge alignment files |

**Reference:** Tiedemann (2012), "Parallel Data, Tools and Interfaces in OPUS", LREC.

### 1.2 Replace Python Alignment with MGIZA++ (IBM4)

**Expected BLEU gain:** +2–4 points over fast_align  
**Why:** fast_align uses HMM (equivalent to IBM2+HMM, not IBM4). MGIZA++ runs IBM Model 1→HMM→3→4 with proper fertility and relative distortion. IBM4 alignment quality consistently outperforms HMM-only by 2–4 BLEU points in phrase-based systems.

**Implementation:**
```bash
# Install MGIZA++ (multi-threaded fork of GIZA++)
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local && make -j$(nproc)

# Run training pipeline
# 1. mkcls (word classes for IBM3/4)
mkcls -n10 -pcorpus.zh -Vcorpus.zh.vcb.classes opt
mkcls -n10 -pcorpus.en -Vcorpus.en.vcb.classes opt

# 2. GIZA++ (src→tgt)
plain2snt corpus.zh corpus.en
snt2cooc corpus.zh_en.cooc corpus.zh.vcb corpus.en.vcb corpus.zh_en.snt
mgiza -S corpus.zh.vcb -T corpus.en.vcb -C corpus.zh_en.snt \
      -CoocurrenceFile corpus.zh_en.cooc -o fwd -model4smoothalpha 0.01

# 3. Symmetrize (grow-diag-final-and)
python3 -m smt.align_mgiza --fwd fwd.A3.final --rev rev.A3.final --out sym.align
```

**Python wrapper needed:** A new `smt/align_mgiza.py` module that:
1. Writes corpus to plain text format
2. Invokes MGIZA++ subprocess (both directions)
3. Parses A3.final output format
4. Runs symmetrization (grow-diag-final-and)
5. Returns `List[Set[Tuple[int, int]]]` — same interface as existing `align_fast.py`

**Time budget for 5M sentences:** ~4–6 hours on 32-core machine with MGIZA++ using 8 threads.

**Reference:** Och & Ney (2003), "A Systematic Comparison of Various Statistical Alignment Models", Computational Linguistics.

### 1.3 KenLM 5-gram Binary Language Model

**Expected BLEU gain:** +1–3 points (over 3-gram)  
**Also enables:** orders of magnitude faster LM loading and querying

**Current pain points:**
- JSON LM: 111s load time for 50K-sentence 5-gram, ~5min for 5M sentences
- 3-gram: weaker fluency modeling, shorter context
- Kneser-Ney in Python: ~10ms per log_prob query

**Target: KenLM 5-gram with Modified Kneser-Ney, probing binary format:**
- Load time: ~100ms (memory-mapped file)
- Query time: ~1μs per log_prob (C++ trie with quantization)
- Size: ~2–5 GB for 5M sentences (vs 5–15 GB JSON)
- Supports out-of-vocabulary handling with `<unk>` fallback

**Implementation options:**

| Option | Effort | Quality | Risk |
|:-------|:-------|:--------|:-----|
| **A: pybind11 wrapper for libkenlm** | 8–12h | Best (native KenLM) | Medium (C++ build) |
| **B: Subprocess KenLM query server** | 4–6h | Best (native KenLM) | Low (IPC overhead) |
| C: Pure Python KenLM binary reader | 15–25h | Good (reimplementation) | Medium (bugs, subtle mismatches) |

**Recommended: Option B first → Option A later.**

Option B implementation:
```python
# smt/kenlm_proxy.py
import subprocess, struct

class KenLMProxy:
    """Thin wrapper around KenLM query binary."""
    def __init__(self, model_path: str):
        # Start KenLM query process
        self.proc = subprocess.Popen(
            ["query", model_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1
        )
    
    def log_prob(self, word: str, history: Tuple[str, ...]) -> float:
        """Query via stdin/stdout protocol."""
        self.proc.stdin.write(f"{' '.join(history)} {word}\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        # Parse: "word=XXX p=0.00123 ppl=456.78"
        ...

class KenLMBinding:
    """pybind11 binding (Option A, later)."""
    def __init__(self, model_path: str):
        self.model = kenlm.Model(model_path)
    
    def log_prob(self, word, history):
        state = kenlm.State()
        # ... efficient incremental scoring
```

**Reference:** Heafield (2011), "KenLM: Faster and Smaller Language Model Queries", WMT.

### 1.4 Lexicalized Reordering Model (msd-bidirectional-fe)

**Expected BLEU gain:** +2–4 points  
**Why:** Current distance-based distortion (`distortion_weight * |jump|`) is blind to word identities. For Chinese→English, reordering patterns are lexicalized — `把`-constructions require object fronting, temporal adverbs move to sentence-initial position, etc.

**Model:** For each phrase pair, learn 3 orientation probabilities:
- **M** (monotone): next source phrase appears immediately to the right in target
- **S** (swap): next source phrase appears immediately to the left in target  
- **D** (discontinuous): next source phrase is non-adjacent in target

Computed bidirectionally (source→target and target→source) = 6 features.

**Implementation in `smt/reordering.py`:**
```python
@dataclass
class ReorderingModel:
    """msd-bidirectional-fe reordering model."""
    # Key: (src_phrase_key, tgt_phrase_key) → (p_m, p_s, p_d)
    fwd_model: Dict[Tuple[str, str], Tuple[float, float, float]]
    bwd_model: Dict[Tuple[str, str], Tuple[float, float, float]]
    
    @classmethod
    def train(cls, src_sents, tgt_sents, alignments, phrase_table) -> "ReorderingModel":
        """Collect orientation statistics during phrase extraction."""
        # For each phrase pair occurrence:
        #   Find the next source phrase
        #   Check its position in target relative to current phrase's target span
        #   Classify as M, S, or D
        ...
    
    def score(self, src_phrase, tgt_phrase, orientation: str) -> float:
        """Return log-probability for a given orientation."""
        ...

# Decoder integration: add 6 reordering features to scoring function
# new_score += reorder_weight * log(p_orientation | phrase_pair)
```

**Integration into decoder:** The reordering model replaces the distance-based distortion cost. Each hypothesis expansion checks the relative position of the next source span in the target and adds the lexicalized probability.

**Reference:** Koehn et al. (2005), "Edinburgh System Description for the 2005 IWSLT Speech Translation Evaluation".

### 1.5 Hierarchical Phrase-Based Model (Chiang 2005/2007)

**Expected BLEU gain:** +2–5 points  
**Why:** Flat phrase pairs cannot model long-distance reordering or syntactic transformations. Hierarchical phrases (synchronous context-free grammar rules) capture structural patterns:

```
X → ⟨ X₁ 的 X₂ , the X₂ that X₁ ⟩   # relative clause reordering
X → ⟨ 把 X₁ X₂ , X₂ X₁ ⟩             # ba-construction object fronting
```

**Implementation as a decoder extension (`smt/hiero_decoder.py`):**
```python
class HieroRule:
    """Synchronous CFG rule: X → ⟨ α, γ, ~ ⟩
    α: source-side (with nonterminals X₁, X₂, ...)
    γ: target-side (with same nonterminals, possibly reordered)
    ~: 1-to-1 alignment between nonterminals in α and γ
    """
    src: List[Union[str, int]]  # e.g., ["X", 1, "的", "X", 2]
    tgt: List[Union[str, int]]  # e.g., ["the", "X", 2, "that", "X", 1]
    features: Dict[str, float]

class HieroDecoder:
    """CKY-style decoder with cube pruning for hierarchical SMT."""
    def decode(self, source_tokens):
        # Parse chart: chart[i][j][X] = list of (target_string, score)
        chart = [[defaultdict(list) for _ in range(n+1)] for _ in range(n)]
        
        # Bottom-up: fill spans of increasing length
        for span_len in range(1, n+1):
            for i in range(n - span_len + 1):
                j = i + span_len
                # Apply terminal rules: X → source_span / target_phrase
                # Apply nonterminal rules: X → α with nonterminals
                # Use cube pruning for efficient enumeration
        
        # Extract best derivation from chart[0][n][S]
```

**This is the largest Phase 1 engineering task.** A full hierarchical decoder is ~2,000–3,000 lines of code. Alternatives:

| Approach | Effort | BLEU Gain | Risk |
|:---------|:-------|:----------|:-----|
| Full Hiero decoder from scratch | 3–4 weeks | +3-5 | High |
| Joshua decoder integration | 1–2 weeks | +3-5 | Medium |
| cdec integration | 1–2 weeks | +2-4 | Low-Medium |
| Skip Hiero, go to neural (Phase 3) | 0 | 0 | Lowest |

**Recommendation:** If the goal is production scale, skip the from-scratch Hiero decoder and integrate Joshua or cdec as C++ subprocess. If the goal is learning/pedagogical, implement a simplified Hiero decoder (only glue rules + hierarchical rules, no full syntactic labels).

**Reference:** Chiang (2007), "Hierarchical Phrase-Based Translation", Computational Linguistics.

### 1.6 Syntax-Based Preordering (Chinese→English)

**Expected BLEU gain:** +2–4 points (Chinese-specific)  
**Why:** Chinese and English have fundamentally different constituent orders. Chinese is largely head-final (modifiers before heads) while English is mixed. Preordering the Chinese source to approximate English word order before translation dramatically reduces the decoder's reordering burden.

```
Source: 我 昨天 在 北京 买 了 一本 书
        I  yesterday in Beijing buy ASP a CL book
Preordered: 我 买 了 一本 书 昨天 在 北京
            I buy ASP a CL book yesterday in Beijing
```

**Implementation using Stanford Parser / HanLP:**
```python
# smt/preorder.py
from hanlp_restful import HanLPClient

class ChinesePreorderer:
    """Rule-based Chinese→English constituent reordering."""
    
    def __init__(self):
        self.parser = HanLPClient(...)
    
    def preorder(self, sentence: str) -> str:
        parse = self.parser.parse(sentence)  # dependency or constituency parse
        # Apply reordering rules:
        # 1. Temporal/locative adverbials → sentence-final
        # 2. Prepositional phrases in NP → post-nominal
        # 3. Relative clauses → post-nominal
        # 4. Serial verb constructions → reorder to SVO
        return self._apply_rules(parse)
```

**Rules for Chinese→English preordering (based on Collins et al. 2005):**
| Pattern | Before | After |
|:--------|:-------|:------|
| Temporal NP | 昨天 我 去了 ... | 我 去了 ... 昨天 |
| Locative PP | 在 北京 举行 | 举行 在 北京 |
| Relative clause | 我 买的 书 | 书 我 买的 |
| `把` construction | 把 X V | V X |

**Reference:** Collins, Koehn & Kučerová (2005), "Clause Restructuring for Statistical Machine Translation", ACL.  
Xia & McCord (2004), "Improving a Statistical MT System with Automatically Learned Rewrite Patterns", COLING.

### 1.7 Production-Grade Decoder Optimization

**Expected BLEU gain:** 0 (quality-neutral) but **10–100× speedup**, enabling larger beam sizes and n-best lists.

| Optimization | Speedup | Effort | Priority |
|:-------------|:--------|:-------|:---------|
| Cube pruning (replaces naive beam) | 3–5× | 3–5 days | High |
| Phrase table prefix index (Trie) | 5–20× on lookup | 2–3 days | High |
| Hypothesis recombination with hash | 2–3× | 1 day | Medium |
| Multi-threaded sentence-level decoding | N× (N=cores) | 2–3 days | Medium |
| Future cost precomputation | 1.5–2× | 1 day | Low |

**Detailed implementation for cube pruning (highest-impact):**

Current decoder expands ALL translation options for ALL hypotheses at each coverage level → O(H × O × S) where H=hypotheses, O=options, S=stack size. Cube pruning limits expansion to the K-best options from a priority queue:

```python
def cube_pruning_expand(hypothesis, options, K=1000):
    """Only expand the K most promising translation options.
    
    Creates a priority queue of (hypothesis, option) pairs sorted by
    upper-bound score (translation score + LM score of best completion).
    Expands K items, ensuring monotonicity.
    """
    import heapq
    heap = []
    for opt in options:
        ub_score = hypothesis.score + opt.features["log_phi_f_e"] + best_lm_completion(opt)
        heapq.heappush(heap, (-ub_score, hypothesis, opt))
    
    expanded = []
    for _ in range(min(K, len(heap))):
        score, hyp, opt = heapq.heappop(heap)
        expanded.append(apply_option(hyp, opt))
    return expanded
```

### 1.8 MERT/BLEU Tuning Pipeline

**Expected BLEU gain:** +2–5 points (over untuned weights)  
**Status:** Och MERT already implemented in `scripts/mert_tune.py` with n-best list extraction and error surface optimization. Needs:

1. **Proper dev/test/train split** — Reserve 2,000 sentences each
2. **k-fold cross-validation** — Avoids overfitting to single dev set
3. **Alternative tuners for comparison:**
   - **MIRA** (Margin Infused Relaxed Algorithm): More robust for small dev sets
   - **PRO** (Pairwise Ranking Optimization): Better for large feature sets
   - **Batch MIRA**: Efficient for 10+ feature weights
4. **Multiple reference handling** — sacrebleu supports multi-reference BLEU

**Reference:** Och (2003), "Minimum Error Rate Training in Statistical Machine Translation", ACL.

### Phase 1: Architecture Summary

```
                         ┌──────────────────────────────┐
                         │   Preordering (ZH-specific)   │  ← HanLP / Stanford
                         └──────────┬───────────────────┘
                                    │
┌───────────────────────────────────▼───────────────────────────────────┐
│                       Phrase-Based SMT Pipeline                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ MGIZA++  │→│  Phrase  │→│  Lexical  │→│  KenLM   │→│  Cube   │ │
│  │  IBM4    │  │  Table   │  │ Reorder.  │  │  5-gram  │  │ Pruning │ │
│  │ (5M sen) │  │ (>2M)    │  │  (msd)    │  │  (2-5GB) │  │ Decoder │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────┬────┘ │
│                                                                │       │
│                                          ┌─────────────────────▼─────┐ │
│                                          │   MERT / MIRA / PRO      │ │
│                                          │   Weight Optimization    │ │
│                                          └───────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

**Phase 1 BLEU target:** 25–30 (ZH→EN, WMT news test set)  
**Phase 1 resource requirement:**
- Hardware: 32–64 core CPU, 128–256 GB RAM, 1–2 TB SSD
- GPU: None required
- Data: 5–10M parallel sentence pairs
- Engineering: 2–3 person-months
- Cost: ~$5,000–15,000 (cloud compute, mainly data storage + CPU hours)

---

## Phase 2: Hybrid Neural-Symbolic Systems (BLEU 25–30 → 32–38)

**Philosophy:** Inject neural components into the SMT pipeline where they provide the most value — language modeling (fluency) and re-ranking (global coherence). Keep the phrase-based backbone for interpretability and robustness.

**Timeline:** 3–6 months (2–3 engineers)  
**GPU requirement:** 1–4 GPUs (24–32 GB VRAM each, e.g., A10/V100/A100)  
**Data requirement:** 10–50M parallel pairs + 50M+ monolingual target sentences

### 2.1 Neural Language Model (RNN/LSTM) for Decoder

**Expected BLEU gain:** +2–4 points (replacing n-gram LM)  
**Why:** n-gram LMs have a fixed context window (5 words) and no representation learning. Neural LMs learn continuous word representations and can condition on arbitrarily long contexts, dramatically improving fluency scoring.

**Architecture: LSTM-LM with 2 layers, 1024 hidden units:**
```
Embedding(50K vocab, 512d) → 2× LSTM(1024) → Softmax(50K)
```

**Integration into decoder:** Replace `self._score_lm()` with neural LM scoring. Key challenge: latency. A neural LM forward pass is ~10ms on GPU vs ~1μs for KenLM.

**Implementation with caching:**
```python
class NeuralLMDecoder:
    def __init__(self, lm_path: str):
        self.model = torch.jit.load(lm_path)  # TorchScript for fast inference
        self.cache = LRUCache(maxsize=10000)   # Cache (context, word) → log_prob
    
    def score_lm(self, target_tokens, new_words):
        # Batch all new words in the hypothesis expansion
        contexts = [target_tokens[-(self.order-1):] for _ in new_words]
        # Single batched forward pass
        log_probs = self.model.batch_score(contexts, new_words)
        return sum(log_probs)
```

**Training:**
- Data: 50M+ English sentences (from parallel data target side + monolingual)
- Framework: PyTorch + PyTorch Lightning
- Training time: ~24–48h on 1× V100/A100
- Quantization: INT8 for decoder inference (50% latency reduction)

**Alternative: Transformer-XL or small GPT-2 for stronger context modeling** but at higher latency cost.

**Reference:** Bengio et al. (2003), "A Neural Probabilistic Language Model", JMLR.  
Mikolov et al. (2010), "Recurrent Neural Network Based Language Model", Interspeech.

### 2.2 Neural Re-Ranker (n-best List Rescoring)

**Expected BLEU gain:** +2–4 points  
**Why:** The SMT decoder produces an n-best list of translation candidates. A neural model can re-rank these globally using richer features (encoder-decoder attention, bidirectional context) that the SMT decoder cannot compute incrementally.

**Two architectures:**

| Model | Pros | Cons |
|:------|:-----|:-----|
| **Bidirectional RNN scorer** | Fast, small model (50M params) | Limited context modeling |
| **Encoder-decoder (small Transformer)** | Full cross-attention, best quality | Slower (50ms per candidate) |

**Recommended: Lightweight encoder-decoder (6-layer Transformer, 256d):**
```
Source: 经济 增长 超出 预期
        ↓ Encoder (3-layer Transformer)
        Context vectors
        ↓ Cross-Attention Decoder (3-layer Transformer)
        Candidate: economic growth exceeded expectations  → score: -1.23
        Candidate: growth economic exceeded expectations  → score: -2.45  (re-ranked lower)
```

**Training data:** Parallel corpus (same as SMT training), trained as standard NMT but used only for scoring, not generation.

**Integration:**
```python
class NeuralReranker:
    def rerank(self, source_tokens: List[str], nbest: List[Tuple[List[str], float]]) -> List[str]:
        """Re-rank n-best list using encoder-decoder model."""
        src_tensor = self.tokenize_source(source_tokens)
        encoder_output = self.encoder(src_tensor)
        
        scored = []
        for hyp_tokens, smt_score in nbest:
            tgt_tensor = self.tokenize_target(hyp_tokens)
            nmt_score = self.decoder.score(tgt_tensor, encoder_output)
            combined = smt_score * self.alpha + nmt_score * (1 - self.alpha)
            scored.append((hyp_tokens, combined))
        
        scored.sort(key=lambda x: x[1])
        return scored[0][0]  # Best re-ranked hypothesis
```

**Reference:** Shen et al. (2004), "Discriminative Reranking for Machine Translation", NAACL.  
Stahlberg et al. (2017), "Neural Machine Translation by Jointly Learning to Align and Translate...", showing encoder-decoder as re-ranker.

### 2.3 Operation Sequence Model (OSM)

**Expected BLEU gain:** +1–2 points  
**Why:** The phrase-based decoder applies translation options independently. OSM models the sequence of translation operations (which phrases to use, in what order, with what reordering) as a structured prediction problem, capturing long-range dependencies between phrase choices.

**Implementation as a feature in the decoder:**
```python
class OSModel:
    """5-gram operation sequence model."""
    # Operations: (phrase_start, phrase_end, is_swap)
    # OSM scores the sequence of operations, not just individual phrase choices
    
    def train(self, src_sents, tgt_sents, alignments):
        # Extract operation sequences from training data
        # Build 5-gram model over operation sequences
        ...
    
    def score_sequence(self, op_history: List[Op], next_op: Op) -> float:
        return self.lm.log_prob(next_op, op_history[-4:])
```

**Reference:** Durrani et al. (2011), "Can Markov Models Over Minimal Translation Units Help Phrase-Based SMT?", ACL.  
Heafield et al. (2012), "Operation Sequence Model", AMTA.

### 2.4 System Combination (Multi-Engine)

**Expected BLEU gain:** +1–3 points  
**Why:** Different SMT configurations have complementary strengths. Combining outputs from multiple systems (different aligners, LM orders, reordering models) via confusion network decoding yields better results than any single system.

```python
def combine_systems(outputs: List[List[Tuple[str, float]]], method="confusion_network"):
    """Combine translations from multiple SMT systems.
    
    Systems:
      1. MGIZA++ IBM4 + KenLM 5-gram + msd reordering
      2. fast_align HMM + n-gram LM + lexicalized reordering  
      3. Hierarchical phrase-based (cdec)
    
    Output: single best translation via confusion network decoding
    """
    # 1. Align all outputs to build confusion network
    # 2. Decode confusion network with LM + word posterior features
    # 3. Return 1-best path
    ...
```

**Reference:** Rosti et al. (2007), "Combining Outputs from Multiple Machine Translation Systems", NAACL.

### 2.5 Unknown Word Handling with Neural Models

**Expected BLEU gain:** +1–2 points  
**Why:** OOV words (currently handled by copy/drop/unk strategy) are a major failure mode. Neural models can: (a) transliterate names, (b) copy numbers/dates with format conversion, (c) look up from bilingual dictionaries.

```python
class NeuralOOVHandler:
    def __init__(self):
        self.transliterator = Seq2SeqModel.load("transliterator_zh2en.pt")
        self.dictionary = load_bilingual_dict("cedict.json")
    
    def handle_oov(self, src_word: str) -> str:
        if is_chinese_name(src_word):
            return self.transliterator.translate(src_word)  # 习近平 → Xi Jinping
        if is_number(src_word):
            return convert_number_format(src_word)          # 一万 → 10,000
        if src_word in self.dictionary:
            return self.dictionary[src_word]                # 人工智能 → artificial intelligence
        return src_word  # copy as last resort
```

### Phase 2: Architecture Summary

```
┌───────────────────────────────────────────────────────────────────────┐
│                        Phase 2: Hybrid System                          │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    SMT Backbone (Phase 1)                        │  │
│  │  MGIZA++ → Phrase Table → Lex Reordering → Cube Pruning Decoder │  │
│  └──────────┬──────────────────────┬───────────────────────────────┘  │
│             │                      │                                   │
│             ▼                      ▼                                   │
│  ┌──────────────────┐  ┌──────────────────────┐                       │
│  │  Neural LM       │  │  Neural Re-Ranker    │                       │
│  │  (LSTM/Transformer│  │  (Enc-Dec, 6-layer) │                       │
│  │   replaces n-gram)│  │  rescore n-best list │                       │
│  └──────────────────┘  └──────────────────────┘                       │
│             │                      │                                   │
│             └──────────┬───────────┘                                   │
│                        ▼                                               │
│  ┌─────────────────────────────────────────┐                          │
│  │       System Combination                │                          │
│  │  (Confusion Network Decoding)           │                           │
│  │  + OSM + Neural OOV Handler             │                           │
│  └─────────────────────────────────────────┘                          │
└───────────────────────────────────────────────────────────────────────┘
```

**Phase 2 BLEU target:** 32–38 (ZH→EN, WMT news)  
**Phase 2 resource requirement:**
- Hardware: 1–4 GPUs (V100/A100/A10, 24+ GB VRAM each), 64+ GB RAM, 500 GB SSD
- Data: 10–50M parallel pairs + 50M+ monolingual target (English)
- Engineering: 4–8 person-months
- Cost: ~$20,000–50,000 (GPU cloud + data storage)

---

## Phase 3: Full Neural Machine Translation (BLEU 32–38 → 42–50)

**Philosophy:** Replace the SMT backbone entirely with a modern Transformer-based NMT system. This is where the industry went (Google GNMT 2016, Microsoft 2018, all major systems by 2020). The SMT system from Phases 1–2 serves as a strong baseline, training data pipeline, and synthetic data generator.

**Timeline:** 6–12 months (3–5 engineers)  
**GPU requirement:** 4–8 GPUs (A100 40/80 GB) for training, 1–2 GPUs for inference  
**Data requirement:** 20–100M+ parallel pairs + 200M+ monolingual sentences

### 3.1 Transformer Base Architecture

**Reference implementation:** fairseq or OpenNMT-py  
**Architecture:** Transformer Big (Vaswani et al. 2017), tuned for ZH→EN

| Component | Specification |
|:----------|:--------------|
| Encoder layers | 6 |
| Decoder layers | 6 |
| Model dimension | 1024 |
| FFN dimension | 4096 |
| Attention heads | 16 |
| Total parameters | ~210M |
| Training data | 20–50M sentence pairs |
| Vocabulary | 32K BPE (joint) |
| Training time | ~7 days on 8× A100 |

**Why not build from scratch:** The Teddy project already demonstrates deep understanding of MT internals by building SMT from scratch. For NMT, using battle-tested frameworks (fairseq, OpenNMT, HuggingFace Transformers) is pragmatic — the value is in training recipes, data engineering, and domain adaptation, not reimplementing Transformer attention.

**Recommended: OpenNMT-py (simpler than fairseq, good for production deployment):**
```bash
# Training
onmt_train -config zh2en_transformer_big.yaml

# Config highlights
# zh2en_transformer_big.yaml:
save_model: models/zh2en_big
data: data/zh2en_bpe
encoder_type: transformer
decoder_type: transformer
layers: 6
heads: 16
hidden_size: 1024
word_vec_size: 1024
transformer_ff: 4096
batch_size: 4096
batch_type: tokens
max_generator_batches: 2
accum_count: 4
optim: adam
learning_rate: 2.0
warmup_steps: 8000
decay_method: noam
train_steps: 200000
```

**Reference:** Vaswani et al. (2017), "Attention Is All You Need", NeurIPS.  
Ott et al. (2019), "fairseq: A Fast, Extensible Toolkit for Sequence Modeling", NAACL.

### 3.2 BPE/Subword Tokenization

**Expected BLEU gain:** +2–5 points (especially for rare/morphological words)  
**Why:** Chinese word segmentation (jieba) and English whitespace tokenization create open vocabularies with high OOV rates. Byte-Pair Encoding (BPE) or SentencePiece creates a joint subword vocabulary that handles any input without OOV.

**Implementation with SentencePiece (joint BPE):**
```bash
# Train joint BPE model on combined ZH+EN corpus
spm_train --input=corpus.zh,corpus.en \
          --model_prefix=zh2en_bpe \
          --vocab_size=32000 \
          --character_coverage=0.9995 \
          --model_type=bpe \
          --split_digits=true

# Encode
spm_encode --model=zh2en_bpe.model < corpus.zh > corpus.bpe.zh
```

**Why BPE for Chinese?** While Chinese characters are already "subword-like", BPE handles:
- English morphology (un-believ-able → subword regularization)
- Chinese multi-character words that jieba segments inconsistently
- Transliterated names (克-里-姆-林-宫 → Kremlin)
- Numbers and mixed-script text

**Reference:** Sennrich et al. (2016), "Neural Machine Translation of Rare Words with Subword Units", ACL.

### 3.3 Back-Translation for Data Augmentation

**Expected BLEU gain:** +3–8 points (largest single NMT quality lever after architecture)  
**Why:** Parallel data is scarce and expensive. Monolingual data is abundant and free. Back-translation uses a target→source NMT model to translate monolingual target text back into synthetic source text, creating unlimited "parallel" training data.

**Implementation pipeline:**
```python
# Step 1: Train EN→ZH NMT model (target→source) on existing parallel data
# Step 2: Use EN→ZH model to back-translate monolingual English corpus
# Step 3: Combine original parallel + synthetic parallel for ZH→EN training

def backtranslate_pipeline():
    # 1. Train reverse model
    train_nmt(en2zh_config, parallel_corpus_zh_en)
    
    # 2. Back-translate English monolingual data
    for batch in monolingual_en:
        synthetic_zh = en2zh_model.translate(batch)
        synthetic_corpus.append((synthetic_zh, batch))
    
    # 3. Train forward model with augmented data
    augmented = parallel_corpus_zh_en + synthetic_corpus
    train_nmt(zh2en_config, augmented)  # Tag synthetic data for weighting
```

**Key techniques for quality:**
- **Tagged back-translation:** Mark synthetic source with a special token so the model learns to trust it less
- **Noised beam search:** Use sampling instead of beam search when generating synthetic data (more diverse)
- **Iterative back-translation:** Retrain reverse model, generate better synthetic data, retrain forward model (2–3 rounds)
- **Filtering:** Remove low-quality synthetic pairs using language model perplexity threshold

**Data sources for monolingual English:** CommonCrawl, Wikipedia dumps, news corpora, BooksCorpus — 200M+ sentences accessible.

**Reference:** Sennrich et al. (2016), "Improving Neural Machine Translation Models with Monolingual Data", ACL.  
Edunov et al. (2018), "Understanding Back-Translation at Scale", EMNLP.

### 3.4 Domain Adaptation

**Expected BLEU gain:** +3–10 points (domain-dependent)  
**Why:** A general-domain model (WMT news) performs poorly on specialized domains (legal, medical, technical). Enterprise translation needs domain-specific quality.

**Multi-stage domain adaptation strategy:**

```
Stage 1: General domain training (20M pairs, WMT + OPUS)
         ↓
Stage 2: Domain-aware fine-tuning (mix general + domain data)
         ↓
Stage 3: Domain-specific fine-tuning (domain data only, small LR)
```

**Implementation approaches:**

| Approach | Pros | Cons | Best For |
|:---------|:-----|:-----|:---------|
| **Fine-tuning** | Simple, effective | Catastrophic forgetting of general domain | Single target domain |
| **Multi-domain training** (domain tag) | One model serves all | Diluted quality per domain | Multiple target domains |
| **Data selection** (TF-IDF, LM perplexity) | Training data quality ↑ | Requires domain corpus selection | Pre-training data curation |
| **Adapter layers** | Parameter-efficient, no forgetting | Slightly lower peak quality | Many small domains |

**Recommended: Domain tag approach for production:**
```
Input: <legal> 合同双方同意... → The parties agree...
       <medical> 患者出现... → The patient presented with...
       <general> 今天天气... → The weather today...
```

The domain tag is prepended to the source during both training and inference, letting the model learn domain-specific translation patterns without separate models.

**Reference:** Chu et al. (2017), "An Empirical Comparison of Domain Adaptation Methods for Neural Machine Translation", ACL.  
Bapna & Firat (2019), "Simple, Scalable Adaptation for Neural Machine Translation", EMNLP.

### 3.5 Knowledge Distillation (Teacher→Student)

**Expected BLEU gain:** Not about BLEU — about **3–10× inference speedup** with <2 BLEU loss  
**Why:** Transformer Big (210M params) is too slow for production (200ms+ per sentence). Distillation trains a smaller "student" model (6-layer, 512d, ~60M params) to mimic the "teacher" (12-layer, 1024d, ~210M params), achieving near-teacher quality at a fraction of the cost.

**Training:**
```python
# Student training loss:
# L = α × CrossEntropy(student_logits, ground_truth) 
#   + (1-α) × KL(student_logits || teacher_logits)

# Where teacher_logits are probability distributions from the big model
# This transfers "dark knowledge" — which words are plausible alternatives
```

**Typical results (ZH→EN, WMT):**
| Model | Params | BLEU | Latency (ms/sent) | Throughput (sent/s/GPU) |
|:------|:-------|:-----|:-----------------|:-----------------------|
| Teacher (Big) | 210M | 42.3 | 250 | 4 |
| Student distilled | 60M | 40.8 | 45 | 22 |
| Student from scratch | 60M | 38.1 | 45 | 22 |

**Reference:** Hinton et al. (2015), "Distilling the Knowledge in a Neural Network".  
Kim & Rush (2016), "Sequence-Level Knowledge Distillation", EMNLP.

### 3.6 Production Inference Optimization

**Essential for deployment.** A model that takes 250ms to translate a sentence is unusable for interactive applications (target: <100ms).

| Technique | Speedup | Quality Loss | Implementation |
|:----------|:--------|:-------------|:---------------|
| INT8 quantization (dynamic) | 2–4× | <0.3 BLEU | PyTorch `torch.quantization` |
| Cache KV states (incremental decoding) | 1.5–2× | 0 | Native in fairseq/OpenNMT |
| Beam search optimization (length penalty, coverage) | 1.3–1.8× | 0 | Tune beam_size=4, diverse beam |
| ONNX Runtime / TensorRT | 1.5–3× | 0 | Export → optimized graph |
| Speculative decoding | 2–3× | 0 | Draft model + verification |
| Batched inference (dynamic batching) | 3–10× throughput | 0 | Triton Inference Server |

**Production deployment stack:**
```
Client (Web/Mobile API)
    ↓
Load Balancer (Nginx/Envoy)
    ↓
Triton Inference Server (dynamic batching, multi-model)
    ↓
ONNX-optimized Student Transformer (INT8, TensorRT)
    ↓
Post-processing (detokenization, BPE merge, format restoration)
    ↓
Response
```

### 3.7 Multilingual NMT (Optional, High-Impact)

**Expected impact:** Support 10+ language pairs with one model (vs. 10 separate models)  
**Why:** A single multilingual model shares parameters across language pairs, benefiting low-resource languages through transfer learning from high-resource pairs.

**Architecture:** Add language token at the start of the source:
```
<2zh> <2en> 你好世界 → Hello world
<2ja> <2en> こんにちは世界 → Hello world
```

**Reference:** Johnson et al. (2017), "Google's Multilingual Neural Machine Translation System: Enabling Zero-Shot Translation", TACL.  
Aharoni et al. (2019), "Massively Multilingual Neural Machine Translation", NAACL.

### 3.8 Document-Level NMT (Cutting Edge)

**Expected BLEU gain:** +1–3 points, +significant improvement in discourse coherence  
**Why:** Sentence-level NMT translates each sentence independently, losing cross-sentence context (pronoun resolution, discourse coherence, lexical consistency). Document-level models condition on previous sentences.

**Architecture:** Extended Transformer with cross-sentence attention or context embedding fusion:
```
Previous sentences: [S₁, S₂, S₃]
    ↓ Context Encoder (additional Transformer layers)
    Context vector
    ↓ Concatenated with source encoding of S₄
    Decoder → Translation of S₄
```

**Reference:** Maruf et al. (2021), "A Survey on Document-level Neural Machine Translation: Methods and Evaluation", ACM Computing Surveys.  
Zhang et al. (2018), "Improving the Transformer Translation Model with Document-Level Context", EMNLP.

### Phase 3: Architecture Summary

```
┌───────────────────────────────────────────────────────────────────────┐
│                      Phase 3: Full NMT System                          │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                  Data Pipeline                                    │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │  │
│  │  │ Parallel 100M │  │ Monolingual  │  │ Back-Translation     │   │  │
│  │  │ (WMT+OPUS+UN)│  │ 200M+ EN/ZH  │  │ (iterative, noised)  │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                   │
│                                    ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              SentencePiece BPE (32K joint vocab)                  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                    │                                   │
│                                    ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │       Transformer Big (6L-6L, 1024d, 16 heads, 210M params)     │  │
│  │       + Domain Tag Training (legal/medical/tech/general)        │  │
│  └────────────────────────────┬────────────────────────────────────┘  │
│                               │                                        │
│                               ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │         Knowledge Distillation (Teacher → Student, 60M params)   │  │
│  └────────────────────────────┬────────────────────────────────────┘  │
│                               │                                        │
│                               ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │          Production Inference Stack                               │  │
│  │  INT8 Quantization → TensorRT → Dynamic Batching → Triton        │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                               │                                        │
│                               ▼                                        │
│                    REST API / gRPC Endpoint                            │
│                    Latency: <50ms/sentence                             │
└───────────────────────────────────────────────────────────────────────┘
```

**Phase 3 BLEU target:** 42–50 (ZH→EN, WMT news)  
**Phase 3 resource requirement:**
- Hardware: 4–8 GPUs (A100-80GB) for training, 1–2 GPUs (A10/T4) for inference
- Data: 20–100M parallel pairs + 200M+ monolingual sentences each language
- Engineering: 8–24 person-months (4× more than Phase 1+2 combined)
- Cost: ~$50,000–200,000 (GPU cloud, data acquisition, engineering)

---

## Cross-Cutting Infrastructure (All Phases)

### Data Pipeline

```
Raw monolingual/parallel text
    ↓
Language detection + filtering (fastText)
    ↓
Deduplication (MinHash LSH / SimHash)
    ↓
Cleaning (HTML removal, script normalization, length filtering)
    ↓
Tokenization / BPE encoding
    ↓
Sharded TFRecord / Arrow format for efficient training I/O
```

### Evaluation Framework

Beyond BLEU, production systems need:

| Metric | What It Measures | Tool |
|:-------|:-----------------|:-----|
| BLEU | N-gram overlap precision | sacrebleu |
| chrF | Character n-gram F-score (better for morphology) | sacrebleu |
| COMET | Neural reference-based metric (highest correlation with human judgment) | Unbabel COMET |
| BLEURT | Learned BLEU replacement | Google BLEURT |
| TER | Translation Edit Rate (post-editing effort) | sacrebleu |
| Latency | P50/P95/P99 response time | Custom monitoring |
| Throughput | Sentences/second at P95 latency | Custom monitoring |

**Human evaluation:** For production deployment, conduct A/B testing with bilingual evaluators on 500–1000 sentences, rating adequacy (1–5) and fluency (1–5).

### CI/CD for Model Updates

```
New training data → Automated retraining → COMET/BLEU gate → A/B test → Gradual roll-out
```

---

## Key Papers by Phase

### Phase 1 — Classical SMT
| Paper | Year | Venue | Relevance |
|:------|:-----|:------|:----------|
| Koehn, Och, Marcu — Statistical Phrase-Based Translation | 2003 | NAACL | Foundation of phrase-based SMT |
| Och — Minimum Error Rate Training in SMT | 2003 | ACL | MERT algorithm |
| Koehn et al. — Moses: Open Source Toolkit for SMT | 2007 | ACL | Reference SMT system |
| Chiang — Hierarchical Phrase-Based Translation | 2007 | CL | Hiero model |
| Heafield — KenLM: Faster and Smaller LM Queries | 2011 | WMT | Production LM |
| Dyer et al. — A Simple, Fast, and Effective Reparameterization of IBM Model 2 | 2013 | NAACL | fast_align |

### Phase 2 — Hybrid Systems
| Paper | Year | Venue | Relevance |
|:------|:-----|:------|:----------|
| Bengio et al. — A Neural Probabilistic Language Model | 2003 | JMLR | Neural LM |
| Mikolov et al. — Recurrent Neural Network Based LM | 2010 | IS | RNN-LM |
| Shen et al. — Discriminative Reranking for MT | 2004 | NAACL | Neural reranking |
| Durrani et al. — Operation Sequence Model | 2011 | ACL | OSM |
| Rosti et al. — Combining Outputs from Multiple MT Systems | 2007 | NAACL | System combination |

### Phase 3 — Neural MT
| Paper | Year | Venue | Relevance |
|:------|:-----|:------|:----------|
| Vaswani et al. — Attention Is All You Need | 2017 | NeurIPS | Transformer architecture |
| Sennrich et al. — NMT of Rare Words with Subword Units | 2016 | ACL | BPE for MT |
| Sennrich et al. — Improving NMT Models with Monolingual Data | 2016 | ACL | Back-translation |
| Edunov et al. — Understanding Back-Translation at Scale | 2018 | EMNLP | Best practices for BT |
| Wu et al. — Google's NMT System | 2016 | arXiv | Production GNMT |
| Hinton et al. — Distilling the Knowledge in a Neural Network | 2015 | arXiv | Knowledge distillation |
| Kim & Rush — Sequence-Level Knowledge Distillation | 2016 | EMNLP | MT-specific distillation |
| Johnson et al. — Google's Multilingual NMT System | 2017 | TACL | Multilingual NMT |
| Maruf et al. — Document-level NMT Survey | 2021 | CSUR | Doc-level MT |

---

## Key Open-Source Projects by Phase

### Phase 1
| Project | Language | Role |
|:--------|:---------|:-----|
| [Moses](https://github.com/moses-smt/mosesdecoder) | C++/Perl | Reference SMT pipeline (GIZA++, KenLM, MERT, phrase extraction) |
| [MGIZA++](https://github.com/moses-smt/mgiza) | C++ | Multi-threaded GIZA++ (IBM4 alignment) |
| [fast_align](https://github.com/clab/fast_align) | C++ | HMM alignment (already integrated) |
| [KenLM](https://github.com/kpu/kenlm) | C++ | Efficient n-gram LM |
| [Joshua](https://github.com/joshua-decoder/joshua) | Java | Hierarchical phrase-based decoder |
| [cdec](https://github.com/redpony/cdec) | C++ | SMT decoder with Hiero support |

### Phase 2
| Project | Language | Role |
|:--------|:---------|:-----|
| [PyTorch](https://pytorch.org/) | Python | Neural network framework |
| [Transformers](https://github.com/huggingface/transformers) | Python | Pre-trained LMs for neural LM/reranker |
| [KenLM Python](https://github.com/kpu/kenlm) | C++/Python | Fast LM for hybrid scoring |

### Phase 3
| Project | Language | Role |
|:--------|:---------|:-----|
| [fairseq](https://github.com/facebookresearch/fairseq) | Python | Production NMT training (Facebook) |
| [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) | Python | Production NMT (Harvard/Systran) |
| [SentencePiece](https://github.com/google/sentencepiece) | C++/Python | BPE/subword tokenization |
| [sacrebleu](https://github.com/mjpost/sacrebleu) | Python | Standardized BLEU |
| [COMET](https://github.com/Unbabel/COMET) | Python | Neural MT evaluation metric |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | C++/Python | Production model serving |
| [TensorRT](https://developer.nvidia.com/tensorrt) | C++ | GPU inference optimization |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | C++ | Cross-platform inference optimization |
| [fastBPE](https://github.com/glample/fastBPE) | C++ | Fast BPE implementation |

---

## Risk Analysis and Mitigation

| Risk | Phase | Severity | Mitigation |
|:-----|:------|:---------|:-----------|
| Data quality (noisy parallel data hurts more than it helps) | 1,2,3 | High | Aggressive cleaning pipeline, quality filtering, data selection methods |
| MGIZA++ compilation issues | 1 | Medium | Pre-built Docker images, fall back to fast_align |
| Neural LM too slow for decoder integration | 2 | High | Quantized model, caching, subprocess batching |
| Catastrophic forgetting in domain adaptation | 3 | Medium | Elastic weight consolidation, data mixing, adapter layers |
| Training instability (Transformer divergence) | 3 | Medium | Learning rate warmup, gradient clipping, FP16 mixed precision |
| Production latency requirements unmet | 3 | High | Aggressive distillation + quantization + Triton batching |
| GPU cost exceeds budget | 2,3 | Medium | Spot/preemptible instances, gradient accumulation on fewer GPUs |
| BLEU ceiling (domain mismatch limits quality) | All | Medium | Domain-specific data acquisition, back-translation, fine-tuning |

---

## Decision Gates

Each phase transition requires a Go/No-Go evaluation:

| Gate | After | Criterion | Threshold |
|:-----|:------|:----------|:----------|
| G1: P1 → P2 | Phase 1 complete | BLEU ≥ 22 on WMT news test | Proceed to neural components |
| G2: P2 → P3 | Phase 2 complete | BLEU ≥ 30 on WMT news test | Proceed to full NMT |
| G3: Production deploy | P3 complete | BLEU ≥ 38 + P95 latency < 100ms + human eval ≥ 4.0 adequacy | Production launch |

**Alternative path:** If Phase 1 BLEU reaches >28 (which is possible with 5M+ sentences + IBM4 + preordering + MERT), consider deploying the SMT system directly for latency-critical applications (SMT inference is inherently faster than NMT). Some production systems still use SMT for specific low-latency use cases (e.g., on-device translation).

---

## Effort and BLEU Roadmap Visualization

```
BLEU
50 │                                              ██████ Phase 3: NMT + distillation
   │                                         ▄▄▄▄▄▄
45 │                                    ▄▄▄▄▄
   │                              ▄▄▄▄▄
40 │                         ▄▄▄▄▄                    ← Google Translate 2018
   │                    ▄▄▄▄▄
35 │               ▄▄▄▄▄
   │          ▄▄▄▄▄    Phase 2: Neural LM + Reranker
30 │     ▄▄▄▄▄
   │▄▄▄▄▄  Phase 1: Production SMT
25 │
   │
20 │
   │
15 │
   │
10 │ ● Current (BLEU≈8)
   │
 5 │
   └─────┬──────────┬──────────┬──────────┬──────────┬──────────
        Now       3 months    6 months    9 months   12 months

Effort (person-months):       2–3          4–8         8–24
GPUs needed:                   0           1–4         4–8
Data (pairs):               5–10M        10–50M      100M+
```

---

## Immediate Next Steps (Week 1–2)

1. **Data acquisition sprint:**
   - Download UN Parallel Corpus (zh-en, ~15M pairs) via OPUS
   - Download OpenSubtitles 2018 (zh-en, ~3M pairs)
   - Run `scripts/clean_wmt.py` extended to handle OPUS format
   - Build deduplication pipeline (MinHash)

2. **MGIZA++ integration:**
   - Install MGIZA++ on dev machine or cloud instance
   - Create `smt/align_mgiza.py` following the same interface as `align_fast.py`
   - Run small-scale test (100K sentences) to verify alignment quality

3. **KenLM compilation and Python binding:**
   - Compile KenLM from source
   - Implement `smt/kenlm_proxy.py` (subprocess-based LM)
   - Train 5-gram model on 5M English sentences
   - Benchmark query latency vs current Python Kneser-Ney

4. **Preordering prototype:**
   - Install HanLP or Stanford CoreNLP
   - Implement 5 core reordering rules for Chinese→English
   - Test on 100 sentences; measure BLEU impact

5. **Dev/test split + evaluation harness:**
   - Split WMT data into train (90%) / dev (5%) / test (5%)
   - Set up automated BLEU + COMET evaluation on each experiment

---

*Generated 2026-06-07 by Prometheus (Strategic Planner / Architect)*
