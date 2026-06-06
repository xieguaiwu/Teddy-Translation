# Research: Open-Source Non-Transformer Machine Translation Systems

## Summary
The dominant pre-neural open-source SMT systems are Moses (phrase-based + hierarchical), cdec (unified hypergraph framework), Joshua (hierarchical with suffix arrays), and Phrasal (Java, with discontinuous phrases). A minimum viable SMT pipeline needs word alignment (GIZA++), phrase extraction, a 3-gram language model (KenLM), and a beam-search decoder — ~130K sentence pairs yields BLEU ~23. Key missing pieces in most toy implementations: MERT tuning, cube pruning for efficient decoding, future cost estimation heuristics, lexicalized reordering models, and compact binary LM/phrase-table formats.

---

## Findings

### 1. Moses SMT — Full Pipeline and Production-Grade Techniques

The Moses pipeline has 9 training steps plus corpus preparation and tuning:

**Corpus Preparation:**
- Tokenisation (punctuation/word splitting)
- Truecasing (normalizing sentence-initial case to reduce sparsity)
- Cleaning (removing long/misaligned pairs, length ratio filtering)

**Training Pipeline (Steps 1–9):**
1. **Prepare data** — factored data with lemmas, POS tags as additional annotation layers
2. **Run GIZA++** — IBM Model 4 word alignment in both directions (source→target, target→source)
3. **Align words** — symmetrization heuristics (`grow-diag-final-and`), merging bidirectional alignments
4. **Lexical translation table** — word-level translation probabilities with lexical weighting
5. **Extract phrases** — collect all phrase pairs consistent with word alignments (up to max phrase length, default 7)
6. **Score phrases** — 5 scores per phrase pair: p(e|f), lexical weight e|f, p(f|e), lexical weight f|e, constant phrase penalty
7. **Build reordering model** — lexicalized reordering with msd-bidirectional-fe (monotone, swap, discontinuous × bidirectional)
8. **Build generation model** — for factored systems mapping between annotation layers
9. **Create configuration file** — `moses.ini` specifying all feature functions and weights

**Language Model:**
- KenLM (included, LGPL): `lmplz` for estimation (modified Kneser-Ney smoothing), `build_binary` for mmap-based fast-loading binary format
- Probing hash table or trie data structures, bit-level packing
- Supports 5-gram and higher; binary format reduces loading from minutes to milliseconds

**Tuning:**
- MERT (Minimum Error Rate Training): Och's algorithm using Powell's line search to minimize BLEU on dev set
- Alternatives: MIRA (Margin Infused Relaxed Algorithm), PRO (Pairwise Ranking Optimization), MBR (Minimum Bayes Risk)
- Tuning script `mert-moses.pl` coordinates multi-iteration decoding

**Decoder (Production-Grade Features):**
- **Beam search** with histogram pruning (stack size limit) and threshold pruning (beam width)
- **Future cost estimation**: pre-computed heuristic for remaining untranslated spans, guiding search toward promising hypotheses
- **Cube pruning**: approximate A* search for hierarchical/chart decoding; pop-limit controls candidate expansion per node
- **Recombination**: merging hypotheses with identical distortion/coverage/LM state
- **Multi-threading**: one thread per sentence (embarrassingly parallel)
- **Binarized phrase tables**: `PhraseDictionaryCompact` format for fast loading via `processPhraseTableMin`
- **Incremental training**: suffix array-based on-demand phrase lookup, enabling corpus updates without full retraining
- **Factored models**: multiple annotation layers (surface form, lemma, POS, morphology) with configurable translation/generation steps
- **Confusion networks and word lattices**: input from ASR systems
- **n-best lists, MBR decoding, lattice MBR, consensus decoding**
- **XML markup**: force translations or specify constraints

[Source: Moses Specification, 265-page manual](http://www2.statmt.org/moses/manual/Moses-Specification.pdf)  
[Source: Moses Baseline Guide](http://www2.statmt.org/moses/?n=Moses.Baseline)

### 2. cdec — Key Innovations

cdec introduced several architectural innovations that influenced later systems:

- **Unified translation forest (hypergraph)**: All model types (word-based, phrase-based, SCFG, tagging) produce a single internal representation. Each node represents a contiguous target-language sequence; edges correspond to synchronous productions. This means rescoring, pruning, and inference algorithms work identically for all model types.
- **Strict separation of concerns**: Model-specific logic only required during forest *construction*; language model integration (rescoring), pruning, and inference algorithms are model-agnostic. New model types benefit from all existing algorithms immediately.
- **Semiring framework**: A generic C++ template system that computes Viterbi derivations, k-best lists, feature expectations, entropy, and expected translation length using a single linear-time algorithm but different semirings (tropical, log, expectation).
- **VEST (Viterbi Envelope Semiring Training)**: MERT implemented directly on hypergraphs rather than n-best approximations, computing error surfaces via semiring operations. Factorable into MapReduce for cluster parallelism.
- **Discriminative training**: Supports gradient-based optimization (LBFGS, RPROP, SGD) for models with millions of sparse features, maximizing conditional log-likelihood.
- **Alignment forests**: Parsing a target reference with the translation grammar produces compact representation of all valid derivations, enabling Viterbi or posterior word alignment extraction.
- **Performance**: In controlled benchmarks, cdec (C++) used 1.0GB RAM and 0.37s/sentence vs Joshua (Java) 1.5GB and 0.98s/sentence for identical Chinese-English translation.
- **Cube pruning and cube growing**: Two strategies for LM integration during forest rescoring, trading speed vs accuracy.

[Source: cdec ACL 2010 System Demo Paper](https://aclanthology.org/P10-4002.pdf)

### 3. Joshua Decoder — Speed Techniques

Joshua's speed came from:

- **Suffix array grammar extraction**: Instead of pre-computing and storing all phrase pairs (which can be terabytes), Joshua stores the sentence-aligned parallel corpus in memory using a suffix array. Translation rules are extracted *on the fly* by pattern-matching against the indexed bitext. This eliminates the disk I/O bottleneck of traditional phrase tables and allows arbitrarily long phrases without storage explosion.
- **Sampling**: To further speed up on-demand extraction, a subset of occurrences is sampled rather than exhausting all matches, with orders-of-magnitude speedup and negligible quality loss.
- **KenLM C++ integration via JNI**: Joshua (Java) calls KenLM (C++) through a JNI bridge, achieving near-native LM query speed. BerkeleyLM (pure Java) also available.
- **Parallel and distributed computing**: Multi-threaded chart parsing, distributed across cluster nodes for training and decoding.
- **Compact compiled LM formats**: Both KenLM and BerkeleyLM support mmap-based binary formats that load faster than ARPA text.
- **Joshua 6.0**: Further optimizations in phrase-based and hierarchical modes, unified feature function architecture.

[Source: Joshua Toolkit Paper](https://aclanthology.org/W09-0424.pdf)  
[Source: Joshua 6 Paper](https://ufal.mff.cuni.cz/pbml/104/art-post-cao-kumar.pdf)  
[Source: Hierarchical Phrase-Based Translation with Suffix Arrays](https://aclanthology.org/D07-1104.pdf)  
[Source: Sampling Phrase Tables](https://ufal.mff.cuni.cz/pbml/104/art-germann.pdf)

### 4. Phrasal (Stanford NLP) — Unique Features

- **Discontinuous (gappy) phrases**: The standout innovation. Unlike conventional phrase-based MT where phrases must be contiguous, Phrasal supports phrases with gaps (e.g., "ne … pas" → "not" with a gap). This provides better generalization and **outperforms hierarchical (Hiero) systems** while staying within the simpler phrase-based framework. [Source](https://aclanthology.org/N10-1140.pdf)
- **Feature API**: `RuleFeaturizer` (static, phrase-table-level) and `DerivationFeaturizer` (dynamic, decoding-level) interfaces. New features added by implementing a Java interface and specifying the class name on the command line — no recompilation of the decoder needed.
- **Online tuning**: AdaGrad-based tuner with pairwise (PRO) or expected BLEU objectives. Much faster than batch MERT — reaches comparable BLEU in 17 minutes vs Moses' 143 minutes on a large Arabic-English system. This is because Phrasal loads LM/phrase-table once, while Moses reloads every tuning epoch.
- **Web service**: RESTful JSON API via J2EE servlet. Supports interactive/prefix decoding for computer-assisted translation (CAT) — as the user types a target prefix, Phrasal suggests completions conditioned on it.
- **MERT implementation**: Uses Cer et al.'s line search for exact corpus-level error minimization.
- **CRF-based post-processor**: Combines truecasing and detokenization in a single pass, trainable to invert any pre-processor.
- **KenLM JNI binding**: Most LM queries execute in C++ for efficiency (>50% of CPU time in typical decoding).

[Source: Phrasal WMT 2014 Paper](https://aclanthology.org/W14-3311.pdf)  
[Source: Stanford NLP Phrasal Page](https://nlp.stanford.edu/phrasal/)  
[Source: Discontinuous Phrases Paper](https://aclanthology.org/N10-1140.pdf)

### 5. Key Techniques Our Implementation Lacks

**MERT Tuning (Minimum Error Rate Training):**
Och's algorithm uses Powell's method to optimize feature weights directly against a corpus-level metric (BLEU). For each feature, it samples weight values, decodes the tuning set, computes the error surface (piecewise linear), and finds the optimum. Requires 10–25 iterations through the tuning set. Modern alternatives: MIRA (faster convergence, better for many features), PRO (pairwise ranking loss, robust for large feature sets), online AdaGrad-based methods.

[Source: MERT Paper](https://aclanthology.org/P03-1021.pdf)  
[Source: Moses MERT script](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/training/mert-moses.pl)

**Cube Pruning:**
An approximate A* search algorithm for hierarchical/chart decoding. At each node in the chart, it maintains a priority queue (k-best) of partial derivations. When combining two spans for a rule, it generates new hypotheses by taking the Cartesian product of the two k-best lists — but since full enumeration is O(k²), cube pruning expands only the most promising candidates using a heuristic (LM estimate + future cost). Equivalent to A* search with a specific heuristic. Pop-limit (typically 100–5000) controls the beam. Critical for practical decoding speed with SCFG grammars.

[Source: Cube Pruning as Heuristic Search](https://aclanthology.org/D09-1007.pdf)

**Future Cost Estimation:**
Before beam search begins, the decoder computes for every contiguous source span a lower-bound estimate of how much it will cost to translate that span. This estimate combines the cheapest phrase translation cost, language model cost, and distortion cost. During decoding, partial hypotheses are scored as `current_cost + future_cost(remaining_spans)`, enabling the beam search to prune unpromising paths early. Without this, the decoder either exhausts memory or gets lost in bad search paths.

[Source: Moses Specification, Section 6.2.5](http://www2.statmt.org/moses/manual/Moses-Specification.pdf)

**Suffix Arrays for Phrase Extraction:**
Traditional phrase extraction enumerates all phrase pairs and stores them on disk as text files — slow to build and massive (hundreds of GB). Suffix arrays keep the entire aligned bitext in memory and retrieve phrase translations on demand by pattern matching. Sampling further reduces lookup time by orders of magnitude with no BLEU loss. Moses supports this via `-suffix-array` in the training pipeline. Critical for large-scale systems and incremental training.

[Source: Callison-Burch et al. 2005](https://aclanthology.org/P05-1032.pdf)  
[Source: Sampling Phrase Tables](https://ufal.mff.cuni.cz/pbml/104/art-germann.pdf)

**KenLM Binary Format:**
KenLM stores n-gram language models in a memory-mapped (mmap) binary file that loads in milliseconds. Two data structures: **probing** (hash table with 64-bit hashed keys, fast for production) and **trie** (bit-level packed trie using minimum bits for word indices and pointers, most compact). The binary format includes modified Kneser-Ney backoff weights, supports lazy loading, and can store 4 billion n-grams at ~23 bits/n-gram. ARPA-to-binary conversion via `build_binary`. BerkeleyLM (Java) provides similar compiled format.

[Source: KenLM README](https://github.com/kpu/kenlm/blob/master/README.md)  
[Source: KenLM Structures](http://kheafield.com/code/kenlm/structures/)  
[Source: BerkeleyLM](http://nlp.cs.berkeley.edu/pubs/Pauls-Klein_2011_LM_paper.pdf)

**Lexicalized Reordering (Distortion) Models:**
Basic distance-based distortion (linear penalty per word skipped) is too weak. Production systems use **lexicalized reordering**: three orientation types (monotone M, swap S, discontinuous D) × bidirectional (source→target and target→source) for 6 features total. Orientation probabilities are conditioned on the actual phrase pair. The `msd-bidirectional-fe` model is standard in Moses. Hierarchical reordering models (Galley & Manning 2008) use a hierarchy of orientation patterns. Without lexicalized reordering, long-distance reordering is essentially random.

[Source: Moses Advanced Models](http://www2.statmt.org/moses/?n=Advanced.Models)  
[Source: Hierarchical Phrase Reordering Model](https://nlp.stanford.edu/pubs/emnlp08-lexorder.pdf)

### 6. Minimum Viable SMT Pipeline and Data Requirements

**Minimum Pipeline:**
1. Parallel corpus (sentence-aligned)
2. Tokenization (lowercase + punctuation splitting)
3. Word alignment (GIZA++ with grow-diag-final-and symmetrization)
4. Phrase extraction (max phrase length 7)
5. Phrase scoring (4 probability scores + phrase penalty)
6. Language model (3-gram, KenLM `lmplz` + `build_binary`)
7. Distance-based reordering (distortion limit 6, linear penalty)
8. Beam search decoder (histogram pruning, stack size ~100)
9. Future cost estimation (pre-computed per-span heuristics)

**Optional but high-impact additions:**
- MERT tuning (dev set of 500–2000 sentences)
- Lexicalized reordering (msd-bidirectional-fe)
- Truecasing
- Binarized phrase tables (compact format for fast loading)
- Compound splitting (for German/Dutch) or word segmentation (for Chinese)

**Data Size vs Quality:**
| Parallel data | Expected BLEU (news domain) | Training time (single machine) |
|---|---|---|
| 10K–50K sentences | 10–18 BLEU | <1 hour |
| 130K sentences (News Commentary) | ~23.5 BLEU | ~1.5h (Moses baseline) |
| 1M sentences | 27–30 BLEU | 6–12h |
| 5M+ sentences | 30–35 BLEU | 1–3 days |

**Hardware minimum:** 2GB RAM, 10GB disk (Moses baseline estimate)
**LM data:** 1M+ monolingual target sentences for decent fluency; much more for production
**Tuning data:** 500–5000 sentence pairs, separate from training and test
**Known lower bound:** The Moses tutorial uses 130K French-English News Commentary sentences and achieves 23.5 BLEU (vs WMT best 30.5). Even very small toy systems (~1K sentences) produce intelligible but poor-quality output.

[Source: Moses Baseline Guide](http://www2.statmt.org/moses/?n=Moses.Baseline)  
[Source: Moses Real-World Usage](https://mt-archive.net/10/EAMT-2014-Schaefer.pdf)

### 7. Chinese-Specific SMT Systems and Techniques

**NiuTrans.SMT:**
Developed by Northeastern University (China) NLP Lab and NiuTrans Team. Fully C++, runs fast and uses less memory. Supports phrase-based, hierarchical phrase-based, and syntax-based (string-to-tree, tree-to-string, tree-to-tree) models. Open source under Apache 2.0. Specifically designed for Chinese↔English — the only major open-source SMT system with first-class Chinese support. [Source](https://github.com/NiuTrans/NiuTrans.SMT)

**Word Segmentation Approaches:**
Chinese word segmentation is *the* critical preprocessing step for Chinese SMT. Key findings from research:
- Dictionary-based vs CRF-based segmentation: **no significant difference** in final BLEU scores [Source](https://aclanthology.org/W08-0335.pdf)
- **Bilingually motivated segmentation** significantly outperforms monolingual segmentation: optimizing segmentation for alignment rather than linguistic correctness gives better translation [Source](https://dl.acm.org/doi/10.1145/1526252.1526255)
- Using **multiple different segmentations** and combining them improves results over any single segmentation [Source](https://aclanthology.org/W08-0335.pdf)
- Unsupervised bilingual segmenters (Dirichlet process models) can provide domain-adaptive segmentation without annotated data, effective on large corpora [Source](https://aclanthology.org/P14-2122.pdf)

**Popular Chinese Segmenters for SMT:**
- **ICTCLAS** (HMM-based, fast, widely used in Chinese SMT research)
- **Stanford Chinese Segmenter** (CRF-based, high accuracy)
- **LDC Segmenter** (rule-based)
- **Jieba** (fast dictionary+HMM hybrid, popular in production)

**CASIA System:**
The Institute of Automation, Chinese Academy of Sciences, built one of the most successful Chinese SMT systems (IWSLT 2008–2009). Their approach: combine multiple phrase-based systems with different preprocessing, using careful preprocessing to reduce OOV rates and data sparsity. [Source](https://aclanthology.org/2009.iwslt-evaluation.13.pdf)

**Character-based alternatives:** Some research suggests character-based models (no segmentation) can work as well as or better than word-based for certain Chinese language pairs, particularly when using hierarchical models. This avoids the segmentation error propagation problem entirely.

---

## Sources

### Kept:
- **Moses Specification (265-page PDF)** — definitive reference for the complete Moses pipeline, decoder internals (future cost, cube pruning, beam search), and all production techniques. http://www2.statmt.org/moses/manual/Moses-Specification.pdf
- **Moses Baseline Guide** — step-by-step instructions for building a real system, includes data sizes, expected BLEU, hardware requirements. http://www2.statmt.org/moses/?n=Moses.Baseline
- **cdec ACL 2010 Paper** — primary source for cdec's architecture and innovations (hypergraph, semirings, VEST). https://aclanthology.org/P10-4002.pdf
- **Joshua W09 Paper** — describes Joshua's suffix array grammar extraction and parsing-based decoding. https://aclanthology.org/W09-0424.pdf
- **Joshua 6 PBML Paper** — updated Joshua with phrase-based and hierarchical modes. https://ufal.mff.cuni.cz/pbml/104/art-post-cao-kumar.pdf
- **Suffix Arrays for Hierarchical MT (EMNLP 2007)** — foundational paper on suffix array-based rule extraction. https://aclanthology.org/D07-1104.pdf
- **Sampling Phrase Tables (PBML 2015)** — practical implementation and evaluation of sampled suffix array phrase tables in Moses. https://ufal.mff.cuni.cz/pbml/104/art-germann.pdf
- **Phrasal WMT 2014 Paper** — direct comparison with Moses, online tuning, feature API. https://aclanthology.org/W14-3311.pdf
- **Discontinuous Phrases (NAACL 2010)** — Phrasal's gappy phrase innovation. https://aclanthology.org/N10-1140.pdf
- **KenLM GitHub** — README with data structures, binary format documentation, API. https://github.com/kpu/kenlm
- **KenLM Structures Documentation** — probing vs trie, build_binary usage. http://kheafield.com/code/kenlm/structures/
- **MERT Original Paper (Och 2003)** — the algorithm, error surface optimization. https://aclanthology.org/P03-1021.pdf
- **Cube Pruning as Heuristic Search (EMNLP 2009)** — formal analysis of cube pruning as A* search. https://aclanthology.org/D09-1007.pdf
- **Moses Advanced Models** — lexicalized reordering (msd-bidirectional-fe). http://www2.statmt.org/moses/?n=Advanced.Models
- **NiuTrans.SMT GitHub** — Chinese-developed C++ open-source SMT system. https://github.com/NiuTrans/NiuTrans.SMT
- **Bilingual Word Segmentation for SMT (ACM 2009)** — bilingually motivated segmentation. https://dl.acm.org/doi/10.1145/1526252.1526255
- **Multiple Chinese Segmentation for SMT** — comparison of segmentation approaches. https://aclanthology.org/W08-0335.pdf
- **Chinese Word Segmentation via Bilingual Constraints (ACL 2014)** — state-of-the-art segmentation method for SMT. https://aclanthology.org/P14-1128.pdf
- **CASIA IWSLT 2009 System** — Chinese SMT system design. https://aclanthology.org/2009.iwslt-evaluation.13.pdf

### Dropped:
- Various duplicate PDF mirrors of the same papers (multiple URLs for cdec, Joshua, etc.)
- OpenAI "Distilling SMT Solver Reasoning" paper — about SMT solvers (Z3, etc.), not statistical machine translation; homonym confusion
- NiuTrans.NMT — neural MT, not statistical; outside scope
- General tutorial slides (AMTA 2012) — redundant with Moses specification

---

## Gaps

1. **Direct comparison of our current implementation vs Moses baseline**: The findings document general techniques; measuring exactly which missing pieces cause the largest quality gap would require a controlled ablation study (e.g., adding MERT alone, adding lexicalized reordering alone, etc.).

2. **Chinese segmentation tool integration**: While approaches are identified, the practical integration of bilingually motivated segmentation (e.g., using the tool from the 2014 ACL paper) into a specific pipeline was not tested.

3. **Minimum BLEU threshold for "useful" translation**: The Moses baseline shows ~23.5 BLEU for 130K sentences, but what constitutes "useful" depends heavily on use case. For gisting, BLEU 15 may suffice; for post-editing, BLEU 25+; for publication, BLEU 30+.

4. **Modern alternatives to GIZA++**: fast_align (Dyer et al.) and the Berkeley Aligner are mentioned as faster/better alternatives, but no direct BLEU comparisons were found for Chinese-English at different data scales.

5. **Chinese-English SMT BLEU ceilings**: No systematic survey was found documenting what BLEU the best Chinese-English SMT systems achieved (pre-neural era) at various training data sizes.

---

*Generated: 2026-06-06*
*Tool: web_search + fetch_content with Perplexity/Exa providers*
