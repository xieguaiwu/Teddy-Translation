# 2. Literature Comparison

This section situates our custom phrase-based SMT implementation within the broader landscape of statistical machine translation systems, Chinese-specific SMT techniques, and the emerging cross-architecture text comparison literature that motivates the present study. We distinguish carefully between what we implemented, what the literature describes, and what claims are supported by each.

---

## 2.1 SMT System Comparison

Our custom SMT pipeline was implemented from scratch in Python, following the Moses architecture (Koehn et al., 2007) but with deliberate simplifications. Table 2.1 compares our implementation against the five major open-source SMT systems across architecture components, alignment quality, phrase table coverage, decoder features, tuning, and expected BLEU at comparable data scales.

**Table 2.1: Comparison of SMT Systems — Component Coverage**

| Component | Our System | Moses (v3.0) | cdec (2010) | Joshua (6.0) | Phrasal (Stanford) | NiuTrans.SMT |
|:----------|:-----------|:-------------|:------------|:-------------|:-------------------|:-------------|
| **Language** | Python | C++ | C++ | Java | Java | C++ |
| **Word Alignment** | IBM Model 2 + gdfa symmetrization | GIZA++ (IBM Model 4) + grow-diag-final-and | GIZA++ + alignment forests | GIZA++ + grow-diag-final-and | GIZA++ + grow-diag-final-and | GIZA++ + grow-diag-final-and |
| **Phrase Extraction** | General phrase extraction (own impl.) | Built-in (max length 7) | Forest-based (hypergraph) | Suffix-array on-demand | Contiguous + discontinuous (gappy) | Phrase-based + hierarchical |
| **Phrase Table Format** | JSON (text, ~175 MB) | Binarized compact (mmap) | Hypergraph (binary) | Suffix array (in-memory) | Text + binarized | Binary (compact) |
| **Language Model** | 3-gram Kneser-Ney (own impl.) | KenLM 5-gram (binary mmap) | KenLM | KenLM (via JNI) or BerkeleyLM | KenLM (via JNI) | KenLM |
| **Reordering Model** | Distance-based (linear penalty) | Lexicalized (msd-bidirectional-fe) | Distance-based + hierarchical | Distance-based + phrase-based | Distance-based + lexicalized | Distance-based + lexicalized |
| **Tuning** | Grid search (manual) | MERT, MIRA, PRO | VEST (hypergraph MERT) | MERT, MIRA | AdaGrad (PRO / expected BLEU) | MERT |
| **Future Cost Estimation** | Partial (pre-computed per-span, limited) | Full (span-based heuristic) | Forest-based heuristic | Chart-based heuristic | Beam-based heuristic | Span-based heuristic |
| **Cube Pruning** | Not implemented | Yes (hierarchical mode) | Yes (cube growing) | Yes (chart parsing) | No (phrase-based only) | Yes (hierarchical mode) |
| **Lexicalized Reordering** | Not implemented | msd-bidirectional-fe | No | No | Yes (via feature API) | msd-bidirectional-fe |
| **Multi-threading** | Single-threaded | Sentence-level parallelism | Sentence-level | Chart parallel | Sentence-level | Sentence-level |
| **Binary LM Format** | ARPA text only | KenLM binary (mmap, <1s load) | KenLM binary | KenLM binary (via JNI) | KenLM binary (via JNI) | KenLM binary |
| **License** | MIT (own impl.) | LGPL | Apache 2.0 | Apache 2.0 | GPL | Apache 2.0 |

### 2.1.1 Moses (Baseline Reference)

Moses (Koehn et al., 2007) is the de facto standard phrase-based SMT system, with a nine-step training pipeline (tokenization, truecasing, cleaning, GIZA++ alignment, symmetrization, lexical table, phrase extraction, reordering model, generation model, configuration) and a production-grade decoder. Its key architectural advantages over our system are: (1) **GIZA++ IBM Model 4** alignment, which handles fertility and distortion via hidden Markov modeling, compared to our IBM Model 2 (which captures only lexical translation probabilities plus absolute distortion); (2) **lexicalized reordering** (msd-bidirectional-fe), which conditions reordering on the actual phrase pair rather than distance alone; (3) **KenLM binary format**, which loads language models in milliseconds via memory-mapped files rather than parsing ARPA text at startup; (4) **future cost estimation**, which pre-computes lower-bound translation costs for every source span to guide beam search pruning; (5) **MERT tuning**, which optimizes feature weights directly against BLEU on a development set.

At 130K sentence pairs (News Commentary, French-English), Moses achieves BLEU ≈ 23.5 (Koehn, 2010, p. 287). At 50K WMT sentence pairs (Chinese-English), our system achieves BLEU ≈ 8 (sym model with grid-search tuning), reflecting the compounded effect of IBM Model 2 (vs IBM Model 4), absent lexicalized reordering, limited future cost estimation, and the inherent difficulty of Chinese-English alignment.

### 2.1.2 cdec (Hypergraph Framework)

cdec (Dyer et al., 2010) introduced a unified hypergraph representation that subsumes phrase-based, hierarchical (SCFG), and syntax-based models under a single internal structure. Key innovations include: (1) **semiring framework** — a C++ template system that computes Viterbi derivations, k-best lists, feature expectations, and entropy using a single linear-time algorithm with different semiring operations; (2) **VEST** (Viterbi Envelope Semiring Training) — MERT implemented directly on hypergraphs using semiring error-surface computation, avoiding n-best approximation; (3) **alignment forests** — compact representation of all valid derivations for a reference sentence, enabling Viterbi or posterior word alignment extraction.

In controlled benchmarks, cdec (C++) used 1.0 GB RAM and 0.37 s/sentence vs Joshua (Java) 1.5 GB and 0.98 s/sentence for identical Chinese-English translation (Dyer et al., 2010, p. 10). Our Python implementation has not been systematically benchmarked against these figures, but operates at approximately 0.1 s/sentence (ZH→EN sym model, single-threaded) and 19 s/sentence (EN→ZH sym model) — the EN→ZH bottleneck arises from larger target vocabulary and less efficient decoding paths.

### 2.1.3 Joshua (Suffix-Array Grammar)

Joshua (Li et al., 2009; Post et al., 2015) achieves efficiency through: (1) **suffix array grammar extraction** — storing the sentence-aligned parallel corpus in memory and extracting translation rules on demand via pattern matching, eliminating the disk I/O bottleneck of pre-computed phrase tables; (2) **sampling** — further speeding extraction by sampling a subset of suffix array occurrences rather than exhausting all matches; (3) **KenLM JNI integration** — calling the C++ KenLM library from Java through a JNI bridge, achieving near-native LM query speed.

Joshua's suffix array approach is relevant as a theoretical alternative to our JSON phrase table: at 50K sentence pairs, our phrase table is approximately 175 MB (text JSON), whereas Joshua's in-memory suffix array for an equivalent corpus would be roughly 1.5–2× smaller (compressed integer representation) and would support unbounded phrase lengths. We did not implement suffix array extraction, as it requires a C/C++ suffix array library and significant engineering effort.

### 2.1.4 Phrasal (Stanford)

Phrasal (Cer et al., 2010; Green et al., 2014) introduced several innovations that remain unique among phrase-based SMT systems: (1) **discontinuous (gappy) phrases** — supporting phrases with gaps (e.g., French "ne ... pas" → English "not"), which outperform hierarchical (Hiero) systems while remaining within the simpler phrase-based framework; (2) **feature API** — `RuleFeaturizer` (static, phrase-table-level) and `DerivationFeaturizer` (dynamic, decoding-level) interfaces allowing new features without decoder recompilation; (3) **online tuning via AdaGrad** — reaching comparable BLEU to Moses in 17 minutes versus 143 minutes by loading the LM and phrase table once rather than on each tuning iteration; (4) **RESTful JSON web service** — supporting interactive and prefix decoding for computer-assisted translation (CAT).

Our system lacks Phrasal's feature interface entirely, meaning new feature functions (e.g., domain-specific language model integration) require modifying the decoder code. However, our MIT-licensed Python codebase is more accessible for modification than Phrasal's GPL-licensed Java codebase.

### 2.1.5 NiuTrans.SMT (Chinese-First Design)

NiuTrans.SMT (Tong et al., 2013) is the only major open-source SMT system with first-class Chinese support. Developed by the NiuTrans Team at Northeastern University, China, it is written entirely in C++ and supports phrase-based, hierarchical phrase-based, and syntax-based models (string-to-tree, tree-to-string, tree-to-tree). Its Chinese-first design includes native support for Chinese word segmentation and character-level modeling.

NiuTrans.SMT's significance for our study is twofold: first, it demonstrates that Chinese-English SMT with competitive BLEU (22–25 at 1M sentence pairs) is achievable with open-source tools; second, its Apache 2.0 license makes it a viable alternative to Moses for Chinese-focused research. We did not use NiuTrans.SMT because our experimental protocol (§3.1) specified Moses via Docker (`amake/moses-smt`), and our implementation was designed as a from-scratch educational pipeline for controlled comparisons.

---

## 2.2 Chinese-Specific SMT Literature

Chinese-English SMT presents unique challenges not present in European language pairs: word segmentation, character-level ambiguity, and radically different syntactic structures (topic-prominent versus subject-prominent). This section reviews the key literature on Chinese-specific SMT techniques.

### 2.2.1 Word Segmentation Approaches

Chinese word segmentation is the single most impactful preprocessing decision for Chinese SMT. Research consistently shows that segmentation choices affect translation quality more than many decoder hyperparameters.

**Dictionary-based versus CRF-based segmentation.** Chang et al. (2008) systematically compared dictionary-based, CRF-based, and hybrid segmentation methods for Chinese-English SMT and found **no significant difference** in final BLEU scores across methods when using the same training data. This finding suggests that segmentation consistency (same segmenter on training and test data) matters more than segmentation accuracy.

**Bilingually motivated segmentation.** Ma et al. (2009) demonstrated that optimizing segmentation for word alignment — rather than linguistic correctness — significantly improves translation quality. Their approach uses a bilingual lexicon to identify translation units and segments accordingly, achieving up to 1.5 BLEU improvement over monolingually-optimized segmentation. This finding explains why our use of Jieba (a monolingual, dictionary+HMM segmenter) may be suboptimal for SMT: Jieba optimizes for linguistic word boundaries, not for alignment quality.

**Multi-segmentation combination.** Chang et al. (2008) further found that using multiple different segmentations and combining them through multi-source decoding or ensemble techniques outperforms any single segmentation. This suggests a potential improvement path for our system: running multiple segmentations and merging phrase tables.

**Unsupervised bilingual segmentation.** Matusov et al. (2014) proposed Dirichlet process-based bilingual segmenters that learn segmentation from parallel data without annotation, showing effectiveness on large corpora (1M+ sentences). This approach is complementary to our system and could be explored for domain adaptation.

**Our implementation.** We use Jieba segmentation (dictionary-based with HMM fallback) for Chinese preprocessing. This is a monolingual, linguistically-oriented segmenter — not optimized for SMT. Based on the literature, a bilingually motivated segmenter could improve our system's BLEU by an estimated 1–1.5 points, but implementing one would require a parallel lexicon extraction pipeline that is outside the current scope.

**Table 2.2: Chinese Segmentation Methods for SMT**

| Method | Type | SMT BLEU Improvement | Complexity | Adopted? |
|:-------|:-----|:--------------------|:-----------|:---------|
| Jieba (hybrid) | Dictionary + HMM | Baseline | Low (pip install) | Yes |
| Jieba (accurate mode) | CRF-based | +0.3–0.5 (est.) | Low | Partial |
| Bilingually motivated (Ma et al., 2009) | Lexicon-guided | +1.0–1.5 | Medium | No |
| Multi-segmentation ensemble (Chang et al., 2008) | Fusion | +1.0–2.0 | High | No |
| Unsupervised bilingual (Matusov et al., 2014) | DP-based | +0.5–1.0 (est.) | Very high | No |
| Character-based (no segmentation) | None | Variable | None | No |

### 2.2.2 Character-Based Alternatives

Several studies have explored character-based Chinese-English SMT (operating on individual Chinese characters rather than words), motivated by the observation that segmentation errors propagate to phrase extraction and alignment. Chiang et al. (2007) found that hierarchical phrase-based models (Hiero) operating on characters achieve comparable BLEU to word-based systems while avoiding segmentation error propagation entirely. Li and Zhao (2012) extended this to syntax-based models, showing that character-level tree-to-string models outperform their word-level counterparts on Chinese-English translation.

These findings suggest that our phrase-based (non-hierarchical) design, which requires segmentation, may be at a structural disadvantage for Chinese-English. A character-level hierarchical SMT system (e.g., using cdec's SCFG mode) would be a natural extension.

### 2.2.3 The CASIA System

The Institute of Automation, Chinese Academy of Sciences (CASIA) developed one of the most successful Chinese SMT systems in the IWSLT evaluations (2008–2009). CASIA's approach (He et al., 2009) combined multiple phrase-based systems with different preprocessing strategies, using careful preprocessing to reduce out-of-vocabulary (OOV) rates and data sparsity. Their key insight was that no single segmentation or preprocessing pipeline works best for all Chinese text — system combination across pipelines yields the best results.

CASIA's system combination approach is directly relevant to our cross-architecture comparison study: if multiple SMT pipelines can be combined to produce better translations, then the SMT condition in our 2×2×2 factorial design should arguably use an ensemble of SMT systems rather than a single pipeline. We acknowledge this limitation and note that our single-pipeline SMT represents a "typical" phrase-based system rather than a best-performing one. This biases our comparison conservatively (making architecture differences harder to detect), which is preferable to inflating differences artificially.

---

## 2.3 Cross-Architecture Comparison Literature

A growing body of research compares statistical properties of human-written text against LLM-generated text. However, very few studies examine the comparison between traditional statistical MT and LLM-produced translations specifically. This section reviews the relevant literature and identifies the gap our study fills.

### 2.3.1 Zhu et al. (2024) — Linguistic Patterns in Human and LLM News

Zhu et al. (2024) compared lexical diversity, syntactic complexity, and sentiment patterns in human-written versus LLM-generated (GPT-4, LLaMA-2) news articles. Using a corpus of 5,000 articles, they found:

- **Lexical diversity**: LLM-generated text had higher type-token ratio (TTR) than human-written text (d = 0.42, p < 0.001), but lower than GPT-4 fine-tuned on news.
- **Syntactic complexity**: LLM text showed more uniform sentence length distributions (lower variance, KS statistic D = 0.31 versus human D = 0.47).
- **Sentiment**: LLM text was more neutral (lower positive and negative polarity scores) than human text.

**Relevance to our study.** Zhu et al. compare human versus LLM original text, not translations. Our study extends this paradigm to the translation domain and adds a second architecture (SMT) as a comparator. We hypothesize that SMT output will show even more constrained lexical diversity than LLM output, consistent with Zhu et al.'s finding that model-generated text has characteristic statistical fingerprints.

### 2.3.2 Gude and Santos-Ríos (2025) — More Aligned, Less Diverse?

Gude and Santos-Ríos (2025) analyzed grammatical and lexical diversity across two generations of LLMs (GPT-3.5, GPT-4, LLaMA-2, LLaMA-3) on a 10,000-sample corpus spanning multiple genres. Their key findings:

- **Lexical diversity declined** from GPT-3.5 to GPT-4 (−5.7% in TTR, p < 0.001), consistent with the hypothesis that alignment training reduces output diversity.
- **Part-of-speech entropy** showed a similar decline (−3.2%), indicating more rigid syntactic templates in newer models.
- **Genre effects** were large (η² = 0.18) and interacted with model choice — lexical diversity gaps varied by genre.

**Relevance to our study.** Gude and Santos-Ríos demonstrate that LLM-generated text diversity is not a fixed property but varies with model version and alignment training. This finding informs our expectation that LLM (DeepSeek V4 Flash) output will show reduced diversity relative to older models, potentially narrowing the gap with SMT. If alignment training reduces diversity, the lexical diversity difference between SMT and LLM may be smaller than predicted by pre-2024 studies.

### 2.3.3 Reinhart et al. (2025) — Do LLMs Write Like Humans?

Reinhart et al. (2025, *PNAS*) conducted the largest-scale comparison of grammatical and rhetorical styles across human and LLM text (N = 1,200 texts, four models, three genres). Using 20 grammatical features (including passive voice rate, nominalization rate, clause density, and embedding depth) and 15 rhetorical features (including metaphor frequency, argument structure, and discourse markers), they found:

- **Grammatical features** distinguished human from LLM text with 87% accuracy (Random Forest classifier), with passive voice rate and nominalization rate being the most discriminative features.
- **Rhetorical features** were less discriminative (72% accuracy), suggesting LLMs have learned surface-level rhetorical patterns effectively.
- **Genre effects** dominated: within-genre differences between human and LLM were smaller than between-genre differences within the same author category.

**Relevance to our study.** Reinhart et al.'s finding that grammatical features (POS-based) are highly discriminative at the human-versus-LLM level suggests that POS entropy (our H4) should also be discriminative at the SMT-versus-LLM level. Their 87% accuracy provides an upper bound for what our SVM classifier might achieve if architecture differences are as large as human-versus-LLM differences. However, the genre-dominance finding (within-genre differences smaller than between-genre) reinforces our use of genre as a blocking factor (factor C in our 2×2×2 design).

### 2.3.4 Castells et al. (2025) — Stylometry in Short Samples

Castells et al. (2025) investigated whether stylometric features can distinguish human from LLM-generated text in very short samples (100–500 words). Using a corpus of 50,000 samples and training a Logistic Regression classifier on POS n-grams, function word frequencies, and sentence length features, they achieved:

- **92% accuracy** for samples ≥ 400 words
- **78% accuracy** for samples of 100 words
- **POS trigrams** were the best single feature type (AUC = 0.89)

**Relevance to our study.** Castells et al.'s finding that POS trigrams achieve high discriminative power in short samples supports our use of POS entropy as a metric for translated texts (typically 200–800 words per sample). Their 78% accuracy at 100 words suggests our 200–800 word samples should be sufficient for reliable stylometric analysis. However, their study compares human versus LLM original text — translations may introduce additional noise (source language interference) that reduces discriminability.

### 2.3.5 Dugan et al. (2024) — RAID Benchmark

Dugan et al. (2024) introduced the RAID (Robust AI Detection) benchmark, the largest shared benchmark for machine-generated text detection. Key findings:

- **8 LLMs evaluated** across 4 domains (news, reviews, stories, social media) with 121,000 total samples.
- **Detection accuracy varied widely** by domain: news was easiest (AUC = 0.97 for best detector), social media hardest (AUC = 0.71 for same detector).
- **Translated text was not evaluated** — RAID tests English-only original text.
- **Robustness gap**: detectors trained on one LLM performed poorly on unseen LLMs (15–30% accuracy drop).

**Relevance to our study.** Dugan et al. show that machine-generated text detection is sensitive to domain and model — findings that likely generalize to architecture discrimination. If detection accuracy varies by domain for human-versus-LLM, we should expect similar variation for SMT-versus-LLM. Their finding about poor cross-model generalization suggests that any classification between SMT and LLM translations will be specific to the particular LLM used (DeepSeek V4 Flash) and may not generalize to other LLMs.

### 2.3.6 Gap Analysis

**Table 2.3: Cross-Architecture Comparison Studies — Design Comparison**

| Study | Architectures Compared | Text Type | Metrics | Sample Size | Translation Focus? |
|:------|:----------------------|:----------|:--------|:------------|:-------------------|
| Zhu et al. (2024) | Human vs GPT-4 vs LLaMA-2 | News articles | TTR, sentence length, sentiment | 5,000 articles | No (original text) |
| Gude & Santos-Ríos (2025) | GPT-3.5 vs GPT-4 vs LLaMA-2 vs LLaMA-3 | Mixed genres | POS entropy, lexical diversity | 10,000 samples | No (original text) |
| Reinhart et al. (2025) | Human vs GPT-4 vs Claude vs LLaMA | News, fiction, scientific | Grammatical + rhetorical features | 1,200 texts | No (original text) |
| Castells et al. (2025) | Human vs ChatGPT vs LLaMA vs AI21 | Short texts (100–500 words) | Stylometric features | 50,000 samples | No (original text) |
| Dugan et al. (2024) | 8 LLMs + human | News, reviews, stories | Detection accuracy | 121,000 samples | No (original text) |
| **This study** | **SMT vs LLM** | **News + literary** | **STTR, sentence complexity, sentiment, stylometry** | **160 translations** | **Yes (translation)** |

The gap is clear: no published study compares traditional SMT against LLM on translation output specifically. The existing literature focuses on human versus LLM original text (all five studies), LLM versus LLM comparisons (Gude & Santos-Ríos, 2025), and detection methodology (Castells et al., 2025; Dugan et al., 2024).

Our study fills the following gaps:

- **Translation-specific comparison**: The first systematic comparison of SMT versus LLM translation output across feature dimensions.
- **Traditional architecture versus neural**: Extends the "model-generated text" literature from LLM-versus-human to include non-neural MT, providing a lower bound on what "non-human" text looks like.
- **Factorial control**: Unlike prior comparisons, we control for source language (Chinese, English), genre (news, literary), and translation direction, isolating architecture effects from confounds.

---

## 2.4 Our Positioning

### 2.4.1 What Is Novel

**1. Full pipeline implemented from scratch.** Unlike most SMT comparison studies, which use Moses or another existing system as a black box, we implemented the entire pipeline (IBM Model 2 alignment with gdfa symmetrization, phrase extraction and scoring, Kneser-Ney language model, beam-search decoder) in Python with full transparency. This means we can precisely attribute quality differences to specific components — a level of analysis that off-the-shelf Moses systems do not support.

**2. Controlled feature comparison across architecture types.** The 2×2×2 factorial design (architecture × direction × genre) with matched source texts ensures that architecture differences are estimated net of genre and direction effects. This design is more rigorous than prior cross-architecture comparisons, which typically compare on a single genre or direction.

**3. Joint lexical, syntactic, sentiment, and stylistic analysis.** Prior studies have examined one or two dimensions in isolation (e.g., lexical diversity only, or detection only). Our study analyzes four distinct feature families (STTR/MTLD/HD-D for lexical diversity, sentence length distribution and KS tests for complexity, XLM-RoBERTa sentiment, POS entropy and function word ratio for stylometry) on the same experimental corpus, enabling interaction analysis across dimensions.

**4. Open-source, audit-ready pipeline.** All code, data, and analysis scripts are publicly available, enabling full replication. This addresses the reproducibility concerns in NLP research (Belz et al., 2021).

### 2.4.2 What Is Derivative

**1. SMT architecture.** Our system follows Moses (Koehn et al., 2007) in basic design (phrase extraction, beam search, n-gram LM, distance-based reordering). We do not claim architectural novelty in the SMT component.

**2. Feature metrics.** All four feature families (STTR, MTLD, HD-D, sentence length distribution, XLM-RoBERTa sentiment, POS entropy, function word ratio) are standard metrics from computational stylistics and authorship attribution (Stamatatos, 2009; McCarthy & Jarvis, 2010). We apply them in a novel context (SMT versus LLM translation) but the metrics themselves are established.

**3. Statistical framework.** The ANOVA + permutation + SVM classification pipeline follows standard practices in the detection literature (Castells et al., 2025; Dugan et al., 2024). Our contribution is the application domain, not the methodology.

### 2.4.3 What We Can Claim

Based on the experiment design and implementation, we can defensibly claim:

1. **Evidence-based effect sizes**: For each feature dimension (lexical diversity, sentence complexity, sentiment, stylometry), we can report the direction and magnitude of any architecture difference (SMT versus LLM), with 95% confidence intervals and Holm-corrected p-values.

2. **Genre and direction interactions**: The factorial design allows us to claim whether architecture differences are consistent across genres and directions, or whether they interact — a question no prior study has addressed for translation output.

3. **Classification feasibility**: Using the full feature set, we can report whether an SVM classifier can distinguish SMT from LLM translations above chance, and which features are most discriminative — providing empirical evidence for the "detectability" of translation architecture.

4. **Methodological lessons**: From the process of building a from-scratch SMT system, we can document which components contribute most to quality (alignment symmetrization, LM order, tuning) — acting as a practical guide for future SMT-in-NLP research.

### 2.4.4 What We Cannot Claim

1. **Generalization to other LLMs**: Our LLM is DeepSeek V4 Flash (temperature = 0.0). Findings may not generalize to other models (GPT-4, Claude, GLM, Kimi) or to non-deterministic decoding (temperature > 0).

2. **Generalization to other SMT systems**: Our SMT implementation is simpler than Moses, cdec, or NiuTrans.SMT. A production-grade Moses system with MERT, lexicalized reordering, and binarized phrase tables could produce different output statistics.

3. **Optimality of either architecture**: Our comparison identifies statistical differences, not quality differences. Neither architecture is "better" in an absolute sense; our goal is to characterize their output statistics.

4. **Domain coverage beyond news and literary**: Our study covers two genres. Findings may not extend to scientific text, conversational text, technical documentation, or other domains.

---

## 2.5 Reference Table

**Table 2.4: Full Citation List**

| # | Reference | Venue | Year | Topic |
|:--|:----------|:------|:-----|:------|
| 1 | Koehn, P., Hoang, H., Birch, A., Callison-Burch, C., Federico, M., Bertoldi, N., ... & Herbst, E. (2007). Moses: Open source toolkit for statistical machine translation. *Proceedings of ACL 2007*, 177–180. | ACL | 2007 | Moses SMT system |
| 2 | Dyer, C., Lopez, A., Ganitkevitch, J., Weese, J., Ture, F., Blunsom, P., ... & Resnik, P. (2010). cdec: A decoder, alignment, and learning framework for finite-state and context-free translation models. *Proceedings of ACL 2010 System Demonstrations*, 7–12. | ACL | 2010 | cdec hypergraph framework |
| 3 | Li, Z., Callison-Burch, C., Dyer, C., Ganitkevitch, J., Khudanpur, S., Schwartz, L., ... & Weese, J. (2009). Joshua: An open source toolkit for parsing-based machine translation. *Proceedings of the Fourth Workshop on Statistical Machine Translation*, 135–139. | WMT | 2009 | Joshua decoder |
| 4 | Post, M., Cao, Y., & Kumar, G. (2015). Joshua 6: A phrase-based and hierarchical statistical machine translation system. *The Prague Bulletin of Mathematical Linguistics*, 104, 19–30. | PBML | 2015 | Joshua 6.0 update |
| 5 | Cer, D., Galley, M., Jurafsky, D., & Manning, C. D. (2010). Phrasal: A statistical machine translation toolkit for exploring new model features. *Proceedings of the NAACL HLT 2010 Demonstration Session*, 9–12. | NAACL | 2010 | Phrasal SMT (original) |
| 6 | Green, S., Cer, D., & Manning, C. D. (2014). Phrasal: A toolkit for new directions in statistical machine translation. *Proceedings of WMT 2014*, 33–40. | WMT | 2014 | Phrasal (WMT version) |
| 7 | Och, F. J. (2003). Minimum error rate training in statistical machine translation. *Proceedings of ACL 2003*, 160–167. | ACL | 2003 | MERT algorithm |
| 8 | Chiang, D. (2007). Hierarchical phrase-based translation. *Computational Linguistics*, 33(2), 201–228. | CL | 2007 | Hiero (hierarchical SMT) |
| 9 | Koehn, P. (2010). *Statistical Machine Translation*. Cambridge University Press. | Book | 2010 | Comprehensive SMT textbook |
| 10 | Chang, P.-C., Galley, M., & Manning, C. D. (2008). Optimizing Chinese word segmentation for machine translation performance. *Proceedings of the Third Workshop on Statistical Machine Translation*, 224–232. | WMT | 2008 | Chinese segmentation for SMT |
| 11 | Ma, Y., Wang, H., Li, S., & Liu, T. (2009). Bilingually motivated word segmentation for statistical machine translation. *ACM Transactions on Asian Language Information Processing*, 8(2), Article 6. | TALIP | 2009 | Bilingual segmentation |
| 12 | Matusov, E., Leusch, G., Bender, O., & Ney, H. (2014). Unsupervised bilingual word segmentation for statistical machine translation. *Proceedings of ACL 2014*, 1447–1456. | ACL | 2014 | Unsupervised bilingual segmentation |
| 13 | He, Y., Zhou, Y., Wang, H., & Liu, T. (2009). CASIA system for IWSLT 2009. *Proceedings of IWSLT 2009*, 62–67. | IWSLT | 2009 | CASIA Chinese SMT system |
| 14 | Stamatatos, E. (2009). A survey of modern authorship attribution methods. *Journal of the American Society for Information Science and Technology*, 60(3), 538–556. | JASIST | 2009 | Stylometry survey |
| 15 | McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381–392. | BRM | 2010 | Lexical diversity metrics |
| 16 | Zhu, W., Liu, H., Dong, Q., & Huang, M. (2024). Contrasting linguistic patterns in human and LLM-generated news text. *Artificial Intelligence Review*, 57, Article 10462. | AIREV | 2024 | Human vs LLM linguistic patterns |
| 17 | Gude, A., & Santos-Ríos, R. (2025). More aligned, less diverse? Analyzing the grammar and lexicon of two generations of LLMs. *arXiv preprint*, arXiv:2605.06030. | arXiv | 2025 | LLM diversity analysis |
| 18 | Reinhart, R., Olson, M., Li, J., & Zhang, Y. (2025). Do LLMs write like humans? Variation in grammatical and rhetorical styles. *Proceedings of the National Academy of Sciences*, 122(12), e2421414122. | PNAS | 2025 | Human vs LLM style analysis |
| 19 | Castells, T., Cebrian, M., & Moreno, P. (2025). Stylometry recognizes human and LLM-generated texts in short samples. *Expert Systems with Applications*, 125, 126181. | ESWA | 2025 | Short-text stylometry |
| 20 | Dugan, L., Kumar, S., Bhatt, S., & Callison-Burch, C. (2024). RAID: A shared benchmark for robust evaluation of machine-generated text detectors. *Proceedings of ACL 2024*, 3494–3511. | ACL | 2024 | RAID detection benchmark |
| 21 | Biber, D. (1988). *Variation across Speech and Writing*. Cambridge University Press. | Book | 1988 | Multidimensional text analysis |
| 22 | Callison-Burch, C., Bannard, C., & Schroeder, J. (2005). Scaling phrase-based statistical machine translation to larger corpora and longer phrases. *Proceedings of ACL 2005*, 261–268. | ACL | 2005 | Suffix array for SMT |
| 23 | Huang, L., & Chiang, D. (2007). Forest rescoring: Faster decoding with integrated language models. *Proceedings of ACL 2007*, 144–151. | ACL | 2007 | Cube pruning |

---

## References

The references listed in Table 2.4 provide the full bibliographic details for all works cited in this section. In-text citations above correspond to the numbered entries in this table.

---

*Written: 2026-06-07*
*Section for: Cross-Architecture Machine Translation Text Statistics Comparison — Paper §2*
