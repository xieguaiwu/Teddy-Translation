# Experiment Design

## 3.1 Research Questions

This study investigates whether machine translation outputs produced by two fundamentally different architectures—traditional phrase-based statistical machine translation (SMT) and decoder-only large language models (LLMs)—exhibit statistically distinguishable properties across four linguistic dimensions. The central research question is:

> **Do SMT and LLM translations of the same source texts differ in lexical diversity, sentence complexity, sentiment orientation, and stylometric features, and can these differences be reliably detected through quantitative text analysis?**

Four specific research questions, each with a directional prediction grounded in architectural properties of each paradigm, guide the investigation:

| ID | Research Question | Dimension | Predicted Direction |
|:--:|:-----------------|:----------|:-------------------|
| **RQ1** | Do SMT and LLM translations differ in lexical diversity, as measured by standardized type-token ratio (STTR)? | Lexical Diversity | LLM > SMT. LLMs are trained on diverse corpora and access a broader vocabulary during generation; SMT is constrained by the coverage of its phrase table (Koehn, 2010). |
| **RQ2** | Do SMT and LLM translations differ in sentence length distribution? | Sentence Complexity | Distributions differ in shape. SMT decoders impose length penalties and n-gram language model constraints that produce more uniform sentence lengths; LLMs lack explicit length control and produce distributions closer to human writing. |
| **RQ3** | Do SMT and LLM translations differ in average sentiment polarity? | Sentiment | Difference is expected to be small. Both architectures tend toward neutralization in translation, but LLMs may preserve more of the source text's affective coloring due to their attention-based contextual modeling. |
| **RQ4** | Do SMT and LLM translations differ in part-of-speech (POS) entropy, a measure of syntactic diversity? | Stylometry | LLM > SMT. LLMs generate more varied syntactic structures; SMT is biased toward fixed phrase templates extracted from parallel data. |

---
---

## 3.2 Experimental Design

### 3.2.1 Factorial Structure

The experiment follows a **2 × 2 × 2 completely between-texts factorial design** with three independent variables:

| Factor | Levels | Description |
|:-------|:------|:------------|
| **Architecture (A)** | SMT, LLM | Translation system used to generate the output text |
| **Language Direction (B)** | ZH→EN, EN→ZH | Source-to-target language pair |
| **Genre (C)** | News, Literary | Discourse genre of the source text |

Each of the 8 cells (2 × 2 × 2) contains 10 source texts, yielding:

- **80 source texts** in total (20 per cell before translation)
- **160 translated outputs** (each source text translated by both architectures)
- **160 analysis units** (the unit of statistical analysis is the full translated text)

### 3.2.2 Sample Size and Power Analysis

Sample size was determined through a prospective power analysis for a two-sided two-sample *t*-test (Cohen's *d* = 0.8, α = 0.05, power = 0.80), which yields a required minimum of 21 observations per group (calculated in `statsmodels.stats.power.TTestIndPower`). With *n* = 40 per architecture group (80 source texts × 1 translation per architecture, collapsed across directions and genres for the main effect test), the study is adequately powered. For interactions and stratified analyses (e.g., within a single genre), *n* = 20 per cell provides power ≥ 0.70 for detecting large effects (*d* ≥ 0.8).

> **Table 1: Power analysis parameters**
>
> | Parameter | Value |
> |:----------|:------|
> | Expected effect size (Cohen's *d*) | 0.8 |
> | Significance level (α) | 0.05 (Holm-corrected per hypothesis family) |
> | Statistical power (1 − β) | 0.80 |
> | Required *n* per group (two-sample *t*-test) | 21 |
> | Actual *n* per architecture (RQ1 main effect) | 40 |
> | Actual *n* per cell (individual condition) | 10 |

### 3.2.3 Randomization and Blinding

Source texts are assigned to experimental conditions by their natural language and genre (non-random assignment, as these are fixed factors). The translation system factor is fully crossed: every source text is translated by both architectures. Outcome assessment through feature extraction is fully automated, eliminating assessor bias.

---
---

## 3.3 Source Text Corpus

### 3.3.1 Corpus Composition

The corpus comprises **80 source texts**, balanced across language (Chinese originals, English originals) and genre (news, literary). Table 2 summarizes the composition.

> **Table 2: Source text corpus composition**
>
> | Genre | Chinese Originals | English Originals | Total |
> |:------|:-----------------:|:-----------------:|:-----:|
> | **News** | 20 texts | 20 texts | 40 |
> | **Literary** | 20 texts | 20 texts | 40 |
> | **Total** | 40 | 40 | **80** |

### 3.3.2 Genre Definitions

**News texts** are defined as published news reports or articles from established journalistic sources. Selection criteria include: (a) original publication date after January 1, 2023 (to minimize overlap with LLM training data), (b) expository journalistic style, and (c) length between 200–800 words (or Chinese character equivalent). Topics span politics, economics, technology, and culture.

**Literary texts** are defined as fiction or creative non-fiction originally intended for artistic expression. Selection criteria include: (a) published literary works or author-approved manuscripts, (b) narrative or descriptive style with dialog and exposition, and (c) length between 200–800 words (or Chinese character equivalent).

### 3.3.3 Source Languages

- **Chinese original texts** (40 texts): 20 news articles from Xinhua News Agency public reports; 20 literary excerpts from Can Xue's *Terracotta Warriors* (兵马俑) and the experimenters' original Chinese short fiction.
- **English original texts** (40 texts): 20 news articles accessed via the NYT Archive API; 20 literary short stories from Project Gutenberg or the experimenters' original English writing.

### 3.3.4 Length Control

All texts are constrained to 200–800 English words (or the Chinese character equivalent, approximately 300–1,200 Chinese characters). This range ensures each text contains enough sentences (typically 8–40) for reliable sentence-length distribution analysis, sentence-level sentiment segmentation, and lexical diversity estimation, while remaining within the context window of all translation systems employed.

### 3.3.5 Acquisition Strategy

| Source Type | Acquisition Method | Timeline |
|:------------|:-------------------|:---------|
| Chinese news | Xinhua public access; automated scraping of post-2023 articles | Week 1 |
| Chinese literary | Manual extraction from existing digital manuscripts (.docx) | Week 1 |
| English news | NYT Archive API query for post-2023 articles | Week 1 |
| English literary | Project Gutenberg short stories + experimenters' original writing | Week 1 |

---
---

## 3.4 Translation Systems

### 3.4.1 Traditional SMT Condition: Moses

The SMT condition employs **Moses** (Koehn et al., 2007), a well-established phrase-based statistical machine translation system, deployed via the `amake/moses-smt` Docker image.

| Component | Tool | Details |
|:----------|:-----|:--------|
| Word alignment | GIZA++ | IBM Model 4, bidirectional alignment with grow-diag-final-and symmetrization |
| Phrase extraction | Moses built-in | Extract phrases consistent with word alignment; phrase table scoring using relative frequency and lexical weighting |
| Language model | KenLM (Heafield, 2011) | 5-gram with modified Kneser-Ney smoothing; trained on News Crawl monolingual data (target language) |
| Tuning | MERT (Minimum Error Rate Training) | Optimize feature weights against BLEU score on a held-out development set |
| Decoding | Moses cube pruning | Phrase-based beam search |

**Training data**: WMT news-commentary v16–v18 parallel corpus (~300K sentence pairs) for both ZH→EN and EN→ZH directions. Monolingual News Crawl data for target-language KenLM training.

**Expected performance**: BLEU scores of approximately 20–25 on held-out test data for both directions, consistent with reported WMT results for phrase-based SMT on this language pair.

**Translation procedure**:
```bash
docker pull amake/moses-smt
# Tokenization → truecasing → cleaning
# GIZA++ word alignment (bidirectional)
# Phrase extraction and scoring
# KenLM 5-gram language model
# MERT tuning
echo "source sentence" | docker run -i --rm \
  -v $(pwd)/model:/model amake/moses-smt \
  moses -f /model/moses.ini
```

**Training time**: Estimated 2–4 hours per direction on CPU (~100K parallel sentences).

### 3.4.2 LLM Condition: DeepSeek V4 Flash

The LLM condition employs **DeepSeek V4 Flash**, a decoder-only large language model accessed via the OpenAI-compatible OpenCode Go API.

| Parameter | Value |
|:----------|:------|
| Model ID | `opencode-go/deepseek-v4-flash` |
| Context window | 1,000,000 tokens |
| Temperature | 0.0 (deterministic output; removes stochasticity as a confound) |
| System prompt | "You are a professional translator." |
| User prompt | Uniform template (see below), identical across all 160 translation tasks |

**Unified prompt template** (identical for all translations):
```
Translate the following [genre] text from [source language] to [target language].
Output only the translation, without any explanations or notes. Preserve the original paragraph structure.

[full source text]
```

**API call** (Python):
```python
import openai
client = openai.OpenAI(
    base_url="https://opencode.ai/zen/go/v1",
    api_key="sk-..."
)
response = client.chat.completions.create(
    model="opencode-go/deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a professional translator."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.0
)
```

**Backup models** (if DeepSeek V4 is unavailable): GLM-5.1 (OpenCode Go, 203K context) or Kimi K2.6 (OpenCode Go, 256K context).

### 3.4.3 Determinism Across Conditions

The LLM temperature parameter is set to 0.0 for all translations, ensuring fully deterministic outputs (barring API-side non-determinism in batched decoding). The SMT condition is fully deterministic by nature. This design choice eliminates within-system output variability as a confound, isolating architecture as the sole independent variable.

---
---

## 3.5 Feature Extraction Pipeline

All 160 translated texts are processed through a uniform automated pipeline that extracts features across four linguistic dimensions. The pipeline is implemented in Python 3.11 and produces a consolidated CSV matrix (see Appendix B for schema).

### 3.5.1 Lexical Diversity

Three complementary measures of lexical diversity are computed. **STTR is designated as the primary indicator**; MTLD and HD-D serve as convergent validity checks.

| Metric | Abbreviation | Definition | Implementation |
|:-------|:-------------|:-----------|:---------------|
| **Standardized Type-Token Ratio** | **STTR** | Mean TTR across consecutive 1,000-token windows | `lexical_diversity.sttr(text)` |
| Measure of Textual Lexical Diversity | MTLD | Mean length of sequential word strings that maintain TTR ≥ 0.72 | `lexical_diversity.mtld(text)` |
| Hypergeometric Distribution *D* | HD-D | Sum of per-type probabilities of occurrence under the hypergeometric distribution | `lexical_diversity.hdd(text)` |

**Formula — STTR:**

$$\text{STTR} = \frac{1}{K} \sum_{k=1}^{K} \frac{|V_k|}{N_k}$$

where $K$ is the number of 1,000-token windows, $|V_k|$ is the number of unique types in window $k$, and $N_k$ is the token count in window $k$ (fixed at 1,000).

**Preprocessing for Chinese texts**: Chinese outputs are first tokenized with `jieba` (for consistency) before lexical diversity computation. English outputs are tokenized on whitespace with punctuation separated.

### 3.5.2 Sentence Complexity

Sentence boundaries are detected using spaCy's sentence segmentation (`zh_core_web_sm` for Chinese, `en_core_web_sm` for English). For each text, the sequence of sentence lengths (in tokens) is characterized by:

- **Central tendency**: Mean sentence length ($\bar{x}$)
- **Dispersion**: Standard deviation of sentence length ($s$)
- **Distribution shape**: Skewness ($\gamma_1$) and excess kurtosis ($\gamma_2$)
- **Cross-architecture distribution comparison**: Two-sample Kolmogorov–Smirnov *D* statistic (comparing SMT vs. LLM sentence length distributions within each source text pair)

**Optional syntactic metric**: Maximum dependency parse depth, computed from spaCy's dependency tree, is extracted for a subset of analyses.

### 3.5.3 Sentiment

Sentiment is computed using the **XLM-RoBERTa** multilingual sentiment classifier (`cardiffnlp/twitter-xlm-roberta-base-sentiment`), which outputs three-class probabilities (positive, neutral, negative) for each input segment. The model supports both Chinese and English with a shared representation space.

For each text, the pipeline extracts:

- **Mean class probabilities**: $\bar{p}_{\text{pos}}$, $\bar{p}_{\text{neu}}$, $\bar{p}_{\text{neg}}$ (averaged across sentences)
- **Polarity score**: $\text{Pol} = \bar{p}_{\text{pos}} - \bar{p}_{\text{neg}}$ (range [−1, +1])
- **Sentiment volatility**: $\sigma_{\text{Pol}} = \sqrt{\frac{1}{S} \sum_{s=1}^{S} (\text{Pol}_s - \overline{\text{Pol}})^2}$, where $S$ is the number of sentences

**Implementation:**
```python
from transformers import pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)
```

### 3.5.4 Stylometry

Stylometric features capture syntactic and functional patterns of the text.

**POS tag distribution**: spaCy's universal POS tagger (18 tags: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X) produces a probability vector $\mathbf{p} = (p_1, \ldots, p_{18})$ for each text, where $p_i$ is the proportion of tokens tagged with POS tag $i$.

**POS Shannon entropy:**

$$H(\mathbf{p}) = -\sum_{i=1}^{18} p_i \log_2 p_i$$

where $\mathbf{p}$ is the POS tag probability vector. Higher entropy indicates greater syntactic diversity. Shannon entropy is computed via `scipy.stats.entropy`.

**Function word ratio**: Predefined function word lists are compiled for each language:

| Language | Source | Size |
|:---------|:-------|:----:|
| English | Universal function word list (the, a, of, in, to, etc.) | ~150 words |
| Chinese | Function word list (的, 了, 在, 是, 有, 和, 就, 等) | ~100 words |

The function word ratio is $f_{\text{func}} = \frac{N_{\text{func}}}{N_{\text{total}}}$, the proportion of tokens matching the function word list.

**Cross-text cosine similarity** (for classification validation): For any two texts $i$ and $j$ with POS probability vectors $\mathbf{p}_i$ and $\mathbf{p}_j$:

$$\text{cosim}(\mathbf{p}_i, \mathbf{p}_j) = \frac{\mathbf{p}_i \cdot \mathbf{p}_j}{\|\mathbf{p}_i\| \|\mathbf{p}_j\|}$$

### 3.5.5 Output Data Format

The pipeline produces a CSV file with the following schema (one row per translated text):

| Column | Type | Description |
|:-------|:-----|:------------|
| `filename` | str | Source text identifier |
| `architecture` | {smt, llm} | Translation system |
| `direction` | {zh2en, en2zh} | Language direction |
| `genre` | {news, literary} | Source genre |
| `src_lang` | {zh, en} | Source language |
| `tgt_lang` | {zh, en} | Target language |
| `sttr` | float | Standardized type-token ratio |
| `mtld` | float | Measure of textual lexical diversity |
| `hdd` | float | Hypergeometric distribution *D* |
| `mean_sent_len` | float | Mean sentence length (tokens) |
| `sd_sent_len` | float | SD of sentence length |
| `sent_skew` | float | Skewness of sentence length distribution |
| `ks_statistic` | float | KS *D* comparing SMT vs. LLM distributions for the source text |
| `pos_entropy` | float | Shannon entropy of POS tag distribution |
| `func_word_ratio` | float | Proportion of function words |
| `sent_pos` | float | Mean positive sentiment probability |
| `sent_neg` | float | Mean negative sentiment probability |
| `sent_neu` | float | Mean neutral sentiment probability |
| `sent_volatility` | float | Standard deviation of sentence-level polarity scores |
| `bleu` | float | BLEU score against reference (if available) |

---
---

## 3.6 Statistical Analysis Plan

The analysis proceeds in five sequential stages.

### 3.6.1 Stage 1: Descriptive Statistics

For each cell of the factorial design (8 cells), we report means, standard deviations, and 95% bootstrap confidence intervals (10,000 resamples, bias-corrected accelerated method). Results are visualized as faceted boxplots showing the distribution of each dependent measure across architecture × direction × genre combinations.

### 3.6.2 Stage 2: Hypothesis Testing

Four principal hypotheses are tested, each corresponding to one research question. Analyses are stratified by language direction (ZH→EN and EN→ZH conducted independently) to avoid conflating cross-linguistic effects with architecture effects.

> **Table 3: Hypothesis testing framework**
>
> | Hypothesis | Dependent Variable | Null Hypothesis | Test | Model |
> |:----------:|:-------------------|:----------------|:-----|:------|
> | **H1** | STTR (lexical diversity) | $\mu_{\text{SMT}} = \mu_{\text{LLM}}$ | Two-way ANOVA | STTR ~ Architecture × Genre |
> | **H2** | Sentence length distribution | Distributions are identical | Two-sample KS test | $D(\text{SMT}, \text{LLM})$ per text pair |
> | **H3** | Polarity score $P_{\text{pos}} - P_{\text{neg}}$ | $\mu_{\text{SMT}} = \mu_{\text{LLM}}$ | Two-way ANOVA or Mann–Whitney *U* | Polarity ~ Architecture × Genre |
> | **H4** | POS Shannon entropy | $\mu_{\text{SMT}} = \mu_{\text{LLM}}$ | Two-way ANOVA | Entropy ~ Architecture × Genre |

**Model specification for ANOVAs:**

$$\text{DV} = \beta_0 + \beta_1 \cdot \text{Architecture} + \beta_2 \cdot \text{Genre} + \beta_3 \cdot (\text{Architecture} \times \text{Genre}) + \varepsilon$$

where Architecture and Genre are categorical fixed effects. The primary test is the Architecture main effect ($\beta_1$). The Architecture × Genre interaction ($\beta_3$) is a secondary, exploratory test.

**Assumption checking**: Before ANOVA, normality of residuals is assessed with the Shapiro–Wilk test (at α = 0.05) and homogeneity of variance with Levene's test. If either assumption is violated, a Welch ANOVA (without homogeneity assumption) or non-parametric Mann–Whitney *U* test is substituted.

### 3.6.3 Stage 3: Effect Sizes

All significant results are reported with effect sizes and 95% confidence intervals:

- **ANOVA**: Partial eta-squared ($\eta_p^2$), with interpretation benchmarks: small = 0.01, medium = 0.06, large = 0.14 (Cohen, 1988).
- **KS test**: Cohen's *d* approximation via $d \approx D \cdot \sqrt{\frac{n_1 n_2}{n_1 + n_2}}$ (where $D$ is the KS statistic).
- **Mann–Whitney *U***: Rank-biserial correlation $r = 1 - \frac{2U}{n_1 n_2}$.

### 3.6.4 Stage 4: SVM Classification Validation

As a convergent validation of any significant univariate findings, a linear support vector machine (SVM) classifier is trained to predict architecture (SMT vs. LLM) from the full feature vector (all 12 continuous features from Section 3.5). Classification is performed separately for each language direction.

- **Cross-validation strategy**: GroupKFold with *K* = 10, where groups are defined by source text identity. This ensures that the same source text's SMT and LLM outputs never appear in different folds, preventing data leakage.
- **Feature scaling**: Standardization to zero mean and unit variance within each fold.
- **Evaluation metrics**: Accuracy, macro-averaged F₁ score, and Matthews Correlation Coefficient (MCC), each reported with 95% CI across folds.
- **Chance baseline**: Permutation test (1,000 label shuffles) to establish the null distribution of each metric.

### 3.6.5 Stage 5: Multiplicity Correction

The four hypothesis tests (H1–H4) are treated as a single family within each language direction. The **Holm–Bonferroni** procedure (Holm, 1979) controls the family-wise error rate at α = 0.05:

1. Sort *p*-values in ascending order: $p_{(1)} \leq p_{(2)} \leq p_{(3)} \leq p_{(4)}$.
2. Reject $H_{(i)}$ if $p_{(i)} \leq \frac{0.05}{4 - i + 1}$.
3. Stop at the first non-rejection.

This correction is applied independently within each language direction (two families of four tests each). No correction is applied across the two directions, as they involve different populations of source texts and different translation directions.

### 3.6.6 Sensitivity Analyses

Three sensitivity analyses assess the robustness of findings:

1. **Metric convergence**: Check whether STTR, MTLD, and HD-D produce the same directional conclusions for H1.
2. **Outlier influence**: Re-run all tests after excluding texts with any feature value exceeding ±3 standard deviations from the cell mean.
3. **Length confound**: Stratify texts by length (short: <400 tokens; long: ≥400 tokens) and test whether architecture effects interact with text length.

---
---

## 3.7 Pre-registration

Prior to any translation output generation, the study design is registered on the **Open Science Framework (OSF)** at [osf.io](https://osf.io). The pre-registration includes:

1. **Source text list**: Complete inventory of 80 source texts with title, source, word count, and language.
2. **Exact statistical tests**: Specification of each hypothesis with its corresponding test, model formula, and effect size metric.
3. **Correction method**: Holm–Bonferroni procedure, as specified in Section 3.6.5.
4. **Exclusion criteria**: Texts are excluded from analysis only if: (a) the translation system produces empty output, (b) output is truncated (≤50% of source length), or (c) output is in the wrong language (determined by automatic language identification). Exclusions are reported with reasons in all cases.
5. **Primary endpoint**: STTR is designated as the primary dependent measure for lexical diversity (H1), with MTLD and HD-D as secondary confirmatory measures.

**Pre-registration timing**: Week 2 of the project timeline (during Moses model training), before any translation outputs are generated.

---
---

## 3.8 Implementation Timeline

The experiment is executed over an 8-week period according to the following schedule:

> **Table 4: Project timeline**
>
> | Week | Phase | Tasks | Deliverables |
> |:----:|:------|:------|:-------------|
> | **1** | Preparation | Install Moses Docker and validate training pipeline; collect 80 source texts; configure OpenCode API credentials | Source text corpus (80 texts in `.txt`); API access verified |
> | **2** | Pre-registration | Register on OSF; download WMT training data; begin Moses training | Pre-registration completed; training data staged |
> | **3** | Training | Complete Moses model training (ZH↔EN); validate translation quality; verify LLM API response format | Trained Moses models (ZH→EN, EN→ZH); API translation script finalized |
> | **4** | Generation | Moses batch translation (80 texts × 2 directions); LLM batch translation via API (80 texts × 2 directions) | 160 translated texts (80 SMT + 80 LLM); quality check log |
> | **5** | Feature extraction | Run unified feature extraction pipeline (lexical, sentence, sentiment, stylometry) | Feature matrix CSV (160 rows × 19 columns) |
> | **6** | Statistical analysis | Execute Stage 1–5 analyses (descriptive, hypothesis tests, effect sizes, SVM, sensitivity); generate visualizations | Analysis report; figure drafts |
> | **7** | Writing | Draft Introduction, Methods, Results, and Discussion sections | Full manuscript first draft |
> | **8** | Revision and submission | Revise based on co-author feedback; finalize figures and tables; submit | Camera-ready manuscript |

### Resource Requirements

| Resource | Estimate | Notes |
|:---------|:---------|:------|
| Moses training | 2–4 hours per direction | CPU-based; ~100K parallel sentences |
| LLM API cost | ~$5–10 total | ~160 translations × ~500 tokens avg |
| Storage | < 1 GB | Source texts, translations, features, models |
| Software | All open-source | Moses (LGPL), spaCy (MIT), scikit-learn (BSD) |

---
---

## References

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

Heafield, K. (2011). KenLM: Faster and smaller language model queries. In *Proceedings of the Sixth Workshop on Statistical Machine Translation* (pp. 187–197).

Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.

Koehn, P. (2010). *Statistical Machine Translation*. Cambridge University Press.

Koehn, P., Hoang, H., Birch, A., et al. (2007). Moses: Open source toolkit for statistical machine translation. In *Proceedings of the 45th Annual Meeting of the ACL* (pp. 177–180).

McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381–392.
