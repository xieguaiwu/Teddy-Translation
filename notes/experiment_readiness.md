# Experiment Readiness Assessment — Brutally Honest

> Date: 2026-06-06
> Assessor: Automated critical review
> Scope: SMT baseline readiness for the cross-architecture MT statistical comparison experiment

---

## Executive Verdict: ❌ NOT READY — Fatal Design Flaws

The experiment is **not viable in its current form**. Two independently fatal problems exist, plus a cascade of secondary issues. Do not proceed to data collection or statistical analysis until these are resolved.

---

## 1. Fatal Flaw #1: Source Texts Are Single Sentences

### Evidence

Every one of the 80 source texts is a **single sentence** (~10-20 words, 55-95 bytes):

| Category | Files | Total bytes | Avg bytes/text | Sentences/text |
|:---------|:-----:|:-----------:|:--------------:|:--------------:|
| zh_news | 20 | 1,301 | 65 | 1 |
| zh_lit | 20 | 1,532 | 77 | 1 |
| en_news | 20 | 1,498 | 75 | 1 |
| en_lit | 20 | 1,582 | 79 | 1 |

### Why This Kills the Experiment

The protocol (§2.2) explicitly requires **"200-800 英文词（或等量中文）"** per text. The actual texts average ~15 words. This makes **every single analysis in §4-5 impossible or meaningless**:

| Metric | Protocol Intent | Reality with Single-Sentence Input |
|:-------|:----------------|:-----------------------------------|
| **STTR** (§4.1) | 1000-word window TTR mean | Single window of ~15 tokens → STTR ≈ 1.0 for all texts. No variance to compare. |
| **MTLD** (§4.1) | Factor mean across TTR-threshold segments | 1 segment per text. Degenerates to TTR. Statistically useless. |
| **Sentence length distribution** (§4.2) | Distribution across multiple sentences | 1 data point per text. You cannot compute a "distribution," skew, or kurtosis from n=1. |
| **KS test on sentence lengths** (§5.1 H2) | Two-sample KS test of distributions | Each category's "distribution" is just 20 individual sentence lengths. The KS test becomes a test of whether two architectures produce different average sentence lengths *for the same single-sentence input* — which is tautological, not insightful. |
| **Sentiment volatility** (§4.3) | Variance of per-sentence sentiment scores | Undefined (n=1 sentence per text). The pipeline will return 0 or NaN. |
| **POS entropy** (§4.4) | Shannon entropy over POS tags | Computable, but based on ~10-15 tokens. Entropy on tiny samples is heavily biased and unreliable. |
| **Func word ratio** (§4.4) | Ratio of function words | With 10 words, a single function word difference shifts the ratio by 10%. No statistical power. |

### The Protocol Itself Contradicts This

The protocol's power analysis (§5.3) assumes **d≈0.8 effect size, n=10 per cell**. This calculation implicitly assumes each observation has *stable, low-noise measurements*. STTR computed from 15 tokens is not stable. MTLD from 1 segment is not low-noise. You're feeding garbage measurements into a power analysis that assumes clean ones.

### Bottom Line

You could run the pipeline and get numbers. They would be **statistically nonsensical**. A reviewer with basic literacy in lexical diversity metrics will spot this in 30 seconds. This is a desk-reject-level flaw.

---

## 2. Fatal Flaw #2: SMT Output Is Not a "Translation Baseline" — It's Word Salad

### News Domain (the "good" case)

**Source (zh_news/001):** 美国总统今日宣布了一项新的经济政策，旨在促进就业增长。
> "The US president announced a new economic policy today, aimed at promoting employment growth."

**SMT output:** `President announced New economy policy , aimed promoting employment growth . has`

**Source (en_news/001):** The president announced a major infrastructure plan to boost the economy.

**SMT output:** `。提振经济基础设施计划向一个重大该总统宣布`
> Literal back-translation: ". Boost economy infrastructure plan toward a major the president announce"

**Assessment:** Keywords are partially preserved, but word order is scrambled, function words are dropped or misplaced, and outputs read like a drunk person's telegram. This is **not a "phrase-based SMT baseline"** in any recognized sense — it's a partially-trained model that never converged.

### Literary Domain (the "bad" case)

**Source (zh_lit/001):** 春风拂过湖面，泛起层层涟漪，柳枝轻摇，仿佛在诉说着古老的故事。
> "Spring breeze brushes the lake surface, ripples spread in layers, willow branches sway gently, as if telling an ancient story."

**SMT output:** `, , In of story . ,`

**Source (zh_lit/002):** 他站在桥上，望着远方的山峦，心中涌起无限的感慨。
> "He stood on the bridge, gazing at distant mountains, his heart surging with boundless emotion."

**SMT output:** `his In at , of , of .`

**Assessment:** This is not translation. This is a Markov chain with a 3-gram language model spitting out the most common English function words. The content words are entirely gone because the SMT phrase table has zero coverage of literary vocabulary trained on 50K WMT news sentences. **The model is effectively producing random draws from a unigram distribution of common function words.**

### Quantifying the SMT Output Quality

| Category | Avg output bytes | Avg source bytes | Output/source ratio | Typical word count |
|:---------|:----------------:|:----------------:|:-------------------:|:------------------:|
| zh_news (ZH→EN) | 53 | 65 | 81% | 5-10 words |
| zh_lit (ZH→EN) | 28 | 77 | 37% | 2-5 words |
| en_news (EN→ZH) | 44 | 75 | 59% | 3-8 chars |
| en_lit (EN→ZH) | 39 | 79 | 50% | 4-8 chars |

The literary ZH→EN output is less than half the source length and consists primarily of punctuation and function words. You are comparing near-empty strings to LLM output and calling it a "statistical comparison of architectures."

---

## 3. Answering the Four Specific Questions

### Q1: Will the statistical comparison produce scientifically valid results?

**No.** The results will be:

1. **STTR:** SMT output will have *higher* STTR (approaching 1.0) because 5-word outputs have almost no repeated tokens. LLM output will have *lower* STTR because it generates actual text with natural lexical repetition. The result shows up as "statistically significant" but means the **opposite** of what the hypothesis predicts. This is a measurement artifact, not a feature of the architecture.

2. **Sentence length:** SMT output is uniformly short (2-8 words). LLM output varies. ANOVA will produce p < 0.001. The conclusion "SMT produces shorter sentences" is trivially true but scientifically empty — you've demonstrated that a broken model produces broken output.

3. **Sentiment:** The zh_lit SMT output contains no content words to classify. Sentiment scores will be near-uniform neutral. LLM output will vary. Again, highly significant but meaningless.

4. **POS entropy:** SMT output POS tags will be dominated by punctuation and determiners. Entropy will appear *lower* than LLM — but this simply reflects that your SMT model can't generate content words.

**The experiment will produce "significant" results on every dimension. Not because architectures differ in interesting ways, but because your baseline is broken.** You're measuring failure modes, not architectural properties.

### Q2: What's the minimum fix needed to make the literary domain usable?

There is no minimum fix. The problem is structural:

- **Training data mismatch:** Your SMT model is trained on 50K WMT news-commentary sentences. Literary Chinese uses completely different vocabulary, syntax, and register. No amount of tinkering with alignment algorithms or epsilon values will make a news-trained model translate literary text.

- **The fix requires:** (a) A literary parallel corpus — at minimum 50K-100K sentence pairs of literary Chinese→English and English→Chinese. This corpus does not exist in easily accessible form. (b) Or, use the actual Moses/GIZA++ pipeline with a much larger WMT corpus (the full ~25M sentence pairs from UN + News Commentary + WikiMatrix), which might incidentally cover some literary vocabulary through sheer scale.

- **Estimated effort:** 2-4 weeks of data acquisition + 1-2 weeks of training + debugging. This is a full project on its own.

### Q3: Would it be better to JUST use news-domain texts (40 pairs) and skip literary?

**Better than proceeding with literary, but still scientifically insufficient.**

Three problems remain even with news-only:

1. **Single-sentence inputs still destroy all metrics.** Same fatal flaw as above — STTR, MTLD, sentence-length distributions are meaningless on single sentences regardless of domain.

2. **The SMT output is still broken.** Even the "best" news outputs show scrambled word order and missing content. A reviewer will ask: "Is this really what phrase-based SMT produces? Or is this what happens when a student project fails to converge?" The distinction matters. If your SMT is so bad that it's not representative of the architecture, your entire comparison is invalid.

3. **Loss of experimental design.** The protocol specifies a 2×2×2 factorial design with 10 observations per cell. Dropping the genre factor reduces it to 2×2 (architecture × direction) with 20 per cell. You lose:
   - The architecture × genre interaction (which was your strongest predicted effect per §6.2)
   - The ability to claim genre-generalizability
   - Statistical power for detecting direction-specific effects

### Q4: What would a reviewer criticize about the current experimental design?

Here's what a reviewer would say, ranked by severity:

#### 🔴 Desk-Reject Level

1. **"Your source texts are single sentences. Your protocol says 200-800 words. This is a fundamental data integrity problem."**
   - The protocol was presumably approved by the instructor. If the submitted paper used these texts while claiming they meet spec, this is academic dishonesty. If the protocol was not approved, the entire experiment is unregistered and ad-hoc.

2. **"Your SMT system does not produce translations. It produces fragments. You cannot compare a broken system to a functional one and attribute differences to architecture."**
   - The phrase table comes from 50K WMT sentences. A reasonable phrase-based SMT trained on 50K news sentences should produce BLEU ~15-25 on in-domain text. Your model's actual output suggests it never learned to reorder phrases or handle unknown words properly. The model is buggy, not "traditional."
   - The critical_review.md claims BLEU≈8 and "半可理解" (semi-comprehensible). This is generous. The actual output is not semi-comprehensible — it's word salad with occasional keyword preservation.

3. **"You cannot compute STTR, MTLD, sentence-length distributions, or sentiment volatility from single-sentence texts. Your entire statistical pipeline is invalid regardless of translation quality."**

#### 🟡 Major Revision Level

4. **"Your SMT is home-built rather than using the standard Moses pipeline specified in the protocol. How do we know your implementation is correct?"**
   - The protocol (§3.1) specifies Moses + GIZA++ + KenLM + MERT. The actual system is a custom Python implementation with IBM2 (not IBM4) alignment, custom phrase extraction, custom decoder, and grid-search tuning instead of Och MERT.
   - The critical_review.md acknowledges this as "partially compliant" (8/10). A reviewer will not be this generous. They will ask: "Prove your custom system is behaviorally equivalent to Moses." You cannot.

5. **"Your training data (50K WMT news-commentary) is 6× smaller than the protocol's minimum (~300K). Your own experiment (213K training) showed no quality improvement, which suggests a bug in your training pipeline, not that 'data is not the bottleneck.'"**
   - The protocol says "~300K 句." You used 50K. The 213K experiment showed only +4% phrase count and no quality improvement. A properly functioning SMT training pipeline would show significant improvements going from 50K→213K. The fact that yours didn't is evidence of a pipeline bug (likely in the symmetrization/gdfa step), not evidence that data doesn't matter.

6. **"Your EN→ZH direction produces outputs with Chinese characters in scrambled order. Chinese is a word-order-sensitive language. A baseline that can't get word order right isn't a baseline — it's noise."**

#### 🟢 Minor/Clarification Level

7. **"Your source texts appear to be teacher-composed rather than extracted from real publications. For example, zh_news/010 '教育部宣布将增加对农村学校的教育投入' reads like a textbook example, not a NYT/Xinhua excerpt."**
   - The protocol §2.2 says sources should come from "NYT / 新华社公开文章（2023年后，确保不在LLM训练数据中）." The actual texts look like they were written by hand for this experiment. This matters because (a) authentic texts have different statistical properties, and (b) hand-written texts may inadvertently be simpler/more regular, inflating apparent SMT performance.

8. **"You have not implemented the feature extraction pipeline (§4.1-4.4). The protocol specifies it should be complete by Week 5. Has any feasibility testing been done?"**

9. **"The protocol mentions 'SVM (linear kernel) classification with GroupKFold.' With n=10 per cell and ~8 features, you have 80 observations total. Training an SVM on 80 samples with ~8 features is borderline underpowered for classification, especially with GroupKFold leaving-one-group-out."**

10. **"Your protocol was finalized on June 3. The experiment is supposed to take 8 weeks. You've spent 3 days building the SMT system and are asking if you can proceed. The rush is concerning."**

---

## 4. What Would Actually Need to Happen

If you're serious about this experiment, here's the minimum viable path:

### Option A: Full Reset (Recommended)

1. **Replace source texts** with actual multi-paragraph texts. Minimum 10 sentences, 200+ words each. Extract from real NYT/Xinhua articles and Project Gutenberg.
2. **Use actual Moses.** The `amake/moses-smt` Docker image exists. 50K WMT sentences through the full Moses pipeline (GIZA++ IBM4 → grow-diag-final-and → KenLM 5-gram → MERT) will produce BLEU 15-25 on news.
3. **Drop literary domain** if you can't find a parallel literary corpus. Replace with a single-domain study (news only, 2×2 design: architecture × direction) with 40 source texts.
4. **Implement the feature extraction pipeline** and validate it on test data before running the full batch.

### Option B: Salvage with Constraints (If time is severely limited)

1. **Replace source texts** with multi-paragraph texts (non-negotiable).
2. **Keep the custom SMT but fix the training pipeline.** The 213K experiment showing no improvement is a red flag. Debug the gdfa symmetrization. Use fast_align for better alignment quality. Train on the full 213K dataset.
3. **Drop literary domain.** Accept that the experiment becomes news-only.
4. **Acknowledge limitations explicitly** in the paper: custom SMT implementation, news-only domain, small training corpus. Frame as an exploratory study rather than a definitive comparison.

### Option C: Change the Research Question (Nuclear option)

If you can't fix the SMT or source texts in time:
- **Don't claim this is an "architecture comparison."** Your SMT isn't representative of phrase-based SMT.
- **Reframe as:** "A Case Study in Building and Diagnosing a Phrase-Based SMT System for Chinese-English Translation" — a methodology paper about what went wrong, why, and what the diagnostics revealed.
- This is actually a more honest and potentially more interesting paper given the extensive bug documentation and model evolution history you already have (7 bugs, 5 model iterations, alignment strategy comparison, MERT tuning).

---

## 5. Summary Severity Matrix

| Issue | Severity | Blocks experiment? | Fix effort |
|:------|:--------:|:-----------------:|:----------:|
| Single-sentence source texts | 🔴 FATAL | Yes — all metrics invalid | 1-2 days to collect real texts |
| SMT output is word salad | 🔴 FATAL | Yes — baseline is not a baseline | 1-3 weeks to retrain or switch to Moses |
| Protocol non-compliance (custom SMT, smaller data) | 🟡 MAJOR | Partial — fixable with documentation | Acknowledge and justify |
| No feature extraction pipeline | 🟡 MAJOR | Yes — can't compute metrics | 2-4 days to implement |
| Literary domain impossible without domain data | 🟡 MAJOR | Yes for full 2x2x2 design | Drop it or acquire literary corpus |
| No LLM translations to compare against | 🟡 MAJOR | Yes — the whole experiment | 1 day via API |
| Source texts look hand-composed | 🟢 MINOR | No | Replace with real articles |
| SVM underpowered with n=80 | 🟢 MINOR | No — analysis still runs | Acknowledge or use simpler classifier |

---

## Final Word

The SMT model-building effort was serious (7 bugs fixed, 5 iterations, multiple alignment strategies tested). The critical_review.md and FINAL_REPORT.md are thorough and honest. But the **experiment design has two independent fatal flaws** that cannot be papered over. Proceeding with the current source texts and SMT output would produce statistically significant but scientifically meaningless results that any competent reviewer would reject.

**Do not run the experiment in its current state.** Fix the source texts first (non-negotiable), then decide whether to fix the SMT or change the research question.
