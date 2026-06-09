# 论文 LaTeX 大纲：跨架构机器翻译文本统计比较

> **用途**: 统计课项目, 4–6 页会议论文, 2026-06-09
> **基于**: 实验协议 v1.0 + 已完成特征矩阵 (833 rows) + 已完成分析
> **最后更新**: 2026-06-09

---

## 总体结构概览

| 章节 | 估计页数 | 估计词数 | 数据就绪度 | 缺失 |
|:-----|:-------:|:-------:|:--------:|:----|
| 1. 引言 (Introduction) | 0.6 | 400 | ✅ 就绪 | — |
| 2. 方法 (Methods) | 1.2 | 800 | ✅ 就绪 | 需精简 (原文3个md ~1000行) |
| 3. 结果 (Results) | 1.8 | 1200 | ✅ 已完成 | 需生成正式图表 |
| 4. 讨论 (Discussion) | 0.6 | 400 | ⚠️ 部分 | 需基于结果撰写 |
| 5. 结论 (Conclusion) | 0.2 | 150 | ⚠️ 部分 | 需基于讨论撰写 |
| 参考文献 | 0.4 | — | ✅ 已有 | — |
| **合计** | **~5** | **~3000** | — | — |

---

## 详细大纲

---

### Section 1: Introduction

#### 1.1 Research Problem (~150 words)

**内容**:
- 机器翻译两大范式: 传统统计机器翻译 (SMT, phrase-based) 与大型语言模型 (LLM, decoder-only Transformer)
- 两种架构在原理上根本不同: SMT 基于离散短语表 + n-gram 语言模型 + 束搜索, LLM 基于连续表示 + 自回归生成
- 已有研究重点比较 SMT 与 LLM 的 BLEU 翻译质量, 但很少关注两者译文的**语言统计特征差异**
- 提出研究问题: 两种架构产生的译文在词汇丰富度、句法复杂度、情感倾向、风格计量学四个维度上是否存在统计显著差异?

**数据需求**:
- [x] 已有: literature_comparison.md 中的 gap analysis (Sect 2.3.6)
- [x] 已有: 实验设计中的 research questions
- [ ] 待补充: 引入句 — 用 1-2 句概括当前 SMT vs LLM 比较研究的现状 (引用 Castells 2025, Dugan 2024 等)

#### 1.2 Contributions (~100 words)

**内容** (3 点):
1. **首次系统比较 SMT 与 LLM 译文的语言统计特征** — 前人工作仅比较人类文本 vs LLM 文本, 未涉及翻译领域
2. **受控 2×2×2 析因设计** — 架构 × 方向 × 体裁, 匹配源文本, 排除体裁/方向混淆
3. **多维度特征矩阵** — 词汇 (STTR/MTLD/HD-D), 句法 (句长分布), 情感 (XLM-RoBERTa), 风格 (POS 熵/功能词比), 共 19 维特征
4. **6 个 LLM 模型的内部比较** — Tukey HSD 检验 6 个模型的差异 (次要分析)

**数据需求**: 无 (纯声明性内容)

#### 1.3 Paper Organization (~50 words)

**内容**: 简要说明后续结构 (Section 2: Methods, Section 3: Results, Section 4: Discussion)

---

### Section 2: Methods

> ⚠️ **重要**: 此节必须严格压缩。原始 material (paper/ 目录下 3 个 md 共 ~1200 行) 包含大量 SMT 实现细节 (7 个 bug, 5 轮迭代等), 仅 10-20% 需进入会议论文。焦点应放在**实验设计、特征提取流水线、统计方法**。

#### 2.1 Experimental Design (~200 words)

**2.1.1 Factorial Structure**
- 2 (Architecture: SMT vs LLM) × 2 (Direction: ZH→EN vs EN→ZH) × 2 (Genre: News vs Literary)
- 80 source texts, 20 per cell
- Each source text translated by 1 SMT system + 6 LLM models → ~560 translations (actual: 833 including SMT variants)

**2.1.2 Source Text Corpus**
- 80 texts: 40 Chinese originals + 40 English originals
- News (40): Xinhua (ZH) + NYT Archive API (EN), post-2023
- Literary (40): Can Xue + other authors (ZH), Project Gutenberg + own writing (EN)
- Length control: 200–800 words / 300–1,200 Chinese characters

**2.1.3 Power Analysis**
- Target: Cohen's d = 0.8, α = 0.05 (Holm-corrected), power = 0.80
- Required n ≥ 21 per group for two-sample t-test
- Actual n = 40 per architecture (collapsed) → adequately powered

**数据需求**:
- [x] 已有: experiment_design.md 中的完整设计描述
- [x] 已有: power analysis 参数
- [ ] **缺失/需确认**: SMT 使用哪个变体? (smt / smt_fa / smt_sym_v2?) — **需在 contact_supervisor 中确认**

**图表需求**:
- Table 1: Factorial design cell breakdown (2×2×2 = 8 cells × n per cell)
- [ ] 待生成: `paper/table_design.tex`

#### 2.2 Translation Systems (~200 words)

**2.2.1 SMT System**
- Custom phrase-based SMT: IBM2 word alignment + gdfa symmetrization + phrase extraction + Kneser-Ney 3-gram LM + beam search decoder
- Training data: 213K WMT news-commentary v12 Chinese-English parallel corpus (50K actively used)
- Final model: fast_align HMM + gdfa, 65,909 phrase pairs
- Deterministic output (no temperature/noise)

**2.2.2 LLM Systems**
- 6 decoder-only LLMs accessed via OpenCode Go API:
  - deepseek-v4-pro, deepseek-v4-flash, glm-5.1, glm-5, kimi-k2.6, qwen3.6-plus
- Temperature = 0.0 (deterministic), identical system prompt ("You are a professional translator.")
- Uniform translation prompt template across all 480 translation tasks

**数据需求**:
- [x] 已有: methodology.md (SMT 详细描述), experiment_design.md §3.4 (LLM 描述)
- [ ] **缺失**: 需确认论文中选用哪个 SMT 变体作为 "the SMT condition" (smt_fa / smt_sym_v2 / 聚合?)
- [ ] **缺失**: qwen3.6-plus 缺失 7 篇译文 (73/80), 论文中需注明

**图表需求**:
- Table 2: Translation systems compared (SMT specs + 6 LLM model IDs, context window sizes, pricing)

#### 2.3 Feature Extraction (~200 words)

**2.3.1 Lexical Diversity**: STTR, MTLD, HD-D (3 metrics); STTR = primary
**2.3.2 Sentence Complexity**: Mean, SD, skewness, kurtosis of sentence length (spaCy segmentation)
**2.3.3 Sentiment**: XLM-RoBERTa multilingual classifier (cardiffnlp/twitter-xlm-roberta-base-sentiment); polarity score = pos − neg
**2.3.4 Stylometry**: POS Shannon entropy (spaCy 18 universal tags), function word ratio
**2.3.5 Supplementary**: α-ratio, punctuation ratio, digit ratio

**数据需求**:
- [x] 已有: experiment_design.md §3.5 中的完整流水线描述
- [x] 已有: feature_matrix.csv 中的 19 个特征列
- [ ] **需生成**: Feature table listing all 14+ extracted features with definitions and implementations

#### 2.4 Statistical Analysis Plan (~200 words)

**2.4.1 Primary Analysis**: Mixed-effects model (lme4 / statsmodels MixedLM)
- Fixed effects: Architecture × Direction × Genre
- Random intercept: source_text_id (to control for paired design)
- DV: STTR (primary), Mean Sentence Length, Sentiment Polarity, POS Entropy (secondary)

**2.4.2 Secondary Analysis**: 
- Two-sample Kolmogorov-Smirnov tests for distribution-level differences
- 6 LLM models × Tukey HSD post-hoc comparison
- SVM classification (linear, GroupKFold by source text ID, 10-fold) for multivariate architecture separation

**2.4.3 Effect Sizes**: Cohen's d for pairwise comparisons; η²_p for ANOVA; MCC for SVM

**2.4.4 Multiplicity Correction**: Holm-Bonferroni per hypothesis family (4 tests per direction × 2 directions = 8 tests)

**数据需求**:
- [x] 已有: experiment_design.md §3.6 中的完整分析计划
- [x] 已有: mixedlm_results.json + analysis_results.json (分析已完成)
- [ ] **待生成**: 正式统计检验结果表 (含 p 值, 效应量, CI)

**图表需求**:
- Table 3: Hypothesis testing framework (H1–H4, DV, test statistic, correction)

---

### Section 3: Results

> **全部分析已完成** (server_results/), 需要的是: 将现有 JSON 结果格式化为正式 LaTeX 表格和图片。

#### 3.1 Descriptive Statistics (~250 words)

**内容**:
- Overall means and SDs for each feature by architecture (SMT vs LLM)
- Stratified by direction and genre
- Key observations highlighted in text

**数据需求**:
- [x] 已有: analysis_results.json 包含所有 cell 的 mean/SD
- [x] 已有: paper_table1.tex (overall) + paper_table2.tex (by direction)

**图表需求**:
- **Table 1** (现存 `paper_table1.tex`): 总体特征对比 (需补充 SD 和 CI)
  - Columns: Feature | SMT Mean (SD) | LLM Mean (SD) | Cohen's d | KS D | p-value
  - Rows: STTR, MTLD, HD-D, Mean Sent Len, SD Sent Len, Sent Polarity, POS Entropy, Func Word Ratio
- **Figure 1**: Faceted boxplots / violin plots for STTR across 2 (architecture) × 2 (direction) × 2 (genre) — 8 subplots
  - [x] 生成数据已有: feature_matrix.csv
  - [ ] **需生成**: `paper/fig1_boxplots_sttr.png` / `.pdf`
- **Figure 2**: Sentence length distribution histograms (SMT overlaid on LLM), faceted by direction
  - [x] 生成数据已有: server_results/mean_sent_len_hist.png
  - [ ] **需升级**: 当前为简易 matplotlib PNG, 需改为 publication-quality 版本

#### 3.2 Hypothesis Tests (~350 words)

**内容** (按 4 个假设组织):

**H1 (Lexical Diversity)**:
- STTR: SMT (0.95) >> LLM (0.74), Cohen's d = +2.83, KS D = 0.92, p < 0.001
- 方向交互: en2zh 差异更大 (d = +5.98) than zh2en (d = +1.88)
- MTLD and HD-D convergent validity

**H2 (Sentence Complexity)**:
- Mean sentence length: LLM (9.3 tokens) >> SMT (3.5 tokens), d = −0.86, p < 0.001
- Strong direction effect: zh2en (d = −5.61) vs en2zh (d = −0.43, ns)
- KS test: D = 0.49, p < 0.001 (distributions differ in shape, not just location)

**H3 (Sentiment)**:
- Polarity: LLM (0.29) more positive than SMT (0.07), d = −0.46, p < 0.001
- Only in zh2en (d = −0.79); en2zh shows no difference (both near zero)

**H4 (Stylometry)**:
- POS entropy: SMT (0.46) vs LLM (0.47), d = −0.02, p < 0.001 (significant but negligible)
- Function word ratio follows similar pattern

**数据需求**:
- [x] 已有: analysis_results.json 中每个假设的完整统计量
- [x] 已有: paper_table1.tex 和 paper_table2.tex 片段
- [ ] **需生成**: 正式的 LaTeX 表格, 包含所有 strata 的 Cohen's d + CI + Holm-corrected p

**图表需求**:
- **Table 2**: Primary hypothesis test results (H1–H4), with p-values, effect sizes, 95% CI
  - [ ] 需从 analysis_results.json 提取并格式化

#### 3.3 Mixed-Effects Model (~200 words)

**内容**:
- MixedLM with source_text_id as random intercept → accounts for paired design
- Model formula: Feature ~ Architecture + Direction + Genre + Architecture:Direction + Architecture:Genre
- Key finding: **Only STTR is robustly significant** after controlling for random effects
  - STTR: β_smt = +0.27, p < 0.001
  - POS entropy: β_smt = −0.04, p = 0.009 (marginal)
  - Mean sent len: β_smt = −0.001, p = 0.996 (ns after controlling direction)
  - Sentiment: ns, Function words: ns
- Interpretation: Architecture differences are driven primarily by lexical diversity

**数据需求**:
- [x] 已有: mixedlm_results.json 中的完整系数和 p 值
- [ ] **需补充**: Random effects variance estimates (σ²_source_text, σ²_residual)
- [ ] **需补充**: AIC/BIC for model comparison (null model vs full model)

**图表需求**:
- **Table 3**: Mixed-effects model coefficients with 95% CI and p-values
  - [ ] 需从 mixedlm_results.json 扩展

#### 3.4 SVM Classification (~150 words)

**内容**:
- Linear SVM, GroupKFold (10-fold by source text ID), standardized features
- Accuracy: 98.0% ± 1.4% (chance = 50%, permutation p < 0.001)
- F1: 98.3% ± 1.1%, MCC: 0.96 ± 0.03
- Top features by weight: STTR (−2.18), Mean Sent Len (+1.89), Func Word Ratio (−1.22)
- Interpretation: Near-perfect separability confirms architectural divergence

**数据需求**:
- [x] 已有: svm_results.json 中的完整指标
- [x] 已有: server_results/svm_confusion.png
- [ ] **需升级**: confusion matrix 改为 publication-quality

**图表需求**:
- **Figure 3**: SVM confusion matrix (heatmap)
  - [x] 原始图存在: server_results/svm_confusion.png
  - [ ] 需改为 colorblind-friendly palette, add percentage labels
- **Figure 4 (optional)**: Feature importance bar chart (SVM coefficients)
  - [ ] 可从 svm_results.json 的 feature_weights 生成

#### 3.5 LLM Inter-Model Comparison (~150 words)

**内容**:
- Tukey HSD comparing 6 LLM models on STTR and Mean Sentence Length
- Most pairwise differences small (d < 0.3 within LLM group)
- Exceptions: glm-5 vs deepseek-v4-flash on STTR? qwen3.6-plus vs others? (需查看实际 Tukey 结果)
- Key conclusion: inter-LLM variance << architecture-level variance

**数据需求**:
- [ ] **缺失**: Tukey HSD 结果未在现有 JSON 中找到 — **需确认是否已计算**
- [ ] 若未计算: 需运行 TukeyHSD 或 pairwise_tukeyhsd 在 STTR 上, 按 model 分组

**图表需求**:
- **Table 4**: Tukey HSD post-hoc comparison matrix (6×6) for STTR
  - [ ] 待生成
- **Figure 5 (optional)**: Forest plot of model-level STTR means with 95% CI

---

### Section 4: Discussion

#### 4.1 Summary of Findings (~150 words)

**内容**:
- SMT and LLM translations are statistically distinguishable across all four dimensions
- Lexical diversity (STTR) is the most discriminative feature (d = +2.83)
- Sentence length distribution is the second most discriminative (d = −0.86)
- However, after controlling for paired design (mixed model), only STTR remains robustly significant
- SVM achieves 98% accuracy, confirming multivariate separability

#### 4.2 Interpretation (~150 words)

**4.2.1 Why does SMT have higher lexical diversity?**
- SMT phrase table recombination: even a "deterministic" SMT decoder has many valid phrase segmentations → higher type diversity
- LLM over-reliance on frequent tokens during deterministic greedy decoding (temperature = 0)
- Consistent with Gude & Santos-Ríos (2025): alignment training reduces LLM output diversity

**4.2.2 Why are LLM sentences longer?**
- SMT imposes explicit length/word penalties; LLMs do not
- LLMs learn to produce "fluent" translation → longer, more natural sentences
- zh2en direction shows extreme difference because SMT fails on Chinese segmentation → many short fragments

**4.2.3 Why no sentiment difference in en2zh?**
- Chinese sentiment expressed differently (lexical rather than syntactic markers)
- XLM-RoBERTa may be less sensitive to Chinese sentiment nuances
- SHORT Chinese texts (1 token per sentence in SMT output) yield zero sentiment signal

**4.2.4 Why near-perfect SVM classification?**
- STTR alone provides near-perfect separation
- Adding sentence length makes classification essentially trivial
- Implication: one could build a simple, interpretable detector (STTR threshold) to distinguish translation origin

#### 4.3 Limitations (~100 words)

1. **Temperature = 0**: LLM results may not generalize to stochastic sampling
2. **Single SMT system**: Our custom SMT is simpler than Moses (no lexicalized reordering, no IBM4 alignment)
3. **Missing qwen3.6-plus data**: 7/80 literary translations missing → slight bias toward other LLMs for en2zh literary
4. **Source text diversity**: Only 2 genres, 2 languages; cannot generalize to other domains/language pairs
5. **No human reference**: Cannot distinguish which architecture produces "better" translations — only that they produce *different* translations
6. **Statistical vs practical significance**: POS entropy is statistically significant (p < 0.001) but Cohen's d = −0.02 (negligible)

#### 4.4 Implications and Future Work (~100 words)

- For NLP practitioners: STTR is a lightweight diagnostic for detecting SMT vs LLM translations
- For translation quality assessment: feature-level analysis complements BLEU scores
- Future: extend to other architectures (encoder-decoder NMT, hybrid systems), other genres, other language pairs
- Future: test if stochastic LLM decoding (temperature > 0) closes the lexical diversity gap
- Future: investigate whether downstream tasks (classification, summarization) are affected by translation architecture

---

### Section 5: Conclusion

(~100 words)

- Summary: SMT and LLM translations are fundamentally different in their statistical fingerprint
- Lexical diversity is the most salient differentiator
- The architecture signal is strong enough for near-perfect classification
- Implications for MT system selection and evaluation

---

## 参考文献

**已有条目** (来自 literature_comparison.md):
1. Koehn et al. (2007) — Moses
2. Dyer et al. (2010) — cdec
3. Li et al. (2009) — Joshua
4. Cer et al. (2010) — Phrasal
5. Tong et al. (2013) — NiuTrans.SMT
6. Och (2003) — MERT
7. Chiang (2007) — Hiero
8. Chang et al. (2008) — Chinese segmentation
9. McCarthy & Jarvis (2010) — MTLD/HD-D
10. Zhu et al. (2024) — Human vs LLM news
11. Gude & Santos-Ríos (2025) — LLM diversity
12. Reinhart et al. (2025) — Grammatical styles
13. Castells et al. (2025) — Stylometry detection
14. Dugan et al. (2024) — RAID benchmark
15. Brown et al. (1993) — IBM models
16. Chen & Goodman (1998) — Kneser-Ney smoothing
17. Dyer et al. (2013) — fast_align
18. Cohen (1988) — Statistical power
19. Holm (1979) — Multiple testing

**需补充的引用**:
- XLM-RoBERTa (Barbieri et al., 2022 or Conneau et al., 2020)
- spaCy (Honnibal et al., 2020)
- Jieba (Sun, 2012 - "Jieba Chinese text segmentation")

---

## 数据状态总结

### ✅ 已有且已完成 (不需要额外工作)

| 项目 | 状态 | 位置 |
|:-----|:----:|:-----|
| 特征矩阵 (833 rows × 25 cols) | ✅ | `data/feature_matrix.csv` |
| 描述统计 (mean, SD, Cohen's d) | ✅ | `server_results/analysis_results.json` |
| KS 检验 | ✅ | 同上 |
| 混合效应模型 | ✅ | `server_results/mixedlm_results.json` |
| SVM 分类 (98%) | ✅ | `server_results/svm_results.json` |
| LaTeX 表格片段 (2 张) | ✅ | `server_results/paper_table1.tex`, `paper_table2.tex` |
| 直方图 (6 张, 简易 matplotlib) | ✅ | `server_results/*.png` |
| 方法学描述 (SMT + 实验设计) | ✅ | `paper/01_methodology.md`, `paper/03_experiment_design.md` |
| 文献对比与引用 | ✅ | `paper/02_literature_comparison.md` |
| 源文本 (80 篇) | ✅ | `data/source_texts/` |
| 译文文件 (LLM: 946 个 / 960 目标) | ✅ | `translations/` |

### ⚠️ 需确认/补充 (在撰写前)

| 项目 | 状态 | 优先级 |
|:-----|:----:|:------:|
| **SMT 哪个变体用作主条件?** (smt / smt_fa / smt_sym_v2 / aggregated?) | ⚠️ 需确认 | 🔴 HIGH |
| **Tukey HSD 6-LLM 间比较** | ⚠️ 已有分析中未找到 → 需确认 | 🟡 MEDIUM |
| qwen3.6-plus 缺失 7 篇 (literary, en2zh?) → 论文中如何处理? | ⚠️ 需记录 | 🟡 MEDIUM |
| Mixed model random effects variance (σ²) / AIC | ⚠️ 未在 JSON 中找到 | 🟢 LOW |

### ❌ 缺失 (需在撰写前/中生成)

| 项目 | 说明 | 优先级 |
|:-----|:-----|:------:|
| **正式 LaTeX 表格** (4-5 张) | 需从 JSON 生成 publication-quality 表格 | 🔴 HIGH |
| **Publication-quality 图表** (3-5 张) | 升级 server_results 中的简易 matplotlib PNG | 🔴 HIGH |
| Introduction 草稿 | 需从 research questions + gap analysis 改写 | 🟡 MEDIUM |
| Discussion 草稿 | 需基于结果撰写, 可参考 rough analysis in EXPERIMENT_REPORT | 🟡 MEDIUM |
| Abstract (150 词) | 最后写 | 🟢 LOW |
| 补充引用 (XLM-RoBERTa, spaCy, jieba) | 3 条引用 | 🟢 LOW |

---

## 图表清单 (论文中需要)

| ID | 类型 | 标题 | 数据来源 | 状态 |
|:--:|:----:|:-----|:--------|:----:|
| **T1** | Table | Experimental design: factorial cell breakdown | experiment_design.md §3.2.1 | ❌ 待生成 |
| **T2** | Table | Translation systems specifications | §2.2 | ❌ 待生成 |
| **T3** | Table | Descriptive statistics by architecture (w/ SD, CI) | analysis_results.json | ⚠️ 有片段, 需补全 |
| **T4** | Table | Hypothesis test results (H1–H4, w/ Holm-corrected p) | analysis_results.json | ❌ 待生成 |
| **T5** | Table | Mixed-effects model coefficients | mixedlm_results.json | ⚠️ 有片段, 需补全 |
| **T6** | Table | SVM classification metrics | svm_results.json | ❌ 待生成 |
| **T7** | Table | Tukey HSD matrix (6 LLM × 6 LLM) | 待计算 | ⚠️ 待确认 |
| **F1** | Figure | STTR boxplots: 2×2×2 faceted | feature_matrix.csv | ❌ 待生成 |
| **F2** | Figure | Sentence length histograms (SMT vs LLM overlay) | feature_matrix.csv | ⚠️ 有初版, 需升级 |
| **F3** | Figure | SVM confusion matrix | svm_results.json | ⚠️ 有初版, 需升级 |
| **F4** | Figure | SVM feature weights bar chart | svm_results.json | ❌ 待生成 |
| **F5** | Figure | Forest plot: 6 LLM model STTR means w/ 95% CI | feature_matrix.csv | ❌ 待生成 |

---

## LaTeX 模板建议

使用 ACL 会议模板 (适用于 4-6 页统计课项目):
- `\documentclass[11pt,a4paper]{article}` 或 ACL-specific style
- `\usepackage{booktabs}` (表格)
- `\usepackage{graphicx}` (图片)
- `\usepackage{siunitx}` (数值格式化)
- `\usepackage{natbib}` 或 `biblatex` (引用)
- `\usepackage{hyperref}` (超链接)

替代方案: 使用 `acmart` 格式 (ACM 会议), 或标准 `article` + 双栏。

---

## 撰写优先级 (建议执行顺序)

1. **Phase 1 (数据确认)**: 确认 SMT 变体选择 + Tukey 结果
2. **Phase 2 (图表生成)**: 生成所有 T1-T5 表格 + F1-F4 图片 (Python/matplotlib) → 这是 LaTeX 撰写的先行条件
3. **Phase 3 (结果草稿)**: Section 3 (Results) — 基于图表, 大部分是数字转文字
4. **Phase 4 (方法精简)**: Section 2 (Methods) — 从 3 个 md 文件压缩到 800 词
5. **Phase 5 (引言+讨论)**: Section 1 + 4 — 需更多创造性写作
6. **Phase 6 (最终)**: Abstract, 排版, 引用检查

---

*End of outline.*
