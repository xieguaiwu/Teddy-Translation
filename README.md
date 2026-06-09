---
license: mit
language:
- zh
- en
tags:
- translation
- smt
- statistical-machine-translation
- phrase-based
- fast-align
- kneser-ney
- traditional-ml
- classic-nlp
pipeline_tag: translation
library_name: custom
model_type: phrase-based-smt
---

# 🧸 Teddy — 短语级统计机器翻译 | Phrase-Based SMT

<p align="center">
  <a href="#english">🇬🇧 English</a> &nbsp;|&nbsp; <a href="#chinese">🇨🇳 中文</a>
</p>

---

<a id="english"></a>

## 🇬🇧 English <sub><a href="#chinese">[切换到中文]</a></sub>

**Teddy** is a **pure-Python phrase-based statistical machine translation (SMT) system** for Chinese↔English, built entirely from scratch. It implements the complete classical SMT pipeline — word alignment (IBM Model 2 + fast_align HMM), phrase extraction, Kneser-Ney N-gram language modeling, and beam-search decoding — with **zero neural network dependencies**. Just NumPy, SciPy, and a C++ fast_align binary.

> 📄 **Research context**: This system was built as the **traditional SMT baseline** for a cross-architecture text statistics comparison experiment (SMT vs LLM translation). The experiment showed SMT and LLM outputs are **statistically distinguishable with 98% SVM accuracy**.

---

### 🔧 Quick Start

```bash
pip install numpy scipy spacy nltk sacrebleu pyyaml scikit-learn
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

```python
from smt.decoder import PhraseDecoder
from smt.language_model import KneserNeyLM
from smt.phrase_table import load_phrase_table

lm = KneserNeyLM.load("models/zh2en_sym/lm.json")
pt = load_phrase_table("models/zh2en_sym/phrase_table.txt")
decoder = PhraseDecoder(lm, pt)

print(decoder.translate("企业推动协议"))
# → "enterprise promoted agreement"
```

---

### 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│               Pipeline (smt.pipeline)                │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│ DataPrep │ Aligner  │  Phrase  │ Language │ Decoder │
│ Tokenize  │ IBM2 /  │  Table   │ Model    │ Beam    │
│ & Clean  │ fast_align│ Extract │ Kneser-  │ Search  │
│          │ + gdfa   │ + Score  │ Ney 3-gram│ + Recomb│
└──────────┴──────────┴──────────┴──────────┴─────────┘
```

| Module | File | Description |
|:-------|:-----|:------------|
| `smt.ibm_align` | IBM Model 1+2 | EM training, Viterbi alignment |
| `smt.align_fast` | fast_align wrapper | HMM alignment + atools symmetrization |
| `smt.phrase_table` | Phrase extraction | 4 features: φ(f\|e), φ(e\|f), lex(f\|e), lex(e\|f) |
| `smt.language_model` | Kneser-Ney LM | 3-gram, JSON + pickle, backoff smoothing |
| `smt.decoder` | Beam search | Recombination, future cost, N-best list |
| `smt.pipeline` | Orchestration | End-to-end train/translate |

---

### 📊 Available Models

| Direction | Model | Align | Phrases | LM vocab | Notes |
|:----------|:------|:------|:--------|:---------|:------|
| **ZH→EN** | sym | IBM2+gdfa | 8,705 | 77K | Semi-readable |
| **ZH→EN** | fa-50K | fast_align+gdfa | 65,909 | 77K | Best 50K |
| **ZH→EN** | fa-213K | fast_align+gdfa | **396,738** | **124K** | 🌟 Largest (LM enhanced) |
| **EN→ZH** | sym | IBM2+gdfa | 8,729 | 81K | Semi-readable |
| **EN→ZH** | fa-50K | fast_align+gdfa | **68,228** | 81K | 🌟 Best 50K |
| **EN→ZH** | fa-213K | fast_align+gdfa | 423,009 | **134K** | Largest (LM enhanced) |

**Key insight**: More phrases ≠ better quality. The enhanced LM (BOOKS + WMT, 124K vocab) significantly improves lexical diversity but OOV on literary text remains challenging.

---

### 🧪 Experiment Results (SMT vs LLM)

| Metric | Effect Size (Cohen's d) | Direction |
|:-------|:-----------------------|:----------|
| **STTR** (lexical diversity) | **2.83** (very large) | SMT more diverse |
| **Sentence length** | −0.86 (large) | LLM 2.6× longer |
| **Sentiment polarity** | −0.46 (medium) | LLM more positive |
| **POS entropy** | 0.20 (small) | Similar |
| **SVM classification** | **98% accuracy** | 🔬 Clearly separable |

> Full results: `server_results/` directory.

---

### 🗂 Repository Structure

```
teddy/
├── smt/               # Core SMT library (5.3K lines)
│   ├── ibm_align.py   # IBM Model 1+2
│   ├── align_fast.py  # fast_align wrapper
│   ├── phrase_table.py# Phrase extraction & scoring
│   ├── language_model.py# Kneser-Ney LM
│   ├── decoder.py     # Beam search decoder
│   └── pipeline.py    # End-to-end pipeline
├── scripts/           # Training, evaluation, batch translation
│   ├── train_symmetrized.py
│   ├── train_fastalign.py
│   ├── llm_batch_translate.py
│   └── extract_features.py
├── models/            # Pre-trained models (Git LFS)
├── app.py             # Gradio demo
├── paper/             # Research paper drafts
├── notes/             # Dev documentation & context
├── server_results/    # Full experiment analysis
└── data/              # Training data (excluded)
```

---

### 📜 License & Acknowledgements

**MIT License**. Training data: WMT news-commentary v12 (permissive research license).

- [fast_align](https://github.com/clab/fast_align) — Dyer et al. 2013
- [Kneser-Ney smoothing](https://en.wikipedia.org/wiki/Kneser–Ney_smoothing) — Kneser & Ney 1995
- WMT Workshop on Statistical Machine Translation

<p align="center">
  <a href="#english">⬆ Back to top</a> &nbsp;|&nbsp; <a href="#chinese">🇨🇳 中文版本</a>
</p>

---

<a id="chinese"></a>

## 🇨🇳 中文 <sub><a href="#english">[Switch to English]</a></sub>

**Teddy** 是一个从零构建的**纯 Python 短语级统计机器翻译系统**，支持中英互译。它完整实现了经典的 SMT 流水线——词对齐（IBM Model 2 + fast_align HMM）、短语抽取、Kneser-Ney N-gram 语言模型和束搜索解码器——**完全无需神经网络依赖**，仅需 NumPy、SciPy 和一个 C++ fast_align 可执行文件。

> 📄 **研究背景**：本项目作为**传统 SMT 基线系统**，用于跨架构文本统计比较实验（SMT 与 LLM 翻译输出的统计特征对比）。实验结果表明：SMT 和 LLM 输出在统计上**可区分性达 98% SVM 准确率**。

---

### 🔧 快速开始

```bash
pip install numpy scipy spacy nltk sacrebleu pyyaml scikit-learn
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

```python
from smt.decoder import PhraseDecoder
from smt.language_model import KneserNeyLM
from smt.phrase_table import load_phrase_table

lm = KneserNeyLM.load("models/zh2en_sym/lm.json")
pt = load_phrase_table("models/zh2en_sym/phrase_table.txt")
decoder = PhraseDecoder(lm, pt)

print(decoder.translate("企业推动协议"))
# → "enterprise promoted agreement"
```

---

### 🏗 系统架构

```
┌─────────────────────────────────────────────────────┐
│               Pipeline (smt.pipeline)                │
├──────────┬──────────┬──────────┬──────────┬─────────┤
│ 数据清洗 │ 词对齐    │ 短语抽取  │ 语言模型  │ 解码器  │
│ 分词     │ IBM2 /   │ + 评分   │ Kneser-  │ 束搜索  │
│ & 过滤   │ fast_align│ 4个特征  │ Ney 3-gram│ 未来成本│
│         │ + gdfa   │          │          │ 重组    │
└──────────┴──────────┴──────────┴──────────┴─────────┘
```

| 模块 | 文件 | 功能 |
|:-----|:-----|:-----|
| `smt.ibm_align` | IBM Model 1+2 | EM 训练，Viterbi 对齐 |
| `smt.align_fast` | fast_align 包装 | HMM 对齐 + atools 对称化 |
| `smt.phrase_table` | 短语抽取 | 4 特征：φ(f\|e), φ(e\|f), 双向词汇化 |
| `smt.language_model` | Kneser-Ney LM | 3-gram，JSON + pickle，回退平滑 |
| `smt.decoder` | 束搜索解码 | 重组（recombination），未来成本估计，N-best 列表 |
| `smt.pipeline` | 流程编排 | 端到端训练 / 翻译 |

---

### 📊 可用模型

| 方向 | 模型 | 对齐 | 短语数 | LM 词表 | 说明 |
|:-----|:-----|:-----|:-------|:--------|:-----|
| **中→英** | sym | IBM2+gdfa | 8,705 | 77K | 半可读 |
| **中→英** | fa-50K | fast_align+gdfa | 65,909 | 77K | 最佳 50K |
| **中→英** | fa-213K | fast_align+gdfa | **396,738** | **124K** | 🌟 最大（LM 增强） |
| **英→中** | sym | IBM2+gdfa | 8,729 | 81K | 半可读 |
| **英→中** | fa-50K | fast_align+gdfa | **68,228** | 81K | 🌟 最佳 50K |
| **英→中** | fa-213K | fast_align+gdfa | 423,009 | **134K** | 最大（LM 增强） |

**核心发现**: 短语多≠质量好。增强 LM（BOOKS 文学 + WMT 新闻，124K 词表）大幅提升词汇丰富度，但文学文本 OOV 问题仍存在。

---

### 🧪 实验结论（SMT vs LLM）

| 统计维度 | 效应量 (Cohen's d) | 方向 |
|:---------|:------------------|:-----|
| **STTR**（词汇丰富度） | **2.83**（极大） | SMT 词汇更多样 |
| **句长** | −0.86（大） | LLM 句子长 2.6× |
| **情感极性** | −0.46（中） | LLM 更积极 |
| **POS 信息熵** | 0.20（小） | 差异不大 |
| **SVM 分类** | **98% 准确率** | 🔬 SMT 与 LLM 明确可区分 |

> 完整结果见 `server_results/` 目录。

---

### 🗂 仓库结构

```
teddy/
├── smt/               # SMT 核心库（5.3K 行）
│   ├── ibm_align.py   # IBM Model 1+2
│   ├── align_fast.py  # fast_align 包装
│   ├── phrase_table.py# 短语抽取与评分
│   ├── language_model.py# Kneser-Ney 语言模型
│   ├── decoder.py     # 束搜索解码器
│   └── pipeline.py    # 端到端流水线
├── scripts/           # 训练、评估、批量翻译
│   ├── train_symmetrized.py
│   ├── train_fastalign.py
│   ├── llm_batch_translate.py
│   └── extract_features.py
├── models/            # 预训练模型（Git LFS）
├── app.py             # Gradio 演示
├── paper/             # 论文草稿
├── notes/             # 开发文档与上下文
├── server_results/    # 完整实验分析结果
└── data/              # 训练数据（不包含在仓库中）
```

---

### 📜 许可证与致谢

**MIT 许可证**。训练数据：WMT news-commentary v12（学术研究许可）。

- [fast_align](https://github.com/clab/fast_align) — Dyer et al. 2013
- [Kneser-Ney smoothing](https://en.wikipedia.org/wiki/Kneser–Ney_smoothing) — Kneser & Ney 1995
- WMT 统计机器翻译研讨会

<p align="center">
  <a href="#chinese">⬆ 返回顶部</a> &nbsp;|&nbsp; <a href="#english">🇬🇧 English Version</a>
</p>
