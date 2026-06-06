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

# 🧸 Teddy — Phrase-Based Statistical Machine Translation (ZH↔EN)

> **传统统计机器翻译系统 | A classic SMT system built from scratch in Python**

Teddy is a **pure-Python phrase-based statistical machine translation system** for Chinese↔English. It implements the full SMT pipeline — word alignment, phrase extraction, language modeling, and beam search decoding — without any neural network dependencies. Just NumPy, SciPy, and a C++ fast_align binary.

---

## 🎯 Overview / 概述

| Direction | Model | Phrases | Quality |
|:----------|:------|:--------|:--------|
| ZH → EN | sym (IBM2+gdfa) | 8,705 | BLEU ≈ 8, semi-readable |
| EN → ZH | sym (IBM2+gdfa) | 8,729 | semi-readable |
| **EN → ZH** | **fast_align+gdfa** | **68,228** | **🌟 Best** |

**🇨🇳 中文**: Teddy 是一个纯 Python 实现的短语级统计机器翻译系统，支持中英互译。完整实现了词对齐（IBM2 + fast_align HMM）、短语抽取、Kneser-Ney 3-gram 语言模型和束搜索解码器。**无需任何神经网络依赖**。

---

## 🏗 Architecture / 架构

```
┌──────────────────────────────────────────────────────────┐
│                    Pipeline (主入口)                       │
├──────────┬──────────┬──────────┬──────────┬──────────────┤
│ DataPrep │ Aligner  │  Phrase  │ Language │   Decoder    │
│ 分词清洗  │ IBM2/HMM │  Table   │ Model    │  束搜索解码   │
│          │ 词对齐   │  短语抽取 │ N-gram   │              │
└──────────┴──────────┴──────────┴──────────┴──────────────┘
```

### Components / 核心组件

| Module | Description |
|:-------|:------------|
| `smt.ibm_align` | IBM Model 1+2 EM training + Viterbi alignment |
| `smt.align_fast` | C++ fast_align wrapper + atools symmetrization |
| `smt.phrase_table` | Phrase extraction with 4 score features (φ, lex bidirectional, penalties) |
| `smt.language_model` | Kneser-Ney smoothing N-gram LM (JSON+pickle dual format) |
| `smt.decoder` | Beam search decoder with recombination + future cost estimation |
| `smt.pipeline` | End-to-end training orchestration |

---

## 🚀 Usage / 使用

### Installation

```bash
pip install numpy scipy spacy nltk sacrebleu pyyaml scikit-learn
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

### Translate / 翻译

```python
from smt.decoder import PhraseDecoder
from smt.language_model import KneserNeyLM
from smt.phrase_table import load_phrase_table
from smt.config import Config

cfg = Config()

# Load models
lm = KneserNeyLM.load("models/zh2en_sym/lm.json")
pt = load_phrase_table("models/zh2en_sym/phrase_table.txt")
decoder = PhraseDecoder(cfg, lm, pt)

# Translate
result = decoder.translate("企业推动协议")
print(result)  # "enterprise promoted agreement"
```

### Train from scratch / 从零训练

```bash
# fast_align (HMM alignment, ~30s on 50K sentences)
python scripts/train_fastalign.py --direction zh2en --max-sentences 50000

# Symmetrized IBM2 alignment (~25min)
python scripts/train_symmetrized.py --direction zh2en --max-sentences 50000
```

---

## 📊 Model Evolution / 模型演进

```
v1 (synthetic 20K)  →  v2 (WMT 10K)  →  v3 (WMT 50K)  →  sym (50K+gdfa)  →  fast_align (50K)
     BLEU=0              gibberish         BLEU≈3          BLEU≈8           65K phrases
     unusable            unusable          word order      semi-readable    🌟 best
                                          混乱
```

**Key insight**: More data ≠ better quality. Symmetrized alignment (gdfa) removes noise at the cost of coverage. fast_align's HMM model achieves both **high quality × high coverage** (7.6× more phrases than sym).

---

## 📦 Models / 模型文件

| Path | Size | Description |
|:-----|:-----|:------------|
| `models/zh2en_sym/` | 127 MB | ZH→EN, IBM2+gdfa, 8.7K phrases |
| `models/en2zh_sym/` | 175 MB | EN→ZH, IBM2+gdfa, 8.7K phrases |
| `models/en2zh_fa/` | 191 MB | EN→ZH, fast_align+gdfa, 68K phrases 🌟 |

> **Note**: ZH→EN fast_align model was lost during repo cleanup. Retrain with `scripts/train_fastalign.py --direction zh2en`.

---

## 📝 Training Data / 训练数据

- **WMT news-commentary v12**: 50,000 Chinese↔English sentence pairs
- Standard WMT workshop data, permissive for research use
- Raw data excluded from this repo (download via `scripts/download_wmt_data.py`)

---

## 🔬 Research Context / 研究背景

This project was built as the **traditional SMT baseline** for a cross-architecture text statistics comparison experiment, comparing SMT (statistical) vs LLM (neural) translation outputs across lexical diversity, sentence complexity, and information-theoretic metrics.

---

## 📁 Repository Structure / 仓库结构

```
teddy/
├── smt/               # Core SMT library (5.3K lines)
├── scripts/           # Training, evaluation, batch translation
├── models/            # Pre-trained models (LFS tracked)
│   ├── zh2en_sym/     # ZH→EN sym model
│   ├── en2zh_sym/     # EN→ZH sym model
│   └── en2zh_fa/      # EN→ZH fast_align model (best)
├── app.py             # Gradio demo (Hugging Face Space)
├── config.json        # HF model metadata
├── paper/             # Research paper drafts
├── notes/             # Development documentation
│   ├── FINAL_REPORT.md
│   └── ...
└── data/              # Training data (excluded from Git)
```

---

## ⚖️ License / 许可证

MIT License. The WMT news-commentary training data is subject to its own permissive license.

---

## 🙏 Acknowledgements / 致谢

- **fast_align**: [Dyer et al. 2013](https://github.com/clab/fast_align) — HMM alignment model
- **WMT**: Workshop on Statistical Machine Translation — training data
- **Kneser-Ney smoothing**: Kneser & Ney 1995 — language model smoothing
