# 🧸 Teddy — 短语级统计机器翻译系统 (ZH↔EN)

<p align="center">
  <a href="README.md">🇬🇧 English</a>
</p>

**Teddy** 是一个从零构建的**纯 Python 短语级统计机器翻译系统**，支持中英互译。它完整实现了经典的 SMT 流水线——词对齐（IBM Model 2 + fast_align HMM）、短语抽取、Kneser-Ney N-gram 语言模型和束搜索解码器——**完全无需神经网络依赖**，仅需 NumPy、SciPy 和一个 C++ fast_align 可执行文件。

## 快速开始

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

## 系统架构

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

## 可用模型

| 方向 | 模型 | 对齐 | 短语数 | LM 词表 | 说明 |
|:-----|:-----|:-----|:-------|:--------|:-----|
| **中→英** | sym | IBM2+gdfa | 8,705 | 77K | 半可读 |
| **中→英** | fa-50K | fast_align+gdfa | 65,909 | 77K | 最佳 50K |
| **中→英** | fa-213K | fast_align+gdfa | **396,738** | **124K** | 🌟 最大（LM 增强） |
| **英→中** | sym | IBM2+gdfa | 8,729 | 81K | 半可读 |
| **英→中** | fa-50K | fast_align+gdfa | **68,228** | 81K | 🌟 最佳 50K |
| **英→中** | fa-213K | fast_align+gdfa | 423,009 | **134K** | 最大（LM 增强） |

**核心发现**: 短语多≠质量好。增强 LM（BOOKS 文学 + WMT 新闻，124K 词表）大幅提升词汇丰富度，但文学文本 OOV 问题仍存在。

## 仓库结构

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
│   ├── demo_smt.py
│   └── eval_bleu.py
├── models/            # 预训练模型（Git LFS）
├── app.py             # Gradio 演示
└── config.yaml        # 训练配置
```

## 许可证

**MIT 许可证**。训练数据：WMT news-commentary v12（学术研究许可）。

## 致谢

- [fast_align](https://github.com/clab/fast_align) — Dyer et al. 2013
- [Kneser-Ney smoothing](https://en.wikipedia.org/wiki/Kneser–Ney_smoothing) — Kneser & Ney 1995
- WMT 统计机器翻译研讨会

<p align="center">
  <a href="README.md">🇬🇧 English</a>
</p>
