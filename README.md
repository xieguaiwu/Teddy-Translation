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

<p align="center">
  <a href="README.zh-CN.md">🇨🇳 中文版</a>
</p>

**Teddy** is a **pure-Python phrase-based statistical machine translation (SMT) system** for Chinese↔English, built entirely from scratch. It implements the complete classical SMT pipeline — word alignment (IBM Model 2 + fast_align HMM), phrase extraction, Kneser-Ney N-gram language modeling, and beam-search decoding — with **zero neural network dependencies**. Just NumPy, SciPy, and a C++ fast_align binary.

## Quick Start

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

## Architecture

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

## Available Models

| Direction | Model | Align | Phrases | LM vocab | Notes |
|:----------|:------|:------|:--------|:---------|:------|
| **ZH→EN** | sym | IBM2+gdfa | 8,705 | 77K | Semi-readable |
| **ZH→EN** | fa-50K | fast_align+gdfa | 65,909 | 77K | Best 50K |
| **ZH→EN** | fa-213K | fast_align+gdfa | **396,738** | **124K** | 🌟 Largest (LM enhanced) |
| **EN→ZH** | sym | IBM2+gdfa | 8,729 | 81K | Semi-readable |
| **EN→ZH** | fa-50K | fast_align+gdfa | **68,228** | 81K | 🌟 Best 50K |
| **EN→ZH** | fa-213K | fast_align+gdfa | 423,009 | **134K** | Largest (LM enhanced) |

**Key insight**: More phrases ≠ better quality. The enhanced LM (BOOKS + WMT, 124K vocab) significantly improves lexical diversity but OOV on literary text remains challenging.

## Repository Structure

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
│   ├── demo_smt.py
│   └── eval_bleu.py
├── models/            # Pre-trained models (Git LFS)
├── app.py             # Gradio demo
└── config.yaml        # Training configuration
```

## License

**MIT License**. Training data: WMT news-commentary v12 (permissive research license).

## Acknowledgements

- [fast_align](https://github.com/clab/fast_align) — Dyer et al. 2013
- [Kneser-Ney smoothing](https://en.wikipedia.org/wiki/Kneser–Ney_smoothing) — Kneser & Ney 1995
- WMT Workshop on Statistical Machine Translation

<p align="center">
  <a href="README.zh-CN.md">🇨🇳 中文版</a>
</p>
