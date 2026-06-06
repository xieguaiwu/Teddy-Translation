"""
Teddy SMT — Hugging Face Space Gradio Demo

Interactive Chinese↔English translation using a classic
phrase-based Statistical Machine Translation system.
"""

import os
import sys
import gradio as gr
from pathlib import Path

# Ensure the smt package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Lazy model loading ──────────────────────────────────────
_MODELS = {}

def get_model(direction: str, variant: str = "sym"):
    """Load and cache an SMT model."""
    key = f"{direction}_{variant}"
    if key in _MODELS:
        return _MODELS[key]

    from smt.decoder import SMTDecoder
    from smt.language_model import LanguageModel
    from smt.phrase_table import PhraseTable
    from smt.config import SMTConfig

    base = Path("models")
    if direction == "zh2en":
        model_dir = base / "zh2en_sym"
    elif direction == "en2zh":
        model_dir = base / f"en2zh_{variant}"

    cfg = SMTConfig()

    # Load language model
    lm_path = model_dir / "lm.json"
    if not lm_path.exists():
        lm_path = model_dir / "lm.pkl"
    lm = LanguageModel.load(str(lm_path))

    # Load phrase table
    pt_path = model_dir / "phrase_table.txt"
    pt = PhraseTable.load(str(pt_path))

    decoder = SMTDecoder(cfg, lm, pt)
    _MODELS[key] = decoder
    return decoder


def detect_language(text: str) -> str:
    """Rough detection: if any CJK char → Chinese, else English."""
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return "zh"
    return "en"


def translate(text: str, model_choice: str) -> str:
    """Translate input text using the selected model."""
    if not text or not text.strip():
        return ""

    # Determine direction from model choice
    direction_map = {
        "ZH → EN (sym)": "zh2en",
        "EN → ZH (sym)": "en2zh",
        "EN → ZH (fast_align)": "en2zh",
    }
    variant_map = {
        "ZH → EN (sym)": "sym",
        "EN → ZH (sym)": "sym",
        "EN → ZH (fast_align)": "fa",
    }

    direction = direction_map.get(model_choice)
    variant = variant_map.get(model_choice)

    if direction is None:
        return "Unknown model selection."

    try:
        decoder = get_model(direction, variant)
        result = decoder.translate(text.strip())
        return result
    except Exception as e:
        return f"[Error] {e}"


# ── Examples ─────────────────────────────────────────────────
EXAMPLES = [
    ["企业推动协议", "ZH → EN (sym)"],
    ["President announced new economic policy aimed at promoting employment growth", "EN → ZH (sym)"],
    ["教育部宣布将增加对农村学校的教育投入", "ZH → EN (sym)"],
    ["The unemployment rate fell to its lowest level in over a decade", "EN → ZH (sym)"],
    ["最新数据显示，中国第三季度GDP增长超出预期", "ZH → EN (sym)"],
]

# ── UI ───────────────────────────────────────────────────────
DESCRIPTION = """
# 🧸 Teddy SMT

**Phrase-Based Statistical Machine Translation** (ZH↔EN)

A classic SMT system — no neural networks, just IBM alignment + phrase tables + beam search.
"""

ARTICLE = """
---
⚠️ **Note**: This is a traditional statistical MT system (circa 2010s tech), not a neural model.
Quality is modest (BLEU ≈ 8) but it demonstrates the full classic pipeline:
word alignment → phrase extraction → language modeling → beam search decoding.

[GitHub](https://github.com) · [Model Card](./README.md)
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Teddy SMT") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=[
                    "ZH → EN (sym)",
                    "EN → ZH (sym)",
                    "EN → ZH (fast_align)",
                ],
                value="ZH → EN (sym)",
                label="Model / Direction",
                info="sym = IBM2+gdfa (smaller, faster) | fa = fast_align (larger, better)"
            )

    with gr.Row():
        with gr.Column():
            src_text = gr.Textbox(
                label="Source Text",
                placeholder="输入中文 / Type English...",
                lines=4,
            )
            translate_btn = gr.Button("Translate / 翻译", variant="primary", size="lg")
        with gr.Column():
            tgt_text = gr.Textbox(
                label="Translation / 译文",
                lines=4,
                interactive=False,
            )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[src_text, model_choice],
        outputs=tgt_text,
        fn=translate,
        cache_examples=True,
    )

    gr.Markdown(ARTICLE)

    translate_btn.click(
        fn=translate,
        inputs=[src_text, model_choice],
        outputs=tgt_text,
    )

    src_text.submit(
        fn=translate,
        inputs=[src_text, model_choice],
        outputs=tgt_text,
    )


if __name__ == "__main__":
    demo.launch()
