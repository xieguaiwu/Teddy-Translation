"""
Teddy SMT — Hugging Face Space Gradio Demo

Interactive Chinese↔English translation using a classic
phrase-based Statistical Machine Translation system.
"""

import os
import sys
import gradio as gr
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Lazy model loading ──────────────────────────────────────
_MODELS = {}


def get_model(direction: str, variant: str = "sym"):
    """Load and cache an SMT model."""
    key = f"{direction}_{variant}"
    if key in _MODELS:
        return _MODELS[key]

    from smt.decoder import PhraseDecoder
    from smt.language_model import KneserNeyLM
    from smt.phrase_table import load_phrase_table

    base = Path("models")
    if direction == "zh2en":
        model_dir = base / "zh2en_sym"
        if variant == "fa":
            model_dir = base / "zh2en_fa"
        elif variant == "213k_fa":
            model_dir = base / "zh2en_213k_fa"
    elif direction == "en2zh":
        model_dir = base / f"en2zh_{variant}"
        if variant == "sym":
            model_dir = base / "en2zh_sym"

    # Load language model
    lm_path = model_dir / "lm.json"
    if not lm_path.exists():
        lm_path = model_dir / "lm.pkl"
    if not lm_path.exists():
        raise FileNotFoundError(f"No LM file found in {model_dir}")
    lm = KneserNeyLM.load(str(lm_path))

    # Load phrase table
    pt_path = model_dir / "phrase_table.txt"
    if not pt_path.exists():
        raise FileNotFoundError(f"No phrase table found in {pt_path}")
    pt = load_phrase_table(str(pt_path))

    decoder = PhraseDecoder(pt, lm)
    _MODELS[key] = decoder
    return decoder


def translate(text: str, model_choice: str) -> str:
    """Translate input text using the selected model."""
    if not text or not text.strip():
        return ""

    direction_map = {
        "ZH → EN (sym)": "zh2en",
        "ZH → EN (fast_align)": "zh2en",
        "EN → ZH (sym)": "en2zh",
        "EN → ZH (fast_align)": "en2zh",
    }
    variant_map = {
        "ZH → EN (sym)": "sym",
        "ZH → EN (fast_align)": "fa",
        "EN → ZH (sym)": "sym",
        "EN → ZH (fast_align)": "fa",
    }

    direction = direction_map.get(model_choice)
    variant = variant_map.get(model_choice)

    if direction is None:
        return "Unknown model selection."

    try:
        from smt.data_prep import tokenize

        decoder = get_model(direction, variant)

        # Tokenize by detected language
        lang = "zh" if direction == "zh2en" else "en"
        tokenized = tokenize(text.strip(), lang)
        tokens = tokenized.split()

        result_tokens, score = decoder.decode(tokens)
        output = " ".join(result_tokens)
        return output if output else "(no translation)"
    except Exception as e:
        return f"[Error] {type(e).__name__}: {e}"


# ── Examples ─────────────────────────────────────────────────
EXAMPLES = [
    ["企业推动协议", "ZH → EN (213K fast_align)"],
    ["President announced new economic policy", "EN → ZH (213K fast_align)"],
    ["经济危机不断加深", "ZH → EN (fast_align)"],
    ["The unemployment rate fell to its lowest level", "EN → ZH (fast_align)"],
    ["最新数据显示中国第三季度GDP增长超出预期", "ZH → EN (213K fast_align)"],
]

# ── UI ───────────────────────────────────────────────────────
DESCRIPTION = """
# 🧸 Teddy SMT

**Phrase-Based Statistical Machine Translation** (ZH↔EN)

A classic SMT system — no neural networks, just IBM/HMM alignment
+ phrase tables + beam search. Built entirely from scratch in Python.
"""

ARTICLE = """
---
⚠️ **Note**: This is a traditional statistical MT system (circa 2010s).
Quality is modest (BLEU ≈ 8) but demonstrates the full classic pipeline:
word alignment → phrase extraction → language modeling → beam search decoding.

📦 [Model Card](./README.md)
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Teddy SMT") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=[
                    "ZH → EN (fast_align)",
                    "ZH → EN (sym)",
                    "ZH → EN (213K fast_align)",
                    "EN → ZH (fast_align)",
                    "EN → ZH (sym)",
                    "EN → ZH (213K fast_align)",
                ],
                value="ZH → EN (fast_align)",
                label="Model / Direction",
                info="50K data = 65K phrases | 213K = 397K+ phrases"
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
        cache_examples=False,
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
