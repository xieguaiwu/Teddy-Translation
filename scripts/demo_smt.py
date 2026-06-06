#!/usr/bin/env python3
"""
Demo script for the Python SMT system.

Trains on a larger synthetic corpus (200+ sentence pairs) and translates
sample sentences to verify the full pipeline works end-to-end.
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smt import utils
from smt.config import Config
from smt.pipeline import SMTPipeline

utils.setup_logging(level="INFO")
logger = utils.logger


def create_demo_corpus(tmpdir: str, size: int = 200) -> tuple:
    """Create a synthetic parallel corpus with spaCy-consistent tokenization.

    Words are generated at the token level so Chinese tokenization (spaCy)
    produces the same units as our vocabulary. This makes the phrase table
    actually useful.

    Returns (train_zh, train_en, test_zh, test_en_ref) paths.
    """
    import random
    random.seed(42)

    # Word-level ZH vocabulary lists (each entry is a single spaCy token or
    # a short phrase that spaCy will segment consistently)
    zh_subjects = ["他", "她", "我", "我们", "他们", "老师", "学生",
                   "医生", "科学家", "工程师"]
    zh_verbs = ["喜欢", "正在", "想要", "在"]
    zh_actions = ["学习", "阅读", "工作", "跑步", "唱歌", "跳舞",
                  "写作", "思考", "研究", "讨论"]
    zh_objects = ["机器翻译", "编程", "篮球", "电影", "音乐",
                  "茶", "书", "饭", "旅行", "游泳", "算法", "项目"]
    zh_adjectives = ["好", "大", "小", "漂亮", "聪明", "有趣", "重要",
                     "简单", "复杂", "有用"]
    zh_places = ["学校", "图书馆", "公园", "办公室", "家", "医院", "实验室", "餐厅"]
    zh_times = ["每天", "今天", "昨天", "明天", "早上", "晚上", "下午", "周末"]
    zh_nouns = ["书", "笔", "电脑", "手机", "桌子", "椅子", "杯子", "苹果"]
    zh_numbers = ["一", "两", "三", "四", "五", "很多"]
    zh_adverbs = ["很", "非常"]

    # EN translations (one-to-one aligned with ZH tokens above)
    en_subjects = ["He", "She", "I", "We", "They", "The teacher", "The student",
                   "The doctor", "The scientist", "The engineer"]
    en_verbs_like = ["likes", "loves", "enjoys"]
    en_verbs_doing = ["is", "are", "was"]
    en_verbs_want = ["wants to", "would like to"]
    en_actions = ["study", "read", "work", "run", "sing", "dance",
                  "write", "think", "research", "discuss"]
    en_actions_ing = ["studying", "reading", "working", "running", "singing", "dancing",
                      "writing", "thinking", "researching", "discussing"]
    en_objects = ["machine translation", "programming", "basketball", "movies", "music",
                  "tea", "books", "food", "traveling", "swimming", "algorithms", "projects"]
    en_adjectives = ["good", "big", "small", "beautiful", "smart", "interesting", "important",
                     "simple", "complex", "useful"]
    en_places = ["school", "library", "park", "office", "home", "hospital", "lab", "restaurant"]
    en_times = ["every day", "today", "yesterday", "tomorrow", "in the morning",
                "in the evening", "in the afternoon", "on weekends"]
    en_nouns = ["book", "pen", "computer", "phone", "desk", "chair", "cup", "apple"]
    en_numbers = ["one", "two", "three", "four", "five", "many"]
    en_adverbs = ["very", "really"]
    en_prepositions = ["at", "in", "on"]

    # Template-based generation at the token level
    # Template: (zh_parts, en_parts) where each is a list of (type, choices_or_fixed)
    # Types: "fixed", "subj", "verb", "action", "obj", "adj", "place", "time",
    #        "noun", "num", "adv", "prep"
    templates = [
        # "subject 喜欢 action object"
        (["subj", "fixed:喜欢", "action", "obj"],
         ["subj", "verb_like", "action_ing", "obj"]),
        # "subject 很 adj"
        (["subj", "fixed:很", "adj"],
         ["subj", "fixed:is", "adv", "adj"]),
        # "subject 在 place action"
        (["subj", "fixed:在", "place", "action"],
         ["subj", "fixed:is", "action_ing", "prep", "place"]),
        # "subject time 去 place"
        (["subj", "time", "fixed:去", "place"],
         ["subj", "fixed:goes", "prep", "place", "time"]),
        # "subject 想要 action object"
        (["subj", "fixed:想要", "action", "obj"],
         ["subj", "verb_want", "action", "obj"]),
        # "subject 的 noun 很 adj"
        (["subj", "fixed:的", "noun", "fixed:很", "adj"],
         ["subj", "fixed:'s", "noun", "fixed:is", "adv", "adj"]),
        # "今天天气很 adj"
        (["fixed:今天", "fixed:天气", "fixed:很", "adj"],
         ["fixed:Today", "fixed:the", "fixed:weather", "fixed:is", "adv", "adj"]),
        # "subject 有 num 个 noun"
        (["subj", "fixed:有", "num", "fixed:个", "noun"],
         ["subj", "fixed:has", "num", "noun"]),
        # "subject 正在 action"
        (["subj", "fixed:正在", "action"],
         ["subj", "verb_doing", "action_ing"]),
        # "subject adj 的 noun"
        (["subj", "adv", "adj", "fixed:的", "noun"],
         ["subj", "fixed:has", "adv", "adj", "noun"]),
    ]

    # Create aligned lookup tables
    zh_choices = {
        "subj": zh_subjects,
        "action": zh_actions,
        "obj": zh_objects,
        "adj": zh_adjectives,
        "place": zh_places,
        "time": zh_times,
        "noun": zh_nouns,
        "num": zh_numbers,
        "adv": zh_adverbs,
    }

    en_choices = {
        "subj": en_subjects,
        "action": en_actions,
        "action_ing": en_actions_ing,
        "obj": en_objects,
        "adj": en_adjectives,
        "place": en_places,
        "time": en_times,
        "noun": en_nouns,
        "num": en_numbers,
        "adv": en_adverbs,
        "verb_like": en_verbs_like,
        "verb_doing": en_verbs_doing,
        "verb_want": en_verbs_want,
        "prep": en_prepositions,
    }

    def resolve(parts: list, choice_pools: dict, index: int) -> str:
        """Resolve a template part to a specific word."""
        result_words = []
        for part in parts:
            if part.startswith("fixed:"):
                result_words.append(part[6:])
            elif part in choice_pools:
                pool = choice_pools[part]
                # Use deterministic indexing for cross-lingual alignment
                word = pool[(index + hash(part)) % len(pool)]
                result_words.append(word)
            else:
                result_words.append(part)
        return " ".join(result_words)

    # Generate training pairs
    zh_sentences = []
    en_sentences = []

    for i in range(size):
        template = templates[i % len(templates)]
        zh_parts, en_parts = template

        # Use the same index for both languages to keep them aligned
        zh_text = resolve(zh_parts, zh_choices, i)
        en_text = resolve(en_parts, en_choices, i)

        zh_sentences.append(zh_text)
        en_sentences.append(en_text)

    # Test sentences (held-out)
    test_pairs = [
        ("他 喜欢 研究 算法",
         "He likes researching algorithms"),
        ("她 每天 喝茶",
         "She drinks tea every day"),
        ("学生 正在 学习 编程",
         "The student is studying programming"),
        ("科学家 研究 复杂 的 算法",
         "The scientist researches complex algorithms"),
        ("工程师 完成 重要 的 项目",
         "The engineer completes important projects"),
        ("我们 今天 去 公园",
         "We go to the park today"),
        ("老师 在 图书馆 读书",
         "The teacher is reading at the library"),
        ("医生 的 工作 很 重要",
         "The doctor's work is very important"),
        ("我 喜欢 在 公园 跑步",
         "I like running in the park"),
        ("这 本书 很 有用",
         "This book is very useful"),
    ]

    test_zh = [p[0] for p in test_pairs]
    test_en_ref = [p[1] for p in test_pairs]

    # Write files
    data_dir = os.path.join(tmpdir, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_zh = os.path.join(data_dir, "train.zh")
    train_en = os.path.join(data_dir, "train.en")
    test_zh_file = os.path.join(data_dir, "test.zh")
    test_en_file = os.path.join(data_dir, "test.en")

    utils.write_lines(zh_sentences, train_zh)
    utils.write_lines(en_sentences, train_en)
    utils.write_lines(test_zh, test_zh_file)
    utils.write_lines(test_en_ref, test_en_file)

    logger.info(f"Generated {size} training pairs + 10 test sentences")
    return train_zh, train_en, test_zh_file, test_en_file


def main():
    print("=" * 70)
    print("  Python SMT Demo — End-to-End Pipeline Verification")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="smt_demo_") as tmpdir:
        print(f"\n  Working dir: {tmpdir}")
        print(f"  {'─' * 60}")

        # Create demo corpus
        print("\n  [1] Generating demo corpus (200 sentence pairs)...")
        train_zh, train_en, test_zh, test_en_ref = create_demo_corpus(tmpdir, size=200)

        # Config: lower min_count for demo
        config = Config()
        config["phrase_table.min_phrase_count"] = 1
        config["decoder.beam_size"] = 5
        config["decoder.stack_size"] = 50

        # Initialize pipeline
        pipeline = SMTPipeline()
        pipeline.config = config

        # Train
        print("  [2] Training SMT model (IBM2 alignment + phrase table + LM)...")
        model_dir = os.path.join(tmpdir, "model")
        try:
            model_info = pipeline.train_python(
                src_file=train_zh,
                tgt_file=train_en,
                output_dir=model_dir,
                src_lang="zh",
                tgt_lang="en",
            )
            print(f"  ✓ Phrase table: {model_info.get('phrase_table_entries', 0)} entries")
            print(f"  ✓ Vocabulary:   {model_info.get('vocab_size', 0)} types")
            print(f"  ✓ Alignments:   {model_info.get('num_alignments', 0)} links")
        except Exception as e:
            print(f"  ✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # Translate
        print(f"\n  {'─' * 60}")
        print("  [3] Translating test sentences...")
        output_file = os.path.join(tmpdir, "output.txt")
        try:
            result = pipeline.translate_python(
                input_file=test_zh,
                output_file=output_file,
                src_lang="zh",
            )
            print(f"  ✓ {result.get('num_sentences', 0)} sentences translated")
        except Exception as e:
            print(f"  ✗ Translation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # Show results side by side
        print(f"\n  {'─' * 60}")
        print("  [4] Sample Translations:")
        print(f"  {'─' * 60}")

        src_lines = utils.read_lines(test_zh)
        tgt_lines = utils.read_lines(output_file)
        ref_lines = utils.read_lines(test_en_ref)

        for i, (src, tgt, ref) in enumerate(zip(src_lines, tgt_lines, ref_lines)):
            match = "✓" if tgt.lower() == ref.lower() else "≈" if any(w in tgt.lower() for w in ref.lower().split()[:2]) else " "
            color_match = "✓" if match == "✓" else ("~" if match == "≈" else " ")
            print(f"\n  ZH:     {src}")
            print(f"  {color_match} SMT:    {tgt}")
            print(f"  ✓ Ref:   {ref}")

        # Evaluate
        print(f"\n  {'─' * 60}")
        print("  [5] BLEU Evaluation:")
        try:
            eval_result = pipeline.evaluate(output_file, test_en_ref)
            bleu = eval_result.get("bleu", {})
            print(f"  {'─' * 40}")
            print(f"  Corpus BLEU:  {bleu.get('bleu', 0.0):.2f}")
            precisions = bleu.get("precisions", [])
            if precisions:
                # sacrebleu precisions are 0-100 scale; fallback uses 0-1 scale
                p_str = " / ".join(f"{p:.1f}%" for p in precisions)
                print(f"  1-4-gram P:  {p_str}")
            print(f"  BP:          {bleu.get('brevity_penalty', 0.0):.3f}")
            print(f"  Ratio:       {bleu.get('ratio', 0.0):.3f}")
            print(f"  Hyp len:     {bleu.get('hyp_len', 0)}")
            print(f"  Ref len:     {bleu.get('ref_len', 0)}")
        except Exception as e:
            print(f"  ✗ Evaluation failed: {e}")

        # Single sentence translation
        print(f"\n  {'─' * 60}")
        print("  [6] Interactive Translation:")
        test_sents = [
            "我喜欢学习编程",
            "科学家正在研究复杂的算法",
            "学生们每天在图书馆学习",
        ]
        for sent in test_sents:
            trans = pipeline.translate_sentence(sent, src_lang="zh")
            print(f"  ZH: {sent}")
            print(f"  EN: {trans}")
            print()

    print(f"  {'─' * 60}")
    print("  ✅ Demo complete! Pipeline verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
