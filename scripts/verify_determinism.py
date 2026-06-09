#!/usr/bin/env python3
"""
S1 确定性验证脚本
==================
验证 temperature=0.0 + seed=42 下各模型输出是否严格一致。

方法：
  1. 用 5 篇样本（3 中 + 2 英）作为源文本
  2. 每篇经每个模型翻译 3 次（共 6×5×3 = 90 次 API 调用）
  3. 计算同一模型对同一源文本的 3 次输出间的 self-BLEU
  4. 报告 self-BLEU < 99 的不稳定模型

输出：
  - translations_verify/{model}/ 下所有译文
  - 终端汇总表
  - verify_report.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import openai

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a professional translator. You translate texts accurately and naturally, preserving the original meaning, tone, and style."
USER_PROMPT_TEMPLATE = """Translate the following {genre} text from {src_lang} to {tgt_lang}.
Output only the translation. Preserve paragraph breaks.

{source_text}"""

DIRECTIONS = {
    "zh2en": ("Chinese", "English"),
    "en2zh": ("English", "Chinese"),
}

ALL_MODELS = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "glm-5.1",
    "glm-5",
    "kimi-k2.6",
    "qwen3.6-plus",
]

# ── 5 篇样本：3 中 2 英，不同体裁 ──────────────────────────────────────
SAMPLE_TEXTS = [
    {
        "id": "sample_zh_news_1",
        "lang": "zh",
        "genre": "news",
        "text": "中国国家统计局周三发布的数据显示，2025年第四季度国内生产总值同比增长5.4%，全年经济增长5.0%。"
                "这一增速符合政府年初设定的目标区间。经济学家表示，消费复苏和出口韧性是主要拉动因素，"
                "但房地产市场的持续调整仍对经济构成下行压力。展望2026年，分析师预计政策将继续保持宽松基调。",
    },
    {
        "id": "sample_zh_lit_1",
        "lang": "zh",
        "genre": "literature",
        "text": "那天下午的阳光很好，从窗户斜斜地照进来，在地板上画出一块明亮的四边形。"
                "她坐在那块光里，手里拿着一本书，却一个字也没读进去。"
                "窗外的梧桐树上，几只麻雀在叽叽喳喳地叫着，声音忽远忽近。"
                "她想，这样的下午，大概就是用来浪费的。",
    },
    {
        "id": "sample_zh_lit_2",
        "lang": "zh",
        "genre": "literature",
        "text": "火车在黑暗中疾驰，窗外的灯火像流星一样掠过。车厢里弥漫着泡面和廉价香水的气味。"
                "对面的老人已经睡了，头靠着玻璃，嘴巴微微张开。"
                "他看了看手表，凌晨三点十七分。还有四个小时才能到站。"
                "他想起临走时母亲塞给他的那袋橘子，现在还搁在行李架上。",
    },
    {
        "id": "sample_en_news_1",
        "lang": "en",
        "genre": "news",
        "text": "Scientists at the European Organization for Nuclear Research (CERN) announced today the discovery of "
                "a new subatomic particle that could reshape our understanding of dark matter. "
                "The particle, tentatively named X-417, was observed during high-energy collisions at the Large Hadron Collider. "
                "If confirmed by independent laboratories, this would be the first major breakthrough in particle physics since the Higgs boson in 2012.",
    },
    {
        "id": "sample_en_lit_1",
        "lang": "en",
        "genre": "literature",
        "text": "The old man sat on the porch every evening, watching the sun set behind the hills. "
                "His dog, a golden retriever named Buddy, lay at his feet, occasionally thumping his tail against the wooden boards. "
                "Thirty-seven years he had lived in this house, and he knew every creak in the floor, every draft through the windows. "
                "Tonight, the sky was a deep orange, streaked with purple clouds. He took a slow breath and smiled.",
    },
]


def model_slug(full: str) -> str:
    return full


def translate_once(client: openai.OpenAI, text: str, lang: str, genre: str,
                   model: str, seed: int = 42) -> tuple[str, dict]:
    """Single translation attempt, return (text, meta)."""
    direction = "zh2en" if lang == "zh" else "en2zh"
    src_lang, tgt_lang = DIRECTIONS[direction]
    prompt = USER_PROMPT_TEMPLATE.format(
        genre=genre, src_lang=src_lang, tgt_lang=tgt_lang, source_text=text
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        seed=seed,
    )
    meta = {
        "model_served": response.model,
        "finish_reason": response.choices[0].finish_reason,
        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
        "completion_tokens": response.usage.completion_tokens if response.usage else None,
    }
    return response.choices[0].message.content.strip(), meta


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Simple sentence-level BLEU (1-gram + 2-gram precision with brevity penalty).

    This is intentionally NOT sacrebleu — we only need relative comparison
    (self-BLEU ~100 = identical), not publication-grade numbers.
    """
    import collections
    import math

    def tokenize(s):
        return s.lower().split()

    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)

    # 1-gram precision
    ref_counts = collections.Counter(ref_tokens)
    hyp_counts = collections.Counter(hyp_tokens)

    matches_1 = sum(min(hyp_counts[g], ref_counts[g]) for g in hyp_counts)
    total_1 = len(hyp_tokens)

    # 2-gram precision
    ref_bigrams = collections.Counter(
        " ".join(ref_tokens[i:i+2]) for i in range(len(ref_tokens) - 1)
    )
    hyp_bigrams = collections.Counter(
        " ".join(hyp_tokens[i:i+2]) for i in range(len(hyp_tokens) - 1)
    )

    matches_2 = sum(min(hyp_bigrams[g], ref_bigrams.get(g, 0)) for g in hyp_bigrams)
    total_2 = len(hyp_tokens) - 1

    # Precision
    p1 = matches_1 / total_1 if total_1 > 0 else 0
    p2 = matches_2 / total_2 if total_2 > 0 else 0

    # Brevity penalty
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1))
    else:
        bp = 1.0

    if p1 == 0 or p2 == 0:
        return 0.0

    return bp * math.exp(0.5 * math.log(p1) + 0.5 * math.log(p2)) * 100


def run_verification(args):
    api_key = args.api_key or os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        log.error("No API key. Set OPENCODE_API_KEY or pass --api-key")
        sys.exit(1)

    # Proxy handling
    import httpx
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    http_client = httpx.Client(proxy=proxy) if proxy else None

    client = openai.OpenAI(
        base_url=args.api_base, api_key=api_key, http_client=http_client,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter models if specified
    models = args.models
    if not models or models.lower() == "all":
        models = ALL_MODELS
    else:
        models = [m.strip() for m in models.split(",")]
        models = [m for m in models if m in ALL_MODELS]

    log.info(f"Models: {[model_slug(m) for m in models]}")
    log.info(f"Samples: {[s['id'] for s in SAMPLE_TEXTS]}")
    log.info(f"Runs per sample per model: {args.runs}")

    total_calls = len(models) * len(SAMPLE_TEXTS) * args.runs
    log.info(f"Total API calls: {total_calls}")
    if total_calls > 100:
        log.warning("That's a lot! Make sure you have budget.")

    results = {}
    all_ok = True

    for model in models:
        slug = model_slug(model)
        model_dir = output_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"\n{'='*50}")
        log.info(f"Model: {slug}")

        for sample in SAMPLE_TEXTS:
            sid = sample["id"]
            direction = "zh2en" if sample["lang"] == "zh" else "en2zh"
            tgt_lang = DIRECTIONS[direction][1]

            # Run N times
            outputs = []
            metadatas = []
            for run in range(1, args.runs + 1):
                log.info(f"  [{slug}] {sid} run {run}/{args.runs}...")
                text, meta = translate_once(
                    client, sample["text"], sample["lang"],
                    sample["genre"], model, args.seed
                )
                outputs.append(text)
                metadatas.append(meta)

                # Save each run
                run_file = model_dir / f"{sid}_{direction}_run{run}.txt"
                run_file.write_text(text, encoding="utf-8")
                meta_file = model_dir / f"{sid}_{direction}_run{run}.meta.json"
                meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

                time.sleep(0.3)  # gentle rate limiting

            # Compute pairwise BLEU
            bleus = []
            for i in range(args.runs):
                for j in range(i + 1, args.runs):
                    b = compute_bleu(outputs[i], outputs[j])
                    bleus.append(b)

            min_bleu = min(bleus) if bleus else 100.0
            mean_bleu = sum(bleus) / len(bleus) if bleus else 100.0
            stable = min_bleu >= 99.0

            if not stable:
                all_ok = False

            log.info(f"  → self-BLEU: mean={mean_bleu:.2f}, min={min_bleu:.2f} "
                     f"{'✅' if stable else '❌ UNSTABLE'}")

            results.setdefault(slug, {})[sid] = {
                "mean_self_bleu": round(mean_bleu, 2),
                "min_self_bleu": round(min_bleu, 2),
                "stable": stable,
                "model_served": metadatas[0]["model_served"],
            }

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  确定性验证报告")
    print(f"{'='*60}")
    print(f"  Seed: {args.seed} | 温度: 0.0 | Runs/样本: {args.runs}")
    print(f"  API 调用总数: {total_calls}")
    print(f"")

    header = f"  {'Model':30s} {'Stable':8s} {'Min BLEU':10s} {'Mean BLEU':10s} {'Model Served'}"
    print(header)
    print(f"  {'-'*len(header)}")

    for slug in sorted(results.keys()):
        min_b_all = min(r["min_self_bleu"] for r in results[slug].values())
        mean_b_all = sum(r["mean_self_bleu"] for r in results[slug].values()) / len(results[slug])
        stable_all = all(r["stable"] for r in results[slug].values())
        served = results[slug][list(results[slug].keys())[0]]["model_served"]
        status = "✅" if stable_all else "❌"
        print(f"  {slug:30s} {status:8s} {min_b_all:<10.2f} {mean_b_all:<10.2f} {served}")

    print(f"")
    print(f"  Overall: {'✅ ALL STABLE' if all_ok else '❌ SOME MODELS UNSTABLE'}")
    print(f"{'='*60}")

    # Save report
    report_path = output_dir / "verify_report.json"
    report = {
        "seed": args.seed,
        "runs_per_sample": args.runs,
        "overall_stable": all_ok,
        "models": results,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info(f"Report saved to {report_path}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="S1 deterministic verification")
    parser.add_argument("--models", "-m", default="all",
                        help="Comma-separated or 'all'")
    parser.add_argument("--api-key", help="OpenCode Go key (or $OPENCODE_API_KEY)")
    parser.add_argument("--api-base", default="https://opencode.ai/zen/go/v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=3, help="Runs per sample (default: 3)")
    parser.add_argument("--output-dir", default="translations_verify")
    args = parser.parse_args()
    sys.exit(run_verification(args))
