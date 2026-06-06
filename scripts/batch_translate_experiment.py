#!/usr/bin/env python3
"""
Batch translation for the cross-architecture experiment protocol.

Reads source texts from a directory structure:
    data/source_texts/
      zh_news/*.txt     (20 files)
      zh_lit/*.txt      (20 files)
      en_news/*.txt     (20 files)
      en_lit/*.txt      (20 files)

For each file:
  - Read content, split into sentences
  - Translate using the appropriate model (zh2en or en2zh)
  - Write translations to output/smt/{group}/{filename}
  - Track: filename, num_sentences, translation_time

No spaCy dependency — uses jieba for Chinese tokenization via monkey-patch.

Usage:
  # Translate all 80 source texts
  python3 scripts/batch_translate_experiment.py \
      --model-zh2en model/smt_20k \
      --model-en2zh model/smt_20k_en2zh

  # Translate a single direction
  python3 scripts/batch_translate_experiment.py \
      --model-zh2en model/smt_20k \
      --direction zh2en

  # Create sample source texts and translate
  python3 scripts/batch_translate_experiment.py \
      --model-zh2en model/smt_20k \
      --model-en2zh model/smt_20k_en2zh \
      --create-samples
"""

import sys, os, argparse, time, json
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Monkey-patch Chinese tokenization to use jieba (no spaCy) ─────────
import smt.data_prep as dp
import jieba
dp.tokenize_zh = lambda t: ' '.join(jieba.cut(t))

from smt.decoder import PhraseDecoder
from smt.phrase_table import load_phrase_table
from smt.language_model import KneserNeyLM
from smt import utils

# ── Constants ─────────────────────────────────────────────────────────

SOURCE_GROUPS = {
    'zh_news': {'src_lang': 'zh', 'tgt_lang': 'en'},
    'zh_lit':  {'src_lang': 'zh', 'tgt_lang': 'en'},
    'en_news': {'src_lang': 'en', 'tgt_lang': 'zh'},
    'en_lit':  {'src_lang': 'en', 'tgt_lang': 'zh'},
}

FILES_PER_GROUP = 20
DEFAULT_SOURCE_BASE = 'data/source_texts'
DEFAULT_OUTPUT_BASE = 'output/smt'


# ── Helpers ───────────────────────────────────────────────────────────

def detokenize(tokens, lang):
    """Detokenize token list for target language."""
    if lang == 'zh':
        return ''.join(tokens)
    else:
        return dp.detokenize_en(tokens)


def tokenize_text(text, lang):
    """Tokenize raw text for source language."""
    return dp.tokenize(text, lang=lang)


def load_smt_model(model_dir):
    """Load phrase table and language model from directory."""
    pt_path = os.path.join(model_dir, 'phrase_table.txt')
    lm_path = os.path.join(model_dir, 'lm.json')
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"phrase_table.txt not found in {model_dir}")
    if not os.path.exists(lm_path):
        raise FileNotFoundError(f"lm.json not found in {model_dir}")
    pt = load_phrase_table(pt_path)
    lm = KneserNeyLM.load(lm_path)
    return pt, lm


def create_decoder(pt, lm, beam_size=5):
    """Create a PhraseDecoder with standard config."""
    return PhraseDecoder(pt, lm, config={
        'beam_size': beam_size,
        'stack_size': 50,
        'distortion_limit': 4,
        'lm_weight': 1.0,
        'translation_weight': 1.0,
        'distortion_weight': 0.3,
        'word_penalty': -0.5,
        'oov_strategy': 'copy',
        'future_cost_estimate': False,
    })


# ── Sample text generator ─────────────────────────────────────────────

_SAMPLE_ZH_NEWS = [
    "美国总统今日宣布了一项新的经济政策，旨在促进就业增长。",
    "据报道，全球气候变化会议将于下周在巴黎举行。",
    "股市昨日大幅上涨，投资者对经济前景持乐观态度。",
    "联合国安理会通过了一项关于中东和平的新决议。",
    "最新数据显示，中国第三季度GDP增长超出预期。",
    "科学家发现了一种新的抗癌药物，临床试验效果显著。",
    "欧盟领导人就贸易协定达成共识，协议将于明年生效。",
    "日本首相表示将加强与中国的外交关系。",
    "科技巨头发布了新一代人工智能芯片，性能提升三倍。",
    "教育部宣布将增加对农村学校的教育投入。",
    "国际货币基金组织上调了全球经济增长预测。",
    "新冠疫苗的普及率在全球范围内持续提高。",
    "非洲联盟呼吁发达国家提供更多的气候援助。",
    "俄罗斯与乌克兰的和平谈判取得了初步进展。",
    "世界卫生组织警告称，抗生素耐药性已成为全球威胁。",
    "巴西政府计划大规模投资可再生能源项目。",
    "特斯拉宣布将在上海建立新的超级工厂。",
    "印度成功发射了一颗通信卫星，进入预定轨道。",
    "英国女王在圣诞致辞中强调了团结的重要性。",
    "加拿大政府将提高移民配额，以应对劳动力短缺。",
]

_SAMPLE_ZH_LIT = [
    "春风拂过湖面，泛起层层涟漪，柳枝轻摇，仿佛在诉说着古老的故事。",
    "他站在桥上，望着远方的山峦，心中涌起无限的感慨。",
    "夜色渐浓，繁星点点，寂静的街道上只有风吹过的声音。",
    "那是一个秋天的午后，阳光透过树叶洒在地上，斑驳陆离。",
    "她轻轻地推开门，房间里弥漫着淡淡的花香，一切都那么安静。",
    "雨水顺着屋檐滴落，发出清脆的声响，像是大自然的乐章。",
    "岁月如歌，转眼间已是白发苍苍，回首往事，恍如昨日。",
    "小城的故事总是在不经意间发生，又在不经意间结束。",
    "他翻开那本泛黄的日记，字里行间都是青春的记忆。",
    "雪花纷纷扬扬地落下，整个世界都披上了银白色的外衣。",
    "那条古老的巷子里，藏着无数人的童年与梦想。",
    "海风带着咸味扑面而来，远处是水天一色的壮丽景象。",
    "在这个喧嚣的城市里，每个人都在寻找属于自己的宁静。",
    "晨曦微露，城市还未完全苏醒，这是一天中最安静的时刻。",
    "她坐在窗前，手中的笔在纸上轻轻划过，写下心中的思绪。",
    "银杏叶黄了，铺满了整条街道，金黄一片，美不胜收。",
    "人生的旅途就像这漫长的夜路，有时明亮，有时昏暗。",
    "月光洒在古老的城墙上，仿佛给这座千年古城披上了神秘的面纱。",
    "那座山上的寺庙，钟声悠远，传递着千百年的信仰。",
    "在这个世界的某个角落，总有人在默默守护着爱与希望。",
]

_SAMPLE_EN_NEWS = [
    "The president announced a major infrastructure plan to boost the economy.",
    "Global carbon emissions have reached a record high despite international agreements.",
    "The central bank raised interest rates by 0.25 percent to curb inflation.",
    "A new study reveals significant improvements in renewable energy efficiency.",
    "International trade negotiations have entered their final stage in Geneva.",
    "The World Health Organization declared the outbreak a global health emergency.",
    "Technology stocks surged after the company reported strong quarterly earnings.",
    "The prime minister will visit three countries next month to strengthen diplomatic ties.",
    "Scientists have made a breakthrough in quantum computing research.",
    "The unemployment rate fell to its lowest level in over a decade.",
    "Several countries have pledged additional funding for climate adaptation.",
    "The parliament passed a landmark bill on data privacy protection.",
    "A powerful earthquake struck the coastal region, causing widespread damage.",
    "The airline announced plans to expand its fleet with fuel-efficient aircraft.",
    "New regulations require companies to disclose their environmental impact.",
    "The Olympic committee selected the host city for the 2036 Summer Games.",
    "Researchers developed a new vaccine that shows promise against multiple virus strains.",
    "The government introduced tax incentives to encourage small business growth.",
    "Global food prices have stabilized after months of volatility.",
    "The space agency successfully landed a rover on the distant planet.",
]

_SAMPLE_EN_LIT = [
    "The old man sat by the window, watching the rain paint patterns on the glass.",
    "She walked through the empty streets at dawn, when the city still belonged to dreams.",
    "The autumn leaves danced in the wind, a silent ballet of red and gold.",
    "He remembered the summer of their youth, when every day felt like an eternity.",
    "The lighthouse stood alone against the storm, its beam cutting through the darkness.",
    "In the quiet of the library, she found a world more real than the one outside.",
    "The river flowed gently beneath the ancient bridge, carrying stories of centuries past.",
    "A single candle flickered in the window, a beacon for a wandering soul.",
    "The garden in spring was a symphony of colors, each flower a different note.",
    "He wrote the letter by moonlight, each word a confession of things left unsaid.",
    "The mountain path wound through mist and memory, each step heavier than the last.",
    "She collected seashells along the shore, each one a tiny fragment of infinity.",
    "The cathedral bells rang out across the valley, marking the passage of another hour.",
    "He traced the outlines of the constellation, connecting stars like dots of fate.",
    "The train whistle echoed through the night, a lonely sound that spoke of departures.",
    "In the crowded market, a thousand stories unfolded simultaneously in silence.",
    "The desert stretched endlessly before them, a canvas of sand and sky.",
    "She found the old photograph tucked between the pages of a forgotten book.",
    "The fog rolled in from the sea, transforming the familiar into the mysterious.",
    "He listened to the rain on the tin roof, a rhythm as old as time itself.",
]


def create_sample_sources(source_base):
    """Create sample source text files for testing the experiment protocol."""
    groups = {
        'zh_news': _SAMPLE_ZH_NEWS,
        'zh_lit': _SAMPLE_ZH_LIT,
        'en_news': _SAMPLE_EN_NEWS,
        'en_lit': _SAMPLE_EN_LIT,
    }

    created = 0
    for group_name, texts in groups.items():
        group_dir = os.path.join(source_base, group_name)
        os.makedirs(group_dir, exist_ok=True)
        for i, text in enumerate(texts[:FILES_PER_GROUP], 1):
            fname = f"text_{i:03d}.txt"
            fpath = os.path.join(group_dir, fname)
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(text + '\n')
            created += 1
    return created


def collect_source_files(source_base):
    """Collect source text files organized by language and genre."""
    result = {}
    for group in SOURCE_GROUPS:
        group_dir = os.path.join(source_base, group)
        if os.path.isdir(group_dir):
            files = sorted([
                os.path.join(group_dir, f) for f in os.listdir(group_dir)
                if f.endswith('.txt') and os.path.isfile(os.path.join(group_dir, f))
            ])
            result[group] = files
        else:
            result[group] = []
    return result


# ── Main translation logic ────────────────────────────────────────────

def translate_file(decoder, src_path, out_path, src_lang, tgt_lang):
    """Translate a single file: read → split → translate → write."""
    # Read source
    with open(src_path, encoding='utf-8') as f:
        source_text = f.read().strip()

    # Split into sentences
    sentences = utils.split_sentences(source_text, lang=src_lang)

    # Translate each sentence
    translations = []
    for sent in sentences:
        src_tokens = tokenize_text(sent, src_lang).split()
        if not src_tokens:
            translations.append('')
            continue
        try:
            tokens, _score = decoder.decode(src_tokens)
            text = detokenize(tokens, tgt_lang)
            translations.append(text)
        except Exception:
            translations.append(sent)  # fallback: copy source

    # Write output
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(translations) + '\n')

    return len(sentences)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Batch translate source texts for the cross-architecture experiment protocol')
    parser.add_argument('--model-zh2en', default=None,
                        help='Path to ZH→EN model directory')
    parser.add_argument('--model-en2zh', default=None,
                        help='Path to EN→ZH model directory')
    parser.add_argument('--direction', choices=['zh2en', 'en2zh', 'both'], default='both',
                        help='Which direction to translate (default: both)')
    parser.add_argument('--source-base', default=DEFAULT_SOURCE_BASE,
                        help=f'Base directory for source texts (default: {DEFAULT_SOURCE_BASE})')
    parser.add_argument('--output-base', default=DEFAULT_OUTPUT_BASE,
                        help=f'Base directory for output translations (default: {DEFAULT_OUTPUT_BASE})')
    parser.add_argument('--create-samples', action='store_true',
                        help=f'Create {FILES_PER_GROUP} sample source texts per group if none exist')
    parser.add_argument('--beam-size', type=int, default=5,
                        help='Decoder beam size (default: 5)')
    parser.add_argument('--report', default=None,
                        help='Save translation report to JSON file (default: <output-base>/report.json)')
    args = parser.parse_args()

    # ── Create sample sources if requested ──────────────────────────
    if args.create_samples:
        groups = collect_source_files(args.source_base)
        total_existing = sum(len(v) for v in groups.values())
        if total_existing < FILES_PER_GROUP * 4:
            print(f"Creating sample source texts in {args.source_base}/ ...")
            created = create_sample_sources(args.source_base)
            print(f"  Created {created} sample files")
        else:
            print(f"Source texts already exist ({total_existing} files), skipping --create-samples")

    # ── Collect source files ────────────────────────────────────────
    all_groups = collect_source_files(args.source_base)
    total_files = sum(len(v) for v in all_groups.values())
    if total_files == 0:
        print(f"ERROR: No source files found under {args.source_base}/")
        print("Run with --create-samples to generate sample source texts.")
        sys.exit(1)

    print(f"Source texts: {total_files} files across {len(all_groups)} groups")
    for group, files in all_groups.items():
        print(f"  {group}: {len(files)} files")

    # ── Determine which models to load ──────────────────────────────
    zh2en_dir = args.model_zh2en
    en2zh_dir = args.model_en2zh

    do_zh2en = args.direction in ('zh2en', 'both')
    do_en2zh = args.direction in ('en2zh', 'both')

    if do_zh2en and not zh2en_dir:
        print("ERROR: --model-zh2en required for zh2en translation")
        sys.exit(1)
    if do_en2zh and not en2zh_dir:
        print("ERROR: --model-en2zh required for en2zh translation")
        sys.exit(1)

    # ── Load models ─────────────────────────────────────────────────
    decoders = {}
    if do_zh2en:
        print(f"\nLoading ZH→EN model from {zh2en_dir}...")
        pt_zh2en, lm_zh2en = load_smt_model(zh2en_dir)
        decoders['zh2en'] = create_decoder(pt_zh2en, lm_zh2en, args.beam_size)
        print(f"  Phrase table: {len(pt_zh2en)} entries, LM: {lm_zh2en.vocab_size} types")

    if do_en2zh:
        print(f"\nLoading EN→ZH model from {en2zh_dir}...")
        pt_en2zh, lm_en2zh = load_smt_model(en2zh_dir)
        decoders['en2zh'] = create_decoder(pt_en2zh, lm_en2zh, args.beam_size)
        print(f"  Phrase table: {len(pt_en2zh)} entries, LM: {lm_en2zh.vocab_size} types")

    # ── Translate ───────────────────────────────────────────────────
    report = {'groups': {}, 'summary': {}}
    total_translated = 0
    total_sentences = 0
    total_time = 0.0

    for group, files in all_groups.items():
        if not files:
            continue

        info = SOURCE_GROUPS[group]
        src_lang, tgt_lang = info['src_lang'], info['tgt_lang']
        direction = f'{src_lang}2{tgt_lang}'

        if direction not in decoders:
            print(f"\nSkipping {group} (no model for {direction})")
            continue

        decoder = decoders[direction]
        out_dir = os.path.join(args.output_base, group)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"Translating {group} ({src_lang}→{tgt_lang}), {len(files)} files")
        print(f"{'=' * 50}")

        group_report = []
        group_start = time.perf_counter()
        group_sentences = 0

        for i, src_path in enumerate(files):
            filename = os.path.basename(src_path)
            out_path = os.path.join(out_dir, filename)

            t0 = time.perf_counter()
            try:
                num_sents = translate_file(decoder, src_path, out_path, src_lang, tgt_lang)
                elapsed = time.perf_counter() - t0
                group_sentences += num_sents

                entry = {
                    'filename': filename,
                    'num_sentences': num_sents,
                    'translation_time': round(elapsed, 3),
                }
                group_report.append(entry)

                if (i + 1) % 10 == 0 or i == len(files) - 1:
                    print(f"  [{i + 1}/{len(files)}] {filename}: {num_sents} sents, {elapsed:.2f}s")

            except Exception as e:
                elapsed = time.perf_counter() - t0
                print(f"  [{i + 1}/{len(files)}] {filename}: ERROR - {e}")
                group_report.append({
                    'filename': filename,
                    'num_sentences': 0,
                    'translation_time': round(elapsed, 3),
                    'error': str(e),
                })

        group_elapsed = time.perf_counter() - group_start
        total_translated += len(files)
        total_sentences += group_sentences
        total_time += group_elapsed

        report['groups'][group] = {
            'direction': direction,
            'files_translated': len(files),
            'total_sentences': group_sentences,
            'total_time': round(group_elapsed, 2),
            'avg_time_per_file': round(group_elapsed / max(len(files), 1), 3),
            'avg_time_per_sentence': round(group_elapsed / max(group_sentences, 1), 3),
            'details': group_report,
        }

        print(f"  Completed {group} in {group_elapsed:.1f}s "
              f"({group_sentences} sentences, "
              f"{group_elapsed / max(group_sentences, 1):.3f}s/sent)")

    # ── Summary ─────────────────────────────────────────────────────
    report['summary'] = {
        'total_files': total_translated,
        'total_sentences': total_sentences,
        'total_time': round(total_time, 2),
        'output_base': args.output_base,
        'beam_size': args.beam_size,
    }

    print(f"\n{'=' * 50}")
    print("BATCH TRANSLATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Files translated: {total_translated}")
    print(f"  Total sentences:  {total_sentences}")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Output directory: {args.output_base}/")

    # ── Save report ─────────────────────────────────────────────────
    report_path = args.report or os.path.join(args.output_base, 'report.json')
    os.makedirs(os.path.dirname(report_path) or '.', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")


if __name__ == '__main__':
    main()
