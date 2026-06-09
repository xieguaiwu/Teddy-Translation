#!/usr/bin/env python3
"""
源文本抽取工具 v2 — from ~/BOOKS/ 目录中的 EPUB
====================================================
功能：从 Chinese/English 书籍中提取段落，组装为符合实验规范的源文本。

依赖：pip3 install EbookLib beautifulsoup4 lxml
      或直接用 calibre 的 ebook-convert（已安装）

输出：{lang}_{genre}_{id:03d}.txt  存入 source_texts/
长度：200-800 英文词（中文按字数*0.6折算）

用法：
  python3 extract_source_texts.py --zh-lit       # 20 篇中文文学
  python3 extract_source_texts.py --all          # 全部 80 篇
"""

import argparse
import os
import random
import re
import sys
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
ZH_LIT_DIR = Path.home() / "BOOKS" / "汉语文学"
OUTPUT_DIR = Path("source_texts")

TARGET_MIN, TARGET_MAX = 200, 800  # 英文词；中文按字数*0.6折算

# 优先选取的作者（风格多样、作品量大）
PREFERRED_AUTHORS = [
    "余华", "莫言", "苏童", "王小波", "张爱玲",
    "鲁迅", "老舍", "沈从文", "萧红", "阿城",
    "贾平凹", "迟子建", "格非", "刘震云", "史铁生",
    "王朔", "林棹", "阎连科", "张大春", "金宇澄",
    "巴金", "曹禺", "穆时英", "施蛰存", "孙甘露",
]


def count_words(text: str, lang: str = "zh") -> int:
    """Rough word count."""
    if lang == "zh":
        cjk = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        return int(cjk * 0.6) + len(text.split())
    return len(text.split())


def extract_text_from_epub(epub_path: Path) -> str:
    """Extract all readable text from an EPUB file using EbookLib."""
    from ebooklib import epub
    from bs4 import BeautifulSoup

    try:
        book = epub.read_epub(str(epub_path))
    except Exception:
        return ""

    texts = []
    for item in book.get_items():
        if item.get_type() == 9:  # ITEM_DOCUMENT
            try:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                # Remove script/style
                for tag in soup(["script", "style", "nav"]):
                    tag.decompose()
                text = soup.get_text(separator="\n")
                texts.append(text)
            except Exception:
                continue

    return "\n".join(texts)


def extract_text_from_azw3(azw3_path: Path) -> str:
    """Convert AZW3 → EPUB using calibre, then extract."""
    import tempfile, subprocess
    with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
        epub_tmp = Path(tmp.name)
    try:
        subprocess.run(
            ["ebook-convert", str(azw3_path), str(epub_tmp)],
            capture_output=True, timeout=30,
        )
        return extract_text_from_epub(epub_tmp)
    except Exception:
        return ""
    finally:
        epub_tmp.unlink(missing_ok=True)


def extract_text_from_file(filepath: Path) -> str:
    """Extract text from any supported ebook format."""
    ext = filepath.suffix.lower()
    if ext == ".epub":
        return extract_text_from_epub(filepath)
    elif ext in (".azw3", ".mobi"):
        return extract_text_from_azw3(filepath)
    elif ext == ".txt":
        try:
            return filepath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""
    elif ext == ".pdf":
        # PDF extraction requires extra tools — skip for now
        return ""
    return ""


def get_author_ebooks(author_dir: Path) -> list[Path]:
    """Get the largest ebook file(s) from an author's directory."""
    ebooks = []
    for ext in ("*.epub", "*.azw3", "*.mobi", "*.txt"):
        ebooks.extend(sorted(author_dir.glob(ext)))
    if not ebooks:
        return []
    # Return up to 3 largest files (by size)
    ebooks.sort(key=lambda p: p.stat().st_size, reverse=True)
    return ebooks[:3]


def extract_paragraphs_from_author(author: str, max_chars: int = 50000) -> list[str]:
    """Extract text paragraphs from one author's ebook(s)."""
    author_dir = ZH_LIT_DIR / author
    if not author_dir.is_dir():
        return []

    ebooks = get_author_ebooks(author_dir)
    if not ebooks:
        return []

    all_text = ""
    for eb in ebooks:
        txt = extract_text_from_file(eb)
        if len(txt) > 500:
            all_text += txt + "\n\n"
            if len(all_text) >= max_chars:
                break

    if not all_text:
        return []

    # Split into paragraphs, filter too short/long
    paragraphs = []
    for p in re.split(r'\n\s*\n', all_text):
        p = p.strip()
        wc = count_words(p, "zh")
        if 15 < wc < 1200 and len(p) > 30:
            paragraphs.append(p)

    return paragraphs


def assemble_source_texts(paragraphs: list[str], count: int = 20) -> list[str]:
    """Greedily assemble paragraphs into texts of target word count."""
    random.shuffle(paragraphs)
    texts = []
    current = []
    current_wc = 0
    target = (TARGET_MIN + TARGET_MAX) // 2

    for para in paragraphs:
        pw = count_words(para, "zh")
        if current_wc + pw <= TARGET_MAX:
            current.append(para)
            current_wc += pw
        else:
            if current_wc >= TARGET_MIN:
                texts.append("\n\n".join(current))
                if len(texts) >= count:
                    break
            # Start new segment
            if pw <= TARGET_MAX:
                current = [para]
                current_wc = pw
            else:
                current, current_wc = [], 0

    if current and current_wc >= TARGET_MIN and len(texts) < count:
        texts.append("\n\n".join(current))

    return texts[:count]


def extract_zh_literature(output_dir: Path, count: int = 20):
    """Extract Chinese literary texts from diverse authors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("zh_literature_*.txt")))
    if existing >= count:
        print(f"  Already have {existing} ZH literature texts, skipping")
        return existing

    # Collect paragraphs from diverse authors
    authors = PREFERRED_AUTHORS.copy()
    random.shuffle(authors)

    all_paragraphs = []
    authors_used = []
    for author in authors:
        paras = extract_paragraphs_from_author(author)
        if len(paras) >= 5:
            all_paragraphs.extend(paras)
            authors_used.append(author)
            print(f"  {author}: {len(paras)} paragraphs")
        else:
            print(f"  {author}: no content found, skipping")

    if not all_paragraphs:
        print("  ✗ No content extracted from any author!")
        print("  Try: check ~/BOOKS/汉语文学/ directory structure")
        return 0

    texts = assemble_source_texts(all_paragraphs, count)
    print(f"\n  Assembled {len(texts)} texts from {len(authors_used)} authors")

    written = 0
    for i, text in enumerate(texts, 1):
        wc = count_words(text, "zh")
        fname = f"zh_literature_{i:03d}.txt"
        fpath = output_dir / fname
        # Add authorship note from first line
        fpath.write_text(text, encoding="utf-8")
        print(f"  ✓ {fname} ({wc} words)")
        written += 1

    return written


def extract_en_literature(output_dir: Path, count: int = 20):
    """English literature — placeholder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("en_literature_*.txt")))
    if existing >= count:
        print(f"  Already have {existing} EN literature texts")
        return
    print(f"  English literature: need {count} texts from external sources")
    print(f"  Create en_literature_001.txt ... en_literature_{count:03d}.txt")
    print(f"  in {output_dir}/")
    return 0


def create_placeholder_news(output_dir: Path, lang: str, count: int = 20):
    """Create placeholder news texts (to be replaced with real content)."""
    prefix = f"{lang}_news"
    existing = len(list(output_dir.glob(f"{prefix}_*.txt")))
    if existing >= count:
        return

    news_pairs = {
        "zh": [
            ("中国人工智能产业政策推动技术突破", "近年来，中国在人工智能领域持续加大政策支持力度。工业和信息化部最新数据显示，全国已有超过300个AI相关产业园区投入运营，涵盖芯片制造、算法研发、应用落地等全产业链环节。业内专家指出，政策引导与市场需求的双重驱动正在加速中国AI产业的全球竞争力提升。据预测，到2030年，核心AI产业规模将突破万亿元人民币大关。"),
            ("新能源汽车出口量创历史新高", "中国汽车工业协会今日发布的数据显示，2026年第一季度新能源汽车出口量达到45.6万辆，同比增长67.3%，创下历史同期新高。其中，对欧洲市场出口占比最大，达到38%。分析人士认为，中国新能源汽车在技术、成本和供应链方面的综合优势是出口增长的主要驱动力。与此同时，东南亚和拉美市场也成为新的增长点。"),
        ],
        "en": [
            ("Global Climate Summit Reaches Historic Agreement", "World leaders gathered in Geneva today announced a landmark agreement to reduce carbon emissions by 60% by 2040. The accord, signed by 195 nations, includes binding targets for renewable energy adoption, deforestation reduction, and carbon capture technology investment. Environmental groups cautiously welcomed the deal while noting that enforcement mechanisms remain weak. \"This is a step forward, but the real work begins now,\" said UN Secretary-General."),
            ("Quantum Computing Milestone Achieved", "Researchers at MIT and Google Quantum AI announced a breakthrough in quantum error correction, achieving a logical qubit with error rates below the critical threshold for practical quantum computing. The demonstration, published in Nature, shows that quantum computers can now perform calculations with sufficient reliability for certain commercial applications. Industry analysts predict the first practical quantum advantage in drug discovery within three years."),
        ],
    }

    texts = news_pairs.get(lang, [])
    for i in range(1, count + 1):
        fname = f"{prefix}_{i:03d}.txt"
        fpath = output_dir / fname
        if fpath.exists():
            continue
        # Cycle through templates
        title, body = texts[(i - 1) % len(texts)]
        # Slightly vary for multiples
        if i > len(texts):
            body = body + f"\n\nThis is article number {i} in our series."
        fpath.write_text(f"{title}\n\n{body}", encoding="utf-8")
        wc = count_words(body, lang)
        print(f"  ✓ {fname} ({wc} words)")


def main():
    parser = argparse.ArgumentParser(description="Extract source texts")
    parser.add_argument("--zh-lit", action="store_true")
    parser.add_argument("--en-lit", action="store_true")
    parser.add_argument("--zh-news", action="store_true")
    parser.add_argument("--en-news", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", default="source_texts")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.zh_lit or args.all:
        print("=== Chinese Literature ===")
        extract_zh_literature(output, args.count)

    if args.en_lit or args.all:
        print("\n=== English Literature ===")
        extract_en_literature(output, args.count)

    if args.zh_news or args.all:
        print("\n=== Chinese News ===")
        create_placeholder_news(output, "zh", args.count)

    if args.en_news or args.all:
        print("\n=== English News ===")
        create_placeholder_news(output, "en", args.count)

    if not any([args.zh_lit, args.en_lit, args.zh_news, args.en_news, args.all]):
        parser.print_help()
        return

    # Summary
    total = 0
    for f in sorted(output.glob("*.txt")):
        wc = count_words(f.read_text(), "zh" if "zh_" in f.name else "en")
        print(f"    {f.name:45s} {wc:4d} words")
        total += 1
    print(f"\n  Total: {total} source texts in {output}/")
    return total


if __name__ == "__main__":
    main()
