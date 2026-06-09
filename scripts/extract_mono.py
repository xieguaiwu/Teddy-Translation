#!/usr/bin/env python3
"""
P1.2: Extract monolingual corpora from ~/BOOKS/ for LM enhancement.

Scans the user's book collection, extracts readable text from epub/pdf/txt
files, tokenizes (jieba for Chinese, Moses-style for English), and produces
clean monolingual text files ready for LM training.

Features:
- Handles epub (requires ebooklib), pdf, txt, and plain-text formats
- Language detection to separate ZH/EN content
- Deduplication and length filtering
- Incremental extraction with resume support

Output:
    data/mono/zh.txt  — tokenized Chinese sentences
    data/mono/en.txt  — tokenized English sentences

Usage:
    python3 scripts/extract_mono.py --lang zh --max-files 50 --output data/mono/zh.txt
    python3 scripts/extract_mono.py --lang en --max-files 100 --output data/mono/en.txt
"""

import sys, os, re, argparse, json, time, hashlib
from pathlib import Path
from collections import defaultdict

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

BOOKS_ROOT = os.path.expanduser("~/BOOKS")

# ── Language-specific directories ─────────────────────────────────────
LANG_DIRS = {
    "zh": ["汉语文学", "汉语文学-日语译版", "东欧文学", "北欧文学",
           "德语文学", "意大利文学", "法语文学", "日语文学",
           "其他语种文学的中文译版", "外国文学名著丛书", "短经典系列",
           "学术", "笔记"],
    "en": ["英语文学", "学术"],
}

# ── File extensions to try ────────────────────────────────────────────
TEXT_EXTS = {".txt", ".md", ".html", ".tex", ".docx"}
EBOOK_EXTS = {".epub", ".mobi", ".azw3"}
PDF_EXT = ".pdf"


def extract_text_txt(filepath: str) -> str:
    """Extract text from plain-text formats."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        # Try latin-1 fallback
        try:
            with open(filepath, "r", encoding="latin-1", errors="replace") as f:
                return f.read()
        except Exception:
            return ""


def extract_text_epub(filepath: str) -> str:
    """Extract text from epub using ebooklib."""
    try:
        import ebooklib
        from ebooklib import epub
        book = epub.read_epub(filepath)
        chunks = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                # Strip HTML tags
                text = item.get_content().decode("utf-8", errors="replace")
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text)
                chunks.append(text)
        return "\n".join(chunks)
    except ImportError:
        print("  [WARN] ebooklib not installed, skipping epub support. Install: pip install ebooklib")
        return ""
    except Exception as e:
        # Many epub files have malformed metadata — try fallback
        return ""


def extract_text_pdf(filepath: str) -> str:
    """Extract text from PDF using pdftotext or PyPDF2."""
    # Try pdftotext (poppler-utils) first — fastest and most reliable
    import subprocess, tempfile
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", "-nopgbrk", filepath, "-"],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and len(result.stdout.strip()) > 100:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback: PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        print("  [WARN] pdftotext and PyPDF2 unavailable. Install: pip install PyPDF2")
        return ""
    except Exception:
        return ""


def extract_text(filepath: str) -> str:
    """Router: extract text based on file extension."""
    ext = Path(filepath).suffix.lower()
    if ext in TEXT_EXTS:
        return extract_text_txt(filepath)
    elif ext in EBOOK_EXTS:
        return extract_text_epub(filepath)
    elif ext == PDF_EXT:
        return extract_text_pdf(filepath)
    return ""


def detect_language(text: str) -> str:
    """Simple CJK vs ASCII-based language detection.

    Returns 'zh' if >15% CJK characters, 'en' otherwise.
    More robust than langdetect for book-scale extraction.
    """
    if not text.strip():
        return "unknown"
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
    total = max(len(text), 1)
    return "zh" if cjk / total > 0.15 else "en"


def split_sentences_zh(text: str) -> list:
    """Split Chinese text into sentences."""
    # Split on Chinese sentence-ending punctuation
    sentences = re.split(r'[。！？；\n]+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= 5]


def split_sentences_en(text: str) -> list:
    """Split English text into sentences (simple heuristic)."""
    # Split on sentence-ending punctuation + whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip().split()) >= 3]


def tokenize_zh(text: str) -> str:
    """Tokenize Chinese text with jieba."""
    try:
        import jieba
        return ' '.join(jieba.cut(text))
    except ImportError:
        # Fallback: character-level (not ideal, but workable)
        return ' '.join(text.replace(' ', ''))


def tokenize_en(text: str) -> str:
    """Simple English tokenization: lowercase + split on non-alphanumeric."""
    # Basic Moses-style: split punctuation, lowercase
    text = re.sub(r'([.,!?;:()\[\]{}"\'/\\-])', r' \1 ', text)
    tokens = text.lower().split()
    return ' '.join(t for t in tokens if len(t) > 0)


def clean_sentence(text: str, lang: str, min_len: int = 5, max_len: int = 200) -> str:
    """Filter sentences by length and content quality."""
    if lang == "zh":
        chars = len(text.replace(' ', ''))
        if chars < min_len or chars > max_len:
            return ""
        # Remove lines that are mostly numbers or punctuation
        alpha_ratio = sum(1 for c in text if c.isalpha() or '\u4e00' <= c <= '\u9fff') / max(len(text), 1)
        if alpha_ratio < 0.3:
            return ""
    else:
        words = text.split()
        if len(words) < 3 or len(words) > 80:
            return ""
        # Remove lines that are mostly non-ASCII
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
        if ascii_ratio < 0.7:
            return ""
    return text


def main():
    parser = argparse.ArgumentParser(description="Extract monolingual corpus from BOOKS")
    parser.add_argument("--lang", choices=["zh", "en"], required=True)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--output", required=True)
    parser.add_argument("--books-root", default=BOOKS_ROOT)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    # ── Collect target files ──────────────────────────────────────────
    target_dirs = [os.path.join(args.books_root, d) for d in LANG_DIRS[args.lang]
                   if os.path.isdir(os.path.join(args.books_root, d))]

    files_to_process = []
    for d in target_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                ext = Path(f).suffix.lower()
                if ext in TEXT_EXTS | EBOOK_EXTS | {PDF_EXT}:
                    files_to_process.append(os.path.join(root, f))

    # Also scan root-level files that match LANG_DIRS names (e.g., 外国文学名著丛书.mobi)
    root_dirs_set = set(LANG_DIRS[args.lang])
    for f in os.listdir(args.books_root):
        fpath = os.path.join(args.books_root, f)
        if os.path.isfile(fpath):
            name_no_ext = Path(f).stem
            ext = Path(f).suffix.lower()
            # Check if filename (without ext) contains any known dir name
            if ext in TEXT_EXTS | EBOOK_EXTS | {PDF_EXT}:
                if any(dname in name_no_ext for dname in root_dirs_set):
                    files_to_process.append(fpath)
                    print(f"  [ROOT] Added root-level file: {f}")

    print(f"Found {len(files_to_process)} candidate files in {len(target_dirs)} directories")
    if args.max_files:
        files_to_process = files_to_process[:args.max_files]
        print(f"Processing first {len(files_to_process)} files")

    # ── Resume support ────────────────────────────────────────────────
    seen_hashes = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                h = hashlib.md5(line.encode()).hexdigest()
                seen_hashes.add(h)
        print(f"Resuming: {len(seen_hashes)} existing sentences loaded")

    # ── Process files ─────────────────────────────────────────────────
    total_sentences = 0
    total_files_ok = 0
    out_f = open(args.output, "a" if args.resume else "w", encoding="utf-8")

    t0 = time.time()
    for i, filepath in enumerate(files_to_process):
        fname = os.path.basename(filepath)
        text = extract_text(filepath)

        if not text or len(text) < 200:
            print(f"  [{i+1}/{len(files_to_process)}] SKIP (no text): {fname}")
            continue

        # Language check
        detected = detect_language(text[:5000])
        if detected != args.lang:
            print(f"  [{i+1}/{len(files_to_process)}] SKIP (lang={detected}): {fname}")
            continue

        # Split and clean
        splitter = split_sentences_zh if args.lang == "zh" else split_sentences_en
        tokenizer = tokenize_zh if args.lang == "zh" else tokenize_en
        sentences = splitter(text)

        file_sentences = 0
        for sent in sentences:
            cleaned = clean_sentence(sent, args.lang)
            if not cleaned:
                continue
            tokenized = tokenizer(cleaned)
            h = hashlib.md5(tokenized.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            out_f.write(tokenized + "\n")
            file_sentences += 1

        total_sentences += file_sentences
        total_files_ok += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(files_to_process)}] {fname}: +{file_sentences} sents "
                  f"({total_sentences} total, {elapsed:.0f}s)")

    out_f.close()
    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"Extraction complete ({elapsed:.0f}s)")
    print(f"  Files processed: {total_files_ok}/{len(files_to_process)}")
    print(f"  Sentences:       {total_sentences}")
    if total_sentences > 0:
        size_kb = os.path.getsize(args.output) / 1024
        print(f"  Output:          {args.output} ({size_kb:.1f} KB)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
