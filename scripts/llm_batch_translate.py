#!/usr/bin/env python3
"""
LLM Batch Translation Script (Multi-Model, Production)
=========================================================
Provider: OpenCode Go (https://opencode.ai/zen/go/v1)
Temperature: 0.0 | Seed: 42 | No reasoning (fast translation mode)
Direction: ZH→EN and EN→ZH (auto-detected from filename)

Models (6):
  deepseek-v4-pro, deepseek-v4-flash, glm-5.1, glm-5, kimi-k2.6, qwen3.6-plus

Usage:
  python3 llm_batch_translate.py --models all --api-key "sk-..."
  python3 llm_batch_translate.py --models deepseek-v4-flash,glm-5.1 --api-key "$KEY"

Features:
  ✓ 6 models in one pass (or select subset)
  ✓ No reasoning mode — fast, cost-efficient translation only
  ✓ Parallel workers (configurable concurrency)
  ✓ Quality gate: language check + finish_reason + length ratio
  ✓ JSON sidecar metadata per file (model, token usage, timestamp, etc.)
  ✓ Atomic checkpoint every translation (crash-safe resume)
  ✓ Proper 429 handling with Retry-After header
  ✓ Seed=42 for reproducibility
  ✓ API key via CLI or OPENCODE_API_KEY env var
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

import openai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("translate.log")],
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a professional translator. You translate texts accurately and naturally, preserving the original meaning, tone, and style."

USER_PROMPT_TEMPLATE = """Translate the following {genre} text from {src_lang} to {tgt_lang}.
Output only the translation. Preserve paragraph breaks.

{source_text}"""

DIRECTIONS = {
    "zh2en": ("Chinese", "English"),
    "en2zh": ("English", "Chinese"),
}

# CJK unified ideographs + CJK Extension A
CJK_RANGES = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x2E80, 0x2EFF),
              (0x3000, 0x303F), (0xFF00, 0xFFEF)]
LATIN_RANGE = (0x0041, 0x007A)

ALL_MODELS = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "glm-5.1",
    "glm-5",
    "kimi-k2.6",
    "qwen3.6-plus",
]

PRICES = {
    "deepseek-v4-pro":    {"input": 8, "output": 24},
    "deepseek-v4-flash":  {"input": 1, "output": 4},
    "glm-5.1":            {"input": 5, "output": 15},
    "glm-5":              {"input": 3, "output": 9},
    "kimi-k2.6":          {"input": 4, "output": 12},
    "qwen3.6-plus":       {"input": 5, "output": 16},
}


# ── Helpers ────────────────────────────────────────────────────────────────

def model_slug(full: str) -> str:
    return full


def classify_lang(text: str) -> str:
    """Detect text language via Unicode range heuristic (no deps)."""
    cjk = latin = 0
    for ch in text[:500]:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in CJK_RANGES):
            cjk += 1
        elif LATIN_RANGE[0] <= cp <= LATIN_RANGE[1]:
            latin += 1
    return "zh" if cjk > latin else "en"


def load_source_texts(input_dir: Path) -> list[dict]:
    """Load source texts with naming convention: {lang}_{genre}_{id}.txt"""
    texts = []
    for fpath in sorted(input_dir.iterdir()):
        if fpath.suffix not in (".txt", ".md"):
            continue
        name = fpath.stem
        parts = name.split("_")
        if len(parts) < 3:
            log.warning(f"Skipping {fpath.name}: unexpected naming format")
            continue
        lang, genre = parts[0], parts[1]
        if lang not in ("zh", "en") or genre not in ("news", "literature"):
            log.warning(f"Skipping {fpath.name}: unrecognized lang/genre ({lang},{genre})")
            continue
        text = fpath.read_text(encoding="utf-8").strip()
        if not text:
            log.warning(f"Skipping {fpath.name}: empty")
            continue
        texts.append({"filename": fpath.name, "stem": name,
                       "lang": lang, "genre": genre, "text": text})
    return texts


def build_jobs(source_texts: list[dict], models: list[str]) -> list[dict]:
    jobs = []
    for model in models:
        for src in source_texts:
            direction = "zh2en" if src["lang"] == "zh" else "en2zh"
            jobs.append({"model": model, "slug": model_slug(model),
                         "stem": src["stem"], "genre": src["genre"],
                         "source_text": src["text"], "direction": direction,
                         "src_lang": src["lang"]})
    return jobs


# ── Checkpoint (thread-safe, atomic) ──────────────────────────────────────

class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self._done: set[tuple[str, str, str]] = set()
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self._done = set(tuple(item) for item in data)
            except (json.JSONDecodeError, OSError):
                self._done = set()

    def contains(self, model: str, stem: str, direction: str) -> bool:
        with self._lock:
            return (model, stem, direction) in self._done

    def add(self, model: str, stem: str, direction: str):
        with self._lock:
            self._done.add((model, stem, direction))
            self._flush()

    def _flush(self):
        """Atomic write via tempfile + rename."""
        data = [list(t) for t in sorted(self._done)]
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.path)  # atomic on POSIX


# ── Quality Gate ──────────────────────────────────────────────────────────

class QualityGate:
    """Validate translation output before accepting."""

    MIN_RATIO = 0.2   # output chars / input chars lower bound
    MAX_RATIO = 5.0   # upper bound (ZH→EN can expand 4-5× due to CJK→Latin)
    REJECT_PATTERNS = [
        r"(?i)I cannot translate",
        r"(?i)I cannot complete",
        r"(?i)unable to translate",
        r"(?i)as an AI",
        r"(?i)as a language model",
        r"(?i)sorry[,!]",
        r"(?i)抱歉",
        r"(?i)我不(能|会|可以)",
        r"(?i)作为AI|作为人工智能",
    ]

    @classmethod
    def check(cls, src_text: str, tgt_text: str,
              target_dir: str, finish_reason: str) -> Optional[str]:
        """Return error message if quality check fails, else None."""
        # 1. Finish reason
        if finish_reason == "length":
            return "truncated (finish_reason=length)"
        if finish_reason == "content_filter":
            return "blocked (content_filter)"

        # 2. Rejection phrases
        for pat in cls.REJECT_PATTERNS:
            if re.search(pat, tgt_text[:200]):
                return f"rejection detected (matched: {pat[:40]})"

        # 3. Language check
        detected = classify_lang(tgt_text)
        if target_dir == "zh" and detected != "zh":
            return f"wrong language: expected ZH, detected {detected.upper()}"
        if target_dir == "en" and detected == "zh":
            return f"wrong language: expected EN, detected ZH"

        # 4. Length ratio check
        ratio = len(tgt_text) / max(len(src_text), 1)
        if ratio < cls.MIN_RATIO:
            return f"too short: ratio={ratio:.2f} (min={cls.MIN_RATIO})"
        if ratio > cls.MAX_RATIO:
            return f"too long: ratio={ratio:.2f} (max={cls.MAX_RATIO})"

        return None


# ── Core Translate ────────────────────────────────────────────────────────

def translate_with_metadata(
    client: openai.OpenAI,
    source_text: str,
    src_lang: str,
    tgt_lang: str,
    genre: str,
    model: str,
    max_retries: int = 3,
    seed: int = 42,
) -> tuple[Optional[str], Optional[dict]]:
    """Translate and return (cleaned_text, metadata_dict) or (None, error_meta)."""
    prompt = USER_PROMPT_TEMPLATE.format(
        genre=genre, src_lang=src_lang, tgt_lang=tgt_lang, source_text=source_text
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                seed=seed,
            )
        except openai.RateLimitError as e:
            retry_after = 5
            try:
                retry_after = int(e.response.headers.get("Retry-After", 5))
            except (AttributeError, ValueError, TypeError):
                pass
            log.warning(f"  429 Rate limited, waiting {retry_after}s (attempt {attempt})")
            time.sleep(retry_after)
            last_error = f"429:{retry_after}s"
            continue
        except openai.APITimeoutError as e:
            log.warning(f"  Timeout (attempt {attempt})")
            last_error = f"timeout:{attempt}"
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            continue
        except openai.APIStatusError as e:
            status = e.status_code
            body = str(e.body)[:200] if e.body else "(no body)"
            log.warning(f"  API {status}: {body}")
            last_error = f"http_{status}"
            if status == 401 and "CreditsError" in body:
                log.error(f"  → Insufficient credits! Top up at https://opencode.ai")
                return None, {"error": "insufficient_credits"}
            if attempt < max_retries and status >= 500:
                time.sleep(2 ** attempt)
            continue
        except Exception as e:
            err_str = str(e)[:100]
            log.warning(f"  Unexpected error (attempt {attempt}): {err_str}")
            last_error = f"unexpected:{err_str}"
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            continue

        # Got a response
        msg = response.choices[0].message
        output_raw = msg.content or ""
        finish_reason = response.choices[0].finish_reason or "unknown"
        model_served = response.model

        # Determine target language direction
        target_dir = "zh" if tgt_lang == "Chinese" else "en"

        # Quality gate
        qc_error = QualityGate.check(source_text, output_raw, target_dir, finish_reason)
        if qc_error:
            log.warning(f"  Quality fail: {qc_error}")
            last_error = qc_error
            if attempt < max_retries:
                continue
            return None, {"error": qc_error, "model": model_served,
                          "finish_reason": finish_reason}

        # Passed
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "total_tokens": response.usage.total_tokens if response.usage else None,
        } if response.usage else {}

        meta = {
            "model_requested": model,
            "model_served": model_served,
            "finish_reason": finish_reason,
            "temperature": 0.0,
            "seed": seed,
            "output_chars": len(output_raw),
            "source_chars": len(source_text),
            **usage,
        }
        return output_raw.strip(), meta

    return None, {"error": last_error or "max_retries_exceeded"}


# ── Save Output ───────────────────────────────────────────────────────────

def save_result(output_dir: Path, stem: str, direction: str,
                model_name: str, text: str, meta: dict):
    """Save translation .txt + .meta.json sidecar."""
    slug = model_slug(model_name)
    model_dir = output_dir / slug
    model_dir.mkdir(parents=True, exist_ok=True)

    txt_path = model_dir / f"{stem}_{direction}.txt"
    txt_path.write_text(text, encoding="utf-8")

    meta_path = model_dir / f"{stem}_{direction}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    return txt_path


# ── Single Job Worker ─────────────────────────────────────────────────────

def process_job(job: dict, client: openai.OpenAI,
                output_dir: Path, checkpoint: Checkpoint,
                seed: int) -> dict:
    """Process one translation job. Returns result dict."""
    key = (job["model"], job["stem"], job["direction"])

    # Skip if done
    if checkpoint.contains(*key):
        return {"status": "skipped", "key": key, "slug": job["slug"], "stem": job["stem"]}

    src_lang, tgt_lang = DIRECTIONS[job["direction"]]
    log.info(f"  [{job['slug']}] {job['stem']} ({src_lang}→{tgt_lang}, {job['genre']})")

    text, meta = translate_with_metadata(
        client=client,
        source_text=job["source_text"],
        src_lang=src_lang, tgt_lang=tgt_lang,
        genre=job["genre"],
        model=job["model"],
        seed=seed,
    )

    if text and meta:
        out_path = save_result(output_dir, job["stem"], job["direction"],
                                job["model"], text, meta)
        checkpoint.add(*key)
        log.info(f"    ✓ {len(text)} chars → {out_path.name}")
        return {"status": "ok", "key": key, "slug": job["slug"],
                "stem": job["stem"], "chars": len(text),
                "tokens": meta.get("total_tokens")}
    else:
        err = meta.get("error", "unknown") if meta else "no_response"
        log.error(f"    ✗ {job['stem']}: {err}")
        return {"status": "fail", "key": key, "slug": job["slug"],
                "stem": job["stem"], "error": err}


# ── Main Pipeline ─────────────────────────────────────────────────────────

def run_pipeline(args):
    # ── Resolve API key ──
    api_key = args.api_key or os.environ.get("OPENCODE_API_KEY")
    if not api_key:
        # Check provider-specific env var as fallback
        if args.provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            log.error(f"API key required for {args.provider}. "
                      f"Pass --api-key or set the appropriate env var.")
            sys.exit(1)

    # ── Resolve models ──
    if args.models.lower() == "all":
        models = ALL_MODELS
    else:
        models = [m.strip() for m in args.models.split(",")]
        models = [m for m in models if m in ALL_MODELS]
        unknown = [m for m in models if m not in ALL_MODELS]
        for m in unknown:
            log.warning(f"Unknown model '{m}', skipping")
        if not models:
            log.error("No valid models specified! Known: {ALL_MODELS}")
            sys.exit(1)

    log.info(f"Models ({len(models)}): {[model_slug(m) for m in models]}")
    for m in models:
        p = PRICES.get(m, {})
        log.info(f"  {model_slug(m):30s} ¥{p.get('input','?')}/¥{p.get('output','?')} per 1M tokens")

    # ── IO setup ──
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        log.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Client (no proxy env handling — rely on system httpx) ──
    import httpx
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    http_client = httpx.Client(proxy=proxy) if proxy else None

    client = openai.OpenAI(
        base_url=args.api_base,
        api_key=api_key,
        http_client=http_client,
    )

    # ── Load & build ──
    sources = load_source_texts(input_dir)
    log.info(f"Source texts: {len(sources)}")
    if not sources:
        log.error("No valid source texts found.")
        sys.exit(1)

    jobs = build_jobs(sources, models)
    log.info(f"Total jobs: {len(jobs)} (= {len(models)} models × {len(jobs)//len(models)} texts)")

    checkpoint = Checkpoint(output_dir / "checkpoint.json")
    skipped = sum(1 for j in jobs if checkpoint.contains(j["model"], j["stem"], j["direction"]))
    log.info(f"Already done: {skipped} / {len(jobs)}")

    # ── Execute (parallel) ──
    workers = min(args.workers, len(jobs))
    log.info(f"Workers: {workers} (parallel)")

    results = {"ok": 0, "fail": 0, "skip": skipped}
    model_stats = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_job, j, client, output_dir, checkpoint, args.seed)
                   for j in jobs]

        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r["status"] == "ok":
                results["ok"] += 1
                s = model_stats.setdefault(r["slug"], {"ok": 0, "fail": 0})
                s["ok"] += 1
            elif r["status"] == "fail":
                results["fail"] += 1
                s = model_stats.setdefault(r["slug"], {"ok": 0, "fail": 0})
                s["fail"] += 1

            # Progress every 20
            if i % 20 == 0 or i == len(jobs) - skipped:
                done = results["ok"] + results["fail"]
                total = len(jobs) - skipped
                log.info(f"  Progress: {done}/{total} ({results['ok']} ok, {results['fail']} fail)")

    # ── Summary ──
    total_done = results["ok"] + results["fail"] + skipped
    log.info(f"\n{'='*60}")
    log.info(f"  COMPLETE")
    log.info(f"  Total: {len(jobs)} | OK: {results['ok']} | Fail: {results['fail']} | Skip: {skipped}")
    log.info(f"")
    log.info(f"  Per model:")
    for slug in sorted(model_stats):
        s = model_stats[slug]
        log.info(f"    {slug:30s}  ✓ {s['ok']:3d}  ✗ {s['fail']:3d}")
    log.info(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"  翻译完成")
    print(f"  Total: {len(jobs)} | OK: {results['ok']} | Fail: {results['fail']} | Skip: {skipped}")
    for m in models:
        slug = model_slug(m)
        d = output_dir / slug
        n = len(list(d.glob("*.txt"))) if d.exists() else 0
        print(f"  {slug:30s} → {d}/  ({n} files)")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model batch LLM translator (SMT experiment)"
    )
    parser.add_argument("-i", "--input-dir", default="source_texts")
    parser.add_argument("-o", "--output-dir", default="translations")
    parser.add_argument("-m", "--models", default="all",
                        help="Comma-separated or 'all'")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--api-base",
                        help="API base URL (default depends on --provider)")
    parser.add_argument("--provider", default="opencode-go",
                        choices=["opencode-go", "deepseek"],
                        help="API provider (default: opencode-go)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Deprecated: checkpoint is now every translation")
    args = parser.parse_args()

    # Provider-specific defaults
    if args.provider == "deepseek":
        if not args.api_base:
            args.api_base = "https://api.deepseek.com"
        if not args.api_key:
            import os
            args.api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not args.api_key:
                try:
                    args.api_key = Path(
                        os.path.expanduser("~/.config/opencode/secrets/deepseek")
                    ).read_text().strip()
                except:
                    pass
        # Only DeepSeek models
        if args.models == "all":
            args.models = "deepseek-v4-pro,deepseek-v4-flash"
    else:  # opencode-go
        if not args.api_base:
            args.api_base = "https://opencode.ai/zen/go/v1"
        if not args.api_key:
            import os
            args.api_key = os.environ.get("OPENCODE_API_KEY")

    run_pipeline(args)


if __name__ == "__main__":
    main()
