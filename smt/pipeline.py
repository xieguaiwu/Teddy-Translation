"""
High-level SMT pipeline orchestrator.

Brings together all SMT components (data prep, alignment, phrase table,
language model, decoder) into a unified training and translation pipeline.
Supports both pure-Python and Moses Docker backends.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any

from . import utils
from . import data_prep, ibm_align, phrase_table, language_model, decoder, evaluation, moses_orch
from .config import Config
from .vocab_manager import VocabularyManager

logger = utils.logger


class SMTPipeline:
    """End-to-end SMT training and translation pipeline.

    Manages the complete workflow:
    1. Data preparation (tokenization, truecasing, cleaning)
    2. Word alignment (IBM Model 1/2)
    3. Phrase extraction and scoring
    4. Language model training
    5. Translation (beam search decoder)
    6. Evaluation (BLEU)
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline with optional config file.

        Args:
            config_path: Path to YAML config file. Uses defaults if None.
        """
        self.config = Config(config_path) if config_path else Config()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging from config."""
        log_level = self.config.get("output.log_level", "INFO")
        log_file = self.config.get("output.log_file")
        utils.setup_logging(level=log_level, log_file=log_file)

    # ─── Data Preparation ───────────────────────────────────────────

    def prepare_data(
        self,
        src_raw: str,
        tgt_raw: str,
        output_prefix: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None,
        truecaser_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """Prepare parallel corpus for SMT training.

        Args:
            src_raw: Raw source file.
            tgt_raw: Raw target file.
            output_prefix: Prefix for output files.
            src_lang: Source language code.
            tgt_lang: Target language code.
            max_src_len: Max source tokens.
            max_tgt_len: Max target tokens.
            truecaser_path: Pre-trained truecaser model path.

        Returns:
            Dict of {step: output_path}.
        """
        src_lang = src_lang or self.config.get("data.src_lang", "zh")
        tgt_lang = tgt_lang or self.config.get("data.tgt_lang", "en")
        max_src_len = max_src_len or self.config.get("data.max_src_len", 80)
        max_tgt_len = max_tgt_len or self.config.get("data.max_tgt_len", 100)
        max_pairs = self.config.get("data.max_train_sentences")

        src_out = f"{output_prefix}.tok.{src_lang}"
        tgt_out = f"{output_prefix}.tok.{tgt_lang}"

        kept, discarded = data_prep.prepare_corpus(
            src_raw=src_raw, tgt_raw=tgt_raw,
            src_out=src_out, tgt_out=tgt_out,
            src_lang=src_lang, tgt_lang=tgt_lang,
            truecaser_model=truecaser_path,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            max_pairs=max_pairs,
        )

        return {
            "src_prepared": src_out,
            "tgt_prepared": tgt_out,
            "kept": str(kept),
            "discarded": str(discarded),
        }

    def train_truecaser(
        self,
        corpus_path: str,
        output_path: str,
        lang: str = "en",
    ) -> str:
        """Train a truecasing model from a tokenized corpus."""
        data_prep.train_truecaser_from_corpus(corpus_path, output_path, lang=lang)
        return output_path

    # ─── Pure-Python SMT Training ────────────────────────────────────

    def train_python(
        self,
        src_file: str,
        tgt_file: str,
        output_dir: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        max_sentences: Optional[int] = None,
        num_workers: int = 0,
        skip_prep: bool = False,
        warm_start_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train a pure-Python phrase-based SMT model.

        Pipeline:
        1. Tokenize and clean corpus
        2. Train IBM Model 2 word alignment
        3. Extract phrase table
        4. Train Kneser-Ney language model

        Args:
            src_file: Source corpus file (raw text).
            tgt_file: Target corpus file (raw text).
            output_dir: Output directory for trained model files.
            src_lang: Source language.
            tgt_lang: Target language.
            max_sentences: Max sentence pairs to use.
            num_workers: Number of parallel workers (0 = auto from config).

        Returns:
            Dict with model metadata and file paths.
        """
        src_lang = src_lang or self.config.get("data.src_lang", "zh")
        tgt_lang = tgt_lang or self.config.get("data.tgt_lang", "en")
        utils.ensure_dir(output_dir)

        # Resolve num_workers from config if not explicitly set
        if num_workers == 0:
            parallel_cfg = self.config.get("parallel", {})
            if parallel_cfg.get("enabled", True):
                num_workers = parallel_cfg.get("num_workers", 0)

        logger.info("=" * 60)
        logger.info("Pure-Python SMT Training Pipeline")
        logger.info("=" * 60)

        # Step 1: Data preparation
        if skip_prep:
            logger.info("[1/4] Skipping data prep (using pre-tokenized files)...")
            src_sentences_raw = utils.read_lines(src_file)
            tgt_sentences_raw = utils.read_lines(tgt_file)
        else:
            logger.info("[1/4] Preparing data...")
            prep_out = self.prepare_data(
                src_raw=src_file, tgt_raw=tgt_file,
                output_prefix=os.path.join(output_dir, "train"),
                src_lang=src_lang, tgt_lang=tgt_lang,
            )
            src_sentences_raw = utils.read_lines(prep_out["src_prepared"])
            tgt_sentences_raw = utils.read_lines(prep_out["tgt_prepared"])

        src_sentences = [s.split() for s in src_sentences_raw]
        tgt_sentences = [t.split() for t in tgt_sentences_raw]

        if max_sentences:
            src_sentences = src_sentences[:max_sentences]
            tgt_sentences = tgt_sentences[:max_sentences]

        logger.info(f"Training on {len(src_sentences)} sentence pairs")

        # Step 2: Word alignment (IBM Model 2)
        align_cfg = self.config.get("alignment", {})

        # Warm-start: load previous IBM table if provided
        warm_t_table = None
        if warm_start_model:
            lex_path = os.path.join(warm_start_model, "lex.table")
            if os.path.exists(lex_path):
                logger.info(f"[Warm-start] Loading IBM table from {lex_path}")
                warm_ibm = ibm_align.IBM2()
                warm_ibm.load(lex_path)
                warm_t_table = warm_ibm.t
                logger.info(f"[Warm-start] Loaded {warm_ibm.num_params_t} parameters")
            else:
                logger.warning(f"[Warm-start] lex.table not found in {warm_start_model}")

        logger.info("[2/4] Training IBM Model 2 alignment...")
        ibm_model = ibm_align.train_ibm(
            src_sentences, tgt_sentences,
            model=align_cfg.get("model", "ibm2"),
            iterations_model1=align_cfg.get("iterations_model1", 5),
            iterations_model2=align_cfg.get("iterations_model2", 5),
            null_prob=align_cfg.get("null_prob", 0.08),
            num_workers=num_workers,
            warm_t_table=warm_t_table,
        )

        # Save alignment model
        ibm_model.save(
            path_t=os.path.join(output_dir, "lex.table"),
            path_a=os.path.join(output_dir, "distortion.table") if hasattr(ibm_model, 'a') else None,
        )

        # Extract alignments
        alignments = ibm_model.extract_alignments(src_sentences, tgt_sentences)
        alignments_set = [set(al) for al in alignments]

        # Step 3: Phrase extraction
        logger.info("[3/4] Building phrase table...")
        pt_cfg = self.config.get("phrase_table", {})
        vocab_cfg = self.config.get("vocabulary", {})
        pt = phrase_table.build_phrase_table(
            src_sentences, tgt_sentences,
            alignments_set,
            t_table=ibm_model.t if hasattr(ibm_model, 't') else getattr(ibm_model, 't', {}),
            max_phrase_len=pt_cfg.get("max_phrase_len", 7),
            min_count=pt_cfg.get("min_phrase_count", 2),
            score_features=pt_cfg.get("score_features", True),
            src_vocab_min_freq=vocab_cfg.get("src_vocab_min_freq"),
            tgt_vocab_min_freq=vocab_cfg.get("tgt_vocab_min_freq"),
            num_workers=num_workers,
        )

        # Save phrase table
        pt_path = os.path.join(output_dir, "phrase_table.txt")
        phrase_table.save_phrase_table(pt, pt_path)

        # Step 4: Language model
        logger.info("[4/4] Training Kneser-Ney language model...")
        lm_cfg = self.config.get("language_model", {})

        # Extract target-side sentences for LM training
        lm_sentences = tgt_sentences
        lm = language_model.train_lm(
            lm_sentences,
            order=lm_cfg.get("order", 5),
            smoothing=lm_cfg.get("smoothing", "kneser_ney"),
            prune_threshold=lm_cfg.get("prune_threshold", 1e-7),
            num_workers=num_workers,
        )

        # Save LM
        lm_path = os.path.join(output_dir, "lm.json")
        lm.save(lm_path)

        # ── Vocabulary Statistics ────────────────────────────────────
        logger.info("Computing vocabulary statistics...")

        # Build source and target vocabularies
        src_vocab = VocabularyManager.build_from_corpus(
            src_sentences,
            min_freq=vocab_cfg.get("min_freq", 2),
            max_size=vocab_cfg.get("max_size"),
        )
        tgt_vocab = VocabularyManager.build_from_corpus(
            tgt_sentences,
            min_freq=vocab_cfg.get("min_freq", 2),
            max_size=vocab_cfg.get("max_size"),
        )

        # Coverage on training data (self-check)
        src_coverage = src_vocab.coverage_report(src_sentences)
        tgt_coverage = tgt_vocab.coverage_report(tgt_sentences)

        # Save vocabularies
        vocab_cfg_dict = self.config.get("vocabulary", {})
        if vocab_cfg_dict.get("report_coverage", True):
            src_vocab.save(os.path.join(output_dir, "src_vocab.json"))
            tgt_vocab.save(os.path.join(output_dir, "tgt_vocab.json"))

        vocab_stats = {
            "source": {
                "types": src_vocab.total_types,
                "tokens": src_vocab.total_tokens,
                "coverage": src_coverage["coverage"],
                "oov_rate": src_coverage["oov_rate"],
                "freq_distribution": src_vocab.get_stats().get("freq_distribution", {}),
                "hapax_legomena": src_vocab.get_stats().get("hapax_legomena", 0),
            },
            "target": {
                "types": tgt_vocab.total_types,
                "tokens": tgt_vocab.total_tokens,
                "coverage": tgt_coverage["coverage"],
                "oov_rate": tgt_coverage["oov_rate"],
                "freq_distribution": tgt_vocab.get_stats().get("freq_distribution", {}),
                "hapax_legomena": tgt_vocab.get_stats().get("hapax_legomena", 0),
            },
            "vocab_config": vocab_cfg_dict,
        }

        logger.info(
            f"  Source vocab: {src_vocab.total_types} types, "
            f"coverage={src_coverage['coverage']:.2%}"
        )
        logger.info(
            f"  Target vocab: {tgt_vocab.total_types} types, "
            f"coverage={tgt_coverage['coverage']:.2%}"
        )

        # Save config with vocabulary stats
        model_info = {
            "type": "python_smt",
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "num_sentence_pairs": len(src_sentences),
            "vocab_size": lm.vocab_size,
            "lm_order": lm.order,
            "phrase_table_entries": len(pt),
            "num_alignments": sum(len(a) for a in alignments_set),
            "vocabulary_stats": vocab_stats,
            "files": {
                "lex_table": "lex.table",
                "phrase_table": "phrase_table.txt",
                "language_model": "lm.json",
                "src_vocab": "src_vocab.json",
                "tgt_vocab": "tgt_vocab.json",
            },
        }
        with open(os.path.join(output_dir, "model_info.json"), "w") as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Training complete! Model saved to {output_dir}")
        logger.info(f"  - {len(pt)} phrase table entries")
        logger.info(f"  - {lm.vocab_size} vocabulary types")
        logger.info(f"  - {lm.order}-gram language model")

        # Store for later use
        self._pt = pt
        self._lm = lm
        self._model_dir = output_dir

        return model_info

    # ─── Translation ─────────────────────────────────────────────────

    def load_model(self, model_dir: str) -> None:
        """Load a trained SMT model from directory.

        Args:
            model_dir: Path to trained model directory.
        """
        pt_path = os.path.join(model_dir, "phrase_table.txt")
        lm_path = os.path.join(model_dir, "lm.json")

        if not os.path.exists(pt_path) or not os.path.exists(lm_path):
            raise FileNotFoundError(
                f"Model files not found in {model_dir}. "
                f"Need: phrase_table.txt, lm.json"
            )

        self._pt = phrase_table.load_phrase_table(pt_path)
        self._lm = language_model.KneserNeyLM.load(lm_path)
        self._model_dir = model_dir

        logger.info(f"Model loaded from {model_dir}")

    def translate_python(
        self,
        input_file: str,
        output_file: str,
        model_dir: Optional[str] = None,
        src_lang: Optional[str] = None,
        beam_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Translate using pure-Python SMT decoder.

        Args:
            input_file: Path to source text (one sentence per line).
            output_file: Path to write translations.
            model_dir: Trained model directory (loads if not already loaded).
            src_lang: Source language.
            beam_size: Beam search width.

        Returns:
            Dict with translation results and stats.
        """
        utils.ensure_dir(os.path.dirname(output_file) or ".")

        # Load model if needed
        if model_dir:
            self.load_model(model_dir)
        if not hasattr(self, '_pt') or not hasattr(self, '_lm'):
            raise RuntimeError("No model loaded. Call load_model() or train_python() first.")

        # Read input
        src_sentences = utils.read_lines(input_file)
        src_lang = src_lang or self.config.get("data.src_lang", "zh")

        # Tokenize input
        logger.info(f"Tokenizing {len(src_sentences)} input sentences...")
        src_tokenized = []
        for sent in src_sentences:
            tok = data_prep.tokenize(sent, lang=src_lang)
            src_tokenized.append(tok.split())

        # Initialize decoder
        decoder_cfg = self.config.get("decoder", {})
        vocab_cfg = self.config.get("vocabulary", {})
        oov_strategy = decoder_cfg.get("oov_strategy") or vocab_cfg.get("oov_strategy", "copy")
        dec = decoder.PhraseDecoder(
            phrase_table=self._pt,
            lm=self._lm,
            config=decoder_cfg,
            oov_strategy=oov_strategy,
        )

        if beam_size:
            dec.beam_size = beam_size

        # Translate
        logger.info(f"Translating with beam_size={dec.beam_size}...")
        results = decoder.batch_translate(dec, src_tokenized, verbose=True)

        # Write output (detokenized)
        translations: List[str] = []
        with open(output_file, "w", encoding="utf-8") as f:
            for tokens, score in results:
                # Detokenize
                if self.config.get("data.tgt_lang", "en") == "zh":
                    text = data_prep.detokenize_zh(tokens)
                else:
                    text = data_prep.detokenize_en(tokens)
                translations.append(text)
                f.write(text + "\n")

        logger.info(f"Translation saved to {output_file} ({len(translations)} sentences)")

        return {
            "num_sentences": len(translations),
            "output_file": output_file,
            "avg_score": sum(s for _, s in results) / max(len(results), 1),
        }

    def translate_sentence(
        self,
        source_text: str,
        src_lang: Optional[str] = None,
    ) -> str:
        """Translate a single sentence.

        Args:
            source_text: Raw source text.
            src_lang: Source language.

        Returns:
            Translated text.
        """
        if not hasattr(self, '_pt') or not hasattr(self, '_lm'):
            raise RuntimeError("No model loaded. Call load_model() or train_python() first.")

        src_lang = src_lang or self.config.get("data.src_lang", "zh")

        # Tokenize
        tok = data_prep.tokenize(source_text, lang=src_lang)
        src_tokens = tok.split()

        # Decode
        decoder_cfg = self.config.get("decoder", {})
        vocab_cfg = self.config.get("vocabulary", {})
        oov_strategy = decoder_cfg.get("oov_strategy") or vocab_cfg.get("oov_strategy", "copy")
        dec = decoder.PhraseDecoder(
            phrase_table=self._pt,
            lm=self._lm,
            config=decoder_cfg,
            oov_strategy=oov_strategy,
        )

        translation, _ = dec.decode(src_tokens)

        # Detokenize
        tgt_lang = self.config.get("data.tgt_lang", "en")
        if tgt_lang == "zh":
            return data_prep.detokenize_zh(translation)
        return data_prep.detokenize_en(translation)

    # ─── Moses Docker Backend ────────────────────────────────────────

    def train_moses(
        self,
        src_file: str,
        tgt_file: str,
        output_dir: str,
        src_lang: str = "zh",
        tgt_lang: str = "en",
        lm_order: int = 5,
        use_mert: bool = False,
        dev_src: Optional[str] = None,
        dev_tgt: Optional[str] = None,
    ) -> bool:
        """Train a Moses SMT model via Docker (if available).

        Falls back to Python SMT if Docker is unavailable.

        Args:
            src_file: Source corpus file.
            tgt_file: Target corpus file.
            output_dir: Model output directory.
            src_lang: Source language code.
            tgt_lang: Target language code.
            lm_order: LM n-gram order.
            use_mert: Whether to run MERT tuning.
            dev_src: Development source for MERT.
            dev_tgt: Development target for MERT.

        Returns:
            True if training succeeded.
        """
        if moses_orch._docker_available():
            return moses_orch.train_moses_model(
                src_file=src_file, tgt_file=tgt_file,
                output_dir=output_dir,
                src_lang=src_lang, tgt_lang=tgt_lang,
                lm_order=lm_order,
                use_mert=use_mert,
                dev_src=dev_src, dev_tgt=dev_tgt,
            )
        else:
            logger.warning("Docker not available. Falling back to Python SMT.")
            self.train_python(
                src_file=src_file, tgt_file=tgt_file,
                output_dir=output_dir,
                src_lang=src_lang, tgt_lang=tgt_lang,
            )
            return True

    def translate_moses(
        self,
        input_file: str,
        output_file: str,
        model_dir: str,
    ) -> bool:
        """Translate via Moses Docker (if available), else Python."""
        if moses_orch._docker_available():
            return moses_orch.translate_moses(input_file, output_file, model_dir)
        else:
            logger.warning("Docker not available. Using Python decoder.")
            self.translate_python(
                input_file=input_file,
                output_file=output_file,
                model_dir=model_dir,
            )
            return True

    # ─── Batch Translation for Experiment ────────────────────────────

    def batch_translate(
        self,
        source_dir: str,
        output_dir: str,
        model_dir: str,
        src_lang: str = "zh",
        tgt_lang: str = "en",
        file_pattern: str = ".txt",
    ) -> Dict[str, Any]:
        """Batch translate multiple files for the experiment protocol.

        Translates all files in source_dir matching the pattern.

        Args:
            source_dir: Directory with source files.
            output_dir: Output directory for translations.
            model_dir: Trained model directory.
            src_lang: Source language.
            tgt_lang: Target language.
            file_pattern: File extension filter.

        Returns:
            Dict mapping (input_file → output_file) for all translations.
        """
        utils.ensure_dir(output_dir)
        self.load_model(model_dir)

        # Get all source files
        all_files = [f for f in os.listdir(source_dir) if f.endswith(file_pattern)]
        all_files.sort()
        logger.info(f"Batch translating {len(all_files)} files from {source_dir}")

        results: Dict[str, str] = {}
        for filename in all_files:
            src_path = os.path.join(source_dir, filename)
            out_path = os.path.join(output_dir, filename)

            self.translate_python(
                input_file=src_path,
                output_file=out_path,
                src_lang=src_lang,
            )
            results[filename] = out_path

        return results

    # ─── Vocabulary Analysis ─────────────────────────────────────────

    def analyze_vocabulary(
        self,
        corpus_path: str,
        lang: str = "en",
        min_freq: int = 2,
        max_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze vocabulary of a corpus file.

        Tokenizes the file, builds vocabulary, and returns statistics.
        Useful for estimating vocabulary size before training or
        diagnosing OOV issues in a test set.

        Args:
            corpus_path: Path to corpus file (one sentence per line).
            lang: Language code ('zh' or 'en') for tokenization.
            min_freq: Minimum word frequency for vocabulary.
            max_size: Maximum vocabulary size.

        Returns:
            Dict with vocabulary statistics:
                types, tokens, hapax_legomena, coverage, freq_dist, top_50.
        """
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus not found: {corpus_path}")

        logger.info(f"Analyzing vocabulary in {corpus_path} (lang={lang})...")

        # Read and tokenize
        raw_lines = utils.read_lines(corpus_path)
        tokenized: List[List[str]] = []
        for line in raw_lines:
            tok = data_prep.tokenize(line, lang=lang)
            tokenized.append(tok.split())

        # Build vocabulary
        vocab = VocabularyManager.build_from_corpus(
            tokenized, min_freq=min_freq, max_size=max_size,
        )

        # Coverage report
        coverage = vocab.coverage_report(tokenized)
        stats = vocab.get_stats()

        result: Dict[str, Any] = {
            "file": corpus_path,
            "language": lang,
            "sentences": len(tokenized),
            "vocab_types": vocab.total_types,
            "vocab_tokens": vocab.total_tokens,
            "min_freq": min_freq,
            "max_size": max_size,
            "hapax_legomena": stats.get("hapax_legomena", 0),
            "mean_freq": stats.get("mean_freq", 0),
            "median_freq": stats.get("median_freq", 0),
            "freq_distribution": stats.get("freq_distribution", {}),
            "coverage": coverage["coverage"],
            "oov_rate": coverage["oov_rate"],
            "sent_coverage": coverage["sent_coverage"],
            "top_50": vocab.get_top_n(50),
        }

        logger.info(
            f"Vocabulary analysis: {vocab.total_types} types / "
            f"{vocab.total_tokens} tokens, "
            f"coverage={coverage['coverage']:.2%}, "
            f"OOV={coverage['oov_rate']:.2%}"
        )

        return result

    # ─── Evaluation ──────────────────────────────────────────────────

    def evaluate(
        self,
        hypotheses_file: str,
        references_file: str,
        tokenize: str = "intl",
    ) -> Dict[str, Any]:
        """Evaluate translations against references.

        Args:
            hypotheses_file: Translated text file.
            references_file: Reference text file.
            tokenize: BLEU tokenization method.

        Returns:
            BLEU evaluation results.
        """
        hyps = utils.read_lines(hypotheses_file)
        refs = utils.read_lines(references_file)

        logger.info(f"Evaluating {len(hyps)} sentences against {len(refs)} references")

        return evaluation.translation_quality_report(
            hyps, refs, tokenize=tokenize,
        )

    # ─── Full Pipeline ──────────────────────────────────────────────

    def run_full_pipeline(
        self,
        train_src: str,
        train_tgt: str,
        test_src: str,
        test_tgt: str,
        output_dir: str,
        src_lang: str = "zh",
        tgt_lang: str = "en",
        use_moses: bool = False,
    ) -> Dict[str, Any]:
        """Run the complete SMT pipeline: train → translate → evaluate.

        Args:
            train_src: Training source file.
            train_tgt: Training target file.
            test_src: Test source file.
            test_tgt: Test reference file.
            output_dir: Output directory.
            src_lang: Source language.
            tgt_lang: Target language.
            use_moses: Use Moses Docker if available.

        Returns:
            Full results dict.
        """
        model_dir = os.path.join(output_dir, "model")
        utils.ensure_dir(model_dir)

        results: Dict[str, Any] = {}

        # Train
        if use_moses:
            logger.info("Training Moses SMT model...")
            self.train_moses(train_src, train_tgt, model_dir, src_lang, tgt_lang)
        else:
            logger.info("Training Python SMT model...")
            self.train_python(train_src, train_tgt, model_dir, src_lang, tgt_lang)

        results["training"] = {
            "model_dir": model_dir,
            "type": "moses" if use_moses else "python_smt",
        }

        # Translate
        test_out = os.path.join(output_dir, "test_translated.txt")
        self.translate_python(
            input_file=test_src,
            output_file=test_out,
            src_lang=src_lang,
        )
        results["translation"] = {"output": test_out}

        # Evaluate
        if os.path.exists(test_tgt):
            eval_result = self.evaluate(test_out, test_tgt)
            results["evaluation"] = eval_result
            logger.info(f"BLEU score: {eval_result.get('bleu', {}).get('bleu', 0.0):.2f}")

        # Save results
        results_path = os.path.join(output_dir, "pipeline_results.json")
        utils.save_json(results, results_path)

        return results
