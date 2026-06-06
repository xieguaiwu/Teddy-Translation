"""
smt — Traditional Phrase-Based Statistical Machine Translation.

Implements the Moses-style phrase-based SMT pipeline described in
the cross-architecture text statistics comparison experimental protocol.
Supports both Docker-based Moses orchestration and a pure-Python fallback
with IBM Model 2 alignment, phrase extraction, Kneser-Ney LM, and beam search decoding.

Submodules:
    config        — YAML-based configuration
    data_prep     — Tokenization, truecasing, corpus cleaning
    ibm_align     — IBM Model 1/2 word alignment (EM algorithm)
    phrase_table  — Phrase extraction and scoring
    language_model— N-gram language model with Kneser-Ney smoothing
    decoder       — Phrase-based beam search decoder
    moses_orch    — Moses Docker orchestration
    pipeline      — High-level training/translation pipeline
    evaluation    — BLEU scoring and quality metrics
    utils         — I/O utilities and helpers
    vocab_manager — Vocabulary management and OOV handling
"""

from . import config, data_prep, ibm_align, phrase_table
from . import language_model, decoder, moses_orch, pipeline, evaluation, utils
from . import vocab_manager
from .pipeline import SMTPipeline
from .vocab_manager import VocabularyManager

__version__ = "1.0.0"
__all__ = [
    "config", "data_prep", "ibm_align", "phrase_table",
    "language_model", "decoder", "moses_orch", "pipeline",
    "evaluation", "utils", "vocab_manager", "SMTPipeline",
    "VocabularyManager",
]
