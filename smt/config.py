"""
Configuration management for the SMT pipeline.

Loads settings from a YAML file with sensible defaults for
the cross-architecture experiment protocol.
"""

import os
from typing import Any, Dict, Optional

from . import utils

logger = utils.logger

# ─── Default Configuration ───────────────────────────────────────────

DEFAULT_CONFIG: Dict[str, Any] = {
    # Data
    "data": {
        "max_train_sentences": 100_000,
        "max_src_len": 80,           # Max source tokens per sentence
        "max_tgt_len": 100,          # Max target tokens per sentence
        "src_lang": "zh",
        "tgt_lang": "en",
    },
    # IBM Alignment
    "alignment": {
        "model": "ibm2",             # "ibm1" or "ibm2"
        "iterations_model1": 5,
        "iterations_model2": 5,
        "null_prob": 0.08,           # NULL insertion probability
    },
    # Phrase table
    "phrase_table": {
        "max_phrase_len": 7,
        "min_phrase_count": 2,       # Minimum frequency to include phrase pair
        "score_features": True,      # Compute all 4 standard features
    },
    # Language model
    "language_model": {
        "order": 5,                  # N-gram order
        "smoothing": "kneser_ney",   # "kneser_ney" or "modified_kn"
        "prune_threshold": 1e-7,     # Prune n-grams below this prob
    },
    # Parallel processing
    "parallel": {
        "enabled": True,             # Use multiprocessing for CPU-bound steps
        "num_workers": 0,            # 0 = auto-detect (cpu_count - 1)
    },
    # Decoder
    "decoder": {
        "beam_size": 10,
        "stack_size": 100,
        "max_phrase_len": 7,
        "distortion_limit": 6,
        "word_penalty": -0.5,        # Word penalty weight
        "lm_weight": 1.0,
        "translation_weight": 1.0,
        "distortion_weight": 0.3,
        "future_cost_estimate": True,
        "oov_strategy": "copy",     # copy | drop | unk
    },
    # Vocabulary
    "vocabulary": {
        "min_freq": 2,               # Minimum word frequency for vocabulary
        "max_size": 50000,           # Maximum vocabulary size (null = unlimited)
        "src_vocab_min_freq": 2,     # Phrase table: min source word frequency
        "tgt_vocab_min_freq": 2,     # Phrase table: min target word frequency
        "oov_strategy": "copy",      # OOV strategy: copy | drop | unk
        "report_coverage": True,     # Generate coverage report after training
    },
    # Moses Docker (optional)
    "moses": {
        "image": "amake/moses-smt",
        "container_name": "moses_train",
        "ram_gb": 8,
        "cpus": 4,
        "working_dir": "/moses/model",
    },
    # Output
    "output": {
        "bleu_tokenize": "zh" if None else "intl",
        "log_file": None,
        "log_level": "INFO",
    },
}


class Config:
    """Experiment configuration with YAML loading and attribute-style access.

    Usage:
        cfg = Config("config.yaml")
        cfg["decoder.beam_size"]   # → 10
        cfg.get("decoder.beam_size", default=10)
    """

    def __init__(self, path: Optional[str] = None):
        self._data: Dict[str, Any] = self._deep_copy(DEFAULT_CONFIG)
        if path and os.path.exists(path):
            self._load_yaml(path)

    def _load_yaml(self, path: str) -> None:
        """Merge YAML file values into defaults."""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed; trying to install...")
            import subprocess
            subprocess.check_call(
                [utils.sys.executable or "python3", "-m", "pip", "install", "pyyaml"]
            )
            import yaml

        with open(path, encoding="utf-8") as f:
            overrides = yaml.safe_load(f)
        if overrides:
            self._deep_merge(self._data, overrides)
            logger.info(f"Loaded config from {path}")

    @staticmethod
    def _deep_copy(d: Dict) -> Dict:
        """Deep copy a dictionary."""
        import copy
        return copy.deepcopy(d)

    @staticmethod
    def _deep_merge(base: Dict, overrides: Dict) -> None:
        """Recursively merge overrides into base."""
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get dotted-key value, e.g. 'decoder.beam_size'."""
        parts = key.split(".")
        node = self._data
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        val = self.get(key)
        if val is None:
            raise KeyError(f"Config key not found: {key}")
        return val

    def __setitem__(self, key: str, value: Any) -> None:
        parts = key.split(".")
        node = self._data
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def as_dict(self) -> Dict[str, Any]:
        """Return a copy of the full config dict."""
        return self._deep_copy(self._data)

    def to_json(self, path: str) -> None:
        """Save current config as JSON."""
        utils.save_json(self._data, path)
