"""Lightweight ONNX embedding model for fact vector search.

Uses all-MiniLM-L6-v2 via onnxruntime — no PyTorch dependency.
384-dim embeddings, mean-pooled + L2-normalized.

Model files are downloaded on first use and cached in the user's
Hermes home directory.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
    from tokenizers import Tokenizer

# Lazy-loaded singletons
_sess: InferenceSession | None = None
_tok: Tokenizer | None = None
_available: bool = False

# Model download URLs
_MODEL_URL = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
_TOKENIZER_URL = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"

# Cache directory (Hermes home)
def _get_cache_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "holographic_embeddings"
    except ImportError:
        return Path.home() / ".hermes" / "holographic_embeddings"

_CACHE_DIR = _get_cache_dir()
_MODEL_PATH = _CACHE_DIR / "model.onnx"
_TOKENIZER_PATH = _CACHE_DIR / "tokenizer.json"

EMBEDDING_DIM = 384


def _download_file(url: str, dest: Path) -> bool:
    """Download a file from URL to destination."""
    try:
        import urllib.request
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest))
        return True
    except Exception:
        return False


def _load() -> None:
    """Lazy-load the ONNX session and tokenizer. Idempotent."""
    global _sess, _tok, _available

    if _sess is not None:
        return  # Already loaded

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer as Tkz

        # Download model files if not present
        if not _MODEL_PATH.exists():
            if not _download_file(_MODEL_URL, _MODEL_PATH):
                _available = False
                return

        if not _TOKENIZER_PATH.exists():
            if not _download_file(_TOKENIZER_URL, _TOKENIZER_PATH):
                _available = False
                return

        _sess = ort.InferenceSession(
            str(_MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
        _tok = Tkz.from_file(str(_TOKENIZER_PATH))
        _available = True
    except Exception:
        _available = False


def is_available() -> bool:
    """Check if the embedding model is available and loaded."""
    _load()
    return _available


def encode(text: str) -> np.ndarray:
    """Encode a text string into a 384-dim L2-normalized embedding.

    Returns None if the model is unavailable.
    """
    _load()
    if not _available or _sess is None or _tok is None:
        return None

    encoded = _tok.encode(text)
    seq_len = len(encoded.ids)

    input_ids = np.array([encoded.ids], dtype=np.int64)
    attention_mask = np.array([[1] * seq_len], dtype=np.int64)
    token_type_ids = np.array([[0] * seq_len], dtype=np.int64)

    outputs = _sess.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })

    # Mean pooling over token embeddings
    token_embeddings = outputs[0][0]  # (seq_len, 384)
    mask = attention_mask[0]
    pooled = (token_embeddings * mask[:, np.newaxis]).sum(axis=0) / mask.sum()

    # L2 normalize
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm

    return pooled.astype(np.float32)


def encode_batch(texts: list[str]) -> list[np.ndarray | None]:
    """Encode multiple texts. Returns list of embeddings (None for failures)."""
    results = []
    for text in texts:
        try:
            results.append(encode(text))
        except Exception:
            results.append(None)
    return results
