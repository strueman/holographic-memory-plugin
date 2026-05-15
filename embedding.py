"""intfloat/multilingual-e5-small INT8 ONNX embedding model for fact vector search.

Uses multilingual-e5-small (384-dim) via onnxruntime — no PyTorch dependency.
INT8 quantized model for fast CPU inference.

Supports 100 languages including Chinese and English.

Uses "query: " / "passage: " prefix pattern required by e5 models.

Model files are downloaded on first use and cached in the user's
Hermes home directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnxruntime import InferenceSession
    from tokenizers import Tokenizer

# Lazy-loaded singletons
_np = None
_sess: InferenceSession | None = None
_tok: Tokenizer | None = None
_available: bool = False

# Model download URLs (Teradata INT8 ONNX export — pre-pooled sentence embeddings)
_MODEL_URL = "https://huggingface.co/Teradata/multilingual-e5-small/resolve/main/onnx/model.onnx"
_TOKENIZER_URL = "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json"

# Cache directory (Hermes home)
def _get_cache_dir() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "mnemoss_embeddings"
    except ImportError:
        return Path.home() / ".hermes" / "mnemoss_embeddings"

_CACHE_DIR = _get_cache_dir()
_MODEL_PATH = _CACHE_DIR / "multilingual-e5-small-int8.onnx"
_TOKENIZER_PATH = _CACHE_DIR / "multilingual-e5-small-tokenizer.json"

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
    global _np, _sess, _tok, _available

    if _sess is not None:
        return  # Already loaded

    try:
        import numpy as np  # noqa: F821
        _np = np
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


_MAX_TOKENS = 512  # e5-small max position embeddings


def encode(text: str, prefix: str = "passage") -> np.ndarray:  # type: ignore[name-defined]
    """Encode a text string into a 384-dim L2-normalized embedding.

    Uses multilingual-e5-small's pre-pooled sentence embedding output.
    Requires the correct prefix for proper e5 embedding quality:
    - "passage" (default) for documents/facts
    - "query" for search queries

    The Teradata INT8 ONNX export already handles mean-pooling and returns
    a [batch, 384] sentence_embedding output directly.

    Long inputs are truncated to _MAX_TOKENS to avoid ONNX runtime errors.

    Returns None if the model is unavailable or encoding fails.
    """
    _load()
    if not _available or _sess is None or _tok is None or _np is None:
        return None

    # e5 models require a prefix for proper embeddings
    text_with_prefix = f"{prefix}: " + text

    encoded = _tok.encode(text_with_prefix)
    # Truncate to model's max position embeddings
    ids = encoded.ids[:_MAX_TOKENS]
    seq_len = len(ids)

    # Safety: ensure we don't exceed model limits
    if seq_len == 0:
        return None

    # Teradata ONNX model expects 2D input (batch_size, seq_len), not 3D
    input_ids = _np.array([encoded.ids[:seq_len]], dtype=_np.int64)
    attention_mask = _np.array([[1] * seq_len], dtype=_np.int64)

    try:
        outputs = _sess.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })
    except Exception:
        return None

    # The Teradata model outputs: [token_embeddings, sentence_embedding]
    # sentence_embedding is already mean-pooled and 384-dim
    sentence_emb = outputs[1]  # (1, 384)

    # L2 normalize
    norm = _np.linalg.norm(sentence_emb)
    if norm > 0:
        sentence_emb = sentence_emb / norm

    return sentence_emb[0].astype(_np.float32)  # Return (384,) not (1, 384)


def encode_batch(texts: list[str]) -> list[np.ndarray | None]:  # type: ignore[name-defined]
    """Encode multiple texts. Returns list of embeddings (None for failures)."""
    results = []
    for text in texts:
        try:
            results.append(encode(text))
        except Exception:
            results.append(None)
    return results
