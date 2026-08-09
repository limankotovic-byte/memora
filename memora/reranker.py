"""Optional cross-encoder reranking for hybrid search.

Adds a precision stage on top of RRF fusion: given the top fused candidates,
a cross-encoder scores each (query, document) pair and reorders the list.

The model is loaded lazily on first use and cached for the process lifetime.
Configuration via environment variables:

    MEMORA_RERANKER_MODEL   model name (default: BAAI/bge-reranker-base)
                            set to "none"/"disabled" to disable reranking

Degrades gracefully: if the model cannot be loaded, rerank() returns
candidates unchanged so hybrid search keeps working.
"""

import logging
import os
import threading
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-base"
DISABLED_VALUES = {"none", "disabled", "off", "0", ""}

_lock = threading.Lock()
_model: Optional[object] = None
_model_name: Optional[str] = None
_model_failed: bool = False


def configured_model() -> Optional[str]:
    """Return the configured model name, or None if reranking is disabled."""
    name = os.environ.get("MEMORA_RERANKER_MODEL", DEFAULT_MODEL)
    if name is None or name.strip().lower() in DISABLED_VALUES:
        return None
    return name.strip()


def _load_model() -> Optional[object]:
    """Load the cross-encoder once. Returns None if disabled or load failed."""
    global _model, _model_name, _model_failed
    with _lock:
        if _model is not None:
            return _model
        name = configured_model()
        if name is None:
            _model_failed = True
            return None
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning(
                "Reranker %s requested but sentence-transformers is not installed; "
                "reranking disabled. Install with: pipx inject memora-mcp sentence-transformers",
                name,
            )
            _model_failed = True
            return None
        try:
            logger.info("Loading reranker model %s (first use only)", name)
            _model = CrossEncoder(name, max_length=512)
            _model_name = name
            return _model
        except Exception:
            logger.exception("Failed to load reranker model %s; reranking disabled", name)
            _model_failed = True
            return None


def rerank(query: str, candidates: Sequence[Tuple[int, str]], top_n: int) -> List[int]:
    """Reorder candidate memory ids by cross-encoder relevance to the query.

    Args:
        query: The search query
        candidates: Sequence of (memory_id, document_text) pairs
        top_n: How many candidates to rerank (the rest keep their order at the tail)

    Returns:
        Memory ids reordered by reranker score, best first. On any failure or
        when disabled, returns candidate ids in original order.
    """
    ids = [cand[0] for cand in candidates]
    if not ids or top_n <= 0:
        return ids
    if len(ids) == 1:
        return ids

    model = _load_model()
    if model is None:
        return ids

    head = candidates[:top_n]
    tail_ids = ids[top_n:]

    pairs = [(query, text) for _, text in head]
    try:
        with _lock:
            scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
    except Exception:
        logger.exception("Reranker inference failed; returning original order")
        return ids

    ranked = sorted(zip(head, scores), key=lambda x: x[1], reverse=True)
    return [cand[0] for cand, _ in ranked] + tail_ids


def reranker_status() -> Dict[str, object]:
    """Report model status for debugging."""
    name = configured_model()
    if name is None:
        return {"enabled": False, "reason": "disabled_by_env"}
    if _model_failed:
        return {"enabled": False, "reason": "load_failed", "model": name}
    if _model is not None:
        return {"enabled": True, "model": _model_name}
    return {"enabled": True, "model": name, "loaded": False}
