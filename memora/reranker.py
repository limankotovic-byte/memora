"""Optional cross-encoder reranking for hybrid search.

Adds a precision stage on top of RRF fusion: given the top fused candidates,
a cross-encoder scores each (query, document) pair and reorders the list.

The model is loaded lazily on first use, cached for the process lifetime,
and unloaded after a period of inactivity to free RAM (TTL, default 10 min).
Configuration via environment variables:

    MEMORA_RERANKER_MODEL   model name (default: BAAI/bge-reranker-base)
                            set to "none"/"disabled" to disable reranking
    MEMORA_RERANKER_TTL     idle seconds before unloading the model to free
                            RAM (default: 600). Set to 0 to keep it loaded
                            for the whole process lifetime.

Degrades gracefully: if the model cannot be loaded, rerank() returns
candidates unchanged so hybrid search keeps working.
"""

import gc
import logging
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-reranker-base"
DISABLED_VALUES = {"none", "disabled", "off", "0", ""}
DEFAULT_TTL = 600  # seconds of inactivity before unloading to free RAM

_lock = threading.Lock()
_model: Optional[object] = None
_model_name: Optional[str] = None
_model_failed: bool = False
_last_use: float = 0.0


def _ttl_seconds() -> float:
    """Idle timeout before the model is unloaded. 0 disables auto-unload."""
    raw = os.environ.get("MEMORA_RERANKER_TTL", str(DEFAULT_TTL))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_TTL
    return max(0.0, value)


def configured_model() -> Optional[str]:
    """Return the configured model name, or None if reranking is disabled."""
    name = os.environ.get("MEMORA_RERANKER_MODEL", DEFAULT_MODEL)
    if name is None or name.strip().lower() in DISABLED_VALUES:
        return None
    return name.strip()


def _unload_model() -> None:
    """Release the model and force a GC pass to actually free the RAM."""
    global _model, _model_name, _model_failed, _last_use
    if _model is not None:
        logger.info("Unloading reranker model %s (idle TTL reached)", _model_name)
    _model = None
    _model_name = None
    _model_failed = False
    _last_use = 0.0
    gc.collect()


def _load_model() -> Optional[object]:
    """Load the cross-encoder once. Returns None if disabled or load failed."""
    global _model, _model_name, _model_failed, _last_use
    with _lock:
        # Reload after idle TTL: drop the stale model, load fresh on next use
        ttl = _ttl_seconds()
        if ttl > 0 and _model is not None and _last_use and (time.time() - _last_use) > ttl:
            _unload_model()
        if _model is not None:
            _last_use = time.time()
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
            _last_use = time.time()
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
            _last_use = time.time()
    except Exception:
        logger.exception("Reranker inference failed; returning original order")
        return ids

    ranked = sorted(zip(head, scores), key=lambda x: x[1], reverse=True)
    return [cand[0] for cand, _ in ranked] + tail_ids


def reranker_status() -> Dict[str, object]:
    """Report model status for debugging."""
    name = configured_model()
    ttl = _ttl_seconds()
    info: Dict[str, object] = {"ttl_seconds": ttl}
    if name is None:
        return {"enabled": False, "reason": "disabled_by_env", **info}
    if _model_failed:
        return {"enabled": False, "reason": "load_failed", "model": name, **info}
    if _model is not None:
        idle = round(time.time() - _last_use, 1) if _last_use else None
        return {"enabled": True, "model": _model_name, "idle_seconds": idle, **info}
    return {"enabled": True, "model": name, "loaded": False, **info}

