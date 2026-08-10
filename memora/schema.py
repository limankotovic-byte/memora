"""Database schema management and connection helpers."""
from __future__ import annotations

import json
import sqlite3
import threading
import weakref

from .backends import D1Connection

# Cache of backends whose schema has already been ensured in this process.
# We stash a *signature* on each backend instance (so the cache entry dies
# with the backend — no id() reuse hazard), with a WeakKeyDictionary fallback
# for weakref-capable backends that reject direct attribute assignment. If a
# backend supports neither (e.g. __slots__ without __weakref__), caching is
# silently disabled for that instance — ensure_schema() just runs every call,
# which is today's behavior, so this is a safe degradation.
#
# The signature for backends with a ``cache_path`` attribute (e.g.
# ``CloudSQLiteBackend``) is the underlying file's ``(st_ino, st_dev)`` pair,
# so when ``sync_before_use()`` replaces the file via ``shutil.move``/
# ``os.rename`` the cache is invalidated automatically. Normal writes (commits)
# keep the same inode, so steady-state tool calls still hit the cache.
#
# For backends without a ``cache_path`` (``D1Backend``, in-memory, bare
# ``SQLiteBackend``), the signature is the sentinel ``True`` — identity-only,
# matching pre-fix behavior.
_schema_lock = threading.Lock()
_schema_ensured_fallback: "weakref.WeakKeyDictionary[object, object]" = weakref.WeakKeyDictionary()


def _backend_schema_signature(storage_backend):
    """Return a value that changes when the underlying DB file is replaced.

    ``None`` means "can't compute a stable signature right now" — the caller
    should treat this as a cache miss and re-run ensure_schema (but not cache
    the result, since the next call would miss again).
    """
    cache_path = getattr(storage_backend, "cache_path", None)
    if cache_path is not None:
        try:
            st = cache_path.stat()
        except (OSError, AttributeError):
            return None
        return (st.st_ino, st.st_dev)
    return True


def _backend_schema_ensured(storage_backend) -> bool:
    current_sig = _backend_schema_signature(storage_backend)
    if current_sig is None:
        return False
    stored = getattr(storage_backend, "_schema_ensured", None)
    if stored is not None and stored == current_sig:
        return True
    try:
        fallback_sig = _schema_ensured_fallback.get(storage_backend)
    except TypeError:
        # Backend not weak-referenceable (e.g. __slots__ without __weakref__).
        return False
    return fallback_sig is not None and fallback_sig == current_sig


def _mark_backend_schema_ensured(storage_backend) -> None:
    sig = _backend_schema_signature(storage_backend)
    if sig is None:
        # Can't compute a signature — don't cache. Next call will re-run
        # ensure_schema, which is the safe fallback.
        return
    try:
        storage_backend._schema_ensured = sig
        return
    except (AttributeError, TypeError):
        pass
    try:
        _schema_ensured_fallback[storage_backend] = sig
    except TypeError:
        # Backend not weak-referenceable and not attribute-settable: caching
        # is disabled for this instance. ensure_schema() will re-run, matching
        # pre-cache behavior.
        pass


def connect(storage_backend, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Create a database connection using the given storage backend.

    For cloud backends, this will automatically sync from cloud before use.

    ``ensure_schema()`` is run once per backend instance per process. On D1 HTTP
    it issues 7–9 round-trips (~4–8 s); the per-call version was the single
    biggest source of tool-call latency. Cache is tied to the backend instance
    lifetime — a new backend triggers a fresh ensure_schema pass.
    """
    conn = storage_backend.connect(check_same_thread=check_same_thread)
    if not _backend_schema_ensured(storage_backend):
        with _schema_lock:
            if not _backend_schema_ensured(storage_backend):
                ensure_schema(conn)
                _mark_backend_schema_ensured(storage_backend)
    return conn


def sync_to_cloud(storage_backend) -> None:
    """Sync database to cloud storage if using a cloud backend."""
    storage_backend.sync_after_write()


def get_backend_info(storage_backend) -> dict:
    """Get information about the current storage backend."""
    return storage_backend.get_info()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            metadata TEXT,
            tags TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT
        )
        """
    )
    conn.commit()
    _ensure_fts(conn)
    _ensure_embeddings_table(conn)
    _ensure_crossrefs_table(conn)
    _ensure_memory_edges_table(conn)
    _ensure_events_table(conn)
    _ensure_actions_table(conn)
    _ensure_importance_columns(conn)
    _ensure_updated_at_column(conn)
    _migrate_legacy_explicit_edges(conn)


def _ensure_fts(conn: sqlite3.Connection) -> None:
    if isinstance(conn, D1Connection):
        return
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'"
    ).fetchone()
    if not table_exists:
        conn.execute(
            """
            CREATE VIRTUAL TABLE memories_fts
            USING fts5(content, metadata, tags)
            """
        )
        conn.commit()


def _ensure_embeddings_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_embeddings (
            memory_id INTEGER PRIMARY KEY,
            embedding TEXT,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.commit()


def _ensure_crossrefs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_crossrefs (
            memory_id INTEGER PRIMARY KEY,
            related TEXT,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def _ensure_memory_edges_table(conn: sqlite3.Connection) -> None:
    """Create durable, typed relationships separate from similarity crossrefs."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_memory_id INTEGER NOT NULL,
            to_memory_id INTEGER NOT NULL,
            relation_type TEXT NOT NULL,
            relation_confidence REAL,
            source TEXT NOT NULL DEFAULT 'manual',
            reason TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(from_memory_id, to_memory_id, relation_type),
            FOREIGN KEY(from_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY(to_memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_edges_from
        ON memory_edges(from_memory_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_edges_to
        ON memory_edges(to_memory_id)
        """
    )
    conn.commit()


def _migrate_legacy_explicit_edges(conn: sqlite3.Connection) -> None:
    """Move legacy typed links out of the computed crossref cache once.

    Older versions stored both embedding-similarity neighbours and explicit
    semantic relationships in ``memories_crossrefs.related``.  Typed links and
    manually-created ``related_to`` links used a score of 1.0, while computed
    links contain cosine similarity scores.  Keep the latter in the cache and
    migrate the former into ``memory_edges``.
    """
    migrated = conn.execute(
        "SELECT value FROM memories_meta WHERE key = 'memory_edges_migrated_v1'"
    ).fetchone()
    if migrated:
        return

    rows = conn.execute(
        "SELECT memory_id, related FROM memories_crossrefs WHERE related IS NOT NULL"
    ).fetchall()
    for memory_id, related_json in rows:
        try:
            related = json.loads(related_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(related, list):
            continue

        computed_refs = []
        for ref in related:
            if not isinstance(ref, dict):
                continue
            try:
                target_id = int(ref.get("id"))
            except (TypeError, ValueError):
                continue
            if memory_id == target_id:
                continue

            relation_type = ref.get("edge_type", "related_to")
            score = ref.get("score", 0.0)
            is_explicit = relation_type != "related_to"
            if relation_type == "related_to":
                try:
                    is_explicit = float(score) >= 0.9999
                except (TypeError, ValueError):
                    is_explicit = False
            if not is_explicit:
                computed_refs.append(ref)
                continue

            conn.execute(
                """
                INSERT INTO memory_edges(
                    from_memory_id, to_memory_id, relation_type, source, reason
                ) VALUES (?, ?, ?, 'legacy_migration', 'Migrated from legacy crossrefs')
                ON CONFLICT(from_memory_id, to_memory_id, relation_type) DO NOTHING
                """,
                (memory_id, target_id, relation_type),
            )

        computed_json = json.dumps(computed_refs, ensure_ascii=False) if computed_refs else None
        conn.execute(
            "UPDATE memories_crossrefs SET related = ? WHERE memory_id = ?",
            (computed_json, memory_id),
        )

    conn.execute(
        """
        INSERT INTO memories_meta(key, value)
        VALUES ('memory_edges_migrated_v1', '1')
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """
    )
    conn.commit()


def _ensure_events_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            tags TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            consumed INTEGER DEFAULT 0,
            FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def _ensure_actions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER,
            action TEXT NOT NULL,
            summary TEXT NOT NULL,
            timestamp TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()


def _ensure_importance_columns(conn: sqlite3.Connection) -> None:
    """Add importance scoring columns to memories table if they don't exist."""
    cursor = conn.execute("PRAGMA table_info(memories)")
    columns = {row[1] for row in cursor.fetchall()}

    if "importance" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 1.0")

    if "last_accessed" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN last_accessed TEXT")

    if "access_count" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0")

    conn.commit()


def _ensure_updated_at_column(conn: sqlite3.Connection) -> None:
    """Add updated_at column to memories table if it doesn't exist."""
    cursor = conn.execute("PRAGMA table_info(memories)")
    columns = {row[1] for row in cursor.fetchall()}

    if "updated_at" not in columns:
        conn.execute("ALTER TABLE memories ADD COLUMN updated_at TEXT")
        conn.commit()


    # Note: memory_absorb stores source/confidence in metadata (not separate columns)
    # to avoid schema migration and keep provenance co-located with other metadata.
