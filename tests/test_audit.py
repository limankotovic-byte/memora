"""Tests for the edge auditor (memora.audit)."""

import json

import memora.audit as audit
import memora.storage as storage


def _seed_crossrefs(conn, memory_id, related):
    conn.execute(
        "INSERT INTO memories_crossrefs(memory_id, related) VALUES (?, ?) "
        "ON CONFLICT(memory_id) DO UPDATE SET related=excluded.related",
        (memory_id, json.dumps(related)),
    )


def test_load_crossref_candidates_filters_by_score_and_limit(local_db):
    """Only pairs above min_score survive; sorting and limit apply."""
    with storage.connect() as conn:
        m1 = storage.add_memory(conn, content="first memory", tags=["t"])
        m2 = storage.add_memory(conn, content="second memory", tags=["t"])
        m3 = storage.add_memory(conn, content="third memory", tags=["t"])
        _seed_crossrefs(
            conn,
            m1["id"],
            [{"id": m2["id"], "score": 0.95}, {"id": m3["id"], "score": 0.5}],
        )
        conn.commit()

        candidates = audit._load_crossref_candidates(conn, min_score=0.8, max_pairs=10)
        assert (m1["id"], m2["id"], 0.95) in candidates

        candidates = audit._load_crossref_candidates(conn, min_score=0.4, max_pairs=10)
        assert (m1["id"], m2["id"], 0.95) in candidates
        assert (m1["id"], m3["id"], 0.5) in candidates

        candidates = audit._load_crossref_candidates(conn, min_score=0.4, max_pairs=1)
        assert len(candidates) == 1
        assert candidates[0][2] == 0.95  # highest score first


def test_load_crossref_candidates_skips_linked_and_bad_rows(local_db):
    """Pairs already present in memory_edges and malformed rows are skipped."""
    with storage.connect() as conn:
        m1 = storage.add_memory(conn, content="memory a", tags=["t"])
        m2 = storage.add_memory(conn, content="memory b", tags=["t"])
        m3 = storage.add_memory(conn, content="memory c", tags=["t"])
        _seed_crossrefs(conn, m1["id"], [{"id": m2["id"], "score": 0.9}])
        _seed_crossrefs(conn, m3["id"], "not a list")
        storage.add_link(conn, m1["id"], m2["id"], edge_type="related_to", source="test")
        conn.commit()

        candidates = audit._load_crossref_candidates(conn, min_score=0.8, max_pairs=10)
        assert candidates == []


def test_load_crossref_candidates_ignores_self_pairs(local_db):
    """A memory linking to itself is not a candidate."""
    with storage.connect() as conn:
        m1 = storage.add_memory(conn, content="solo memory", tags=["t"])
        _seed_crossrefs(conn, m1["id"], [{"id": m1["id"], "score": 1.0}])
        conn.commit()

        candidates = audit._load_crossref_candidates(conn, min_score=0.8, max_pairs=10)
        assert candidates == []
