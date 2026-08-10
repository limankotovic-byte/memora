"""Regression tests for core storage operations."""

import memora
import memora.storage as storage


def test_add_memory_crud(local_db):
    """Basic create/read/update/delete cycle."""
    with storage.connect() as conn:
        mem = storage.add_memory(conn, content="Test CRUD memory content here", tags=["test"])
        assert mem["id"] is not None
        mid = mem["id"]

        fetched = storage.get_memory(conn, mid)
        assert fetched is not None
        assert fetched["content"] == "Test CRUD memory content here"
        assert "test" in fetched["tags"]

        updated = storage.update_memory(conn, mid, content="Updated CRUD memory content here")
        assert updated is not None
        assert updated["content"] == "Updated CRUD memory content here"

        storage.delete_memory(conn, mid)
        assert storage.get_memory(conn, mid) is None


def test_update_tags_recomputes_fts(local_db):
    """Updating tags should refresh the FTS index."""
    with storage.connect() as conn:
        mem = storage.add_memory(conn, content="FTS reindex test memory content", tags=["old-tag"])
        mid = mem["id"]

        storage.update_memory(conn, mid, tags=["new-tag-fts"])

        row = conn.execute(
            "SELECT tags FROM memories_fts WHERE rowid = ?", (mid,)
        ).fetchone()
        assert row is not None
        assert "new-tag-fts" in row[0].lower()


def test_update_tags_recomputes_embedding(local_db):
    """Updating tags should refresh the embedding."""
    with storage.connect() as conn:
        mem = storage.add_memory(conn, content="Embedding reindex test memory content", tags=["alpha"])
        mid = mem["id"]

        old_emb = conn.execute(
            "SELECT embedding FROM memories_embeddings WHERE memory_id = ?", (mid,)
        ).fetchone()

        storage.update_memory(conn, mid, tags=["completely-different-tag"])

        new_emb = conn.execute(
            "SELECT embedding FROM memories_embeddings WHERE memory_id = ?", (mid,)
        ).fetchone()

        assert old_emb is not None and new_emb is not None
        assert old_emb[0] != new_emb[0]


def test_update_metadata_recomputes_embedding(local_db):
    """Updating metadata should refresh the embedding."""
    with storage.connect() as conn:
        mem = storage.add_memory(
            conn,
            content="Metadata reindex test memory content",
            tags=["meta"],
            metadata={"section": "docs"},
        )
        mid = mem["id"]

        old_emb = conn.execute(
            "SELECT embedding FROM memories_embeddings WHERE memory_id = ?", (mid,)
        ).fetchone()

        storage.update_memory(conn, mid, metadata={"section": "api-reference"})

        new_emb = conn.execute(
            "SELECT embedding FROM memories_embeddings WHERE memory_id = ?", (mid,)
        ).fetchone()
        updated = storage.get_memory(conn, mid)

        assert old_emb is not None and new_emb is not None
        assert old_emb[0] != new_emb[0]
        assert updated is not None
        assert updated["metadata"]["section"] == "api-reference"


def test_update_metadata_merges_existing_keys(local_db):
    """Partial metadata updates should preserve omitted metadata keys."""
    with storage.connect() as conn:
        mem = storage.add_memory(
            conn,
            content="Metadata merge safety test memory content",
            tags=["meta"],
            metadata={
                "type": "todo",
                "status": "open",
                "priority": "high",
                "category": "release",
            },
        )
        mid = mem["id"]

        updated = storage.update_memory(conn, mid, metadata={"status": "closed"})

        assert updated is not None
        assert updated["metadata"]["type"] == "todo"
        assert updated["metadata"]["status"] == "closed"
        assert updated["metadata"]["priority"] == "high"
        assert updated["metadata"]["category"] == "release"


def test_update_metadata_null_deletes_key(local_db):
    """Patch-style metadata updates should delete keys explicitly set to None."""
    with storage.connect() as conn:
        mem = storage.add_memory(
            conn,
            content="Metadata delete safety test memory content",
            tags=["meta"],
            metadata={"status": "closed", "closed_reason": "done"},
        )
        mid = mem["id"]

        updated = storage.update_memory(conn, mid, metadata={"closed_reason": None})

        assert updated is not None
        assert updated["metadata"]["status"] == "closed"
        assert "closed_reason" not in updated["metadata"]


def test_update_metadata_replace_option_allows_full_replacement(local_db):
    """Explicit replacement remains available for callers that need it."""
    with storage.connect() as conn:
        mem = storage.add_memory(
            conn,
            content="Metadata replace safety test memory content",
            tags=["meta"],
            metadata={"type": "todo", "status": "open", "priority": "high"},
        )
        mid = mem["id"]

        updated = storage.update_memory(
            conn,
            mid,
            metadata={"status": "closed"},
            replace_metadata=True,
        )

        assert updated is not None
        assert updated["metadata"] == {"status": "closed"}


def test_update_content_validates(local_db):
    """Updating with too-short content should raise ValueError."""
    with storage.connect() as conn:
        mem = storage.add_memory(conn, content="Valid content for update validation test", tags=["test"])
        mid = mem["id"]

        try:
            storage.update_memory(conn, mid, content="hi")
            assert False, "Expected ValueError for short content"
        except ValueError:
            pass


def test_semantic_search_basic(local_db):
    """Basic semantic search should find relevant memories."""
    with storage.connect() as conn:
        storage.add_memory(conn, content="Python programming language tutorial guide", tags=["code"])
        storage.add_memory(conn, content="Recipe for chocolate cake baking dessert", tags=["cooking"])

        results = storage.semantic_search(conn, "python programming")
        assert len(results) > 0
        assert any("python" in r["memory"]["content"].lower() for r in results)


def test_explicit_edges_survive_crossref_rebuild_and_keep_lineage(local_db):
    """Similarity maintenance must never overwrite durable semantic edges."""
    with storage.connect() as conn:
        old = storage.add_memory(
            conn,
            content="Service uses the old configuration value for production",
            tags=["config"],
        )
        current = storage.add_memory(
            conn,
            content="Service uses the current configuration value for production",
            tags=["config"],
        )

        storage.add_link(
            conn,
            current["id"],
            old["id"],
            edge_type="supersedes",
            relation_confidence=0.93,
            source="llm",
            reason="Current configuration fully replaces the old value",
        )
        storage.rebuild_crossrefs(conn)

        edges = storage.get_explicit_edges(conn, current["id"])
        assert len(edges) == 1
        assert edges[0]["id"] == old["id"]
        assert edges[0]["edge_type"] == "supersedes"
        assert edges[0]["relation_confidence"] == 0.93
        assert edges[0]["source"] == "llm"
        assert storage.get_crossrefs(conn, current["id"]) != edges

        active = storage.hybrid_search(conn, "production configuration", follow="active")
        active_ids = {entry["memory"]["id"] for entry in active}
        assert current["id"] in active_ids
        assert old["id"] not in active_ids


def test_legacy_typed_crossrefs_migrate_to_durable_edges(local_db):
    with storage.connect() as conn:
        old = storage.add_memory(conn, content="Legacy previous service configuration", tags=["config"])
        current = storage.add_memory(conn, content="Legacy current service configuration", tags=["config"])
        storage._store_crossrefs(
            conn,
            old["id"],
            [{"id": current["id"], "score": 0.7, "edge_type": "related_to"}],
        )
        storage._store_crossrefs(
            conn,
            current["id"],
            [{"id": old["id"], "score": 1.0, "edge_type": "supersedes"}],
        )
        conn.execute("DELETE FROM memories_meta WHERE key = 'memory_edges_migrated_v1'")

        from memora.schema import _migrate_legacy_explicit_edges

        _migrate_legacy_explicit_edges(conn)

        assert storage.get_crossrefs(conn, old["id"])[0]["score"] == 0.7
        assert storage.get_crossrefs(conn, current["id"]) == []
        migrated = storage.get_explicit_edges(conn, current["id"])
        assert migrated[0]["id"] == old["id"]
        assert migrated[0]["edge_type"] == "supersedes"
        assert migrated[0]["source"] == "legacy_migration"


def test_explicit_edge_rejects_invalid_confidence_and_supersession_cycles(local_db):
    with storage.connect() as conn:
        first = storage.add_memory(conn, content="First version of the deployment plan", tags=["plan"])
        second = storage.add_memory(conn, content="Second version of the deployment plan", tags=["plan"])

        try:
            storage.add_link(
                conn,
                second["id"],
                first["id"],
                edge_type="supersedes",
                relation_confidence=1.1,
            )
            assert False, "Expected confidence validation error"
        except ValueError:
            pass

        storage.add_link(conn, second["id"], first["id"], edge_type="supersedes")
        try:
            storage.add_link(conn, first["id"], second["id"], edge_type="supersedes")
            assert False, "Expected supersession cycle validation error"
        except ValueError:
            pass


def test_relationship_expansion_uses_explicit_edges_not_similarity_cache(local_db):
    with storage.connect() as conn:
        seed = storage.add_memory(
            conn,
            content="How to deploy the API service safely",
            tags=["deployment"],
        )
        extension = storage.add_memory(
            conn,
            content="Rollback procedure for a failed API service deployment",
            tags=["deployment"],
        )
        unrelated = storage.add_memory(
            conn,
            content="Recipe for a chocolate cake dessert",
            tags=["cooking"],
        )
        storage.add_link(
            conn,
            seed["id"],
            extension["id"],
            edge_type="extends",
            relation_confidence=0.9,
            source="manual",
        )

        results = storage.hybrid_search(
            conn,
            "deploy API service",
            top_k=5,
            relationship_expansion=True,
        )
        by_id = {entry["memory"]["id"]: entry["memory"] for entry in results}
        assert extension["id"] in by_id
        assert by_id[extension["id"]]["relationship_context"] == [{
            "from_id": seed["id"],
            "edge_type": "extends",
            "relation_confidence": 0.9,
            "source": "manual",
            "reason": None,
        }]
        # Unrelated memory may appear via base RRF (min_score=None in semantic leg),
        # but it must NOT have relationship_context — expansion did not pull it.
        if unrelated["id"] in by_id:
            assert "relationship_context" not in by_id[unrelated["id"]], (
                "unrelated memory should not have relationship_context"
            )


def test_hybrid_search_tags_all_filters_semantic_leg(local_db):
    """Phase 0 regression: tags_all must filter both legs of hybrid_search.

    Before the fix, the semantic leg only honored metadata_filters, so fused
    results could include rows that violated tags_all.
    """
    with storage.connect() as conn:
        storage.add_memory(conn, content="Python programming language overview", tags=["a", "b"])
        storage.add_memory(conn, content="Python programming basics intro", tags=["a"])
        storage.add_memory(conn, content="Python programming advanced patterns", tags=["b"])

        results = storage.hybrid_search(conn, "python programming", tags_all=["a", "b"])

        assert len(results) >= 1
        for entry in results:
            memory = entry.get("memory", entry)
            tags = set(memory.get("tags") or [])
            assert {"a", "b"}.issubset(tags), (
                f"hybrid_search returned row with tags {tags} violating tags_all=['a','b']"
            )


def test_hybrid_search_selective_tag_filter_surfaces_matches(local_db):
    """Phase 0 regression: selective filters must still surface the matching row
    from the semantic leg even when it would lie outside the unfiltered top-k.
    """
    with storage.connect() as conn:
        # Populate many non-matching rows to push the needle below the default
        # top_k * 3 window for semantic search.
        for i in range(20):
            storage.add_memory(
                conn,
                content=f"Distractor document about python programming number {i}",
                tags=["distract"],
            )
        # The single row that should match the filter.
        storage.add_memory(
            conn,
            content="Rare needle memory for python programming query",
            tags=["needle"],
        )

        results = storage.hybrid_search(
            conn, "python programming", tags_all=["needle"], top_k=5
        )

        assert len(results) == 1, (
            f"Expected 1 needle row, got {len(results)} — filter did not push "
            f"into semantic leg"
        )
        memory = results[0].get("memory", results[0])
        assert "needle" in (memory.get("tags") or [])


def test_hybrid_search_date_filter_applies_to_semantic_leg(local_db):
    """Phase 0 regression: date_from/date_to must filter the semantic leg."""
    import sqlite3 as _sqlite3

    with storage.connect() as conn:
        old = storage.add_memory(
            conn, content="Python programming early historical note", tags=["old"]
        )
        new = storage.add_memory(
            conn, content="Python programming recent current note", tags=["new"]
        )

        # Force the old row's created_at backward so date_from filters it out.
        conn.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ("2020-01-01T00:00:00", old["id"]),
        )
        conn.commit()

        results = storage.hybrid_search(
            conn, "python programming", date_from="2024-01-01T00:00:00"
        )

        ids = {entry.get("memory", entry)["id"] for entry in results}
        assert new["id"] in ids
        assert old["id"] not in ids, (
            "hybrid_search returned a row older than date_from — semantic leg "
            "ignored the date filter"
        )


def test_list_memories_filtered_pagination(local_db):
    """Phase 1 regression: offset/limit must apply to filtered results, not raw SQL rows.

    Before the fix, SQL LIMIT/OFFSET ran before Python-side tag filtering,
    so filtered pagination underfilled and offset skipped wrong rows.
    """
    with storage.connect() as conn:
        # Create 20 memories: 10 matching (tags=["match"]) and 10 non-matching
        for i in range(20):
            tag = "match" if i % 2 == 0 else "nomatch"
            storage.add_memory(
                conn,
                content=f"Filtered pagination test memory number {i:02d}",
                tags=[tag],
            )

        # Without filters: 20 total. With tags_all=["match"]: 10 matching.
        all_matching = storage.list_memories(conn, tags_all=["match"], limit=-1)
        assert len(all_matching) == 10

        # Page 1: first 5 filtered results
        page1 = storage.list_memories(conn, tags_all=["match"], limit=5, offset=0)
        assert len(page1) == 5, f"Page 1 expected 5 rows, got {len(page1)}"

        # Page 2: next 5 filtered results
        page2 = storage.list_memories(conn, tags_all=["match"], limit=5, offset=5)
        assert len(page2) == 5, f"Page 2 expected 5 rows, got {len(page2)}"

        # Pages should not overlap and should cover all 10 matching rows
        page1_ids = {r["id"] for r in page1}
        page2_ids = {r["id"] for r in page2}
        assert page1_ids.isdisjoint(page2_ids), "Pages overlap"
        assert len(page1_ids | page2_ids) == 10, "Pages don't cover all matches"

        # All returned rows must have the "match" tag
        for r in page1 + page2:
            assert "match" in r["tags"], f"Row {r['id']} missing 'match' tag"


def test_tag_whitelist_enforcement(local_db, monkeypatch):
    """Adding memory with invalid tag should raise when whitelist is active."""
    monkeypatch.setattr(memora, "TAG_WHITELIST", {"allowed-tag"})

    with storage.connect() as conn:
        try:
            storage.add_memory(conn, content="Memory with blocked tag content here", tags=["not-allowed"])
            assert False, "Expected ValueError for invalid tag"
        except ValueError as e:
            assert "not-allowed" in str(e).lower() or "whitelist" in str(e).lower() or "allowed" in str(e).lower()
