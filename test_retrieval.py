"""Tests for chunking and search.

Checks that relevant passages come back and that irrelevant questions return
nothing, since a retriever that always returns its least bad chunk is how a
grounded system ends up confidently wrong.

Run with: python test_retrieval.py
"""

from pathlib import Path

from retrieval import Index, chunk_document, format_context

SAMPLE_DOC = Path(__file__).parent / "sample_docs" / "tasting_room_faq.md"

DOCS = {
    "botanicals.md": (
        "Signature Gin leads on juniper with coriander seed, angelica root and "
        "orris root. Orris root acts as a fixative that holds the aromatics "
        "together. Spiced Gin brings cardamom and cassia bark forward."
    ),
    "events.md": (
        "The Signature Distillery Tour lasts approximately 90 minutes and "
        "includes a finishing cocktail. Private event enquiries should be "
        "answered within one business day."
    ),
    "vouchers.md": (
        "Gift vouchers are purchased through the online store. Confirm the "
        "current expiry terms with a manager before telling a customer that a "
        "voucher does not expire."
    ),
}


def build_index() -> Index:
    index = Index()
    for source, text in DOCS.items():
        index.add(chunk_document(text, source))
    return index


def test_chunking_produces_chunks_with_provenance():
    chunks = chunk_document("word " * 900, "big.md")
    assert len(chunks) > 1, "long text should split into multiple chunks"
    assert all(c.source == "big.md" for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_empty_document_produces_nothing():
    assert chunk_document("", "empty.md") == []
    assert chunk_document("   \n  ", "blank.md") == []


def test_index_reports_size_and_sources():
    index = build_index()
    assert len(index) == len(DOCS)
    assert set(index.sources) == set(DOCS)


def test_retrieves_the_relevant_document():
    index = build_index()
    assert index.search("what does orris root do?")[0].chunk.source == "botanicals.md"
    assert index.search("how long does the tour take?")[0].chunk.source == "events.md"
    assert index.search("do gift vouchers expire?")[0].chunk.source == "vouchers.md"


def test_auxiliary_verbs_do_not_outrank_content_words():
    """Regression. "how long does the tour take" used to return the voucher
    document, which says "does not expire", ahead of the one about tours, because
    sklearn's English stop-word list does not include "does"."""
    hits = build_index().search("how long does the tour take?")
    assert hits, "should retrieve something for a question about the tour"
    assert hits[0].chunk.source == "events.md", (
        f"expected the events document, got {hits[0].chunk.source}"
    )


def test_unrelated_question_returns_nothing():
    hits = build_index().search("what is the average rainfall in Peru")
    assert hits == [], "an unrelated question must not return a weak match"


def test_empty_index_returns_nothing():
    assert Index().search("anything") == []


def test_blank_question_returns_nothing():
    assert build_index().search("   ") == []


def test_results_are_ordered_by_score():
    scores = [h.score for h in build_index().search("juniper coriander orris", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_k_limits_the_number_of_hits():
    index = build_index()
    assert len(index.search("gin tour voucher", k=1)) <= 1
    assert len(index.search("gin tour voucher", k=2)) <= 2


def test_reindexing_a_source_replaces_it_rather_than_duplicating():
    index = Index()
    index.add(chunk_document("first version about juniper", "doc.md"))
    index.add(chunk_document("second version about juniper", "doc.md"))

    assert len(index) == 1, "re-adding a source must replace its chunks"
    assert "second version" in index.chunks[0].text


def test_removing_a_source_makes_it_unsearchable():
    index = build_index()
    index.remove_source("vouchers.md")
    assert "vouchers.md" not in index.sources
    assert all(h.chunk.source != "vouchers.md" for h in index.search("voucher expiry"))


def test_context_carries_citations():
    context = format_context(build_index().search("orris root"))
    assert "botanicals.md" in context
    assert "chunk 0" in context


def test_sample_document_answers_its_suggested_questions():
    """The preloaded document must support the buttons shown in the UI."""
    assert SAMPLE_DOC.exists(), "sample document ships with the app"
    index = Index()
    index.add(chunk_document(SAMPLE_DOC.read_text(encoding="utf-8"), SAMPLE_DOC.name))

    for question in [
        "What do we offer guests who don't drink alcohol?",
        "How long is the Signature tour and what's included?",
        "Can I tell a customer their gift voucher never expires?",
        "What does orris root do in the Signature Gin?",
    ]:
        assert index.search(question), f"no passage retrieved for: {question}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
