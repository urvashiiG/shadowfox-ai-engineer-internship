from io import BytesIO
from pathlib import Path
import pytest
from app.ingestion.loader import DocumentLoadError, load_document
from app.processing.cleaner import clean_text
from app.processing.chunker import chunk_pages
from app.grounding.validator import SAFE_FALLBACK, validate_grounding
from app.retrieval.filters import build_context
from app.retrieval.reranker import rerank
from app.workflow.graph import run_query

TOWER = "The Eiffel Tower is located in Paris. It was completed in 1889. The tower is made primarily from iron."

class FakeLLM:
    def complete(self, prompt):
        if "population" in prompt.lower():
            return "I cannot find the population in the provided excerpts."
        return "The Eiffel Tower is located in Paris [Source 1]."

def test_text_cleaning_and_chunk_overlap():
    assert clean_text(" A\t  B\r\n\r\n\r\nC ") == "A B\n\nC"
    chunks = chunk_pages([{"text": "abcdefghij klmnopqrst uvwxyz"}], document_id="x", document_name="a.txt", source_type="txt", size=12, overlap=3)
    assert len(chunks) >= 2
    assert chunks[0]["source_text"][-3:] == chunks[1]["source_text"][:3]
    assert chunks[0]["chunk_id"] == "x-0"

@pytest.mark.parametrize("name", ["notes.txt", "notes.md"])
def test_text_and_markdown_ingestion(name):
    pages = load_document(BytesIO(b"# Hello\n\nWorld"), name)
    assert pages[0]["text"] == "# Hello\n\nWorld"

def test_bad_and_empty_uploads():
    with pytest.raises(DocumentLoadError):
        load_document(BytesIO(b"x"), "data.exe")
    with pytest.raises(DocumentLoadError):
        load_document(BytesIO(b""), "empty.txt")

def test_pdf_extraction():
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(DocumentLoadError, match="No extractable text"):
        load_document(BytesIO(stream.getvalue()), "blank.pdf")

def test_upload_and_scoped_faiss_retrieval(service):
    first = service.upload(BytesIO(TOWER.encode()), "tower.txt")
    service.upload(BytesIO(b"The moon orbits Earth."), "moon.md")
    results = service.store.search("Where is Eiffel Tower located?", 8, first["document_id"])
    assert results and {r["document_id"] for r in results} == {first["document_id"]}
    assert results[0]["document_name"] == "tower.txt"
    assert results[0]["chunk_id"]

def test_semantic_lexical_reranking_and_context_filtering(service):
    doc = service.upload(BytesIO(TOWER.encode()), "tower.txt")
    candidates = service.store.search("iron tower Paris", 8)
    ranked = rerank("iron tower Paris", candidates, 2, 0.0)
    context, sources = build_context(ranked, 1000)
    assert ranked and sources and "Source" in context
    assert all("score" in source for source in sources)

def test_grounding_rejects_missing_citation_or_unsupported_claim():
    source = {"source_text": TOWER}
    assert validate_grounding("Paris is where the Eiffel Tower is located [Source 1].", [source])[0]
    ok, answer = validate_grounding("The population is 20 million [Source 1].", [source])
    assert not ok and answer == SAFE_FALLBACK

def test_langgraph_end_to_end_and_abstention(service):
    service.upload(BytesIO(TOWER.encode()), "eiffel.txt")
    found = run_query(service.store, FakeLLM(), "Where is the Eiffel Tower located?")
    assert found["grounded"] and "Paris" in found["answer"] and found["sources"]
    unsupported = run_query(service.store, FakeLLM(), "What is the population of Paris?")
    assert not unsupported["grounded"] and unsupported["answer"] == SAFE_FALLBACK

def test_delete_removes_document_vectors(service):
    doc = service.upload(BytesIO(TOWER.encode()), "eiffel.txt")
    assert service.delete(doc["document_id"])
    assert not service.store.search("Eiffel Paris", 8)
    assert not service.delete(doc["document_id"])
