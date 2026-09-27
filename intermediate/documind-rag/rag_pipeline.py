"""
rag_pipeline.py

This is the CORE AI engineering module. It wires together every stage of
the retrieval-augmented generation pipeline:

  USER QUESTION
        v
  QUERY VALIDATION
        v
  QUERY EMBEDDING            (embeddings.py, local sentence-transformers)
        v
  FAISS SIMILARITY SEARCH    (vector_store.py)
        v
  TOP-K RELEVANT CHUNKS
        v
  CONTEXT FILTERING          (character budget via MAX_CONTEXT_CHARS)
        v
  GROUNDED PROMPT            (explicit "use only this context" instructions)
        v
  OPENROUTER LLM             (llm_service.py)
        v
  ANSWER + SOURCES

Only the retrieved chunks are ever sent to the LLM -- never the full
uploaded document(s). This is what makes this a real RAG pipeline rather
than "paste the whole file into the prompt".
"""

from typing import Dict, Any, List

from config import TOP_K, MAX_CONTEXT_CHARS
from validators import validate_query
from embeddings import embed_query
from vector_store import VectorStore
from llm_service import generate_grounded_answer


def retrieve_context(
    query: str, vector_store: VectorStore, top_k: int = TOP_K
) -> List[Dict[str, Any]]:
    """
    Embed the query and run FAISS similarity search against the vector
    store, returning the top_k most relevant chunks (with metadata and
    similarity scores).
    """
    if vector_store.is_empty():
        return []

    query_vector = embed_query(query)
    return vector_store.search(query_vector, top_k=top_k)


def _truncate_to_budget(chunks: List[Dict[str, Any]], max_chars: int) -> List[Dict[str, Any]]:
    """
    Apply a character budget across the retrieved chunks so the prompt
    sent to the LLM stays a reasonable size. Chunks are kept in
    relevance order and dropped once the budget is exhausted.
    """
    budgeted: List[Dict[str, Any]] = []
    used = 0
    for chunk in chunks:
        chunk_len = len(chunk["text"])
        if used + chunk_len > max_chars and budgeted:
            # Keep at least one chunk even if it alone exceeds the budget.
            break
        budgeted.append(chunk)
        used += chunk_len
    return budgeted


def build_grounded_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Build a grounded prompt that explicitly instructs the LLM to answer
    using ONLY the supplied context, to avoid inventing information, and
    to say clearly when the answer cannot be found in the context.
    """
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        page_info = f"Page: {chunk['page']}" if chunk.get("page") is not None else "Page: N/A"
        context_blocks.append(
            f"[Source {i}]\n"
            f"File: {chunk['source']}\n"
            f"{page_info}\n"
            f"Content:\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""You must answer the user's question using ONLY the document context provided below.

Rules:
- Answer strictly based on the supplied context. Do not use outside knowledge.
- Do not invent or assume any information that is not present in the context.
- If the answer cannot be found in the context, clearly say so instead of guessing.
- When you are uncertain, state your uncertainty explicitly rather than presenting a guess as fact.
- Where relevant, refer to the source(s) you used (e.g. "According to Source 1...").

Document context:
{context_text}

User question:
{query}

Answer:"""
    return prompt


def answer_question(query: str, vector_store: VectorStore, top_k: int = TOP_K) -> Dict[str, Any]:
    """
    Run the full RAG pipeline for a user question and return a
    structured result:

        {
            "success": bool,
            "answer": str or None,
            "sources": list of retrieved chunk dicts (with scores),
            "error": str or None,
        }
    """
    is_valid, error = validate_query(query)
    if not is_valid:
        return {"success": False, "answer": None, "sources": [], "error": error}

    if vector_store.is_empty():
        return {
            "success": False,
            "answer": None,
            "sources": [],
            "error": "No documents have been processed yet. Please upload and process a document first.",
        }

    retrieved_chunks = retrieve_context(query, vector_store, top_k=top_k)
    if not retrieved_chunks:
        return {
            "success": False,
            "answer": None,
            "sources": [],
            "error": "No relevant content was found for your question.",
        }

    budgeted_chunks = _truncate_to_budget(retrieved_chunks, MAX_CONTEXT_CHARS)
    prompt = build_grounded_prompt(query, budgeted_chunks)

    llm_result = generate_grounded_answer(prompt)

    if not llm_result["success"]:
        return {
            "success": False,
            "answer": None,
            "sources": budgeted_chunks,
            "error": llm_result["error"],
        }

    return {
        "success": True,
        "answer": llm_result["answer"],
        "sources": budgeted_chunks,
        "error": None,
    }
