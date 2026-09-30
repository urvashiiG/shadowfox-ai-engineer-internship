import re
from langgraph.graph import END, START, StateGraph
from app.config.settings import get_settings
from app.generation.prompts import grounded_prompt
from app.grounding.validator import SAFE_FALLBACK, validate_grounding
from app.retrieval.filters import build_context
from app.retrieval.reranker import rerank
from app.retrieval.retriever import retrieve_candidates
from app.schemas.queries import QueryRequest
from app.workflow.state import RAGState

def build_graph(store, llm):
    settings = get_settings()
    def validate(state: RAGState):
        try:
            QueryRequest(question=state.get("original_query", ""), document_id=state.get("document_id"))
            return {"error": None}
        except Exception as exc:
            return {"error": str(exc), "answer": "Please enter a non-empty question of at most 2,000 characters.", "grounded": False}
    def rewrite(state: RAGState):
        # Retrieval-friendly cleanup preserves user intent; generation sees the original query.
        query = re.sub(r"\s+", " ", state["original_query"]).strip()
        return {"rewritten_query": query}
    def retrieve(state: RAGState):
        rows = retrieve_candidates(store, state["rewritten_query"], settings.top_k, state.get("document_id"))
        return {"retrieved_chunks": rows, "candidate_count": len(rows)}
    def rank(state: RAGState):
        return {"reranked_chunks": rerank(state["rewritten_query"], state.get("retrieved_chunks", []), settings.rerank_top_k, settings.similarity_threshold)}
    def build_context_node(state: RAGState):
        text, sources = build_context(state.get("reranked_chunks", []), settings.max_context_chars)
        return {"context": text, "sources": sources}
    def generate(state: RAGState):
        if state.get("error"):
            return {}
        if not state.get("sources"):
            return {"answer": SAFE_FALLBACK, "grounded": False, "error": "No sufficiently relevant document passages were found."}
        try:
            return {"answer": llm.complete(grounded_prompt(state["original_query"], state["context"])), "error": None}
        except Exception as exc:
            return {"answer": "Answer generation is unavailable. Check the OpenRouter configuration and try again.", "error": str(exc)}
    def grounding(state: RAGState):
        if state.get("error") or not state.get("sources"):
            return {"grounded": False}
        valid, answer = validate_grounding(state.get("answer", ""), state["sources"])
        return {"grounded": valid, "answer": answer}
    graph = StateGraph(RAGState)
    for name, node in [("validate", validate), ("rewrite", rewrite), ("retrieve", retrieve), ("rank", rank), ("build_context", build_context_node), ("generate", generate), ("grounding", grounding)]:
        graph.add_node(name, node)
    graph.add_edge(START, "validate")
    graph.add_conditional_edges("validate", lambda s: "rewrite" if not s.get("error") else END)
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rank")
    graph.add_edge("rank", "build_context")
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", "grounding")
    graph.add_edge("grounding", END)
    return graph.compile()

def run_query(store, llm, question: str, document_id: str | None = None) -> dict:
    result = build_graph(store, llm).invoke({"original_query": question, "document_id": document_id})
    return {"answer": result.get("answer", SAFE_FALLBACK), "grounded": result.get("grounded", False),
            "grounding_status": "Grounded" if result.get("grounded") else "Insufficient evidence",
            "rewritten_query": result.get("rewritten_query", ""), "sources": result.get("sources", []),
            "candidate_count": result.get("candidate_count", 0), "context_count": len(result.get("sources", [])),
            "error": result.get("error")}
