import re
from app.retrieval.reranker import terms

SAFE_FALLBACK = "I couldn't find enough supporting information in the uploaded documents to answer this reliably."

def validate_grounding(answer: str, sources: list[dict]) -> tuple[bool, str]:
    if not answer.strip() or not sources:
        return False, SAFE_FALLBACK
    valid_labels = {f"Source {i}" for i, _ in enumerate(sources, start=1)}
    citations = {re.sub(r"\s+", " ", item.title()) for item in re.findall(r"\[(Source\s+\d+)\]", answer, flags=re.I)}
    if not citations or not citations.issubset(valid_labels):
        return False, SAFE_FALLBACK
    evidence = set().union(*(terms(row["source_text"]) for row in sources))
    claims = [terms(sentence) for sentence in re.split(r"(?<=[.!?])\s+", answer) if terms(sentence)]
    supported = sum(bool(claim & evidence) and len(claim & evidence) / len(claim) >= 0.25 for claim in claims)
    if not claims or supported / len(claims) < 0.5:
        return False, SAFE_FALLBACK
    return True, answer
