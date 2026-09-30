import re

STOPWORDS = {"the", "a", "an", "is", "are", "was", "were", "in", "of", "to", "and", "what", "where", "when", "how", "who", "does", "do", "it", "for", "on"}

def terms(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[\w'-]+", text) if len(word) > 2 and word.lower() not in STOPWORDS}

def rerank(query: str, candidates: list[dict], limit: int, threshold: float) -> list[dict]:
    qterms = terms(query)
    ranked = []
    for item in candidates:
        body = terms(item["source_text"])
        lexical = len(qterms & body) / max(1, len(qterms))
        semantic = max(0.0, float(item.get("semantic_score", 0.0)))
        # Semantic similarity leads; lexical coverage rewards explicit evidence.
        score = 0.72 * semantic + 0.28 * lexical
        ranked.append({**item, "lexical_score": lexical, "score": score})
    ranked.sort(key=lambda row: row["score"], reverse=True)
    return [item for item in ranked if item["semantic_score"] >= threshold or item["lexical_score"] >= 0.5][:limit]
