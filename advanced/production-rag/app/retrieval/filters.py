def build_context(chunks: list[dict], max_chars: int = 9000) -> tuple[str, list[dict]]:
    selected, blocks, used, seen = [], [], 0, set()
    for item in chunks:
        text = item["source_text"].strip()
        if not text or text in seen:
            continue
        label = f"[Source {len(selected)+1}: {item['document_name']} | page {item['page_number'] if item.get('page_number') else 'N/A'} | chunk {item['chunk_id']}]"
        block = f"{label}\n{text}"
        if used + len(block) > max_chars:
            if selected:
                break
            block = block[:max_chars]
        selected.append(item)
        blocks.append(block)
        seen.add(text)
        used += len(block)
    return "\n\n".join(blocks), selected
