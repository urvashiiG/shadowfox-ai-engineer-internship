# DocuMind AI

**DocuMind AI** is a document-based Q&A assistant that lets you upload your own PDF or TXT
study material and ask natural-language questions about it. Every answer is grounded in the
actual content of your documents, and every answer is shown alongside the exact source
snippets it was built from.

This is the **ShadowFox AI Engineer Internship — Intermediate Task**.

---

## Problem

A generic LLM chatbot answers from whatever it memorized during training. It has no access
to *your* private notes, textbooks, or lecture PDFs, so it will either say it doesn't know,
or — worse — confidently make something up ("hallucinate") that sounds plausible but is
wrong. For studying from your own material, that's a real problem: you need answers that
are actually tied to what's in your documents, with a way to verify where an answer came
from.

## Solution

DocuMind AI implements a real **Retrieval-Augmented Generation (RAG)** pipeline:

1. Your documents are parsed and split into small, overlapping chunks.
2. Each chunk is converted into a numeric vector (an *embedding*) using a local embedding
   model.
3. When you ask a question, your question is embedded the same way, and a vector similarity
   search (FAISS) finds the chunks most relevant to your question.
4. Only those *retrieved* chunks — never the whole document — are sent to an LLM, along with
   explicit instructions to answer only from that context.
5. The answer is displayed together with the source chunks it came from, so you can verify
   it yourself.

This grounds every answer in retrieved evidence instead of the model's memory alone.

## Features

- PDF and TXT upload (multiple files)
- Text extraction (page-aware for PDFs)
- Whitespace-normalized, overlapping chunking
- Local embeddings (no embedding API cost)
- FAISS vector similarity search
- Grounded, source-cited answers via an LLM
- Source snippet display (file, page, similarity score)
- Input validation (files, queries) and graceful error handling throughout
- Unit tests with no API key and no network calls required

## Architecture

```
Document (PDF/TXT)
        │
        ▼
  Text Extraction        (document_processor.py)
        │
        ▼
     Chunking             (document_processor.py)
        │
        ▼
   Embeddings             (embeddings.py — local, sentence-transformers)
        │
        ▼
   FAISS Index            (vector_store.py — in-memory)
        │
        ▼
     Retrieval            (rag_pipeline.py — top-k similarity search)
        │
        ▼
   Context Filtering      (rag_pipeline.py — character budget)
        │
        ▼
   Grounded Prompt        (rag_pipeline.py)
        │
        ▼
  OpenRouter LLM          (llm_service.py)
        │
        ▼
   Answer + Sources       (app.py)
```

## RAG Pipeline

1. **Ingestion** — `document_processor.extract_text_from_file` reads PDF (via `pypdf`) or
   TXT files and returns page-aware text.
2. **Chunking** — `document_processor.chunk_text` normalizes whitespace and splits the text
   into overlapping chunks, each carrying `source`, `chunk_id`, and `page` metadata.
3. **Embedding** — `embeddings.embed_documents` / `embed_query` use a local
   `sentence-transformers` model to turn text into normalized vectors.
4. **Indexing** — `vector_store.VectorStore.add` stores vectors and metadata in an in-memory
   FAISS `IndexFlatIP` index.
5. **Retrieval** — `rag_pipeline.retrieve_context` embeds the user's question and performs a
   FAISS similarity search, returning the top-k most relevant chunks with similarity scores.
6. **Grounded prompting** — `rag_pipeline.build_grounded_prompt` builds a prompt that lists
   each retrieved chunk as a numbered `[Source N]` block and instructs the model to answer
   using only that context.
7. **Generation** — `llm_service.generate_grounded_answer` sends the prompt to OpenRouter and
   returns a structured result (success/answer/error), never a raw exception.
8. **Display** — `app.py` renders the answer plus a source list (file name, page, similarity
   score, and excerpt) so the user can verify grounding.

Only the retrieved chunks are ever sent to the LLM — the full document is never passed in
a single prompt. This is what makes it a genuine RAG pipeline rather than "paste the whole
file into the model".

## Chunking Strategy

- **Chunk size:** `800` characters (default, configurable via `CHUNK_SIZE`)
- **Chunk overlap:** `150` characters (default, configurable via `CHUNK_OVERLAP`)

Overlap matters because a sentence or idea near a chunk boundary can otherwise be split in
half, losing context for both the retriever and the LLM. A moderate overlap (roughly 15–20%
of the chunk size here) keeps each chunk self-contained enough to be independently useful
during retrieval, without producing so much redundant text that the index balloons in size.

## Embeddings

Embeddings are generated locally with **`sentence-transformers/all-MiniLM-L6-v2`**:

- **No embedding API cost** — embedding happens on your own machine.
- **Reproducible** — the same text always produces the same vector, with no dependency on an
  external service's availability or versioning.
- **Well suited to an internship/demo application** — it's small (~80MB), fast on CPU, and
  well documented, so it's easy to set up and run anywhere.
- **Keeps retrieval independent from generation** — the embedding/retrieval layer has no
  dependency on OpenRouter (or any LLM provider) at all; only the final answer-generation
  step calls out to an LLM.

## Vector Search

Each chunk embedding is L2-normalized, and FAISS's `IndexFlatIP` (inner product) index is
used — for normalized vectors, inner product is mathematically equivalent to cosine
similarity. At query time, the question is embedded the same way, and FAISS returns the
**top-k** chunks (default `k = 4`, configurable via `TOP_K`) with the highest similarity
score to the question.

## Grounding

Retrieved chunks are formatted into numbered `[Source N]` blocks (file name, page, content)
and inserted into a prompt that explicitly instructs the model to:

- answer using **only** the supplied context,
- **not invent** information that isn't present in the context,
- **say clearly** when the answer cannot be found in the context, and
- reference the relevant source(s) where useful.

This significantly reduces the risk of hallucinated answers compared to an ungrounded
chatbot, because the model is working from a fixed, retrieved evidence set rather than open
recall. It does **not** eliminate hallucination entirely — a language model can still
misread or over-generalize from the provided context, which is exactly why the source
snippets are always shown alongside the answer, so you can verify it yourself.

## Project Structure

```
documind-rag/
│
├── app.py                  # Streamlit UI
├── config.py                # Centralized configuration
├── document_processor.py    # Text extraction + chunking
├── embeddings.py             # Local embedding generation
├── vector_store.py          # FAISS vector store
├── rag_pipeline.py          # Core RAG orchestration
├── llm_service.py           # OpenRouter LLM client
├── validators.py            # Input validation
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
└── tests/
    ├── test_validators.py
    ├── test_chunking.py
    └── test_vector_store.py
```

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Then create your local environment file:

```powershell
copy .env.example .env
```

Open `.env` and paste your OpenRouter API key into `OPENROUTER_API_KEY`
(get a free key at https://openrouter.ai). The app will still start and let you browse the
UI without a key — answer generation will show a friendly message asking you to configure it.

## Run

```powershell
streamlit run app.py
```

## Testing

```powershell
pytest
```

Tests cover validation, chunking, and vector-store logic using synthetic in-memory data.
They do **not** require an OpenRouter API key and make **no network calls**, so they run
the same way in any environment.

## Design Decisions

- **Local embeddings (sentence-transformers):** no embedding API cost, fully reproducible,
  and decouples retrieval quality from any LLM provider's availability.
- **FAISS in-memory index:** simple, fast, and sufficient for a session-scoped demo/internship
  app; avoids the operational overhead of a persistent vector database.
- **OpenRouter free model router (`openrouter/free`):** lets the generation layer run without
  requiring a paid subscription, while still using a standard OpenAI-compatible client.
- **Separation of ingestion / retrieval / generation:** `document_processor.py` and
  `embeddings.py` and `vector_store.py` have no knowledge of the LLM provider, and
  `llm_service.py` has no knowledge of embeddings or FAISS — each stage can be swapped
  independently.
- **In-memory vector store (no persistence):** keeps the project simple and avoids storing
  user document content on disk between runs; documents are re-uploaded and re-indexed each
  session.

## Limitations

- The in-memory FAISS index resets whenever the Streamlit app restarts — documents must be
  re-uploaded each session.
- Scanned or image-only PDFs are not guaranteed to extract text (no OCR is performed).
- Retrieval quality depends on chunk size/overlap and the embedding model's ability to
  represent your document's language and terminology.
- OpenRouter's free model tier availability and rate limits can vary over time and are
  outside this project's control.

## Future Improvements

- Persistent vector database (e.g. on-disk FAISS index or a dedicated vector DB)
- Re-ranking of retrieved chunks before generation
- OCR support for scanned/image-based PDFs
- Support for organizing documents into named collections
- Multi-turn conversation memory
- Observability/logging of retrieval quality and answer latency
- A small evaluation dataset to measure retrieval and answer quality over time

## Internship Mapping

| ShadowFox Intermediate Requirement | Implementation |
|---|---|
| Upload one or more PDF/TXT documents | `app.py` file uploader + `validators.validate_uploaded_files` |
| Extract text from documents | `document_processor.extract_text_from_pdf` / `extract_text_from_txt` |
| Split extracted text into meaningful chunks | `document_processor.chunk_text` |
| Generate embeddings for the chunks | `embeddings.embed_documents` (local `all-MiniLM-L6-v2`) |
| Store/search embeddings via vector similarity | `vector_store.VectorStore` (FAISS `IndexFlatIP`) |
| Ask natural-language questions | `app.py` question input + `validators.validate_query` |
| Retrieve the most relevant chunks | `rag_pipeline.retrieve_context` |
| Generate an answer grounded only in retrieved context | `rag_pipeline.build_grounded_prompt` + `llm_service.generate_grounded_answer` |
| Display supporting source information/snippets | `app.py` source display section |
| Handle invalid files, empty documents, empty queries, API errors gracefully | `validators.py`, `llm_service.py` structured error handling, `app.py` user-facing messages |
