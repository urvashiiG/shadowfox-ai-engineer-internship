"""
app.py

Streamlit UI for DocuMind AI.

This module is presentation only: file upload, document processing
controls/status, question input, answer display, and source display.
All embedding / chunking / vector-search / LLM logic lives in the other
modules (document_processor, embeddings, vector_store, rag_pipeline,
llm_service) and is only orchestrated from here.

IMPORT STRATEGY (why the page renders instantly):
`embeddings.py` imports sentence-transformers (which pulls in torch), and
`rag_pipeline.py` imports both `embeddings` and `llm_service` (which
imports the OpenAI client). Importing those is expensive. If they were
imported at module level, Python would load them the instant this script
runs -- before any UI is drawn -- so the browser would sit on a blank
page until that finished, on every rerun.

So only lightweight modules (config, validators, document_processor,
vector_store -- none of which pull in torch/openai) are imported at the
top level. `embeddings.embed_documents` and `rag_pipeline.answer_question`
are imported lazily, inside the "Process documents" and "Ask DocuMind"
click handlers, so the heavy import (and the model load, cached inside
embeddings.py) only happens the first time the user actually triggers
that action -- never merely to paint the page.

HTML-BLOCK STRATEGY (why there is no empty "ghost" card):
Streamlit renders each st.markdown() call as its own independent HTML
fragment. Opening a <div> in one st.markdown() call and closing it in a
*later*, separate call does not actually wrap the elements in between --
the browser closes the unmatched tag at the end of that single fragment,
leaving a visible, empty, styled box with nothing in it. Every "card" in
this file is therefore built as ONE self-contained st.markdown() call
(opening tag, content, and closing tag all in the same string). Sections
that must contain live Streamlit widgets (the file uploader, buttons,
the text input) are given a plain heading block instead of a bordered
card wrapper, since a real card box cannot safely wrap a widget across
multiple calls.
"""

import streamlit as st

from config import is_llm_configured, MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
from validators import validate_uploaded_files, validate_extracted_text, validate_query
from document_processor import extract_text_from_file, chunk_text
from vector_store import VectorStore

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Premium dark theme (CSS) -- a plain string constant, so it costs nothing
# to compute on rerun; only the (cheap) st.markdown call runs each time.
# ---------------------------------------------------------------------------
_DM_CSS = """
<style>
    :root {
        --bg: #0b0b12;
        --bg-soft: #111119;
        --card: #15151f;
        --card-hover: #191926;
        --card-border: #26263a;
        --accent: #8b7dff;
        --accent-2: #6f63d9;
        --accent-soft-bg: rgba(139, 125, 255, 0.10);
        --text: #edeef5;
        --text-muted: #9a9ab0;
        --text-faint: #6f6f85;
        --success: #6fd88a;
        --warn: #e0a84f;
        --error: #e07272;
    }

    .stApp {
        background: radial-gradient(1100px 550px at 15% -10%, rgba(139,125,255,0.07), transparent),
                    radial-gradient(900px 450px at 100% 0%, rgba(111,99,217,0.05), transparent),
                    var(--bg);
        color: var(--text);
    }

    /* Hide only the hamburger menu and the small "Made with Streamlit"
       footer text. This does not touch Streamlit's top header bar or its
       reserved space, so normal page flow is unaffected. */
    #MainMenu, footer {visibility: hidden;}

    section[data-testid="stSidebar"] {
        background-color: var(--bg-soft);
        border-right: 1px solid var(--card-border);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Normal document flow only: no negative margins, no fixed/absolute
       positioning, no transforms. A modest, fixed top padding is enough
       to clear Streamlit's own header bar without leaving a big gap. */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 2rem;
        max-width: 1100px;
        margin: 0 auto;
    }

    /* ---------- Header ---------- */
    .dm-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-bottom: 0.9rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid var(--card-border);
    }
    .dm-header-brand {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        min-width: 0;
    }
    .dm-header-icon {
        font-size: 1.4rem;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        width: 38px;
        height: 38px;
        min-width: 38px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 16px rgba(139,125,255,0.22);
    }
    .dm-header-title {
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.1;
        color: var(--text);
    }
    .dm-header-subtitle {
        font-size: 0.8rem;
        color: var(--text-muted);
    }
    .dm-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid var(--card-border);
        background-color: var(--card);
        white-space: nowrap;
    }
    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .dot-ok { background-color: var(--success); box-shadow: 0 0 8px rgba(111,216,138,0.6); }
    .dot-warn { background-color: var(--warn); box-shadow: 0 0 8px rgba(224,168,79,0.5); }

    /* ---------- Hero (compact) ---------- */
    .dm-badge {
        display: inline-block;
        background-color: var(--accent-soft-bg);
        color: var(--accent);
        border: 1px solid rgba(139, 125, 255, 0.35);
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.55rem;
    }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 0.35rem;
        background: linear-gradient(135deg, #ffffff 30%, #b9b2ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        max-width: 46rem;
    }
    .hero-subtitle {
        color: var(--text-muted);
        font-size: 0.92rem;
        max-width: 620px;
        margin-bottom: 0.9rem;
        line-height: 1.5;
    }

    /* ---------- Stat cards ---------- */
    .dm-stat-card {
        background-color: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
        text-align: left;
        transition: border-color 0.15s ease;
        overflow-wrap: anywhere;
    }
    .dm-stat-card:hover { border-color: rgba(139,125,255,0.4); }
    .dm-stat-label {
        color: var(--text-faint);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }
    .dm-stat-value {
        color: var(--text);
        font-size: 1.2rem;
        font-weight: 700;
    }
    .dm-stat-value.small {
        font-size: 0.92rem;
    }

    /* ---------- Plain section headings (no card wrapper) used right
       before a live Streamlit widget, so nothing can be left "unclosed" ---------- */
    .dm-section-block {
        margin-top: 1.3rem;
        margin-bottom: 0.5rem;
    }
    .dm-section-heading {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.15rem;
    }
    .dm-section-sub {
        color: var(--text-muted);
        font-size: 0.85rem;
    }

    /* ---------- Generic content card (only ever used for fully
       self-contained HTML, never split across calls) ---------- */
    .dm-card {
        background-color: var(--card);
        border: 1px solid var(--card-border);
        border-radius: 14px;
        padding: 1rem 1.15rem;
        margin-top: 0.9rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.16);
    }

    /* ---------- Upload dropzone ---------- */
    div[data-testid="stFileUploaderDropzone"] {
        background-color: var(--bg-soft);
        border: 1.5px dashed var(--card-border);
        border-radius: 12px;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent);
    }

    /* ---------- Document status rows ---------- */
    .dm-doc-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.4rem 0.2rem;
        border-bottom: 1px solid var(--card-border);
        font-size: 0.9rem;
        color: var(--text);
        overflow-wrap: anywhere;
    }
    .dm-doc-row:last-child { border-bottom: none; }
    .dm-doc-check {
        color: var(--success);
        font-weight: 700;
        flex-shrink: 0;
    }

    /* ---------- Answer card ---------- */
    .dm-answer-card {
        background: linear-gradient(180deg, rgba(139,125,255,0.06), transparent 40%), var(--card);
        border: 1px solid rgba(139,125,255,0.30);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin-top: 0.6rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 8px 26px rgba(139,125,255,0.07);
        overflow-wrap: anywhere;
    }

    /* ---------- Source cards ---------- */
    .dm-source-card {
        background-color: var(--card);
        border: 1px solid var(--card-border);
        border-left: 3px solid var(--accent);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.55rem;
        transition: background-color 0.15s ease;
        overflow-wrap: anywhere;
    }
    .dm-source-card:hover { background-color: var(--card-hover); }
    .dm-source-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.4rem;
        flex-wrap: wrap;
        gap: 0.35rem;
    }
    .dm-source-badge {
        display: inline-block;
        background-color: var(--accent-soft-bg);
        color: var(--accent);
        border: 1px solid rgba(139, 125, 255, 0.35);
        border-radius: 999px;
        padding: 0.12rem 0.65rem;
        font-size: 0.72rem;
        font-weight: 700;
    }
    .dm-source-meta {
        color: var(--text-muted);
        font-size: 0.82rem;
    }
    .dm-source-score {
        color: var(--accent);
        font-weight: 700;
        font-size: 0.82rem;
    }
    .dm-source-text {
        color: var(--text-muted);
        font-size: 0.88rem;
        line-height: 1.45;
        margin-top: 0.35rem;
    }

    /* ---------- Empty state ---------- */
    .dm-empty-state {
        text-align: center;
        padding: 1.6rem 1.2rem;
        border: 1.5px dashed var(--card-border);
        border-radius: 14px;
        color: var(--text-muted);
        background-color: var(--bg-soft);
        margin-top: 0.6rem;
        margin-bottom: 0.5rem;
    }
    .dm-empty-icon {
        font-size: 1.7rem;
        margin-bottom: 0.4rem;
    }
    .dm-empty-title {
        color: var(--text);
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.25rem;
    }

    /* ---------- Footer ---------- */
    .dm-footer {
        text-align: center;
        color: var(--text-faint);
        font-size: 0.78rem;
        margin-top: 1.4rem;
        padding-top: 0.9rem;
        border-top: 1px solid var(--card-border);
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        background: linear-gradient(135deg, var(--accent-2), var(--accent));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: filter 0.15s ease, transform 0.05s ease;
    }
    .stButton>button:hover {
        filter: brightness(1.08);
    }
    .stButton>button:active {
        transform: scale(0.99);
    }

    section[data-testid="stSidebar"] .stButton>button {
        background: var(--card);
        border: 1px solid var(--card-border);
        color: var(--text-muted);
        width: 100%;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        border-color: var(--error);
        color: var(--error);
        filter: none;
    }

    div[data-testid="stTextInput"] input {
        background-color: var(--bg-soft);
        border: 1px solid var(--card-border);
        color: var(--text);
        border-radius: 10px;
        padding: 0.6rem 0.85rem;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent);
    }

    /* ---------- Small-screen safety ---------- */
    @media (max-width: 640px) {
        .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
        .hero-title { font-size: 1.4rem; }
        .dm-header { flex-direction: column; align-items: flex-start; }
    }
</style>
"""

st.markdown(_DM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization (cheap: plain Python objects only --
# VectorStore() just sets a couple of attributes; the FAISS index itself
# is created lazily on the first `.add()` call, so this never loads or
# computes anything expensive).
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []  # list of filenames processed
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "processing_errors" not in st.session_state:
    st.session_state.processing_errors = []

vector_store: VectorStore = st.session_state.vector_store
# is_llm_configured() only checks an in-memory config value (read once at
# import time from the environment) -- no network/API call is made here.
llm_ready = is_llm_configured()

# ---------------------------------------------------------------------------
# Header -- one self-contained st.markdown call, so nothing is left
# "open" and nothing can render as a stray empty box.
# ---------------------------------------------------------------------------
status_dot_class = "dot-ok" if llm_ready else "dot-warn"
status_text = "AI Connected" if llm_ready else "API key not configured"

st.markdown(
    f"""<div class="dm-header">
    <div class="dm-header-brand">
        <div class="dm-header-icon">🧠</div>
        <div>
            <div class="dm-header-title">DocuMind AI</div>
            <div class="dm-header-subtitle">Grounded Document Intelligence</div>
        </div>
    </div>
    <div class="dm-status-pill">
        <span class="dot {status_dot_class}"></span> {status_text}
    </div>
</div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧠 DocuMind AI")
    st.caption("Your documents. Your answers.")
    st.markdown("---")

    st.markdown("**Status**")
    if llm_ready:
        st.markdown('<span style="color:#6fd88a;">● AI Connected</span>', unsafe_allow_html=True)
        st.caption(f"Model: `{MODEL_NAME}`")
    else:
        st.markdown('<span style="color:#e0a84f;">● API key not configured</span>', unsafe_allow_html=True)
        st.caption("Add OPENROUTER_API_KEY to your .env file.")

    st.markdown("---")
    st.markdown("**Knowledge base**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Documents", vector_store.source_count())
    with col_b:
        st.metric("Chunks", vector_store.chunk_count())

    st.markdown("---")
    if st.button("🗑️ Clear all documents"):
        vector_store.clear()
        st.session_state.processed_files = []
        st.session_state.last_result = None
        st.session_state.processing_errors = []
        st.rerun()

    st.markdown("---")
    st.caption("Retrieval-augmented Q&A over your own PDF/TXT documents.")

# ---------------------------------------------------------------------------
# Hero section (compact, one self-contained call)
# ---------------------------------------------------------------------------
st.markdown(
    """<div class="dm-badge">✨ AI-Powered Document Q&A</div>
<div class="hero-title">Your documents. Your knowledge. Instantly searchable.</div>
<div class="hero-subtitle">Upload PDFs or TXT files and get grounded answers with the exact sources behind every answer.</div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Statistics row (real, dynamic values only -- computed from cheap,
# already-in-memory VectorStore metadata; no model or API call involved).
# Each card is its own fully self-contained markdown call.
# ---------------------------------------------------------------------------
retrieval_status = "Ready" if not vector_store.is_empty() else "No data"
ai_status = "Online" if llm_ready else "Not configured"

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
with stat_col1:
    st.markdown(
        f"""<div class="dm-stat-card">
        <div class="dm-stat-label">Documents</div>
        <div class="dm-stat-value">{vector_store.source_count()}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with stat_col2:
    st.markdown(
        f"""<div class="dm-stat-card">
        <div class="dm-stat-label">Chunks Indexed</div>
        <div class="dm-stat-value">{vector_store.chunk_count()}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with stat_col3:
    st.markdown(
        f"""<div class="dm-stat-card">
        <div class="dm-stat-label">Retrieval Status</div>
        <div class="dm-stat-value small">{retrieval_status}</div>
        </div>""",
        unsafe_allow_html=True,
    )
with stat_col4:
    st.markdown(
        f"""<div class="dm-stat-card">
        <div class="dm-stat-label">AI Status</div>
        <div class="dm-stat-value small">{ai_status}</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Upload section
#
# No bordered "card" div wraps this section: st.file_uploader and
# st.button are separate Streamlit components rendered as siblings, and a
# <div> opened before them cannot be safely closed after them (see the
# HTML-BLOCK STRATEGY note at the top of this file) -- that mismatch is
# exactly what produced the empty rounded bar. The heading below is a
# single, fully self-contained markdown call instead.
# ---------------------------------------------------------------------------
st.markdown(
    """<div class="dm-section-block">
    <div class="dm-section-heading">📁 Add your documents</div>
    <div class="dm-section-sub">Upload PDF or TXT files to build your knowledge base.</div>
</div>""",
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

process_clicked = st.button("⚙️ Process documents", disabled=not uploaded_files, use_container_width=True)

if process_clicked:
    is_valid, error = validate_uploaded_files(uploaded_files)
    if not is_valid:
        st.error(error)
    else:
        # Lazy import: this is the first point at which the local
        # embedding model is actually needed, so this is the first point
        # at which sentence-transformers/torch get imported and the
        # model gets loaded (and cached) -- never merely to draw the UI.
        from embeddings import embed_documents

        progress = st.progress(0.0, text="Starting document processing...")
        total = len(uploaded_files)
        new_errors = []

        for i, file in enumerate(uploaded_files):
            filename = file.name
            progress.progress((i) / total, text=f"Extracting text from {filename}...")

            if filename in st.session_state.processed_files:
                new_errors.append(f"'{filename}' was already processed; skipping duplicate.")
                continue

            try:
                pages = extract_text_from_file(file, filename)
            except Exception as exc:
                new_errors.append(f"Failed to read '{filename}': {exc}")
                continue

            full_text = " ".join(p["text"] for p in pages)
            is_valid_text, text_error = validate_extracted_text(full_text, filename)
            if not is_valid_text:
                new_errors.append(text_error)
                continue

            chunks = chunk_text(pages, source=filename)
            if not chunks:
                new_errors.append(f"'{filename}' produced no usable chunks.")
                continue

            progress.progress((i + 0.6) / total, text=f"Embedding {filename}...")
            try:
                vectors = embed_documents(chunks)
            except Exception as exc:
                new_errors.append(f"Failed to embed '{filename}': {exc}")
                continue

            vector_store.add(vectors, chunks)
            st.session_state.processed_files.append(filename)

        progress.progress(1.0, text="Done.")
        st.session_state.processing_errors = new_errors

        if new_errors:
            for err in new_errors:
                st.warning(err)
        if st.session_state.processed_files:
            st.success(f"Processed {len(uploaded_files)} file(s) successfully.")
        st.rerun()

# ---------------------------------------------------------------------------
# Document status card / empty state
#
# Both are single, fully self-contained st.markdown calls (opening tag,
# content, and closing tag together), so they render as real boxes with
# real content -- never as an empty leftover box.
# ---------------------------------------------------------------------------
if st.session_state.processed_files:
    rows_html = "".join(
        f'<div class="dm-doc-row"><span class="dm-doc-check">✓</span> {name}</div>'
        for name in st.session_state.processed_files
    )
    st.markdown(
        f"""<div class="dm-card">
        <div class="dm-section-heading" style="margin-bottom:0.5rem;">📄 Document status</div>
        {rows_html}
        <div class="dm-section-sub" style="margin-top:0.7rem;">
            {vector_store.source_count()} document(s) · {vector_store.chunk_count()} chunks indexed ·
            chunk size {CHUNK_SIZE} / overlap {CHUNK_OVERLAP}
        </div>
        </div>""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """<div class="dm-empty-state">
        <div class="dm-empty-icon">🗂️</div>
        <div class="dm-empty-title">No documents yet</div>
        <div>Upload a PDF or TXT file above to start building your knowledge base.</div>
        </div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Ask section (no bordered card wrapper -- same reasoning as Upload)
# ---------------------------------------------------------------------------
st.markdown(
    """<div class="dm-section-block">
    <div class="dm-section-heading">💬 Ask your documents</div>
    <div class="dm-section-sub">Ask a natural-language question — DocuMind will retrieve the most relevant passages and answer using only that context.</div>
</div>""",
    unsafe_allow_html=True,
)

query = st.text_input(
    "Ask a question",
    placeholder="e.g. What is normalization according to my notes?",
    label_visibility="collapsed",
)

ask_clicked = st.button("✨ Ask DocuMind →", use_container_width=True)

if ask_clicked:
    is_valid, error = validate_query(query)
    if not is_valid:
        st.error(error)
    elif vector_store.is_empty():
        st.warning("Please upload and process at least one document first.")
    else:
        # Lazy import: only pulls in embeddings/openai (via rag_pipeline)
        # at the moment the user actually asks a question.
        from rag_pipeline import answer_question

        with st.spinner("Retrieving relevant context and generating a grounded answer..."):
            result = answer_question(query, vector_store, top_k=TOP_K)
        st.session_state.last_result = result

# ---------------------------------------------------------------------------
# Answer + Sources display
# ---------------------------------------------------------------------------
result = st.session_state.last_result
if result is not None:
    st.markdown('<div class="dm-section-heading">🧩 Answer</div>', unsafe_allow_html=True)

    if result["success"]:
        # The answer's own Markdown (bold, lists, code blocks, etc.) is
        # rendered correctly here because it is included INSIDE the same
        # single st.markdown call as the opening/closing card <div> --
        # Streamlit's Markdown renderer supports inline HTML mixed with
        # Markdown content in one fragment. Splitting this across
        # separate calls is exactly what caused the empty-box bug
        # elsewhere, so the whole card is intentionally built as one call.
        st.markdown(
            f'<div class="dm-answer-card">\n\n{result["answer"]}\n\n</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error(result["error"] or "Something went wrong while generating the answer.")

    if result.get("sources"):
        st.markdown('<div class="dm-section-heading">📚 Sources</div>', unsafe_allow_html=True)
        for i, src in enumerate(result["sources"], start=1):
            page_label = f"Page {src['page']}" if src.get("page") is not None else "No page info"
            snippet = src["text"][:500] + ("..." if len(src["text"]) > 500 else "")
            st.markdown(
                f"""<div class="dm-source-card">
                <div class="dm-source-top">
                    <span class="dm-source-badge">Source {i}</span>
                    <span class="dm-source-score">Similarity {src['score']:.2f}</span>
                </div>
                <div class="dm-source-meta"><strong style="color:var(--text);">{src['source']}</strong> · {page_label}</div>
                <div class="dm-source-text">{snippet}</div>
                </div>""",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="dm-footer">DocuMind AI • ShadowFox AI Engineer Internship</div>',
    unsafe_allow_html=True,
)