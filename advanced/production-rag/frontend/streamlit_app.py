import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="DocuMind AI", page_icon="✦", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
.stApp{background:radial-gradient(ellipse at 78% 0%,#27204c 0%,#11121b 38%,#0b0c12 100%);color:#f0eff7;font-family:'DM Sans',sans-serif}
.block-container{max-width:1160px;padding-top:2rem;padding-bottom:4rem}
[data-testid="stSidebar"]{background:#10111a;border-right:1px solid #292936}
h1,h2,h3{font-family:'Manrope',sans-serif!important;letter-spacing:-.035em}
.hero{padding:1.4rem 0 1.5rem}.eyebrow{color:#b7a8ff;text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:700}
.hero h1{font-size:3rem;margin:.25rem 0}.hero p{color:#aaa9bb;font-size:1.05rem}
.panel{background:linear-gradient(145deg,rgba(29,29,43,.94),rgba(20,20,31,.92));border:1px solid #343344;border-radius:18px;padding:1.15rem 1.35rem;margin:.6rem 0 1rem}
.metric{color:#a6a5b7;font-size:.78rem}.answer{font-size:1.05rem;line-height:1.75;color:#f2f0fa}
.source{border:1px solid #353448;border-radius:12px;background:#171722;padding:.85rem 1rem;margin:.5rem 0;color:#cac8d8}
div.stButton>button{border-radius:10px;border:1px solid #7465da;background:linear-gradient(100deg,#6757dc,#8c6cec);color:white;font-weight:700;min-height:2.8rem}
div[data-testid="stFileUploader"]{border:1px dashed #615d7e;border-radius:14px;background:#171722;padding:.6rem}
small{color:#aaa9bb}
</style>
""", unsafe_allow_html=True)

def api_error(response):
    try:
        payload = response.json()
        return payload.get("detail", "The request could not be completed.")
    except Exception:
        return "The API is unavailable. Start the backend and try again."

st.markdown('<div class="hero"><div class="eyebrow">Production RAG · ShadowFox Advanced</div><h1>DocuMind AI</h1><p>Ask grounded questions across your documents with source-backed answers.</p></div>', unsafe_allow_html=True)
try:
    response = requests.get(f"{API_URL}/documents", timeout=4)
    response.raise_for_status()
    documents = response.json()
    api_ready = True
except requests.RequestException:
    documents, api_ready = [], False
if not api_ready:
    st.warning("The DocuMind API is not reachable. Start the backend with `uvicorn app.main:app --reload`.")

with st.container(border=True):
    st.subheader("Document workspace")
    st.caption("Upload a PDF, TXT, or Markdown file. DocuMind extracts, chunks, embeds, and indexes it locally.")
    uploaded = st.file_uploader("Choose documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("＋  Index documents", disabled=not uploaded or not api_ready, use_container_width=True):
        bar = st.progress(0)
        for i, file in enumerate(uploaded):
            try:
                result = requests.post(f"{API_URL}/documents/upload", files={"file": (file.name, file.getvalue(), file.type)}, timeout=180)
                if result.status_code >= 400:
                    st.error(f"{file.name}: {api_error(result)}")
                else:
                    st.success(f"Indexed {file.name}")
            except requests.RequestException:
                st.error(f"Could not index {file.name}. Check that the backend is running.")
            bar.progress((i + 1) / len(uploaded))
        st.rerun()

left, right = st.columns([1.2, 2], gap="large")
with left:
    st.subheader("Your library")
    total_chunks = sum(d["chunk_count"] for d in documents)
    st.markdown(f'<div class="panel"><span class="metric">DOCUMENTS</span><h2>{len(documents)}</h2><span class="metric">INDEXED CHUNKS</span><h2>{total_chunks}</h2></div>', unsafe_allow_html=True)
    if documents:
        selected = st.selectbox("Search scope", [None] + [d["document_id"] for d in documents], format_func=lambda value: "All documents" if value is None else next(d["document_name"] for d in documents if d["document_id"] == value))
        for doc in documents:
            c1, c2 = st.columns([4, 1])
            c1.caption(f"**{doc['document_name']}**  ·  {doc['source_type'].upper()}  ·  {doc['chunk_count']} chunks")
            if c2.button("×", key=f"delete-{doc['document_id']}", help="Remove document"):
                try:
                    r = requests.delete(f"{API_URL}/documents/{doc['document_id']}", timeout=30)
                    if r.ok:
                        st.rerun()
                    st.error(api_error(r))
                except requests.RequestException:
                    st.error("The document could not be removed. Check that the backend is running and try again.")
    else:
        selected = None
        st.info("Your library is empty. Upload documents above to begin.")

with right:
    st.subheader("Ask your documents")
    question = st.text_area("Question", placeholder="Ask something about your uploaded documents…", height=120, label_visibility="collapsed", max_chars=2000)
    ask = st.button("✦  Ask DocuMind", disabled=not api_ready, use_container_width=True)
    if ask:
        if not question.strip():
            st.warning("Enter a question to continue.")
        else:
            with st.spinner("Retrieving evidence and checking the answer…"):
                try:
                    response = requests.post(f"{API_URL}/query", json={"question": question, "document_id": selected}, timeout=90)
                    if response.status_code >= 400:
                        st.error(api_error(response))
                    else:
                        result = response.json()
                        st.session_state["last_result"] = result
                except requests.RequestException:
                    st.error("The query could not reach the backend. Please retry.")
    result = st.session_state.get("last_result")
    if result:
        status = "✓ Grounded" if result["grounded"] else "⚠ Insufficient evidence"
        with st.container(border=True):
            st.markdown(f"**{status}**")
            st.markdown(result["answer"])
        with st.expander(f"Sources · {len(result['sources'])}", expanded=True):
            for i, source in enumerate(result["sources"], 1):
                page = f"Page {source['page_number']}" if source.get("page_number") else f"Chunk {source['chunk_index'] + 1}"
                with st.container(border=True):
                    st.markdown(f"**Source {i} · {source['document_name']}** · {page} · score {source['score']:.2f}")
                    st.caption(source["source_text"])
        with st.expander("Retrieval details"):
            st.write({"Refined query": result["rewritten_query"], "Candidates": result["candidate_count"], "Final contexts": result["context_count"], "Grounding": result["grounding_status"]})
