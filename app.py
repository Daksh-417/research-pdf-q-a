import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from fastembed import TextEmbedding
import re, hashlib

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Research PDF Q&A", page_icon="📄", layout="centered")

# ── Constants ────────────────────────────────────────────────
MAX_PDFS      = 10
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
TOP_K         = 5
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"

# ── Model (loaded once) ──────────────────────────────────────
@st.cache_resource
def load_model():
    return TextEmbedding(MODEL_NAME)

model = load_model()

def norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)

def embed(texts: list[str]) -> np.ndarray:
    vecs = np.stack(list(model.embed(texts)))   # fastembed returns a generator
    return norm(vecs).astype("float32")

# ── Helpers ──────────────────────────────────────────────────
def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(p.get_text("text") for p in doc)
    doc.close()
    return text

def chunk_text(text: str, source: str) -> list[dict]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            lp = text.rfind(". ", start, end)
            if lp > start + CHUNK_SIZE // 2:
                end = lp + 1
        chunks.append({"text": text[start:end].strip(), "source": source})
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if len(c["text"]) > 40]

def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

# ── Session state ────────────────────────────────────────────
ss = st.session_state
ss.setdefault("index", None)
ss.setdefault("chunks", [])
ss.setdefault("loaded_files", {})
ss.setdefault("history", [])

# ── Sidebar: upload ──────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDFs")
    st.caption(f"Up to {MAX_PDFS} research papers")

    uploaded = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded:
        if len(uploaded) > MAX_PDFS:
            st.warning(f"Max {MAX_PDFS} files. Extra ignored.")
            uploaded = uploaded[:MAX_PDFS]

        new = [(f, file_hash(f.getvalue())) for f in uploaded
               if file_hash(f.getvalue()) not in ss.loaded_files]

        if new:
            with st.spinner(f"Indexing {len(new)} file(s)…"):
                all_chunks = list(ss.chunks)
                for f, h in new:
                    raw = extract_text(f.getvalue())
                    if not raw.strip():
                        st.error(f"⚠️ No text in **{f.name}**")
                        continue
                    all_chunks.extend(chunk_text(raw, f.name))
                    ss.loaded_files[h] = f.name

                texts = [c["text"] for c in all_chunks]
                if texts:
                    vecs = embed(texts)
                    idx = faiss.IndexFlatIP(vecs.shape[1])
                    idx.add(vecs)
                    ss.index = idx
                    ss.chunks = all_chunks

        st.success(f"✅ {len(ss.loaded_files)} file(s) · {len(ss.chunks)} chunks")

    if ss.loaded_files:
        st.markdown("**Loaded:**")
        for name in ss.loaded_files.values():
            st.text(f"  • {name}")
        if st.button("🗑️ Clear all"):
            ss.index, ss.chunks, ss.loaded_files, ss.history = None, [], {}, []
            st.rerun()

# ── Main: Q&A ────────────────────────────────────────────────
st.title("📄 Research PDF Q&A")
st.caption("Upload papers in the sidebar → ask → get the most relevant excerpts.")

if ss.index is None:
    st.info("👈 Upload at least one PDF to start.")
    st.stop()

if q := st.chat_input("Ask a question about the papers…"):
    ss.history.append({"role": "user", "content": q})
    qv = embed([q])
    scores, ids = ss.index.search(qv, TOP_K)

    parts = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        c = ss.chunks[i]
        parts.append(f"**[{len(parts)+1}]** *(score {score:.3f})* — `{c['source']}`\n\n> {c['text']}")

    ss.history.append({"role": "assistant",
                       "content": "\n\n---\n\n".join(parts) or "No relevant passages found."})

for msg in ss.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
