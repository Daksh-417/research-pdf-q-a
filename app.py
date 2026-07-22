import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from fastembed import TextEmbedding
import re, hashlib

try:                                  # optional: app still runs without it
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Research PDF Q&A", page_icon="📄", layout="centered")

# ── Constants ────────────────────────────────────────────────
MAX_PDFS      = 10
CHUNK_SIZE    = 800          # chars, sentence-bounded
OVERLAP_SENT  = 1
TOP_K         = 6
FETCH         = 18           # candidates before MMR
THRESH        = 0.25         # min cosine to keep
LAM           = 0.6          # MMR relevance vs diversity
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.3-70b-versatile"   # swap to llama-3.1-8b-instant for max speed

# ── Model (loaded once) ──────────────────────────────────────
@st.cache_resource
def load_model():
    return TextEmbedding(MODEL_NAME)

model = load_model()

def norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return (v / np.clip(n, 1e-12, None)).astype("float32")

def embed(texts: list[str]) -> np.ndarray:
    return norm(np.stack(list(model.embed(texts))))

# ── Text extraction + cleaning ───────────────────────────────
def clean_page(page) -> str:
    lines = []
    for ln in page.get_text("text", sort=True).splitlines():
        s = ln.strip()
        if not s or len(s) < 3:
            continue
        if re.fullmatch(r"\d{1,4}", s):          # bare page number
            continue
        lines.append(s)
    return " ".join(lines)

def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n\n".join(clean_page(p) for p in doc)
    doc.close()
    return re.sub(r"[ \t]+", " ", text).strip()

def is_reference_block(chunk: str) -> bool:
    if re.search(r"\b(References|Bibliography|Works Cited)\b", chunk):
        return True
    if chunk.count("http") >= 2:                  # reference lists carry URLs
        return True
    return False

def chunk_text(text: str, source: str) -> list[dict]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks, buf, buflen = [], [], 0
    for s in sentences:
        if buflen + len(s) > CHUNK_SIZE and buf:
            chunks.append(" ".join(buf))
            buf, buflen = buf[-OVERLAP_SENT:], sum(len(x) for x in buf[-OVERLAP_SENT:])
        buf.append(s); buflen += len(s) + 1
    if buf:
        chunks.append(" ".join(buf))
    out = []
    for c in chunks:
        if len(c) > 40 and not is_reference_block(c):
            out.append({"text": c, "source": source})
    return out

def file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()

# ── MMR search (relevance + diversity, thresholded) ──────────
def mmr_search(q_vec: np.ndarray, k: int) -> list[dict]:
    m = min(FETCH, len(ss.chunks))
    scores, ids = ss.index.search(q_vec, m)
    valid = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0 and s >= THRESH]
    if not valid:
        return []
    ids = np.array([i for i, _ in valid]); rel = np.array([s for _, s in valid])
    cvecs = ss.embeddings[ids]
    sel, rem = [], list(range(len(ids)))
    while len(sel) < k and rem:
        rel_rem = rel[rem]
        if not sel:
            score = rel_rem
        else:
            sim = cvecs[rem] @ cvecs[sel].T
            score = LAM * rel_rem - (1 - LAM) * sim.max(axis=1)
        pick = int(np.argmax(score)); r = rem[pick]
        sel.append(r); rem.pop(pick)
    return [{**ss.chunks[int(ids[p])], "score": float(rel[p])} for p in sel]

# ── LLM synthesis (optional) ─────────────────────────────────
def groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None

def synthesize(question: str, results: list[dict]):
    ctx = "\n\n".join(f"[{i+1}] ({r['source']}) {r['text']}" for i, r in enumerate(results))
    client = Groq(api_key=groq_key())
    return client.chat.completions.create(
        model=GROQ_MODEL, temperature=0.2, max_tokens=1200, stream=True,
        messages=[
            {"role": "system", "content":
             "Answer using ONLY the provided research excerpts. Cite with [n] matching "
             "the excerpt numbers. If the excerpts don't contain the answer, say so plainly. "
             "Be concise and precise."},
            {"role": "user", "content": f"Question: {question}\n\nExcerpts:\n{ctx}"},
        ],
    )

# ── Session state ────────────────────────────────────────────
ss = st.session_state
for k, v in {"index": None, "chunks": [], "embeddings": None,
             "loaded_files": {}, "history": []}.items():
    ss.setdefault(k, v)

# ── Sidebar: upload ──────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload PDFs")
    st.caption(f"Up to {MAX_PDFS} research papers")
    uploaded = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if uploaded:
        if len(uploaded) > MAX_PDFS:
            st.warning(f"Max {MAX_PDFS} files. Extra ignored."); uploaded = uploaded[:MAX_PDFS]
        new = [(f, file_hash(f.getvalue())) for f in uploaded
               if file_hash(f.getvalue()) not in ss.loaded_files]
        if new:
            with st.spinner(f"Indexing {len(new)} file(s)…"):
                all_chunks = list(ss.chunks)
                for f, h in new:
                    raw = extract_text(f.getvalue())
                    if not raw.strip():
                        st.error(f"⚠️ No text in **{f.name}**"); continue
                    all_chunks.extend(chunk_text(raw, f.name))
                    ss.loaded_files[h] = f.name
                if all_chunks:
                    vecs = embed([c["text"] for c in all_chunks])
                    idx = faiss.IndexFlatIP(vecs.shape[1]); idx.add(vecs)
                    ss.index, ss.chunks, ss.embeddings = idx, all_chunks, vecs
        st.success(f"✅ {len(ss.loaded_files)} file(s) · {len(ss.chunks)} chunks")

    if ss.loaded_files:
        st.markdown("**Loaded:**")
        for name in ss.loaded_files.values():
            st.text(f"  • {name}")
        if st.button("🗑️ Clear all"):
            ss.index = ss.chunks = ss.embeddings = ss.history = None
            ss.loaded_files = {}; ss.chunks = ss.history = ss.embeddings = None
            ss.chunks, ss.history, ss.embeddings = [], [], None
            st.rerun()

    use_llm = GROQ_OK and bool(groq_key())
    st.caption("🟢 LLM answers ON" if use_llm else "⚪ LLM answers OFF (excerpts only)")

# ── Render helpers ───────────────────────────────────────────
def render_msg(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    st.markdown(f"**[{s['n']}]** *(score {s['score']:.3f})* — `{s['src']}`\n\n> {s['text']}")

def excerpts_block(results):
    return "\n\n---\n\n".join(
        f"**[{i+1}]** *(score {r['score']:.3f})* — `{r['source']}`\n\n> {r['text']}"
        for i, r in enumerate(results)) or "No relevant passages found."

# ── Main ─────────────────────────────────────────────────────
st.title("📄 Research PDF Q&A")
st.caption("Upload papers in the sidebar → ask → get a cited answer.")

if ss.index is None:
    st.info("👈 Upload at least one PDF to start."); st.stop()

for msg in ss.history:        # render everything already in history
    render_msg(msg)

if question := st.chat_input("Ask a question about the papers…"):
    ss.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    results = mmr_search(embed([question]), TOP_K)
    sources = [{"n": i+1, "src": r["source"], "text": r["text"], "score": r["score"]}
               for i, r in enumerate(results)]

    with st.chat_message("assistant"):
        if use_llm and results:
            ph = st.empty(); full = ""
            for ch in synthesize(question, results):
                d = ch.choices[0].delta.content or ""
                full += d; ph.markdown(full)
            content = full or "Could not generate an answer."
        else:
            content = excerpts_block(results)
            st.markdown(content)
        if sources:
            with st.expander(f"📚 Sources ({len(sources)})"):
                for s in sources:
                    st.markdown(f"**[{s['n']}]** *(score {s['score']:.3f})* — `{s['src']}`\n\n> {s['text']}")

    ss.history.append({"role": "assistant", "content": content, "sources": sources if use_llm else []})
