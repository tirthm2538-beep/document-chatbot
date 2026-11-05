import os
import time
import numpy as np
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="centered")

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=30000),
)

MODELS = [
    "gemini-flash-latest",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
]
MAX_CHUNKS = 120
BATCH_SIZE = 20

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #4f46e5, #06b6d4);
    margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: 0.4rem 0 0 0; color: white; opacity: 0.92;}
.chip {
    display: inline-block;
    padding: 2px 12px;
    margin: 6px 6px 0 0;
    border-radius: 999px;
    background: rgba(79, 70, 229, 0.15);
    font-size: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_index():
    data = np.load("index.npz")
    vectors = data["vectors"].astype("float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, data["texts"], data["sources"]


def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def embed_batch(chunks):
    for attempt in range(3):
        try:
            r = client.models.embed_content(
                model="gemini-embedding-001", contents=chunks
            )
            return [e.values for e in r.embeddings]
        except Exception:
            time.sleep(5)
    return None


def process_upload(uploaded, progress):
    reader = PdfReader(uploaded)
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        return None, "No readable text found. Scanned PDFs are not supported."
    chunks = chunk_text(text)[:MAX_CHUNKS]
    vectors = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        result = embed_batch(batch)
        if result is None:
            return None, "Google was busy while reading the file. Please try again."
        vectors.extend(result)
        progress.progress(min(1.0, (i + BATCH_SIZE) / len(chunks)))
        time.sleep(1)
    arr = np.array(vectors, dtype="float32")
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    return {
        "name": uploaded.name,
        "vectors": arr,
        "texts": np.array(chunks),
        "sources": np.array([uploaded.name] * len(chunks)),
    }, None


def embed_question(question):
    try:
        r = client.models.embed_content(
            model="gemini-embedding-001", contents=question
        )
        return np.array(r.embeddings[0].values, dtype="float32"), None
    except Exception as e:
        return None, f"Embedding step failed: {str(e)[:200]}"


def generate_with_retry(prompt):
    start = time.time()
    last_error = None
    for model in MODELS:
        if time.time() - start > 45:
            break
        try:
            reply = client.models.generate_content(model=model, contents=prompt)
            return reply.text, None
        except Exception as e:
            last_error = f"{model} failed: {str(e)[:200]}"
    return None, last_error


b_vectors, b_texts, b_sources = load_index()
builtin_names = sorted({str(s) for s in b_sources})

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🧠 DocuMind AI")
    st.write(
        "An AI assistant that answers questions using only your documents, "
        "and shows where every answer came from."
    )

    st.subheader("📤 Upload your own PDF")
    uploaded = st.file_uploader("Drop a PDF here", type="pdf")
    if uploaded is None:
        st.session_state.pop("custom", None)
    elif st.session_state.get("custom", {}).get("name") != uploaded.name:
        bar = st.progress(0.0, text="Reading your document...")
        custom, err = process_upload(uploaded, bar)
        bar.empty()
        if err:
            st.error(err)
        else:
            st.session_state["custom"] = custom
            st.success(f"Ready: {custom['name']}")

    custom = st.session_state.get("custom")
    scope = "Built-in documents"
    if custom:
        scope = st.radio(
            "Search in",
            ["My uploaded document", "Built-in documents", "Both"],
        )

    st.subheader("📄 Loaded documents")
    if custom:
        st.write(f"• {custom['name']} (yours)")
    for name in builtin_names:
        st.write(f"• {name}")

    st.subheader("💡 Try asking")
    examples = [
        "Summarize the main idea of this document",
        "What are the key findings?",
        "What problem does this document try to solve?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
    st.caption("Built by Tirth · Python, Gemini, Streamlit")

# ---------- Header ----------
st.markdown(
    """
<div class="hero">
  <h1>🧠 DocuMind AI</h1>
  <p>Ask anything about your documents. Get answers with sources.</p>
</div>
""",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.info("👈 Upload a PDF or pick an example question in the sidebar, or type your own below.")


def show_message(m):
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            chips = "".join(f'<span class="chip">📄 {s}</span>' for s in m["sources"])
            st.markdown(chips, unsafe_allow_html=True)
        if m.get("error"):
            st.caption(m["error"])


for m in st.session_state.messages:
    show_message(m)

# ---------- Chat ----------
typed = st.chat_input("Ask a question about the documents...")
question = typed or st.session_state.pop("pending", None)

if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    show_message(user_msg)

    parts_v, parts_t, parts_s = [], [], []
    if custom and scope in ("My uploaded document", "Both"):
        parts_v.append(custom["vectors"])
        parts_t.append(custom["texts"])
        parts_s.append(custom["sources"])
    if (not custom) or scope in ("Built-in documents", "Both"):
        parts_v.append(b_vectors)
        parts_t.append(b_texts)
        parts_s.append(b_sources)
    vectors = np.concatenate(parts_v)
    texts = np.concatenate(parts_t)
    source_names = np.concatenate(parts_s)

    answer = "Sorry, the AI service did not respond. Please try again in a moment."
    sources = []
    error_note = None

    with st.spinner("Searching the documents..."):
        q, error_note = embed_question(question)
        if q is not None:
            q = q / np.linalg.norm(q)
            scores = vectors @ q
            top = np.argsort(scores)[::-1][:4]
            context = "\n\n".join(str(texts[i]) for i in top)
            found = sorted({str(source_names[i]) for i in top})

            prompt = (
                "Answer the question using only the context below. "
                "If the answer is not in the context, say you don't know.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            text, error_note = generate_with_retry(prompt)
            if text is not None:
                answer = text
                sources = found
                error_note = None

    bot_msg = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "error": error_note,
    }
    st.session_state.messages.append(bot_msg)
    show_message(bot_msg)