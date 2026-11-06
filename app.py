import base64
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from supabase import create_client
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

SUPABASE_URL = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
if "sb" not in st.session_state:
    st.session_state.sb = create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = st.session_state.sb
if "user" not in st.session_state:
    st.session_state.user = None


def get_secret(name):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name)
    except Exception:
        return None


ADMIN_EMAIL = (get_secret("ADMIN_EMAIL") or "").strip().lower()

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

HERO = """
<div class="hero">
  <h1>🧠 DocuMind AI</h1>
  <p>Ask anything about your documents. Get answers with sources.</p>
</div>
"""


def authenticate(mode, email, password):
    try:
        if mode == "Sign up":
            res = supabase.auth.sign_up({"email": email, "password": password})
        else:
            res = supabase.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
    except Exception as e:
        return None, str(e)[:200]
    if res.user and res.session:
        return res.user, None
    return None, "Account created. Check your email to confirm it, then log in."


def sign_out():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.user = None
    st.session_state.messages = []
    st.session_state.pop("my_docs", None)
    st.session_state.pop("docs_error", None)
    st.session_state.handled_files = set()


# ---------- Login gate ----------
if not st.session_state.user:
    st.markdown(HERO, unsafe_allow_html=True)
    mode = st.radio("Welcome", ["Log in", "Sign up"], horizontal=True)
    login_email = st.text_input("Email")
    login_password = st.text_input("Password (at least 6 characters)", type="password")
    if st.button(mode, type="primary"):
        new_user, message = authenticate(mode, login_email, login_password)
        if new_user:
            st.session_state.user = new_user
            st.rerun()
        else:
            st.warning(message)
    st.stop()

user = st.session_state.user
is_admin = bool(ADMIN_EMAIL) and (user.email or "").lower() == ADMIN_EMAIL


def log_question(question, answered):
    try:
        supabase.table("questions").insert(
            {
                "user_email": user.email,
                "question": question,
                "answered": answered,
            }
        ).execute()
        st.session_state.pop("log_error", None)
    except Exception as e:
        st.session_state["log_error"] = str(e)[:150]


def show_admin():
    st.markdown(HERO, unsafe_allow_html=True)
    st.subheader("📊 Admin dashboard")
    try:
        rows = (
            supabase.table("questions")
            .select("*")
            .order("created_at", desc=True)
            .limit(500)
            .execute()
            .data
        )
    except Exception as e:
        st.error(f"Could not load questions: {str(e)[:200]}")
        return
    if not rows:
        st.info("No questions have been asked yet.")
        return

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    answered_count = int(df["answered"].fillna(False).astype(bool).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total questions", len(df))
    c2.metric("Answered", answered_count)
    c3.metric("Users", df["user_email"].nunique())

    st.subheader("Questions per day")
    per_day = df.groupby(df["created_at"].dt.date).size()
    st.bar_chart(per_day)

    st.subheader("Recent questions")
    st.dataframe(
        df[["created_at", "user_email", "question", "answered"]],
        use_container_width=True,
        hide_index=True,
    )


@st.cache_resource
def load_index():
    data = np.load("index.npz")
    vectors = data["vectors"].astype("float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    texts = data["texts"]
    if "pages" in data.files:
        pages = data["pages"]
    else:
        pages = np.zeros(len(texts), dtype=int)
    return vectors, texts, data["sources"], pages


def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def encode_vec(vec):
    arr = np.asarray(vec, dtype="float16")
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_vec(text):
    raw = base64.b64decode(text)
    return np.frombuffer(raw, dtype="float16").astype("float32")


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
    chunks, chunk_pages = [], []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").replace("\x00", "")
        if not page_text.strip():
            continue
        for chunk in chunk_text(page_text):
            chunks.append(chunk)
            chunk_pages.append(page_number)
    if not chunks:
        return None, "No readable text found. Scanned PDFs are not supported."
    chunks = chunks[:MAX_CHUNKS]
    chunk_pages = chunk_pages[:MAX_CHUNKS]
    vectors = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        result = embed_batch(batch)
        if result is None:
            return None, "Google was busy while reading the file. Remove the file and upload it again in a minute."
        vectors.extend(result)
        progress.progress(min(1.0, (i + BATCH_SIZE) / len(chunks)))
        time.sleep(1)
    return {
        "name": uploaded.name,
        "texts": chunks,
        "pages": chunk_pages,
        "vectors": np.array(vectors, dtype="float32"),
    }, None


def delete_document(doc_id):
    try:
        supabase.table("documents").delete().eq("id", doc_id).execute()
        return None
    except Exception as e:
        return str(e)[:150]


def save_document(doc):
    try:
        res = (
            supabase.table("documents")
            .insert({"name": doc["name"], "chunk_count": len(doc["texts"])})
            .execute()
        )
        doc_id = res.data[0]["id"]
    except Exception as e:
        return str(e)[:150]
    try:
        total = len(doc["texts"])
        for i in range(0, total, 20):
            rows = []
            for j in range(i, min(i + 20, total)):
                rows.append(
                    {
                        "document_id": doc_id,
                        "page": int(doc["pages"][j]),
                        "content": doc["texts"][j],
                        "embedding": encode_vec(doc["vectors"][j]),
                    }
                )
            supabase.table("chunks").insert(rows).execute()
    except Exception as e:
        delete_document(doc_id)
        return str(e)[:150]
    return None


def load_my_documents():
    try:
        docs = (
            supabase.table("documents")
            .select("id,name")
            .order("created_at")
            .execute()
            .data
        )
        if not docs:
            return [], None
        rows = []
        start = 0
        while True:
            batch = (
                supabase.table("chunks")
                .select("document_id,page,content,embedding")
                .order("id")
                .range(start, start + 999)
                .execute()
                .data
            )
            rows.extend(batch)
            if len(batch) < 1000:
                break
            start += 1000
    except Exception as e:
        return [], str(e)[:150]

    by_doc = {}
    for r in rows:
        by_doc.setdefault(r["document_id"], []).append(r)

    result = []
    for d in docs:
        items = by_doc.get(d["id"], [])
        if not items:
            continue
        vecs = np.array([decode_vec(r["embedding"]) for r in items], dtype="float32")
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        result.append(
            {
                "id": d["id"],
                "name": d["name"],
                "vectors": vecs,
                "texts": np.array([r["content"] for r in items]),
                "pages": np.array([r["page"] or 0 for r in items]),
            }
        )
    return result, None


def refresh_documents():
    st.session_state.my_docs, st.session_state.docs_error = load_my_documents()


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


def build_history(messages):
    lines = []
    for m in messages[-7:-1]:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content'][:500]}")
    return "\n".join(lines) if lines else "(none)"


b_vectors, b_texts, b_sources, b_pages = load_index()
builtin_names = sorted({str(s) for s in b_sources})

if "messages" not in st.session_state:
    st.session_state.messages = []
if "handled_files" not in st.session_state:
    st.session_state.handled_files = set()
if "my_docs" not in st.session_state:
    with st.spinner("Loading your documents..."):
        refresh_documents()
my_docs = st.session_state.my_docs

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🧠 DocuMind AI")
    st.caption(f"Signed in as {user.email}")
    if st.button("Log out", use_container_width=True):
        sign_out()
        st.rerun()

    view = "Chat"
    if is_admin:
        view = st.radio("View", ["Chat", "Admin"], horizontal=True)

    st.write(
        "An AI assistant that answers questions using only your documents, "
        "and shows where every answer came from."
    )

    st.subheader("📤 Upload your own PDF")
    uploaded = st.file_uploader("Drop a PDF here", type="pdf")
    if uploaded is None:
        st.session_state.handled_files = set()
    else:
        file_key = f"{uploaded.name}-{uploaded.size}"
        if file_key not in st.session_state.handled_files:
            st.session_state.handled_files.add(file_key)
            if any(d["name"] == uploaded.name for d in my_docs):
                st.info("You already saved this document.")
            else:
                bar = st.progress(0.0, text="Reading your document...")
                doc, err = process_upload(uploaded, bar)
                bar.empty()
                if err:
                    st.error(err)
                else:
                    save_err = save_document(doc)
                    if save_err:
                        st.error(f"Could not save the document: {save_err}")
                    else:
                        st.success(f"Saved: {doc['name']} ({len(doc['texts'])} passages)")
                        refresh_documents()
                        my_docs = st.session_state.my_docs

    scope = "Built-in documents"
    if my_docs:
        scope = st.radio(
            "Search in",
            ["My documents", "Built-in documents", "Both"],
            index=2,
        )

    st.subheader("📄 Documents")
    if st.session_state.get("docs_error"):
        st.caption("Could not load your documents: " + st.session_state["docs_error"])
    for d in my_docs:
        c1, c2 = st.columns([5, 1])
        c1.write(f"• {d['name']} (yours)")
        if c2.button("🗑️", key=f"del-{d['id']}"):
            del_err = delete_document(d["id"])
            if del_err:
                st.error(f"Could not delete: {del_err}")
            else:
                refresh_documents()
                st.rerun()
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
    if st.session_state.get("log_error"):
        st.caption("Could not save question: " + st.session_state["log_error"])
    st.caption("Built by Tirth · Python, Gemini, Streamlit")

# ---------- Admin view ----------
if view == "Admin":
    show_admin()
    st.stop()

# ---------- Header ----------
st.markdown(HERO, unsafe_allow_html=True)

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

    parts_v, parts_t, parts_s, parts_p = [], [], [], []
    if my_docs and scope in ("My documents", "Both"):
        for d in my_docs:
            parts_v.append(d["vectors"])
            parts_t.append(d["texts"])
            parts_s.append(np.array([d["name"]] * len(d["texts"])))
            parts_p.append(d["pages"])
    if (not my_docs) or scope in ("Built-in documents", "Both"):
        parts_v.append(b_vectors)
        parts_t.append(b_texts)
        parts_s.append(b_sources)
        parts_p.append(b_pages)
    vectors = np.concatenate(parts_v)
    texts = np.concatenate(parts_t)
    source_names = np.concatenate(parts_s)
    page_numbers = np.concatenate(parts_p)

    previous_questions = [
        m["content"] for m in st.session_state.messages[:-1] if m["role"] == "user"
    ]
    search_text = question
    if len(question.split()) < 6 and previous_questions:
        search_text = previous_questions[-1] + " " + question
    history = build_history(st.session_state.messages)

    answer = "Sorry, the AI service did not respond. Please try again in a moment."
    sources = []
    error_note = None

    with st.spinner("Searching the documents..."):
        q, error_note = embed_question(search_text)
        if q is not None:
            q = q / np.linalg.norm(q)
            scores = vectors @ q
            top = np.argsort(scores)[::-1][:4]

            context_parts = []
            found = []
            for i in top:
                page = int(page_numbers[i])
                label = str(source_names[i])
                if page > 0:
                    label = f"{label} · p.{page}"
                context_parts.append(f"[{label}]\n{texts[i]}")
                if label not in found:
                    found.append(label)
            context = "\n\n".join(context_parts)

            prompt = (
                "You answer questions about documents. Use only the context "
                "below. If the answer is not in the context, say you don't "
                "know. Use the conversation so far to understand follow-up "
                "questions.\n\n"
                f"Conversation so far:\n{history}\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            text, error_note = generate_with_retry(prompt)
            if text is not None:
                answer = text
                sources = found
                error_note = None

    log_question(question, bool(sources))

    bot_msg = {
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "error": error_note,
    }
    st.session_state.messages.append(bot_msg)
    show_message(bot_msg)