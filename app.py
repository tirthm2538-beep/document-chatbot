import os
import time
import numpy as np
import streamlit as st
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

MODELS = [
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
]

@st.cache_resource
def load_index():
    data = np.load("index.npz")
    vectors = data["vectors"].astype("float32")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, data["texts"], data["sources"]

def embed_question(question):
    for attempt in range(4):
        try:
            r = client.models.embed_content(
                model="gemini-embedding-001", contents=question
            )
            return np.array(r.embeddings[0].values, dtype="float32")
        except Exception:
            time.sleep(3)
    return None

def generate_with_retry(prompt):
    for model in MODELS:
        for attempt in range(2):
            try:
                reply = client.models.generate_content(
                    model=model, contents=prompt
                )
                return reply.text
            except Exception:
                time.sleep(2)
    return None

st.title("Document Chatbot")
st.caption("Ask questions about my documents")

vectors, texts, source_names = load_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

question = st.chat_input("Ask a question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    busy = "The AI service is very busy right now. Please try again in a moment."
    answer = busy

    with st.spinner("Thinking..."):
        q = embed_question(question)
        if q is not None:
            q = q / np.linalg.norm(q)
            scores = vectors @ q
            top = np.argsort(scores)[::-1][:4]
            context = "\n\n".join(str(texts[i]) for i in top)
            sources = sorted({str(source_names[i]) for i in top})

            prompt = (
                "Answer the question using only the context below. "
                "If the answer is not in the context, say you don't know.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            text = generate_with_retry(prompt)
            if text is not None:
                answer = text + "\n\nSources: " + ", ".join(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)