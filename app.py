import os
import time
import chromadb
import streamlit as st
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("docs")

MODELS = [
    "gemini-flash-latest",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
]

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

    with st.spinner("Thinking..."):
        q = client.models.embed_content(
            model="gemini-embedding-001", contents=question
        )
        results = collection.query(
            query_embeddings=[q.embeddings[0].values], n_results=4
        )
        context = "\n\n".join(results["documents"][0])
        sources = sorted({m["source"] for m in results["metadatas"][0]})

        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        text = generate_with_retry(prompt)

    if text is None:
        answer = "The AI service is very busy right now. Please try again in a moment."
    else:
        answer = text + "\n\nSources: " + ", ".join(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)