import os
import time
import numpy as np
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(timeout=60000),
)

PROGRESS = "embed_progress.npy"
BATCH = 10
PAUSE = 7


def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def embed_batch(chunks):
    for attempt in range(8):
        try:
            r = client.models.embed_content(
                model="gemini-embedding-001", contents=chunks
            )
            return [e.values for e in r.embeddings]
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                print("Google says slow down. Waiting 60 seconds...")
                time.sleep(60)
            else:
                print("Retrying...", msg[:80])
                time.sleep(10)
    raise RuntimeError("Google kept failing. Run this again in a few minutes.")


texts, sources, pages = [], [], []
for filename in sorted(os.listdir("docs")):
    if not filename.lower().endswith(".pdf"):
        continue
    reader = PdfReader(os.path.join("docs", filename))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        for chunk in chunk_text(text):
            texts.append(chunk)
            sources.append(filename)
            pages.append(page_number)
    print(f"Read {filename}")

print(f"{len(texts)} chunks to embed")

vectors = []
if os.path.exists(PROGRESS):
    vectors = np.load(PROGRESS).tolist()
    if len(vectors) > len(texts):
        vectors = []
    print(f"Resuming from {len(vectors)}")

for i in range(len(vectors), len(texts), BATCH):
    vectors.extend(embed_batch(texts[i:i + BATCH]))
    np.save(PROGRESS, np.array(vectors, dtype="float32"))
    print(f"Embedded {len(vectors)} of {len(texts)}")
    time.sleep(PAUSE)

np.savez_compressed(
    "index.npz",
    vectors=np.array(vectors, dtype="float16"),
    texts=np.array(texts),
    sources=np.array(sources),
    pages=np.array(pages),
)
os.remove(PROGRESS)
print("Saved index.npz")