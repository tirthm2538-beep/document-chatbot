import os
from pypdf import PdfReader
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection("docs")

def chunk_text(text, size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

count = 0
for filename in os.listdir("docs"):
    if not filename.endswith(".pdf"):
        continue
    reader = PdfReader(os.path.join("docs", filename))
    text = " ".join(page.extract_text() or "" for page in reader.pages)
    for i, chunk in enumerate(chunk_text(text)):
        result = client.models.embed_content(
            model="gemini-embedding-001", contents=chunk
        )
        collection.add(
            ids=[f"{filename}-{i}"],
            embeddings=[result.embeddings[0].values],
            documents=[chunk],
            metadatas=[{"source": filename}],
        )
        count += 1
        print(f"Stored chunk {count}")

print("Done!")