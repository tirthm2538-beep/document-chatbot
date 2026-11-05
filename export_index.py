import chromadb
import numpy as np

db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_collection("docs")
data = collection.get(include=["embeddings", "documents", "metadatas"])

vectors = np.array(data["embeddings"], dtype="float16")
texts = np.array(data["documents"])
sources = np.array([m["source"] for m in data["metadatas"]])

np.savez_compressed("index.npz", vectors=vectors, texts=texts, sources=sources)
print(f"Saved {len(texts)} chunks")