from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, chunks):
    faiss.write_index(index, "faiss.index")
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def load_index():
    index = faiss.read_index("faiss.index")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
