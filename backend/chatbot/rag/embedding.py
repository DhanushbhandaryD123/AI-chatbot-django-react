from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os

VECTOR_DB_PATH = "vector_db"


def create_vectorstore(docs):
    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(docs, embeddings)

    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    db.save_local(VECTOR_DB_PATH)
