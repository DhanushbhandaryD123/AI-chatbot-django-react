from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# vectorstore will be created in embedding.py and reused
from .embedding import get_vectorstore


def chat_with_pdf(question: str) -> str:
    """
    Takes a user question and returns answer from PDF using RAG
    """

    # Load vectorstore (FAISS / Chroma etc.)
    vectorstore = get_vectorstore()

    if vectorstore is None:
        return "No document uploaded yet."

    # LLM (API key is read automatically from environment variable)
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    # Retrieval-based QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    result = qa_chain.invoke({"query": question})

    return result["result"]
