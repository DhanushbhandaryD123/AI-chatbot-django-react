from chatbot.rag import loader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the Prompt Template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

def chat_with_pdf(question):
    # Check if the vectorstore is initialized in the loader
    if loader.vectorstore is None:
        return "No documents indexed. Please upload a PDF and wait for indexing."

    # Initialize the LLM (free Groq-hosted Llama model)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    # Create the retriever from the existing FAISS vectorstore
    retriever = loader.vectorstore.as_retriever(search_kwargs={"k": 3})

    # Retrieve relevant chunks and combine them into the context
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build and invoke the LCEL chain (prompt -> llm -> string output)
    chain = QA_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "input": question})