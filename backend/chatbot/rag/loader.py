import os
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings # Updated: Free alternative
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. FIND THE .ENV FILE
BASE_DIR = Path(__file__).resolve().parent.parent.parent
env_path = BASE_DIR / '.env'

# 2. LOAD AND VALIDATE
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print(f"Found .env at: {env_path}")
else:
    print(f"Could NOT find .env at: {env_path}")

# 3. SETUP DOCUMENT PATHS
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "documents")

def get_vectorstore():
    """Function to read PDFs and create a searchable index using local embeddings."""
    text = ""
    
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        return None

    # Process PDFs
    pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"DEBUG: No PDFs found in {UPLOAD_DIR}")
        return None

    for file in pdf_files:
        print(f"DEBUG: Indexing {file}...")
        try:
            reader = PdfReader(os.path.join(UPLOAD_DIR, file))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not text.strip():
        print("DEBUG: No text extracted from PDFs.")
        return None

    # Create the Vector Store
    # 1,000 character chunks with 200 character overlap to preserve context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Updated: Using a free local model instead of OpenAI
    # 'all-MiniLM-L6-v2' is fast and accurate for general document search
    print("DEBUG: Generating local embeddings (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return FAISS.from_texts(chunks, embeddings)

# Initialize the global variable when the server starts
# This will now succeed because it no longer calls the OpenAI API for embeddings
vectorstore = get_vectorstore()