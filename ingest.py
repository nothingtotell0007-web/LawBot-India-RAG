# ingest.py

import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
DATA_PATH = "knowledge_base"
# Renamed to distinguish from the old OpenAI-based index
VECTOR_DB_PATH = "faiss_index_free" 
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200

def create_vector_db():
    # 1. Load documents from the 'knowledge_base' directory
    print("Loading documents...")
    
    # We use glob="**/*" to load all files, and silent_errors=True to skip unsupported files
    loader = DirectoryLoader(DATA_PATH, glob="**/*", silent_errors=True)
    documents = loader.load()
    
    if not documents:
        print(f"No documents loaded from {DATA_PATH}. Please check the folder content.")
        return

    print(f"Loaded {len(documents)} documents.")

    # 2. Split documents into small, context-rich chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Standard separators for text and code
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""] 
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    # 3. Create embeddings (numerical representations) using a FREE model
    print("Creating free HuggingFace embeddings and FAISS index...")
    
    # Using 'all-MiniLM-L6-v2', a fast, small, and highly-rated free embedding model
    # It will download the model the first time it runs, then use it locally.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
    
    # 4. Create the FAISS vector store and save it locally
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTOR_DB_PATH)
    print(f"\nFAISS index saved successfully to {VECTOR_DB_PATH}")

if __name__ == "__main__":
    # Check that the necessary data folder exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Directory '{DATA_PATH}' not found. Please create it and add your legal documents.")
    else:
        create_vector_db()