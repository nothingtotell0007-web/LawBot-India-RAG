# ingest.py

import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- CONFIGURATION ---
DATA_PATH = "knowledge_base"
VECTOR_DB_PATH = "faiss_index_free" 
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200

def create_vector_db():
    print("Loading documents...")
    
    loader = DirectoryLoader(DATA_PATH, glob="**/*", silent_errors=True)
    documents = loader.load()
    
    if not documents:
        print(f"No documents loaded from {DATA_PATH}. Please check the folder content.")
        return

    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""] 
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    print("Creating free HuggingFace embeddings and FAISS index...")
    
    # Using 'all-MiniLM-L6-v2' (fast, free, local embedding model)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(VECTOR_DB_PATH)
    print(f"\nFAISS index saved successfully to {VECTOR_DB_PATH}")

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Error: Directory '{DATA_PATH}' not found. Please create it and add your legal documents.")
    else:
        create_vector_db()