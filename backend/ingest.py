print("Starting script...")  # <--- Add this at the very top
# ... rest of your code
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from extract import extract_text_from_pdf  # Importing your function from the previous step

# 1. Setup - Define paths
PDF_PATH = "sample.pdf"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    print(f"--- 1. Extracting text from {PDF_PATH} ---")
    raw_text = extract_text_from_pdf(PDF_PATH)
    if not raw_text:
        print("Error: No text extracted. Check your PDF.")
        return

    # 2. Chunk the text
    # We split by 500 characters, with 50 overlap to keep context between chunks
    print("--- 2. Splitting text into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Embeddings
    # We use 'sentence-transformers/all-MiniLM-L6-v2' (Small, Fast, Free)
    print("--- 3. Creating Embeddings (This may take a moment) ---")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Create and Save Vector Store
    print("--- 4. Saving to FAISS Vector DB ---")
    db = FAISS.from_texts(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    print(f"Success! Vector DB saved to: {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()