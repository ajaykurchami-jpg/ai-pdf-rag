print("Starting script...")
import os
import shutil
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from extract import extract_text_from_pdf 

# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found. Check your .env file.")
    exit()

# 2. Setup Paths
PDF_PATH = "sample.pdf"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    # --- SAFETY CHECK: Delete old DB to prevent format conflicts ---
    if os.path.exists(DB_FAISS_PATH):
        print(f"Removing old vectorstore at {DB_FAISS_PATH}...")
        try:
            shutil.rmtree(DB_FAISS_PATH)
        except Exception as e:
            print(f"Warning: Could not delete old DB automatically. Please delete '{DB_FAISS_PATH}' manually. Error: {e}")

    print(f"--- 1. Extracting text from {PDF_PATH} ---")
    
    # Ensure extract.py exists and works
    try:
        raw_text = extract_text_from_pdf(PDF_PATH)
    except Exception as e:
        print(f"Error calling extract_text_from_pdf: {e}")
        return

    if not raw_text:
        print("Error: No text extracted. Check if PDF exists and is readable.")
        return

    # 3. Chunk the text
    print("--- 2. Splitting text into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(raw_text)
    print(f"Created {len(chunks)} chunks.")

    # 4. Create Embeddings (Must match main.py!)
    print("--- 3. Creating Google Embeddings ---")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    except Exception as e:
        print(f"Error initializing Embeddings: {e}")
        return

    # 5. Create and Save Vector Store
    print("--- 4. Saving to FAISS Vector DB ---")
    try:
        db = FAISS.from_texts(chunks, embeddings)
        db.save_local(DB_FAISS_PATH)
        print(f"Success! Vector DB saved to: {DB_FAISS_PATH}")
    except Exception as e:
        print(f"Error saving DB: {e}")

if __name__ == "__main__":
    create_vector_db()