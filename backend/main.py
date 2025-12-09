from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Setup
load_dotenv()
app = FastAPI(title="AI PDF Assistant API")

# Define Paths
DB_FAISS_PATH = "vectorstore/db_faiss"
UPLOAD_FOLDER = "uploaded_files"
SQLITE_DB = "history.db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# 2. Load Brain
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Warning: GOOGLE_API_KEY not found.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
# Switch to Flash Lite to bypass the 2.5 Daily Limit
llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite", google_api_key=api_key)

# --- DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    # Table for Uploaded Documents
    c.execute('''CREATE TABLE IF NOT EXISTS documents 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, upload_date TEXT)''')
    # Table for Chat History
    c.execute('''CREATE TABLE IF NOT EXISTS chats 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT, timestamp TEXT)''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

# 3. Models
class QueryRequest(BaseModel):
    question: str

# 4. Endpoints

@app.get("/")
def home():
    return {"message": "AI PDF API is Running with Smart Prompts!"}

# --- HISTORY ENDPOINTS ---
@app.get("/documents")
def get_documents():
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute("SELECT filename, upload_date FROM documents ORDER BY id DESC")
    docs = [{"filename": row[0], "date": row[1]} for row in c.fetchall()]
    conn.close()
    return {"documents": docs}

@app.get("/history")
def get_chat_history():
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chats ORDER BY id ASC")
    history = [{"type": row[0], "text": row[1]} for row in c.fetchall()]
    conn.close()
    return {"history": history}

@app.delete("/clear")
def clear_history():
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute("DELETE FROM chats")
    c.execute("DELETE FROM documents") 
    conn.commit()
    conn.close()
    return {"message": "History Cleared"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Save file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Ingest
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Save Vector DB
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    # --- SAVE TO DB ---
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    c.execute("INSERT INTO documents (filename, upload_date) VALUES (?, ?)", 
              (file.filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()
    
    return {"filename": file.filename, "status": "Indexed"}

@app.post("/summarize")
async def summarize_document():
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={'k': 10}) 
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert summarizer. 
        Read the following context and provide a concise summary.
        Context: {context}
        """)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = ({"context": retriever | format_docs} | prompt | llm | StrOutputParser())
        summary = chain.invoke("Give me a comprehensive overview") 
        
        # --- SAVE AI SUMMARY TO HISTORY ---
        conn = sqlite3.connect(SQLITE_DB)
        c = conn.cursor()
        c.execute("INSERT INTO chats (role, content, timestamp) VALUES (?, ?, ?)", 
                  ("ai", summary, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        return {"summary": summary}

    except Exception as e:
        return {"summary": f"SYSTEM ERROR (Summarize): {str(e)}"}

@app.post("/query")
async def ask_question(request: QueryRequest):
    try:
        # --- SAVE USER QUESTION TO DB ---
        conn = sqlite3.connect(SQLITE_DB)
        c = conn.cursor()
        c.execute("INSERT INTO chats (role, content, timestamp) VALUES (?, ?, ?)", 
                  ("user", request.question, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={'k': 5})

        # --- UPDATED PROMPT: SMART HANDLING FOR SUMMARIES/TRANSLATIONS ---
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant.
        Answer the question based on the following context.
        
        RULES:
        1. If the answer is found, cite the page number like [Page X].
        2. If the user asks for a SUMMARY, TRANSLATION, or OVERVIEW, ignore the "not found" rule and generate the best response possible from the context.
        3. If the user asks for a specific fact (e.g., "What is the email?") and it is NOT in the context, output EXACTLY: "polite_fallback_trigger"
        4. Answer in the same language as the question (e.g., Hindi for Hindi).

        Context:
        {context}

        Question: {question}
        """)
        
        def format_docs(docs):
            return "\n\n".join(f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}" for doc in docs)

        rag_chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
        
        raw_answer = rag_chain.invoke(request.question)
        
        clean_answer = raw_answer.strip().lower()
        negative_triggers = ["polite_fallback_trigger", "i don't know", "not mentioned", "no information", "cannot answer"]

        if any(trigger in clean_answer for trigger in negative_triggers):
            final_answer = "I checked the document for you, but it doesn't seem to mention that. Is there a different section you'd like me to summarize?"
        else:
            final_answer = raw_answer

        # --- SAVE AI ANSWER TO DB ---
        c.execute("INSERT INTO chats (role, content, timestamp) VALUES (?, ?, ?)", 
                  ("ai", final_answer, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        return {"answer": final_answer}

    except Exception as e:
        return {"answer": f"SYSTEM ERROR (Query): {str(e)}"}