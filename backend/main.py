from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
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
    print("Warning: GOOGLE_API_KEY not found. Ensure it is set in Railway Variables.")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

# --- FIX: Changed to 'gemini-1.5-flash-001' (Stable Version) ---
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", google_api_key=api_key)

# 3. Models
class QueryRequest(BaseModel):
    question: str

# 4. Endpoints

@app.get("/")
def home():
    return {"message": "AI PDF API is Running with Summarization!"}

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
    
    return {"filename": file.filename, "status": "Indexed"}

# --- SUMMARIZATION ENDPOINT ---
@app.post("/summarize")
async def summarize_document():
    try:
        # Load DB
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={'k': 10}) 
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert summarizer. 
        Read the following context and provide a concise summary.
        
        Summary Guidelines:
        1. Start with "This document is about..."
        2. Provide 3-5 key bullet points.
        3. Keep it professional.

        Context:
        {context}
        """)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Send a valid query string
        summary = chain.invoke("Give me a comprehensive overview of this document") 
        return {"summary": summary}

    except Exception as e:
        return {"summary": f"SYSTEM ERROR (Summarize): {str(e)}"}

@app.post("/query")
async def ask_question(request: QueryRequest):
    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={'k': 5})

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant.
        Answer the question based ONLY on the following context.
        
        STRICT RULES:
        1. If the answer is found, cite the page number like [Page X].
        2. If the answer is NOT in the context, output EXACTLY this string: 
           "polite_fallback_trigger"
        
        Context:
        {context}

        Question: {question}
        """)
        
        def format_docs(docs):
            return "\n\n".join(f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}" for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        raw_answer = rag_chain.invoke(request.question)

        # --- FORCE FIX ---
        clean_answer = raw_answer.strip().lower()
        
        negative_triggers = [
            "polite_fallback_trigger",
            "i don't know",
            "not mentioned",
            "no information",
            "not present in the context",
            "cannot answer"
        ]

        if any(trigger in clean_answer for trigger in negative_triggers):
            final_answer = "I checked the document for you, but it doesn't seem to mention that. Is there a different section you'd like me to summarize?"
        else:
            final_answer = raw_answer

        return {"answer": final_answer}

    except Exception as e:
        return {"answer": f"SYSTEM ERROR (Query): {str(e)}"}