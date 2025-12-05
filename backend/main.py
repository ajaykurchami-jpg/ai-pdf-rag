from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import shutil
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
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
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount the "uploaded_files" folder to the "/static" route
app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# 2. Load the "Brain" (Gemini + Embeddings)
api_key = os.getenv("GOOGLE_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# 3. API Input Models
class QueryRequest(BaseModel):
    question: str

# 4. API Endpoints

@app.get("/")
def home():
    return {"message": "AI PDF API is Running!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    1. Saves the uploaded PDF.
    2. Processes it (Chunks & Embeddings).
    3. Updates the Vector DB.
    """
    # Save file locally
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process PDF (Ingestion Logic)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Save to FAISS
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    return {"filename": file.filename, "status": "Successfully Processed & Indexed"}

@app.post("/query")
async def ask_question(request: QueryRequest):
    """
    Handles user questions with Multilingual Support and Citations.
    """
    try:
        # Load the DB
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={'k': 3})

        # WEEK 5 UPDATED PROMPT: Multilingual & Citations
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful AI assistant.
        Answer the question based ONLY on the following context.
        
        Rules:
        1. ALWAYS cite the page number (e.g., [Page 2]) where you found the information.
        2. If the user asks in a specific language (e.g., Hindi, Spanish), ANSWER IN THAT SAME LANGUAGE.
        3. If the answer is not in the context, say "I don't know".

        <context>
        {context}
        </context>

        Question: {question}
        """)
        
        # Helper to include Page Numbers
        def format_docs(docs):
            return "\n\n".join(f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}" for doc in docs)

        # Build the Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Run the Chain
        answer = rag_chain.invoke(request.question)
        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))