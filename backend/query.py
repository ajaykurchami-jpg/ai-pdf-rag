
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: GOOGLE_API_KEY not found. Check your .env file.")
    exit()

# 2. Setup Memory (Vector DB)
DB_FAISS_PATH = "vectorstore/db_faiss"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("--- Loading Vector DB ---")
try:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Error loading DB: {e}")
    exit()

# 3. Setup Brain (UPDATED FOR DEC 2025)
# Using 'gemini-2.5-flash' which replaced the 1.5 series
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

# 4. Define Helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. Create Chain (The Modern "Pipe" Way)
prompt = ChatPromptTemplate.from_template("""
Answer the question based ONLY on the following context:
{context}

Question: {question}
""")

retriever = db.as_retriever(search_kwargs={'k': 3})

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Run Loop
if __name__ == "__main__":
    print("\n--- AI PDF Assistant Ready! (Type 'exit' to quit) ---")
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() == "exit":
            break
        
        try:
            response = rag_chain.invoke(user_input)
            print("\n>> AI Answer:")
            print(response)
        except Exception as e:
            print(f"Error: {e}")