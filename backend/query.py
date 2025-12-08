import os
from dotenv import load_dotenv
# --- FIX 1: Use Google Embeddings to match main.py ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
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

print("--- Loading Vector DB ---")
# --- FIX 1: Using Google Embeddings ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

try:
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Error loading DB: {e}")
    print("TIP: Did you run ingest.py? Make sure your ingestion also uses GoogleEmbeddings!")
    exit()

# 3. Setup Brain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", google_api_key=api_key)

# 4. Define Helper
def format_docs(docs):
    return "\n\n".join(f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}" for doc in docs)

# 5. Create Chain (Matching main.py Logic)
template = """
You are a helpful AI assistant.
Answer the question based ONLY on the following context.

STRICT RULES:
1. If the answer is found, cite the page number like [Page X].
2. If the answer is NOT in the context, output EXACTLY this string: 
   "polite_fallback_trigger"

Context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

retriever = db.as_retriever(search_kwargs={'k': 5})

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
            raw_answer = rag_chain.invoke(user_input)
            
            # --- FIX 2: Python Force Logic ---
            clean_answer = raw_answer.strip().lower()
            negative_triggers = [
                "polite_fallback_trigger",
                "i don't know",
                "not mentioned",
                "no information",
                "cannot answer"
            ]

            if any(trigger in clean_answer for trigger in negative_triggers):
                final_answer = "I checked the document for you, but it doesn't seem to mention that. Is there a different section you'd like me to summarize?"
            else:
                final_answer = raw_answer

            print("\n>> AI Answer:")
            print(final_answer)

        except Exception as e:
            print(f"Error: {e}")