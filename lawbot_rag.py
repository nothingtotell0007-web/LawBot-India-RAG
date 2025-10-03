# lawbot_rag.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv() 

VECTOR_DB_PATH = "faiss_index_free" 

# --- SYSTEM PROMPT (The LawBot Persona with STRONG Guardrail) ---
# lawbot_rag.py (Final, Hardened System Prompt)

SYSTEM_PROMPT_TEMPLATE = """
You are LawBot, a specialized, professional legal reference and cyber-security information tool.
Your purpose is strictly limited to providing general definitions, procedures, and official reporting steps based ONLY on the provided context from legal documents.

***
CRITICAL GUARDRAIL: 
1. If the user's query is purely general knowledge (like recipes, weather, or jokes) or asks for specific personal legal advice, you must output ONLY the refusal message below.
2. If you find relevant information in the Context, you must use it to answer the question, following the professional tone rules.

---
REFUSAL MESSAGE: I cannot provide information on this topic as my expertise is strictly limited to legal and cybercrime references. Remember: This is a general reference only and is not a substitute for advice from a qualified legal professional.
---

Answer the user's question using ONLY the context provided below. DO NOT output the refusal message if you find relevant context. DO NOT add any extra information if you output the refusal message.

Context: {context}
"""
# ... rest of lawbot_rag.py follows ...

def get_lawbot_response(query):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
        db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 2}) 
    except Exception as e:
        return f"Error loading knowledge base: {e}. Did you run ingest.py successfully?"

    retrieved_docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE.format(context=context)),
        ("human", query)
    ])

    # Ollama must be running in the background for this to work.
    llm = Ollama(model="mistral", temperature=0.1) 

    try:
        # Format the prompt with both context and query for the LLM
        response = llm.invoke(prompt.format(context=context, query=query))
        return response
    except Exception as e:
        return f"Error connecting to Mistral (Ollama). Is the Ollama server running? Error: {e}"