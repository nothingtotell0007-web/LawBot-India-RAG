# lawbot_rag.py (Final Code for Cloud Deployment)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# OLD: from langchain_community.llms import Ollama 
from langchain_community.llms import HuggingFaceHub # <-- NEW LLM IMPORT
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st # Need Streamlit to access secrets

load_dotenv() 

VECTOR_DB_PATH = "faiss_index_free" 

# --- SYSTEM PROMPT (The LawBot Persona with STRONG Guardrail) ---
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

def get_lawbot_response(query):
    try:
        # Load the index using the free embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
        db = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 2}) 
    except Exception as e:
        # This error is now the fault of the deployment, not a local server
        return f"Error loading knowledge base: {e}. Check deployment logs."

    # 1. Access the API Token securely via Streamlit Secrets
    try:
        hf_api_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except KeyError:
        return "ERROR: HuggingFace API Token not found in Streamlit Secrets."


    retrieved_docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE.format(context=context)),
        ("human", query)
    ])

    # 2. Initialize the LLM using the Hosted API Endpoint
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2", # Using a highly-rated free hosted model
        huggingfacehub_api_token=hf_api_token, # Passing the token from Streamlit Secrets
        model_kwargs={"temperature": 0.1, "max_length": 1000},
    ) 

    # 3. Generate response
    try:
        response = llm.invoke(prompt.format(context=context, query=query))
        return response
    except Exception as e:
        return f"Error during generation: The model failed to respond. Check API status. Error: {e}"