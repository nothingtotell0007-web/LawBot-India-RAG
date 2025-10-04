# app.py

import streamlit as st
from lawbot_rag import get_lawbot_response

# --- Configuration and Styling ---
st.set_page_config(page_title="LawBot: Legal Reference & Cybercrime Info", layout="centered")

st.title("⚖️ LawBot: Legal Reference and Cybercrime Information")
st.markdown("""
    A specialized, high-accuracy RAG tool built for informational and educational purposes.
""")
st.markdown("---")

# 🚨 MANDATORY DISCLAIMER
st.warning("""
    **MANDATORY DISCLAIMER:** LawBot provides **general legal information and educational references** only. 
    It is **NOT legal advice** and is not a substitute for a qualified legal professional or law enforcement. 
    Do not share confidential information. By using LawBot, you accept these terms.
""")
st.markdown("---")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to LawBot. I am ready to provide factual legal references and cybercrime guidance based on my knowledge base. Please state your query."}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask LawBot a question about legal definitions or cybercrime..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("LawBot is consulting its reference library and drafting a professional response..."):
        response = get_lawbot_response(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)