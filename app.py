# app.py

import streamlit as st
from lawbot_rag import get_lawbot_response

# app.py

# ... other imports ...

# CHANGE THIS LINE:
st.set_page_config(page_title="LawBot: Legal Reference & Cybercrime Info", layout="wide") # <-- CHANGED TO "wide"

# app.py (Add this CSS block near the top after imports)

custom_css = """
<style>
/* Remove the top right 'Deploy' button and Streamlit footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Customizing the chat containers (bubbles) for a professional look */
.stChatMessage {
    padding: 0.5rem 1rem;
    border-radius: 10px;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* User's message bubble style */
.stChatMessage [data-testid="stChatMessageContent"] {
    background-color: #3B82F6; /* Bright blue for user */
    color: white;
    border-radius: 15px 15px 5px 15px; 
    padding: 10px;
}

/* Assistant (LawBot) message bubble style */
.stChatMessage:not([data-testid="stChatMessageContent"]) {
    background-color: #1E2835; /* Same as app background */
}

/* Make LawBot's icon a law-related emoji for branding */
[data-testid="stChatMessageContent"] .st-emotion-cache-1r3j1j { 
    content: "⚖️"; /* Attempts to set custom avatar content for assistant */
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

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