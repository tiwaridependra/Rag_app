# app.py

import streamlit as st
from rag_utils import RAGPipeline
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from rag_utils import RAGPipeline

st.set_page_config(page_title="📚 Chat with your PDF", layout="wide")
st.title("📄💬 RAG Chat: Ask Questions About Your PDF")

# openai_key = st.text_input("🔑 Enter OpenAI API Key", type="password")


if "rag" not in st.session_state:
        st.session_state.rag = None
        st.session_state.chat_history = []
        st.session_state.pdf_name = None

pdf_file = st.file_uploader("📤 Upload a PDF", type="pdf")
 
if pdf_file:
        # Detect if a different PDF was uploaded
        if st.session_state.pdf_name != pdf_file.name:
            st.session_state.pdf_name = pdf_file.name
            st.session_state.chat_history = []
            st.session_state.rag = RAGPipeline()

            with st.spinner("🔍 Processing new PDF..."):
                text = st.session_state.rag.read_pdf(pdf_file)
                chunks = st.session_state.rag.chunk_text(text)
                st.session_state.rag.build_index(chunks)
                st.success("✅ PDF indexed! Start chatting below.")

if st.session_state.rag:
        # Chat input UI
        user_input = st.chat_input("Ask a question about the PDF...")

        if user_input:
            with st.spinner("💬 Thinking..."):
                answer = st.session_state.rag.chat(user_input)
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("AI", answer))

        # Display chat history
        for role, message in st.session_state.chat_history:
            with st.chat_message("user" if role == "You" else "assistant"):
                st.markdown(message)
