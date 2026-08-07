"""
app.py

Streamlit application for the RAG Chatbot.
"""

import os
import shutil
import streamlit as st

from ingest import main as ingest_documents

from embeddings.embedding_model import EmbeddingModel
from vectordb.chroma_manager import ChromaManager
from rag.rag_pipeline import RAGPipeline

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


st.set_page_config(
    page_title="RAG Chatbot",
    layout="wide"
)


st.title("RAG Chatbot")
st.markdown(
    "Ask questions about **Ai + Deep Learning + Machine Learning + RAG**"
)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.header("Knowledge Base")

if st.sidebar.button("Build Knowledge Base"):

    with st.spinner("Building Knowledge Base..."):

        ingest_documents()

    st.sidebar.success("Knowledge Base Built Successfully!")

# ----------------------------
# Initialize RAG
# ----------------------------

embedding_model = (
    EmbeddingModel()
    .get_embedding_model()
)

chroma_manager = ChromaManager(
    embedding_model
)

rag = RAGPipeline(
    chroma_manager
)

# ----------------------------
# Chat History
# ----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ----------------------------
# User Input
# ----------------------------

question = st.chat_input(
    "Ask a question..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Thinking..."):

        result = rag.ask(question)

    answer = result["answer"]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)

        with st.expander(
            "📚 Retrieved Chunks"
        ):

            for i, chunk in enumerate(
                result["retrieved_chunks"],
                start=1,
            ):

                st.markdown(
                    f"### Chunk {i}"
                )

                st.write(
                    chunk.page_content
                )

        with st.expander(
            "📄 Sources"
        ):

            for source in result["sources"]:

                st.write(source)

# ----------------------------
# Clear Chat
# ----------------------------

if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []

    st.rerun()