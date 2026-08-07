"""
chroma_manager.py

Manages ChromaDB operations for the RAG Chatbot.

Responsibilities:
- Create a vector database
- Persist vectors to disk
- Load an existing database
- Create a retriever
"""

from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import (
    CHROMA_DB_DIR,
    TOP_K,
)

from utils.logger import get_logger

logger = get_logger()


class ChromaManager:
    """
    Handles all ChromaDB operations.
    """

    def __init__(self, embedding_model):

        self.embedding_model = embedding_model

        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embedding_model,
        )

    def add_documents(
        self,
        documents: List[Document],
    ) -> None:
        """
        Add document chunks into ChromaDB.
        """

        if not documents:
            logger.warning("No chunks found to index.")
            return

        logger.info(
            f"Adding {len(documents)} chunks to ChromaDB..."
        )

        self.vector_store.add_documents(documents)

        logger.info("Documents indexed successfully.")

    def get_vector_store(self):
        """
        Return the Chroma vector database.
        """
        return self.vector_store

    def get_retriever(
        self,
        top_k: int = TOP_K,
    ):
        """
        Return a similarity retriever.
        """

        logger.info(
            f"Retriever initialized with Top-K = {top_k}"
        )

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": top_k
            }
        )