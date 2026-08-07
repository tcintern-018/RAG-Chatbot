"""
retriever.py

Handles semantic retrieval from ChromaDB.
"""

from typing import List

from langchain_core.documents import Document

from config import TOP_K
from utils.logger import get_logger

logger = get_logger()


class Retriever:
    """
    Retrieves relevant document chunks from ChromaDB.
    """

    def __init__(self, chroma_manager):
        """
        Initialize the Retriever.

        Args:
            chroma_manager: Instance of ChromaManager.
        """
        self.chroma_manager = chroma_manager
        self.retriever = self.chroma_manager.get_retriever(
            top_k=TOP_K
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve the most relevant chunks.

        Args:
            query (str): User question.

        Returns:
            List[Document]
        """
        try:
            logger.info(f"Searching documents for: {query}")

            documents = self.retriever.invoke(query)

            logger.info(
                f"{len(documents)} relevant chunk(s) retrieved."
            )

            return documents

        except Exception as error:
            logger.exception(
                f"Error during retrieval: {error}"
            )
            return []

    @staticmethod
    def build_context(documents: List[Document]) -> str:
        """
        Convert retrieved chunks into a single context string.

        Args:
            documents (List[Document])

        Returns:
            str
        """
        if not documents:
            return ""

        context = "\n\n".join(
            doc.page_content.strip()
            for doc in documents
        )

        return context

    @staticmethod
    def get_sources(documents: List[Document]) -> List[str]:
        """
        Extract unique source filenames.

        Args:
            documents (List[Document])

        Returns:
            List[str]
        """
        sources = []

        for doc in documents:

            source = doc.metadata.get("source", "Unknown")

            if source not in sources:
                sources.append(source)

        return sources