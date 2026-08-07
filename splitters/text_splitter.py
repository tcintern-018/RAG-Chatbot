"""
text_splitter.py

Splits loaded documents into smaller chunks for embedding.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.logger import get_logger

logger = get_logger()


class TextSplitter:
    """
    Splits documents into chunks using RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Split documents into smaller chunks.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List[Document]: Chunked documents.
        """

        if not documents:
            logger.warning("No documents found for chunking.")
            return []

        logger.info(f"Splitting {len(documents)} document(s)...")

        chunks = self.splitter.split_documents(documents)

        logger.info(f"Created {len(chunks)} chunks.")

        return chunks