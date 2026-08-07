"""
embedding_model.py

Initializes and returns the Gemini Embedding model
used for creating vector embeddings.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
)

from utils.logger import get_logger

logger = get_logger()


class EmbeddingModel:
    """
    Creates and manages the Gemini Embedding model.
    """

    def __init__(self):

        logger.info("Initializing Gemini Embedding Model...")

        try:

            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=GOOGLE_API_KEY,
            )

            logger.info("Gemini Embedding Model initialized successfully.")

        except Exception as error:

            logger.exception(
                f"Failed to initialize embedding model: {error}"
            )

            raise

    def get_embedding_model(self):
        """
        Returns the initialized embedding model.

        Returns:
            GoogleGenerativeAIEmbeddings
        """
        return self.embedding_model