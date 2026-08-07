"""
gemini_llm.py

Initializes the Gemini Chat Model for the RAG Chatbot.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    GOOGLE_API_KEY,
    CHAT_MODEL,
)

from utils.logger import get_logger

logger = get_logger()


class GeminiLLM:
    """
    Creates and manages the Gemini Chat Model.
    """

    def __init__(self):

        logger.info("Initializing Gemini Chat Model...")

        try:

            self.llm = ChatGoogleGenerativeAI(
                model=CHAT_MODEL,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.3,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

            logger.info("Gemini Chat Model initialized successfully.")

        except Exception as error:

            logger.exception(
                f"Failed to initialize Gemini LLM: {error}"
            )

            raise

    def get_llm(self):
        """
        Returns the initialized Gemini model.

        Returns:
            ChatGoogleGenerativeAI
        """
        return self.llm