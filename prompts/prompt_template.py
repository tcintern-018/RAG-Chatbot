from langchain_core.prompts import ChatPromptTemplate

from utils.logger import get_logger

logger = get_logger()


class PromptBuilder:
    """
    Creates the prompt template for the RAG pipeline.
    """

    def __init__(self):

        logger.info("Initializing Prompt Template...")

        self.prompt = ChatPromptTemplate.from_template(
            """
You are an intelligent AI assistant.

Answer the user's question ONLY using the provided context.

If the answer cannot be found in the context, respond with:

"I couldn't find the answer in the provided documents."

Do not make up information.

---

Context:
{context}

---

Question:
{question}

Answer:
"""
        )

        logger.info("Prompt Template initialized successfully.")

    def get_prompt(self):
        """
        Returns the prompt template.

        Returns:
            ChatPromptTemplate
        """
        return self.prompt