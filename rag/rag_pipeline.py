"""
rag_pipeline.py

Complete Retrieval-Augmented Generation (RAG) pipeline.
"""

from retriever.retriever import Retriever
from prompts.prompt_template import PromptBuilder
from llm.gemini_llm import GeminiLLM
from utils.logger import get_logger

logger = get_logger()


class RAGPipeline:
    """
    End-to-End RAG Pipeline.
    """

    def __init__(self, chroma_manager):

        logger.info("Initializing RAG Pipeline...")

        self.retriever = Retriever(chroma_manager)

        self.prompt = (
            PromptBuilder()
            .get_prompt()
        )

        self.llm = (
            GeminiLLM()
            .get_llm()
        )

        logger.info("RAG Pipeline initialized successfully.")

    def ask(self, question: str):
        """
        Execute the complete RAG workflow.

        Args:
            question (str): User question.

        Returns:
            dict
        """

        logger.info(f"User Question: {question}")

        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(question)

        # Step 2: Build context
        context = self.retriever.build_context(
            retrieved_docs
        )

        # Step 3: Build prompt
        prompt = self.prompt.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        # Step 4: Generate answer
        response = self.llm.invoke(prompt)

        # Step 5: Get document sources
        sources = self.retriever.get_sources(
            retrieved_docs
        )

        logger.info("Response generated successfully.")

        return {
            "question": question,
            "answer": response.content,
            "sources": sources,
            "retrieved_chunks": retrieved_docs,
        }