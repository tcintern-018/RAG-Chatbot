"""
ingest.py

Loads documents, splits them into chunks,
creates embeddings, and stores them in ChromaDB.
"""

from loaders.document_loader import DocumentLoader
from splitters.text_splitter import TextSplitter
from embeddings.embedding_model import EmbeddingModel
from vectordb.chroma_manager import ChromaManager
from utils.logger import get_logger

logger = get_logger()


def main():
    """
    Build the vector database.
    """

    try:

        logger.info("=" * 60)
        logger.info("Starting RAG Ingestion Pipeline")
        logger.info("=" * 60)

        # ----------------------------------
        # Step 1: Load Documents
        # ----------------------------------
        loader = DocumentLoader("data")

        documents = loader.load_documents()

        if not documents:
            logger.warning("No documents found.")
            return

        logger.info(f"Loaded {len(documents)} documents.")

        # ----------------------------------
        # Step 2: Split Documents
        # ----------------------------------
        splitter = TextSplitter()

        chunks = splitter.split_documents(documents)

        logger.info(f"Generated {len(chunks)} chunks.")

        # ----------------------------------
        # Step 3: Initialize Embedding Model
        # ----------------------------------
        embedding_model = (
            EmbeddingModel()
            .get_embedding_model()
        )

        # ----------------------------------
        # Step 4: Store in ChromaDB
        # ----------------------------------
        chroma_manager = ChromaManager(
            embedding_model
        )

        chroma_manager.add_documents(chunks)

        logger.info("=" * 60)
        logger.info("Knowledge Base Created Successfully!")
        logger.info("=" * 60)

    except Exception as error:

        logger.exception(
            f"Ingestion Pipeline Failed: {error}"
        )


if __name__ == "__main__":
    main()