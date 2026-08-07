"""
document_loader.py

Loads PDF, DOCX, and TXT documents for the RAG pipeline.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from utils.logger import get_logger

logger = get_logger()


class DocumentLoader:
    """
    Loads supported documents from a directory.
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
    }

    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)

    def load_documents(self) -> List[Document]:
        """
        Load all supported documents.

        Returns:
            List[Document]
        """

        documents = []

        if not self.data_directory.exists():
            logger.error(f"Directory not found: {self.data_directory}")
            raise FileNotFoundError(
                f"{self.data_directory} does not exist."
            )

        files = list(self.data_directory.iterdir())

        if not files:
            logger.warning("No files found in the uploads directory.")
            return []

        logger.info(f"Found {len(files)} file(s).")

        for file in files:

            extension = file.suffix.lower()

            if extension not in self.SUPPORTED_EXTENSIONS:
                logger.warning(
                    f"Skipping unsupported file: {file.name}"
                )
                continue

            try:

                logger.info(f"Loading {file.name}")

                loader_class = self.SUPPORTED_EXTENSIONS[extension]

                if extension == ".txt":
                    loader = loader_class(
                        str(file),
                        encoding="utf-8",
                    )
                else:
                    loader = loader_class(str(file))

                docs = loader.load()

                for doc in docs:

                    doc.metadata["source"] = file.name
                    doc.metadata["file_type"] = extension

                documents.extend(docs)

                logger.info(
                    f"Loaded {len(docs)} page(s) from {file.name}"
                )

            except Exception as error:

                logger.exception(
                    f"Failed to load {file.name}: {error}"
                )

        logger.info(
            f"Successfully loaded {len(documents)} document(s)."
        )

        return documents