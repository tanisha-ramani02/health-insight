"""PDF extraction engine using PyMuPDF with page metadata preservation."""

import re
from pathlib import Path
from typing import List, Optional
import fitz  # PyMuPDF
from langchain_core.documents import Document
from config.settings import settings
from config.logger import logger


class PDFDocumentLoader:
    """Extracts and sanitizes text from PDF pages with 1-indexed metadata."""

    def __init__(self, pdf_path: Optional[Path] = None):
        self.pdf_path = pdf_path or settings.PDF_PATH

    def clean_text(self, text: str) -> str:
        """Sanitize text by collapsing excessive spaces while preserving paragraphs."""
        if not text:
            return ""
        # Remove null characters and soft hyphens
        text = text.replace("\x00", "").replace("\xad", "")
        # Normalize whitespace per line
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
        # Remove consecutive blank lines
        cleaned = "\n".join(line for i, line in enumerate(lines) if line or (i > 0 and lines[i - 1]))
        return cleaned.strip()

    def load_pages(self) -> List[Document]:
        """Parse all PDF pages into LangChain Documents with 1-indexed page numbers."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

        logger.info(f"Opening PDF document: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        documents = []

        # Extract Table of Contents for section mapping
        toc = doc.get_toc()
        page_to_section = {item[2]: item[1] for item in toc if len(item) >= 3}
        current_section = "General Information"

        for page_idx in range(len(doc)):
            page_num = page_idx + 1  # 1-indexed page number
            page = doc[page_idx]
            raw_text = page.get_text("text")
            cleaned_text = self.clean_text(raw_text)

            if page_num in page_to_section:
                current_section = page_to_section[page_num]

            doc_obj = Document(
                page_content=cleaned_text if cleaned_text else "[Blank Page]",
                metadata={
                    "source": str(self.pdf_path.name),
                    "source_page": page_num,
                    "total_pages": len(doc),
                    "section": current_section,
                    "character_count": len(cleaned_text)
                }
            )
            documents.append(doc_obj)

        doc.close()
        logger.info(f"Successfully extracted {len(documents)} pages from PDF.")
        return documents


pdf_loader = PDFDocumentLoader()
