
"""Tool 1: Read text from selectable-text PDFs; render scanned pages for vision."""

from langchain_core.tools import tool
import base64
import pymupdf as fitz

MAX_PAGES = 5
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024


# ==========================================
# TOOL 1: PDF READER
# Function: read_invoice_pdf()
# Called by: backend.py -> pdf_reader_node()
# Purpose: Read PDF text and scanned-page images
# LangChain Tool: @tool decorator
# ==========================================

@tool
def read_invoice_pdf(pdf_bytes: bytes) -> dict:
    """Return PDF text and, when needed, page images for the AI extractor."""

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("This file is not a valid PDF.")

    if len(pdf_bytes) > MAX_DOCUMENT_BYTES:
        raise ValueError("PDF exceeds the 10 MB limit.")

    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:

        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")

        if not 1 <= len(document) <= MAX_PAGES:
            raise ValueError(f"PDF must have 1 to {MAX_PAGES} pages.")

        pages_text = [page.get_text("text") for page in document]
        text = "\n\n".join(pages_text).strip()

        # Send page images only if text extraction was insufficient. Text-based invoices
        # generally avoid the additional cost of image/vision processing.

        images = []

        if len(text) < 80:
            for page in document:
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(1.5, 1.5),
                    alpha=False
                )

                images.append(
                    base64.b64encode(
                        pixmap.tobytes("jpeg")
                    ).decode("ascii")
                )

    return {
        "pdf_text": text[:30000],
        "page_images": images,
        "page_count": len(pages_text)
    }
