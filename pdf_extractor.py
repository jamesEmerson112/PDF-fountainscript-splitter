"""
PDF text extraction with page boundary tracking.
"""
import fitz  # PyMuPDF
from models import PageText


def extract_text_with_pages(pdf_path: str) -> tuple[str, list[PageText]]:
    """
    Extract text from PDF while tracking page boundaries.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Tuple of (full_text, list of PageText objects)
    """
    doc = fitz.open(pdf_path)
    pages: list[PageText] = []
    full_text_parts = []
    char_offset = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Ensure text ends with newline for clean concatenation
        if text and not text.endswith('\n'):
            text += '\n'

        pages.append(PageText(
            page_number=page_num + 1,  # 1-indexed
            text=text,
            char_start=char_offset,
            char_end=char_offset + len(text)
        ))

        full_text_parts.append(text)
        char_offset += len(text)

    doc.close()
    return ''.join(full_text_parts), pages


def get_page_for_char_offset(pages: list[PageText], char_offset: int) -> int:
    """
    Find which page a character offset falls on.

    Args:
        pages: List of PageText with char ranges
        char_offset: Character position in full text

    Returns:
        Page number (1-indexed)
    """
    for page in pages:
        if page.char_start <= char_offset < page.char_end:
            return page.page_number
    # If past end, return last page
    return pages[-1].page_number if pages else 1


def get_total_pages(pdf_path: str) -> int:
    """Get total page count of a PDF."""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count
