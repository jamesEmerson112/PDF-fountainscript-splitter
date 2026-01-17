"""
Data models for PDF screenplay splitter.
"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PageText:
    """Text content from a single PDF page."""
    page_number: int          # 1-indexed
    text: str
    char_start: int           # Character offset in full text
    char_end: int


@dataclass
class SceneLocation:
    """Location of a scene heading in the document."""
    heading: str
    line_number: int          # Line in full text
    char_offset: int          # Character offset in full text
    page_number: int          # Page where scene starts


@dataclass
class SceneSpan:
    """A scene with its page range."""
    scene_index: int
    heading: str
    start_page: int           # First page of scene (1-indexed)
    end_page: int             # Last page of scene (inclusive)
    page_count: int


@dataclass
class Chunk:
    """A group of scenes forming one output chunk."""
    chunk_index: int
    scenes: list[SceneSpan]
    start_page: int
    end_page: int
    page_count: int


@dataclass
class SplitResult:
    """Result of PDF splitting operation."""
    chunks: list[Chunk]
    total_pages: int
    total_scenes: int
    avg_pages_per_chunk: float
    warnings: list[str] = field(default_factory=list)
    used_llm_fallback: bool = False
    confidence: Literal["high", "medium", "low"] = "high"
