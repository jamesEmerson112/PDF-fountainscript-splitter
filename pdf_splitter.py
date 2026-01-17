"""
PDF Splitter - Intelligent screenplay PDF chunking.

Follows heuristics-first, LLM-fallback pattern.
"""
import os
import re
import json
from pathlib import Path
from typing import Optional, Literal

import fitz  # PyMuPDF

from models import SplitResult, Chunk, PageText, SceneSpan
from pdf_extractor import extract_text_with_pages
from page_scene_mapper import find_scene_locations, build_scene_spans, map_headings_to_pages
from chunk_grouper import group_scenes_into_chunks, group_by_page_count


# Quick Fountain check pattern (from format_detector.py)
SCENE_HEADING_PATTERN = re.compile(
    r'^[ \t]*(?:\d+[ \t]*)?(?:INT\.?|EXT\.?|INT/EXT\.?|I/E\.?)[ \t]+.+',
    re.IGNORECASE | re.MULTILINE
)

# LLM prompt for scene boundary detection when heuristics fail
SCENE_DETECTION_PROMPT = """Analyze this screenplay text and identify ALL scene headings.

Scene headings typically look like:
- INT. LOCATION - TIME
- EXT. LOCATION - TIME
- INT/EXT. LOCATION - TIME
- I/E. LOCATION - TIME

Sometimes scene headings may be non-standard or missing the INT/EXT prefix.

Return ONLY a JSON array of scene heading strings found in the text, in order:
["INT. KITCHEN - DAY", "EXT. PARK - NIGHT", ...]

If no scene headings are found, return: []

Text to analyze:
"""


def _quick_fountain_check(text: str) -> bool:
    """Quick heuristic check for Fountain format indicators."""
    headings = SCENE_HEADING_PATTERN.findall(text)
    return len(headings) >= 2


def split_pdf(
    pdf_path: str,
    *,
    target_pages: int = 4,
    min_pages: int = 2,
    max_pages: int = 8,
    use_llm_fallback: bool = True,
    openai_api_key: Optional[str] = None,
    page_start: int = 1,
    page_end: Optional[int] = None
) -> SplitResult:
    """
    Split a screenplay PDF into intelligent chunks.

    Args:
        pdf_path: Path to the PDF file
        target_pages: Target pages per chunk (default 4)
        min_pages: Minimum chunk size
        max_pages: Maximum chunk size before forcing split
        use_llm_fallback: Whether to use LLM for scene detection
        openai_api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
        page_start: First page to process (1-indexed, default 1)
        page_end: Last page to process (inclusive, default all pages)

    Returns:
        SplitResult with chunks and metadata
    """
    warnings: list[str] = []
    used_llm = False
    confidence: Literal["high", "medium", "low"] = "high"

    # Step 1: Extract text with page tracking
    try:
        full_text_all, all_pages = extract_text_with_pages(pdf_path)
    except Exception as e:
        return SplitResult(
            chunks=[],
            total_pages=0,
            total_scenes=0,
            avg_pages_per_chunk=0,
            warnings=[f"PDF extraction failed: {e}"],
            used_llm_fallback=False,
            confidence="low"
        )

    # Filter to page range
    if page_end is None:
        page_end = len(all_pages)

    pages = [p for p in all_pages if page_start <= p.page_number <= page_end]
    total_pages = len(pages)

    if total_pages == 0:
        return SplitResult(
            chunks=[],
            total_pages=0,
            total_scenes=0,
            avg_pages_per_chunk=0,
            warnings=["No pages in specified range"],
            used_llm_fallback=False,
            confidence="low"
        )

    # Build text for the specified page range
    full_text = '\n'.join(p.text for p in pages)

    if not full_text.strip():
        warnings.append("PDF appears to be empty or image-only")
        # Fall back to page-count splitting
        chunks = group_by_page_count(total_pages, target_pages)
        # Adjust chunk page numbers to match actual pages
        for chunk in chunks:
            chunk.start_page = chunk.start_page + page_start - 1
            chunk.end_page = chunk.end_page + page_start - 1
            for scene in chunk.scenes:
                scene.start_page = scene.start_page + page_start - 1
                scene.end_page = scene.end_page + page_start - 1
        return SplitResult(
            chunks=chunks,
            total_pages=total_pages,
            total_scenes=0,
            avg_pages_per_chunk=target_pages,
            warnings=warnings,
            used_llm_fallback=False,
            confidence="low"
        )

    # Step 2: Use LLM as PRIMARY method for scene detection
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    scene_locations = []

    if api_key and use_llm_fallback:
        print("  Using LLM for scene detection...")
        llm_scenes = _llm_find_scenes(full_text, api_key)
        if llm_scenes:
            scene_locations = map_headings_to_pages(llm_scenes, full_text, pages)
            used_llm = True
            print(f"  LLM found {len(scene_locations)} scenes")

    # Step 3: Fall back to regex if LLM didn't find enough
    if len(scene_locations) < 2:
        if used_llm:
            warnings.append(f"LLM only found {len(scene_locations)} scene(s), trying regex")
        regex_locations = find_scene_locations(full_text, pages)
        if len(regex_locations) > len(scene_locations):
            scene_locations = regex_locations
            warnings.append(f"Regex found {len(scene_locations)} scenes")

    # Step 4: Evaluate confidence
    if len(scene_locations) >= 5:
        confidence = "high"
    elif len(scene_locations) >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    # Step 5: If still no scenes, fall back to page-count splitting
    if len(scene_locations) < 2:
        warnings.append("Falling back to page-count splitting (no scene boundaries)")
        chunks = group_by_page_count(total_pages, target_pages)
        # Adjust chunk page numbers to match actual pages
        for chunk in chunks:
            chunk.start_page = chunk.start_page + page_start - 1
            chunk.end_page = min(chunk.end_page + page_start - 1, page_end)
            for scene in chunk.scenes:
                scene.start_page = scene.start_page + page_start - 1
                scene.end_page = min(scene.end_page + page_start - 1, page_end)
        return SplitResult(
            chunks=chunks,
            total_pages=total_pages,
            total_scenes=0,
            avg_pages_per_chunk=_calc_avg(chunks),
            warnings=warnings,
            used_llm_fallback=used_llm,
            confidence="low"
        )

    # Step 6: Build scene spans with page ranges
    scene_spans = build_scene_spans(scene_locations, page_end)

    # Step 7: Group into chunks
    chunks = group_scenes_into_chunks(
        scene_spans,
        target_pages=target_pages,
        min_pages=min_pages,
        max_pages=max_pages
    )

    return SplitResult(
        chunks=chunks,
        total_pages=total_pages,
        total_scenes=len(scene_spans),
        avg_pages_per_chunk=_calc_avg(chunks),
        warnings=warnings,
        used_llm_fallback=used_llm,
        confidence=confidence
    )


def _llm_find_scenes(text: str, api_key: str, max_chars: int = 30000) -> list[str]:
    """
    Use OpenAI GPT-4o-mini to find scene headings.

    Args:
        text: Extracted screenplay text
        api_key: OpenAI API key
        max_chars: Maximum characters to send (default 30000)

    Returns:
        List of scene heading strings
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Truncate if too long (GPT-4o-mini has 128k context but we limit for cost)
        sample = text[:max_chars]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a screenplay analyst. Return only valid JSON arrays."},
                {"role": "user", "content": f"{SCENE_DETECTION_PROMPT}\n\n{sample}"}
            ],
            temperature=0
        )

        result = response.choices[0].message.content.strip()

        # Parse JSON array (handle markdown code blocks)
        if result.startswith('```'):
            # Extract content between ```
            parts = result.split('```')
            if len(parts) >= 2:
                result = parts[1]
                if result.startswith('json'):
                    result = result[4:]
                result = result.strip()

        return json.loads(result)

    except json.JSONDecodeError as e:
        print(f"  LLM returned invalid JSON: {e}")
        return []
    except Exception as e:
        print(f"  LLM scene detection failed: {e}")
        return []


def _calc_avg(chunks: list[Chunk]) -> float:
    """Calculate average pages per chunk."""
    if not chunks:
        return 0.0
    return round(sum(c.page_count for c in chunks) / len(chunks), 1)


def split_to_pdf_files(
    source_pdf: str,
    result: SplitResult,
    output_dir: str,
    filename_template: str = "chunk_{index:02d}_pages_{start}-{end}.pdf"
) -> list[str]:
    """
    Split PDF into separate files based on chunks.

    Args:
        source_pdf: Path to source PDF
        result: SplitResult from split_pdf()
        output_dir: Directory for output files
        filename_template: Template for output filenames

    Returns:
        List of created file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(source_pdf)
    output_files = []

    for chunk in result.chunks:
        # Create new PDF with just these pages
        new_doc = fitz.open()

        # Pages are 0-indexed in PyMuPDF
        for page_num in range(chunk.start_page - 1, chunk.end_page):
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)

        filename = filename_template.format(
            index=chunk.chunk_index + 1,
            start=chunk.start_page,
            end=chunk.end_page
        )
        output_path = str(Path(output_dir) / filename)
        new_doc.save(output_path)
        new_doc.close()
        output_files.append(output_path)

    doc.close()
    return output_files


def split_to_markdown_files(
    pages: list[PageText],
    result: SplitResult,
    output_dir: str,
    filename_template: str = "chunk_{index:02d}_pages_{start}-{end}.md"
) -> list[str]:
    """
    Create markdown files for each chunk with extracted text.

    Args:
        pages: List of PageText objects with extracted text
        result: SplitResult from split_pdf()
        output_dir: Directory for output files
        filename_template: Template for output filenames

    Returns:
        List of created file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_files = []

    for chunk in result.chunks:
        # Build markdown content
        lines = [
            f"# Chunk {chunk.chunk_index + 1}: Pages {chunk.start_page}-{chunk.end_page}",
            "",
            f"## Scenes in this chunk ({len(chunk.scenes)}):",
        ]

        for scene in chunk.scenes:
            lines.append(f"- {scene.heading}")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Extract text for pages in this chunk
        for page in pages:
            if chunk.start_page <= page.page_number <= chunk.end_page:
                lines.append(f"### Page {page.page_number}")
                lines.append("")
                lines.append(page.text.strip())
                lines.append("")

        # Write markdown file
        filename = filename_template.format(
            index=chunk.chunk_index + 1,
            start=chunk.start_page,
            end=chunk.end_page
        )
        output_path = Path(output_dir) / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        output_files.append(str(output_path))

    return output_files


def write_summary(
    result: SplitResult,
    output_dir: str,
    source_pdf: str
) -> str:
    """
    Write a human-readable summary file.

    Args:
        result: SplitResult from split_pdf()
        output_dir: Directory for output
        source_pdf: Original PDF path

    Returns:
        Path to summary file
    """
    summary_path = Path(output_dir) / "split_summary.txt"

    lines = [
        f"PDF Split Summary",
        f"=" * 50,
        f"Source: {source_pdf}",
        f"Total pages: {result.total_pages}",
        f"Scenes detected: {result.total_scenes}",
        f"Chunks created: {len(result.chunks)}",
        f"Average pages per chunk: {result.avg_pages_per_chunk}",
        f"Confidence: {result.confidence}",
        f"Used LLM fallback: {result.used_llm_fallback}",
        "",
    ]

    if result.warnings:
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  - {w}")
        lines.append("")

    lines.append("Chunks:")
    lines.append("-" * 50)

    for chunk in result.chunks:
        lines.append(f"\nChunk {chunk.chunk_index + 1}: pages {chunk.start_page}-{chunk.end_page} ({chunk.page_count} pages)")
        lines.append(f"  Scenes ({len(chunk.scenes)}):")
        for scene in chunk.scenes[:5]:  # Show first 5 scenes
            heading_preview = scene.heading[:60] + "..." if len(scene.heading) > 60 else scene.heading
            lines.append(f"    - {heading_preview}")
        if len(chunk.scenes) > 5:
            lines.append(f"    ... and {len(chunk.scenes) - 5} more")

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return str(summary_path)


def get_page_ranges(result: SplitResult) -> list[dict]:
    """
    Get page ranges without creating files.

    Useful for preview or when caller handles PDF splitting.

    Returns:
        List of dicts with chunk info
    """
    return [
        {
            "chunk": chunk.chunk_index + 1,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "page_count": chunk.page_count,
            "scenes": [s.heading for s in chunk.scenes]
        }
        for chunk in result.chunks
    ]
