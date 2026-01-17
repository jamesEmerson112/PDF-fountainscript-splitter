"""
Maps scene boundaries to page numbers.
"""
import re
from models import PageText, SceneLocation, SceneSpan

# Scene heading pattern from format_detector.py
SCENE_HEADING_PATTERN = re.compile(
    r'^[ \t]*(?:\d+[ \t]*)?(?:INT\.?|EXT\.?|INT/EXT\.?|INT\.?/EXT\.?|EXT\.?/INT\.?|I/E\.?|I\.?/E\.?)[ \t]+.+',
    re.IGNORECASE | re.MULTILINE
)


def find_scene_locations(
    full_text: str,
    pages: list[PageText]
) -> list[SceneLocation]:
    """
    Find all scene headings and their page locations.

    Args:
        full_text: Complete extracted text
        pages: Page boundary information

    Returns:
        List of SceneLocation objects
    """
    scenes = []
    lines = full_text.split('\n')
    char_offset = 0

    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if stripped and SCENE_HEADING_PATTERN.match(stripped):
            page_num = _get_page_for_offset(pages, char_offset)
            scenes.append(SceneLocation(
                heading=stripped,
                line_number=line_num,
                char_offset=char_offset,
                page_number=page_num
            ))
        char_offset += len(line) + 1  # +1 for newline

    return scenes


def _get_page_for_offset(pages: list[PageText], offset: int) -> int:
    """Binary search for page containing offset."""
    for page in pages:
        if page.char_start <= offset < page.char_end:
            return page.page_number
    return pages[-1].page_number if pages else 1


def build_scene_spans(
    scenes: list[SceneLocation],
    total_pages: int
) -> list[SceneSpan]:
    """
    Convert scene locations to spans with page ranges.

    Args:
        scenes: List of scene start locations
        total_pages: Total pages in document

    Returns:
        List of SceneSpan with start/end pages
    """
    if not scenes:
        # No scenes found - treat entire doc as one scene
        return [SceneSpan(
            scene_index=0,
            heading="DOCUMENT",
            start_page=1,
            end_page=total_pages,
            page_count=total_pages
        )]

    spans = []
    for i, scene in enumerate(scenes):
        start_page = scene.page_number

        # End page is the page before next scene starts
        # (or same page if next scene on same page)
        if i + 1 < len(scenes):
            next_start = scenes[i + 1].page_number
            # Scene ends on page before next scene, or same page if same page
            end_page = max(start_page, next_start - 1) if next_start > start_page else start_page
        else:
            # Last scene goes to end of document
            end_page = total_pages

        spans.append(SceneSpan(
            scene_index=i,
            heading=scene.heading,
            start_page=start_page,
            end_page=end_page,
            page_count=end_page - start_page + 1
        ))

    return spans


def map_headings_to_pages(
    headings: list[str],
    full_text: str,
    pages: list[PageText]
) -> list[SceneLocation]:
    """
    Map LLM-detected headings back to page locations.

    Args:
        headings: List of scene heading strings found by LLM
        full_text: Complete extracted text
        pages: Page boundary information

    Returns:
        List of SceneLocation objects
    """
    locations = []
    search_start = 0

    for heading in headings:
        # Find this heading in the text
        idx = full_text.find(heading, search_start)
        if idx == -1:
            # Try case-insensitive
            lower_text = full_text.lower()
            idx = lower_text.find(heading.lower(), search_start)

        if idx != -1:
            page_num = _get_page_for_offset(pages, idx)
            locations.append(SceneLocation(
                heading=heading,
                line_number=full_text[:idx].count('\n'),
                char_offset=idx,
                page_number=page_num
            ))
            search_start = idx + len(heading)

    return locations
