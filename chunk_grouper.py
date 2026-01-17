"""
Groups scenes into ~4-page chunks without breaking scenes.
"""
from models import SceneSpan, Chunk

DEFAULT_TARGET_PAGES = 4
DEFAULT_MIN_PAGES = 2
DEFAULT_MAX_PAGES = 8


def group_scenes_into_chunks(
    scenes: list[SceneSpan],
    target_pages: int = DEFAULT_TARGET_PAGES,
    min_pages: int = DEFAULT_MIN_PAGES,
    max_pages: int = DEFAULT_MAX_PAGES
) -> list[Chunk]:
    """
    Group scenes into chunks targeting ~4 pages each.

    Algorithm:
    1. Never split a scene mid-way (scene integrity)
    2. Accumulate scenes until adding another would exceed target
    3. If a single scene exceeds target, it becomes its own chunk
    4. Balance chunks to avoid tiny final chunks

    Args:
        scenes: List of SceneSpan objects
        target_pages: Target pages per chunk (default 4)
        min_pages: Minimum acceptable chunk size
        max_pages: Maximum before forcing new chunk

    Returns:
        List of Chunk objects
    """
    if not scenes:
        return []

    chunks = []
    current_scenes: list[SceneSpan] = []
    chunk_start_page = scenes[0].start_page
    current_end_page = chunk_start_page - 1  # Will be updated when scenes are added

    def get_current_pages() -> int:
        """Calculate actual page count from start to current end."""
        if not current_scenes:
            return 0
        return current_end_page - chunk_start_page + 1

    def get_pages_with_scene(scene: SceneSpan) -> int:
        """Calculate page count if we add this scene."""
        if not current_scenes:
            return scene.end_page - scene.start_page + 1
        new_end = max(current_end_page, scene.end_page)
        return new_end - chunk_start_page + 1

    for scene in scenes:
        scene_pages = scene.page_count
        current_pages = get_current_pages()
        pages_with_scene = get_pages_with_scene(scene)

        # Case 1: Scene by itself meets or exceeds target
        # Make it its own chunk (don't split the scene)
        if scene_pages >= target_pages:
            # First, close out any accumulated scenes
            if current_scenes:
                chunks.append(_make_chunk(
                    len(chunks),
                    current_scenes,
                    chunk_start_page
                ))
                current_scenes = []

            # Add oversized scene as its own chunk
            chunks.append(_make_chunk(
                len(chunks),
                [scene],
                scene.start_page
            ))
            chunk_start_page = scene.end_page + 1
            current_end_page = chunk_start_page - 1
            continue

        # Case 2: Adding this scene would exceed max
        if pages_with_scene > max_pages and current_scenes:
            chunks.append(_make_chunk(
                len(chunks),
                current_scenes,
                chunk_start_page
            ))
            current_scenes = []
            chunk_start_page = scene.start_page
            current_end_page = chunk_start_page - 1

        # Recalculate after potential flush
        current_pages = get_current_pages()
        pages_with_scene = get_pages_with_scene(scene)

        # Case 3: Adding this scene would exceed target
        # Decide whether to include or start new chunk
        if pages_with_scene > target_pages and current_scenes:
            # Check if current chunk is already at minimum
            if current_pages >= min_pages:
                # Close current chunk, start new one
                chunks.append(_make_chunk(
                    len(chunks),
                    current_scenes,
                    chunk_start_page
                ))
                current_scenes = [scene]
                chunk_start_page = scene.start_page
                current_end_page = scene.end_page
                continue

        # Add scene to current chunk
        current_scenes.append(scene)
        current_end_page = max(current_end_page, scene.end_page)

    # Don't forget final chunk
    if current_scenes:
        chunks.append(_make_chunk(
            len(chunks),
            current_scenes,
            chunk_start_page
        ))

    # Rebalance if final chunk is tiny
    chunks = _rebalance_final_chunk(chunks, min_pages)

    return chunks


def group_by_page_count(
    total_pages: int,
    target_pages: int = DEFAULT_TARGET_PAGES
) -> list[Chunk]:
    """
    Fallback: Split by page count only (no scene awareness).

    Used when no scene boundaries are detected.

    Args:
        total_pages: Total pages in document
        target_pages: Target pages per chunk

    Returns:
        List of Chunk objects
    """
    chunks = []
    current_page = 1

    while current_page <= total_pages:
        end_page = min(current_page + target_pages - 1, total_pages)

        # Create a dummy scene span for the chunk
        dummy_scene = SceneSpan(
            scene_index=len(chunks),
            heading=f"Pages {current_page}-{end_page}",
            start_page=current_page,
            end_page=end_page,
            page_count=end_page - current_page + 1
        )

        chunks.append(Chunk(
            chunk_index=len(chunks),
            scenes=[dummy_scene],
            start_page=current_page,
            end_page=end_page,
            page_count=end_page - current_page + 1
        ))

        current_page = end_page + 1

    return chunks


def _make_chunk(index: int, scenes: list[SceneSpan], start_page: int) -> Chunk:
    """Create a Chunk from scenes."""
    end_page = scenes[-1].end_page
    return Chunk(
        chunk_index=index,
        scenes=scenes,
        start_page=start_page,
        end_page=end_page,
        page_count=end_page - start_page + 1
    )


def _rebalance_final_chunk(
    chunks: list[Chunk],
    min_pages: int
) -> list[Chunk]:
    """
    If final chunk is too small, merge with previous.
    """
    if len(chunks) < 2:
        return chunks

    final = chunks[-1]
    if final.page_count < min_pages:
        # Merge with previous chunk
        prev = chunks[-2]
        merged_scenes = prev.scenes + final.scenes
        merged = Chunk(
            chunk_index=prev.chunk_index,
            scenes=merged_scenes,
            start_page=prev.start_page,
            end_page=final.end_page,
            page_count=final.end_page - prev.start_page + 1
        )
        return chunks[:-2] + [merged]

    return chunks
