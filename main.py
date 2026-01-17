"""
PDF Screenplay Splitter - CLI Interface

Split fountain-style screenplay PDFs into ~4 page chunks
without cutting scenes mid-way.

Usage:
    python main.py input.pdf [--target-pages N] [--preview] [-o OUTPUT_DIR]
"""
import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from pdf_splitter import split_pdf, split_to_pdf_files, split_to_markdown_files, write_summary, get_page_ranges
from pdf_extractor import get_total_pages, extract_text_with_pages


def main():
    parser = argparse.ArgumentParser(
        description="Split screenplay PDFs into scene-aware chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py script.pdf                    # Split with defaults
    python main.py script.pdf --preview          # Preview without creating files
    python main.py script.pdf --target-pages 6   # Target 6 pages per chunk
    python main.py script.pdf -o my_output/      # Custom output directory
        """
    )
    parser.add_argument(
        "input",
        help="Input PDF file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        help="Output directory (default: <input>_split/)"
    )
    parser.add_argument(
        "--target-pages",
        type=int,
        default=4,
        help="Target pages per chunk (default: 4)"
    )
    parser.add_argument(
        "--min-pages",
        type=int,
        default=2,
        help="Minimum pages per chunk (default: 2)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=8,
        help="Maximum pages per chunk (default: 8)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show page ranges without creating files"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM fallback for scene detection"
    )
    parser.add_argument(
        "--pages",
        type=str,
        help="Page range: '20' for first 20 pages, '10-30' for pages 10-30"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output"
    )

    args = parser.parse_args()

    # Parse page range if specified
    page_start = 1
    page_end = None  # None means all pages
    if args.pages:
        if '-' in args.pages:
            parts = args.pages.split('-')
            page_start = int(parts[0])
            page_end = int(parts[1])
        else:
            page_start = 1
            page_end = int(args.pages)

    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    if not input_path.suffix.lower() == '.pdf':
        print(f"Error: Not a PDF file: {args.input}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(input_path.parent / f"{input_path.stem}_split")

    # Show progress for large PDFs
    total_pages = get_total_pages(str(input_path))

    # Validate and adjust page range
    if page_end is None:
        page_end = total_pages
    page_end = min(page_end, total_pages)
    page_start = max(1, page_start)

    if args.pages:
        print(f"\nProcessing: {input_path.name} (pages {page_start}-{page_end} of {total_pages})")
    else:
        print(f"\nProcessing: {input_path.name} ({total_pages} pages)")

    # Split the PDF
    print("Analyzing scene boundaries (using LLM)...")
    with tqdm(total=3, desc="Progress", disable=not sys.stdout.isatty()) as pbar:
        pbar.set_description("Extracting text")
        result = split_pdf(
            str(input_path),
            target_pages=args.target_pages,
            min_pages=args.min_pages,
            max_pages=args.max_pages,
            use_llm_fallback=not args.no_llm,
            page_start=page_start,
            page_end=page_end
        )
        pbar.update(1)

        pbar.set_description("Grouping scenes")
        pbar.update(1)

        if args.preview:
            pbar.set_description("Generating preview")
        else:
            pbar.set_description("Creating PDF files")
        pbar.update(1)

    # Show warnings
    if result.warnings and args.verbose:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Total pages: {result.total_pages}")
    print(f"Scenes detected: {result.total_scenes}")
    print(f"Chunks: {len(result.chunks)}")
    print(f"Avg pages/chunk: {result.avg_pages_per_chunk}")
    print(f"Confidence: {result.confidence}")
    if result.used_llm_fallback:
        print("(Used LLM fallback for scene detection)")
    print(f"{'='*50}")

    # Preview mode
    if args.preview:
        print("\nPreview - Page ranges:")
        ranges = get_page_ranges(result)
        for r in ranges:
            print(f"\n  Chunk {r['chunk']}: pages {r['start_page']}-{r['end_page']} ({r['page_count']} pages)")
            for scene in r['scenes'][:3]:
                scene_preview = scene[:55] + "..." if len(scene) > 55 else scene
                print(f"    - {scene_preview}")
            if len(r['scenes']) > 3:
                print(f"    ... and {len(r['scenes']) - 3} more scenes")
        print(f"\nTo create files, run without --preview flag")
        return

    # Create output files with organized subfolders
    pdf_output_dir = str(Path(output_dir) / "pdf")
    text_output_dir = str(Path(output_dir) / "text")
    print(f"\nCreating files in: {output_dir}")
    print(f"  PDFs:     {pdf_output_dir}")
    print(f"  Markdown: {text_output_dir}")

    # Extract pages for markdown output (filtered to page range)
    _, all_pages = extract_text_with_pages(str(input_path))
    pages = [p for p in all_pages if page_start <= p.page_number <= page_end]

    with tqdm(total=len(result.chunks) * 2 + 1, desc="Writing files") as pbar:
        # Write PDF files
        pbar.set_description("Writing PDFs")
        pdf_files = split_to_pdf_files(str(input_path), result, pdf_output_dir)
        pbar.update(len(result.chunks))

        # Write Markdown files
        pbar.set_description("Writing Markdown")
        md_files = split_to_markdown_files(pages, result, text_output_dir)
        pbar.update(len(result.chunks))

        # Write summary
        pbar.set_description("Writing summary")
        summary_path = write_summary(result, output_dir, str(input_path))
        pbar.update(1)

    print(f"\nCreated {len(pdf_files)} PDF files + {len(md_files)} Markdown files:")
    for f in pdf_files[:3]:
        print(f"  - {Path(f).name}")
    if len(pdf_files) > 3:
        print(f"  ... and {len(pdf_files) - 3} more PDFs")
    for f in md_files[:3]:
        print(f"  - {Path(f).name}")
    if len(md_files) > 3:
        print(f"  ... and {len(md_files) - 3} more Markdown files")

    print(f"\nSummary: {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
