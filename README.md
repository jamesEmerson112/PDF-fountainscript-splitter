# PDF Screenplay Splitter

Split fountain-style screenplay PDFs into ~4 page chunks without cutting scenes mid-way.

## Features

- **Scene-aware splitting**: Uses LLM (GPT-4o-mini) to detect scene boundaries
- **Never splits mid-scene**: Chunks respect scene integrity
- **Dual output**: Creates both PDF and Markdown files
- **Page range support**: Process specific page ranges
- **Preview mode**: See chunk boundaries before creating files

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate pdf-splitter

# Or install with pip
pip install pymupdf openai tqdm
```

## Setup

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your-key-here  # Linux/Mac
set OPENAI_API_KEY=your-key-here     # Windows
```

## Usage

```bash
# Basic usage - split entire PDF
python main.py script.pdf

# Preview without creating files
python main.py script.pdf --preview

# Split first 20 pages only
python main.py script.pdf --pages 20

# Split pages 10-30
python main.py script.pdf --pages 10-30

# Custom output directory
python main.py script.pdf -o my_output/

# Target different chunk size (default: 4 pages)
python main.py script.pdf --target-pages 6
```

## Output Structure

```
script_split/
├── pdf/
│   ├── chunk_01_pages_2-5.pdf
│   ├── chunk_02_pages_6-9.pdf
│   └── ...
├── text/
│   ├── chunk_01_pages_2-5.md
│   ├── chunk_02_pages_6-9.md
│   └── ...
└── split_summary.txt
```

## CLI Options

| Option | Description |
|--------|-------------|
| `input` | Input PDF file (required) |
| `-o, --output-dir` | Output directory (default: `<input>_split/`) |
| `--target-pages` | Target pages per chunk (default: 4) |
| `--min-pages` | Minimum pages per chunk (default: 2) |
| `--max-pages` | Maximum pages per chunk (default: 8) |
| `--pages` | Page range: `20` or `10-30` |
| `--preview` | Show page ranges without creating files |
| `--no-llm` | Disable LLM scene detection (use regex only) |
| `-v, --verbose` | Show detailed output |

## How It Works

1. **Extract text** from PDF with page tracking
2. **Detect scenes** using GPT-4o-mini (falls back to regex)
3. **Map scenes to pages** using character offsets
4. **Group scenes** into ~4 page chunks (respecting scene boundaries)
5. **Output** split PDFs + Markdown files

## Requirements

- Python 3.11+
- PyMuPDF (pymupdf)
- OpenAI API key
- tqdm
