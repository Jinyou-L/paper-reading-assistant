import fitz  # PyMuPDF
import json
from pathlib import Path


PDF_DIR = Path("data/pdfs")
OUTPUT_DIR = Path("data/parsed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "paper_id": pdf_path.stem,
            "page": page_idx + 1,
            "text": text
        })

    return pages


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/pdfs/")
        return

    for pdf_path in pdf_files:
        print(f"Parsing {pdf_path.name}...")
        parsed_pages = parse_pdf(pdf_path)

        output_path = OUTPUT_DIR / f"{pdf_path.stem}_parsed.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed_pages, f, ensure_ascii=False, indent=2)

        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()