import json
import re
from pathlib import Path
from typing import List, Dict


PARSED_DIR = Path("data/parsed")
OUTPUT_PATH = Path("data/chunks.json")

CHUNK_WORDS = 420
OVERLAP_WORDS = 80

BAD_CHUNK_IDS = {
    "rag_2020_p12_c2",
    "rag_2020_p13_c2",
    "react_2022_p13_c1",
    "toolformer_2023_p14_c2",
}


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = text.replace("\u0003", " ")

    # Normalize spaces and newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text.strip()


def looks_like_reference_chunk(text: str) -> bool:
    """
    Detect chunks that look like bibliography/reference entries.
    This should skip only the chunk, not stop the whole paper.
    """
    lowered = text.lower()

    citation_count = len(re.findall(r"\[\d+\]", text))
    year_count = len(re.findall(r"\b(19|20)\d{2}\b", text))
    url_count = lowered.count("http") + lowered.count("doi") + lowered.count("arxiv")
    bibliography_words = (
        lowered.count("proceedings")
        + lowered.count("conference")
        + lowered.count("journal")
        + lowered.count("transactions")
        + lowered.count("association for computational linguistics")
    )

    # Clear bibliography pattern
    if citation_count >= 3 and year_count >= 5:
        return True

    if year_count >= 10 and url_count >= 2:
        return True

    if year_count >= 12 and bibliography_words >= 4:
        return True

    return False


def looks_like_garbage(text: str) -> bool:
    """
    Conservative garbage filter.
    We only remove very bad chunks/pages.
    """
    if not text or len(text) < 80:
        return True

    letters = sum(ch.isalpha() for ch in text)
    total = max(len(text), 1)
    letter_ratio = letters / total

    if letter_ratio < 0.30:
        return True

    # Detect some common PyMuPDF figure/OCR garbage from weird decoded text.
    weird_markers = [
        "QXeU",
        "GeneUa",
        "VecWion",
        "VXppoUWV",
        "QXeVWiRQ",
        "$FW",
        "7KLQN",
        "2EV",
        "GUDZHU",
    ]

    marker_hits = sum(marker in text for marker in weird_markers)

    # If multiple weird decoded tokens appear, skip this page/chunk.
    if marker_hits >= 3:
        return True

    return False


def should_stop_at_references_heading(text: str) -> bool:
    """
    Only stop when there is an explicit References/Bibliography heading.
    Do NOT stop just because the page has many citations.
    """
    lowered = text.lower().strip()
    lines = lowered.splitlines()

    # Check first 20 lines for standalone reference heading
    for line in lines[:20]:
        line_clean = line.strip().lower()
        line_clean = re.sub(r"[^a-z0-9 .]", "", line_clean).strip()

        if re.fullmatch(r"references", line_clean):
            return True
        if re.fullmatch(r"\d+\.?\s*references", line_clean):
            return True
        if re.fullmatch(r"bibliography", line_clean):
            return True
        if re.fullmatch(r"\d+\.?\s*bibliography", line_clean):
            return True

    return False


def guess_section(text: str, current_section: str) -> str:
    section_patterns = [
        "abstract",
        "introduction",
        "related work",
        "background",
        "method",
        "methods",
        "approach",
        "experiments",
        "experiment",
        "results",
        "analysis",
        "discussion",
        "limitations",
        "limitation",
        "conclusion",
        "appendix",
    ]

    lines = text.splitlines()[:25]

    for line in lines:
        clean_line = line.strip().lower()

        # remove numbering like "2", "2.1", "3.2.1"
        clean_line = re.sub(r"^\d+(\.\d+)*\s*", "", clean_line)

        # normalize weird spacing
        clean_line = re.sub(r"[^a-z ]", " ", clean_line)
        clean_line = re.sub(r"\s+", " ", clean_line).strip()

        for sec in section_patterns:
            if clean_line == sec or clean_line.startswith(sec + " "):
                return sec.title()

    return current_section


def chunk_words(words: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]

        if len(chunk) >= 80:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def process_parsed_file(parsed_path: Path) -> List[Dict]:
    with open(parsed_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    all_chunks = []
    current_section = "Unknown"
    stop = False

    for page_obj in pages:
        paper_id = page_obj.get("paper_id", parsed_path.stem.replace("_parsed", ""))
        page_num = page_obj.get("page")
        raw_text = page_obj.get("text", "")

        text = clean_text(raw_text)

        if not text:
            continue

        # Stop only at explicit References heading.
        if should_stop_at_references_heading(text):
            stop = True

        if stop:
            continue

        # Skip only very garbled pages.
        if looks_like_garbage(text):
            continue

        current_section = guess_section(text, current_section)

        words = text.split()
        page_chunks = chunk_words(words, CHUNK_WORDS, OVERLAP_WORDS)

        for idx, chunk in enumerate(page_chunks):
            chunk_text = " ".join(chunk)

            # Skip bibliography-like chunks, but do not stop the whole paper.
            if looks_like_reference_chunk(chunk_text):
                continue

            if looks_like_garbage(chunk_text):
                continue

            chunk_id = f"{paper_id}_p{page_num}_c{idx + 1}"

            if chunk_id in BAD_CHUNK_IDS:
                continue

            all_chunks.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "page_start": page_num,
                "page_end": page_num,
                "section": current_section,
                "text": chunk_text
            })

    return all_chunks


def main():
    parsed_files = sorted(PARSED_DIR.glob("*_parsed.json"))

    if not parsed_files:
        print("No parsed JSON files found in data/parsed/")
        return

    all_chunks = []

    for parsed_path in parsed_files:
        print(f"Chunking {parsed_path.name}...")
        chunks = process_parsed_file(parsed_path)
        print(f"  created {len(chunks)} chunks")
        all_chunks.extend(chunks)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_chunks)} total chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
