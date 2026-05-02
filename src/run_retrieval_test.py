import csv
import json
from pathlib import Path

import pandas as pd

from retrieval import ChunkRetriever


QUESTIONS_PATH = Path("data/evaluation_questions.csv")
OUTPUT_PATH = Path("outputs/retrieval_test_outputs.csv")


def main():
    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Could not find {QUESTIONS_PATH}")

    questions = pd.read_csv(QUESTIONS_PATH)
    retriever = ChunkRetriever()

    rows = []

    for _, q in questions.iterrows():
        question_id = q["question_id"]
        paper_id = q["paper_id"]
        task_type = q["task_type"]
        question = q["question"]

        results = retriever.search(
            query=question,
            paper_id=paper_id,
            top_k=5
        )

        cited_evidence = []
        for r in results:
            cited_evidence.append({
                "chunk_id": r["chunk_id"],
                "paper_id": r["paper_id"],
                "section": r["section"],
                "page_start": r["page_start"],
                "score": r["score"],
                "preview": r["preview"]
            })

        rows.append({
            "question_id": question_id,
            "paper_id": paper_id,
            "task_type": task_type,
            "question": question,
            "retrieved_evidence_json": json.dumps(cited_evidence, ensure_ascii=False)
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)

    print(f"Saved retrieval test outputs to {OUTPUT_PATH}")
    print(f"Processed {len(rows)} questions")


if __name__ == "__main__":
    main()
