import json
import pandas as pd
from pathlib import Path

QUESTIONS_PATH = Path("data/evaluation_questions.csv")
RETRIEVAL_PATH = Path("outputs/retrieval_comparison_outputs.csv")
OUTPUT_PATH = Path("outputs/retrieval_metrics_by_method.csv")


def parse_gold_pages(value):
    if pd.isna(value):
        return set()

    value = str(value).strip()

    if value == "" or value.lower() in ["n/a", "na", "none"]:
        return set()

    pages = set()
    for part in value.replace(",", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            pages.add(int(float(part)))
        except ValueError:
            pass

    return pages


def compute_metrics(gold_pages, retrieved_pages):
    """
    Page-level retrieval metrics.
    Hit@k: whether any gold page appears in retrieved pages.
    MRR: reciprocal rank of first matching page.
    Recall@k: whether retrieved pages cover at least one gold page.
              For page-level multi-gold, use fraction of gold pages found.
    """
    if not gold_pages:
        return {
            "is_answerable": False,
            "hit_at_5": "",
            "recall_at_5": "",
            "mrr": "",
            "first_hit_rank": ""
        }

    first_hit_rank = None

    for i, page in enumerate(retrieved_pages, start=1):
        if page in gold_pages:
            first_hit_rank = i
            break

    found_pages = set(retrieved_pages).intersection(gold_pages)

    hit_at_5 = 1 if first_hit_rank is not None else 0
    recall_at_5 = len(found_pages) / len(gold_pages) if gold_pages else 0
    mrr = 1 / first_hit_rank if first_hit_rank is not None else 0

    return {
        "is_answerable": True,
        "hit_at_5": hit_at_5,
        "recall_at_5": recall_at_5,
        "mrr": mrr,
        "first_hit_rank": first_hit_rank if first_hit_rank is not None else ""
    }


def main():
    questions = pd.read_csv(QUESTIONS_PATH)
    retrieval = pd.read_csv(RETRIEVAL_PATH)

    q_gold = questions[["question_id", "gold_evidence_page"]].copy()
    merged = retrieval.merge(q_gold, on="question_id", how="left")

    rows = []

    for _, row in merged.iterrows():
        evidence = json.loads(row["retrieved_evidence_json"])

        retrieved_pages = []
        retrieved_chunk_ids = []

        for item in evidence[:5]:
            page = item.get("page_start", "")
            try:
                retrieved_pages.append(int(float(page)))
            except Exception:
                pass

            retrieved_chunk_ids.append(item.get("chunk_id", ""))

        gold_pages = parse_gold_pages(row.get("gold_evidence_page", ""))

        metrics = compute_metrics(gold_pages, retrieved_pages)

        rows.append({
            "question_id": row["question_id"],
            "paper_id": row["paper_id"],
            "task_type": row["task_type"],
            "retrieval_method": row["retrieval_method"],
            "question": row["question"],
            "gold_evidence_page": row.get("gold_evidence_page", ""),
            "retrieved_pages_top5": ";".join(map(str, retrieved_pages)),
            "retrieved_chunks_top5": ";".join(retrieved_chunk_ids),
            **metrics
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    answerable = out[out["is_answerable"] == True].copy()

    summary = (
        answerable
        .groupby("retrieval_method")
        .agg(
            n=("question_id", "count"),
            hit_at_5=("hit_at_5", "mean"),
            recall_at_5=("recall_at_5", "mean"),
            mrr=("mrr", "mean")
        )
        .reset_index()
    )

    print("Retrieval metrics summary:")
    print(summary)

    print(f"\nSaved detailed retrieval metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
