import pandas as pd
from pathlib import Path

INPUT_PATH = Path("outputs/retrieval_comparison_outputs_scored.csv")
OUTPUT_PATH = Path("outputs/retrieval_comparison_summary.csv")
TASK_OUTPUT_PATH = Path("outputs/retrieval_comparison_by_task.csv")

df = pd.read_csv(INPUT_PATH)

df["retrieval_relevance_score"] = pd.to_numeric(
    df["retrieval_relevance_score"],
    errors="coerce"
)

summary = (
    df.groupby("retrieval_method")
    .agg(
        n=("question_id", "count"),
        avg_relevance=("retrieval_relevance_score", "mean"),
        direct_evidence_rate=("direct_evidence_found", lambda x: (x == "yes").mean())
    )
    .reset_index()
)

task_summary = (
    df.groupby(["task_type", "retrieval_method"])
    .agg(
        n=("question_id", "count"),
        avg_relevance=("retrieval_relevance_score", "mean"),
        direct_evidence_rate=("direct_evidence_found", lambda x: (x == "yes").mean())
    )
    .reset_index()
)

summary.to_csv(OUTPUT_PATH, index=False)
task_summary.to_csv(TASK_OUTPUT_PATH, index=False)

print("Overall retrieval comparison:")
print(summary)

print("\nBy task type:")
print(task_summary)

print(f"\nSaved to {OUTPUT_PATH}")
print(f"Saved to {TASK_OUTPUT_PATH}")
