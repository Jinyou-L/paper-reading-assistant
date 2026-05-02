import pandas as pd
from pathlib import Path

BASELINE_PATH = Path("outputs/baseline_outputs.csv")
AGENT_PATH = Path("outputs/agent_outputs.csv")
OUTPUT_PATH = Path("outputs/evaluation_results_full.csv")

base = pd.read_csv(BASELINE_PATH)
agent = pd.read_csv(AGENT_PATH)

combined = pd.concat([base, agent], ignore_index=True)

for col in [
    "correctness_score",
    "evidence_quality_score",
    "grounding_score",
    "navigation_score",
    "not_answerable_score",
    "failure_type",
    "notes"
]:
    if col not in combined.columns:
        combined[col] = ""

combined.to_csv(OUTPUT_PATH, index=False)

print(f"Saved merged evaluation file to {OUTPUT_PATH}")
print(combined[["question_id", "paper_id", "system", "latency_seconds", "num_tool_calls"]])
print("Rows:", len(combined))
