import pandas as pd
from pathlib import Path

BASELINE_PATH = Path("outputs/baseline_outputs.csv")
AGENT_PATH = Path("outputs/agent_outputs.csv")
CHECKER_PATH = Path("outputs/agent_checker_outputs.csv")

OUTPUT_PATH = Path("outputs/evaluation_results_three_systems.csv")

base = pd.read_csv(BASELINE_PATH)
agent = pd.read_csv(AGENT_PATH)
checker = pd.read_csv(CHECKER_PATH)

combined = pd.concat([base, agent, checker], ignore_index=True)

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

print(f"Saved merged three-system evaluation file to {OUTPUT_PATH}")
print(combined[["question_id", "paper_id", "task_type", "system", "latency_seconds", "num_tool_calls"]])
print("Rows:", len(combined))
print("\nCounts:")
print(combined["system"].value_counts())
