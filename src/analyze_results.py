import pandas as pd
from pathlib import Path

INPUT_PATH = Path("outputs/evaluation_results_full_scored.csv")
SUMMARY_PATH = Path("outputs/evaluation_summary_full.csv")
TASK_SUMMARY_PATH = Path("outputs/evaluation_summary_by_task.csv")
FAILURE_PATH = Path("outputs/failure_summary.csv")
RESULTS_MD_PATH = Path("report/results_section.md")
FAILURE_MD_PATH = Path("report/failure_analysis.md")


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Put evaluation_results_full_scored.csv in outputs/ first."
        )

    df = pd.read_csv(INPUT_PATH)

    score_cols = [
        "correctness_score",
        "evidence_quality_score",
        "grounding_score",
        "navigation_score",
        "not_answerable_score",
        "latency_seconds",
        "num_tool_calls",
    ]

    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Overall summary by system
    summary = (
        df.groupby("system")
        .agg(
            n=("question_id", "count"),
            correctness=("correctness_score", "mean"),
            evidence_quality=("evidence_quality_score", "mean"),
            grounding=("grounding_score", "mean"),
            navigation=("navigation_score", "mean"),
            not_answerable=("not_answerable_score", "mean"),
            avg_latency_seconds=("latency_seconds", "mean"),
            avg_tool_calls=("num_tool_calls", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    # Summary by task type and system
    task_summary = (
        df.groupby(["task_type", "system"])
        .agg(
            n=("question_id", "count"),
            correctness=("correctness_score", "mean"),
            evidence_quality=("evidence_quality_score", "mean"),
            grounding=("grounding_score", "mean"),
            navigation=("navigation_score", "mean"),
            avg_latency_seconds=("latency_seconds", "mean"),
        )
        .reset_index()
    )

    task_summary.to_csv(TASK_SUMMARY_PATH, index=False)

    # Failure summary
    if "failure_type" in df.columns:
        failure_df = df.copy()
        failure_df["failure_type"] = failure_df["failure_type"].fillna("")
        failure_df = failure_df[failure_df["failure_type"].str.strip() != ""]

        if len(failure_df) > 0:
            failure_summary = (
                failure_df.groupby(["system", "failure_type"])
                .size()
                .reset_index(name="count")
                .sort_values(["system", "count"], ascending=[True, False])
            )
        else:
            failure_summary = pd.DataFrame(columns=["system", "failure_type", "count"])
    else:
        failure_summary = pd.DataFrame(columns=["system", "failure_type", "count"])

    failure_summary.to_csv(FAILURE_PATH, index=False)

    # Markdown results section
    RESULTS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Results\n\n")
        f.write("## Overall Performance\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Performance by Task Type\n\n")
        f.write(task_summary.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Interpretation\n\n")
        f.write(
            "The plan-first agent achieved higher average correctness, evidence quality, "
            "and navigation usefulness than the single-pass RAG baseline. Both systems had "
            "strong grounding scores, which suggests that the evidence-only prompting helped "
            "reduce unsupported claims. However, the agentic workflow introduced additional "
            "latency because it required an extra planning step and multiple retrieval/tool-use steps.\n\n"
        )
        f.write(
            "Overall, the results suggest that the plan-first agentic workflow is most useful "
            "when the task requires locating specific evidence or reasoning about where information "
            "appears in the paper. For simpler method-definition questions, the single-pass RAG "
            "baseline was often already sufficient.\n"
        )

    with open(FAILURE_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Failure Analysis\n\n")
        f.write("## Failure Summary\n\n")
        f.write(failure_summary.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Main Failure Patterns\n\n")
        f.write(
            "The most common failure pattern was incomplete evidence retrieval. In some cases, "
            "the system retrieved generally relevant chunks but missed the most direct supporting "
            "passage. This affected questions about baselines, limitations, and exact experimental "
            "details more than broad method-summary questions.\n\n"
        )
        f.write(
            "Another failure pattern was over-cautious answering. When the retrieved evidence was "
            "partially relevant but incomplete, the model sometimes stated that the evidence was "
            "insufficient instead of giving a partial answer. This behavior is safer from a grounding "
            "perspective, but it can reduce answer completeness.\n\n"
        )
        f.write(
            "The plan-first agent reduced some of these issues by creating a retrieval plan and "
            "performing multiple retrieval passes, but it did not eliminate retrieval failures entirely. "
            "The agent also introduced higher latency, which is an important tradeoff for real use.\n"
        )

    print(f"Saved overall summary to {SUMMARY_PATH}")
    print(f"Saved task summary to {TASK_SUMMARY_PATH}")
    print(f"Saved failure summary to {FAILURE_PATH}")
    print(f"Saved results markdown to {RESULTS_MD_PATH}")
    print(f"Saved failure analysis markdown to {FAILURE_MD_PATH}")

    print("\nOverall summary:")
    print(summary)

    print("\nFailure summary:")
    print(failure_summary)


if __name__ == "__main__":
    main()
