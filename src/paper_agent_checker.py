import json
import re
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from retrieval import ChunkRetriever


QUESTIONS_PATH = Path("data/evaluation_questions.csv")
OUTPUT_PATH = Path("outputs/agent_checker_outputs.csv")

MODEL_NAME = "deepseek-chat"

TOP_K_INITIAL = 5
TOP_K_RETRY = 5
FINAL_TOP_K = 6


def call_llm(client, messages, temperature=0):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content


def build_plan_prompt(question, paper_id, task_type):
    return f"""
You are a careful academic paper reading assistant.

Your task is to create a retrieval plan before answering a question about one paper.

Paper ID:
{paper_id}

Task type:
{task_type}

Question:
{question}

Return your response in this exact format:

Search Plan:
1. What kind of information is needed?
2. Which sections are likely relevant?
3. What evidence would be sufficient?

Search Query:
...
""".strip()


def extract_search_query(text, fallback_question):
    marker = "Search Query:"
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if line.strip().lower().startswith(marker.lower()):
            query = line.split(":", 1)[-1].strip()
            if query:
                return query
            if i + 1 < len(lines) and lines[i + 1].strip():
                return lines[i + 1].strip()

    return fallback_question


def format_context(results):
    blocks = []

    for i, r in enumerate(results, start=1):
        block = f"""[Evidence {i}]
chunk_id: {r['chunk_id']}
paper_id: {r['paper_id']}
section: {r['section']}
page: {r['page_start']}
text:
{r['text']}
"""
        blocks.append(block)

    return "\n\n".join(blocks)


def build_checker_prompt(question, plan_text, context):
    return f"""
You are checking whether retrieved evidence is sufficient to answer an academic paper question.

Question:
{question}

Retrieval Plan:
{plan_text}

Retrieved Evidence:
{context}

Decide whether the evidence is sufficient.

Use one of these labels exactly:
- sufficient
- partially_sufficient
- insufficient

Return your response in this exact format:

Sufficiency: sufficient / partially_sufficient / insufficient
Missing Information: ...
Suggested New Search Query: ...
""".strip()



def parse_sufficiency(check_text):
    """
    Robustly parse sufficiency from the checker output.
    Handles exact labels, natural language, and formatting variations.
    """
    lowered = check_text.lower()

    # Strongest signal first, because "insufficient" contains "sufficient"
    if "partially_sufficient" in lowered or "partially sufficient" in lowered:
        return "partially_sufficient"

    if (
        "insufficient" in lowered
        or "not sufficient" in lowered
        or "not enough evidence" in lowered
        or "evidence is not enough" in lowered
    ):
        return "insufficient"

    # Match standalone sufficient, but avoid counting it inside insufficient
    if re.search(r"\bsufficient\b", lowered):
        return "sufficient"

    return "unknown"


def extract_retry_query(check_text, fallback_question):
    marker = "Suggested New Search Query:"
    lines = check_text.splitlines()

    for i, line in enumerate(lines):
        if line.strip().lower().startswith(marker.lower()):
            query = line.split(":", 1)[-1].strip()
            if query:
                return query
            if i + 1 < len(lines) and lines[i + 1].strip():
                return lines[i + 1].strip()

    return fallback_question


def build_answer_prompt(question, plan_text, check_text, context):
    return f"""
You are answering a question about an academic paper.

Use ONLY the provided evidence.
Do not use outside knowledge.
If the evidence is insufficient, say: "The provided evidence is insufficient."

Question:
{question}

Initial Retrieval Plan:
{plan_text}

Evidence Sufficiency Check:
{check_text}

Final Evidence:
{context}

Return your response in this exact format:

Answer:
...

Supporting Evidence:
- chunk_id: ..., page: ..., explanation: ...

Uncertainty:
...
""".strip()


def merge_results(*result_lists, max_results=6):
    merged = []
    seen = set()

    for results in result_lists:
        for r in results:
            if r["chunk_id"] not in seen:
                merged.append(r)
                seen.add(r["chunk_id"])

    return merged[:max_results]


def main():
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not found. Create a .env file first.")

    if not QUESTIONS_PATH.exists():
        raise FileNotFoundError(f"Could not find {QUESTIONS_PATH}")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

    retriever = ChunkRetriever()
    questions = pd.read_csv(QUESTIONS_PATH)

    rows = []

    for _, q in questions.iterrows():
        question_id = q["question_id"]
        paper_id = q["paper_id"]
        task_type = q["task_type"]
        question = q["question"]
        gold_answer = q.get("gold_answer", "")

        print("=" * 100)
        print(f"{question_id} | {paper_id} | {task_type} | {question}")

        start_time = time.time()

        # Step 1: planning
        plan_prompt = build_plan_prompt(question, paper_id, task_type)
        plan_text = call_llm(
            client,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful academic reading assistant. Create concise retrieval plans."
                },
                {
                    "role": "user",
                    "content": plan_prompt
                }
            ],
            temperature=0
        )

        planned_query = extract_search_query(plan_text, question)

        # Step 2: initial retrieval
        initial_results = retriever.search(
            query=planned_query,
            paper_id=paper_id,
            top_k=TOP_K_INITIAL
        )

        # Step 3: fallback retrieval
        fallback_results = retriever.search(
            query=question,
            paper_id=paper_id,
            top_k=TOP_K_INITIAL
        )

        first_pass_results = merge_results(
            initial_results,
            fallback_results,
            max_results=FINAL_TOP_K
        )

        first_context = format_context(first_pass_results)

        # Step 4: evidence sufficiency check
        checker_prompt = build_checker_prompt(question, plan_text, first_context)
        check_text = call_llm(
            client,
            messages=[
                {
                    "role": "system",
                    "content": "You judge evidence sufficiency for academic paper QA. Be strict and concise."
                },
                {
                    "role": "user",
                    "content": checker_prompt
                }
            ],
            temperature=0
        )

        sufficiency = parse_sufficiency(check_text)
        retry_query = extract_retry_query(check_text, question)

        retry_results = []
        did_retry = False

        # Step 5: retry retrieval if evidence is not enough
        if sufficiency in ["insufficient", "partially_sufficient", "unknown"]:
            did_retry = True
            retry_results = retriever.search(
                query=retry_query,
                paper_id=paper_id,
                top_k=TOP_K_RETRY
            )

        final_results = merge_results(
            first_pass_results,
            retry_results,
            max_results=FINAL_TOP_K
        )

        final_context = format_context(final_results)

        # Step 6: final answer
        answer_prompt = build_answer_prompt(
            question=question,
            plan_text=plan_text,
            check_text=check_text,
            context=final_context
        )

        answer = call_llm(
            client,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful academic reading assistant. You answer only from provided evidence."
                },
                {
                    "role": "user",
                    "content": answer_prompt
                }
            ],
            temperature=0
        )

        latency = time.time() - start_time

        tool_trace = [
            {
                "tool": "plan_search",
                "input_question": question,
                "output_plan": plan_text,
                "extracted_search_query": planned_query
            },
            {
                "tool": "search_sections",
                "query": planned_query,
                "paper_id": paper_id,
                "top_k": TOP_K_INITIAL,
                "returned_chunk_ids": [r["chunk_id"] for r in initial_results]
            },
            {
                "tool": "fallback_search",
                "query": question,
                "paper_id": paper_id,
                "top_k": TOP_K_INITIAL,
                "returned_chunk_ids": [r["chunk_id"] for r in fallback_results]
            },
            {
                "tool": "judge_evidence_sufficiency",
                "input_chunk_ids": [r["chunk_id"] for r in first_pass_results],
                "sufficiency": sufficiency,
                "checker_output": check_text
            },
            {
                "tool": "reformulate_query_and_retry",
                "did_retry": did_retry,
                "retry_query": retry_query if did_retry else "",
                "returned_chunk_ids": [r["chunk_id"] for r in retry_results]
            },
            {
                "tool": "read_chunks",
                "chunk_ids": [r["chunk_id"] for r in final_results]
            }
        ]

        cited_evidence = [
            {
                "chunk_id": r["chunk_id"],
                "paper_id": r["paper_id"],
                "section": r["section"],
                "page_start": r["page_start"],
                "score": r["score"],
                "preview": r["preview"]
            }
            for r in final_results
        ]

        rows.append({
            "question_id": question_id,
            "paper_id": paper_id,
            "task_type": task_type,
            "system": "plan_checker_agent",
            "question": question,
            "gold_answer": gold_answer,
            "system_answer": answer,
            "cited_evidence": json.dumps(cited_evidence, ensure_ascii=False),
            "tool_trace": json.dumps(tool_trace, ensure_ascii=False),
            "sufficiency": sufficiency,
            "did_retry": did_retry,
            "retry_query": retry_query if did_retry else "",
            "latency_seconds": round(latency, 2),
            "num_tool_calls": 6 if did_retry else 5,
            "correctness_score": "",
            "evidence_quality_score": "",
            "grounding_score": "",
            "navigation_score": "",
            "not_answerable_score": "",
            "failure_type": "",
            "notes": ""
        })

        print("Planned query:", planned_query)
        print("Sufficiency:", sufficiency)
        print("Did retry:", did_retry)
        if did_retry:
            print("Retry query:", retry_query)
        print(answer[:500])
        print(f"Latency: {latency:.2f}s")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved checker agent outputs to {OUTPUT_PATH}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
