import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from retrieval import ChunkRetriever


QUESTIONS_PATH = Path("data/evaluation_questions.csv")
OUTPUT_PATH = Path("outputs/agent_outputs.csv")

MODEL_NAME = "deepseek-chat"
TOP_K_INITIAL = 5
TOP_K_FOLLOWUP = 3


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

Your job is to create a short retrieval plan before answering.
The user is asking a question about one academic paper.

Paper ID:
{paper_id}

Task type:
{task_type}

Question:
{question}

Write a short search plan in this exact format:

Search Plan:
1. What kind of information is needed?
2. Which paper sections are likely relevant?
3. What search query should be used?

Search Query:
...
""".strip()


def extract_search_query(plan_text, fallback_question):
    """
    Extract the line after 'Search Query:'.
    If extraction fails, fall back to the original question.
    """
    marker = "Search Query:"
    if marker.lower() not in plan_text.lower():
        return fallback_question

    lines = plan_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(marker.lower()):
            query = line.split(":", 1)[-1].strip()
            if query:
                return query

            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line:
                    return next_line

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


def build_answer_prompt(question, plan_text, context):
    return f"""
You are answering a question about an academic paper.

You already created this retrieval plan:
{plan_text}

Now answer the question using ONLY the provided evidence.
Do not use outside knowledge.
If the evidence is insufficient, say: "The provided evidence is insufficient."

Question:
{question}

Evidence:
{context}

Return your response in this exact format:

Answer:
...

Supporting Evidence:
- chunk_id: ..., page: ..., explanation: ...

Uncertainty:
...
""".strip()


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
        print(f"{question_id} | {paper_id} | {question}")

        start_time = time.time()

        # Step 1: ask LLM to create a plan
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

        search_query = extract_search_query(plan_text, question)

        # Step 2: use retrieval tool based on the planned query
        initial_results = retriever.search(
            query=search_query,
            paper_id=paper_id,
            top_k=TOP_K_INITIAL
        )

        # Step 3: simple follow-up retrieval using original question as backup
        followup_results = retriever.search(
            query=question,
            paper_id=paper_id,
            top_k=TOP_K_FOLLOWUP
        )

        # Merge by chunk_id while preserving order
        merged = []
        seen = set()

        for r in initial_results + followup_results:
            if r["chunk_id"] not in seen:
                merged.append(r)
                seen.add(r["chunk_id"])

        final_results = merged[:5]

        context = format_context(final_results)

        # Step 4: answer using evidence
        answer_prompt = build_answer_prompt(question, plan_text, context)
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
                "extracted_search_query": search_query
            },
            {
                "tool": "search_sections",
                "query": search_query,
                "paper_id": paper_id,
                "top_k": TOP_K_INITIAL,
                "returned_chunk_ids": [r["chunk_id"] for r in initial_results]
            },
            {
                "tool": "search_sections",
                "query": question,
                "paper_id": paper_id,
                "top_k": TOP_K_FOLLOWUP,
                "returned_chunk_ids": [r["chunk_id"] for r in followup_results]
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
            "system": "plan_first_agent",
            "question": question,
            "gold_answer": gold_answer,
            "system_answer": answer,
            "cited_evidence": json.dumps(cited_evidence, ensure_ascii=False),
            "tool_trace": json.dumps(tool_trace, ensure_ascii=False),
            "latency_seconds": round(latency, 2),
            "num_tool_calls": 4,
            "correctness_score": "",
            "evidence_quality_score": "",
            "grounding_score": "",
            "navigation_score": "",
            "not_answerable_score": "",
            "failure_type": "",
            "notes": ""
        })

        print("Search query:", search_query)
        print(answer[:600])
        print(f"Latency: {latency:.2f}s")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved agent outputs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
