import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from retrieval import ChunkRetriever


QUESTIONS_PATH = Path("data/evaluation_questions.csv")
OUTPUT_PATH = Path("outputs/baseline_outputs.csv")

# DeepSeek model
MODEL_NAME = "deepseek-chat"
TOP_K = 5


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


def build_prompt(question, context):
    return f"""
You are answering a question about an academic paper.

Use ONLY the provided evidence.
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


def call_llm(client, prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a careful academic reading assistant. You answer only from provided evidence."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content


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

        results = retriever.search(
            query=question,
            paper_id=paper_id,
            top_k=TOP_K
        )

        context = format_context(results)
        prompt = build_prompt(question, context)
        answer = call_llm(client, prompt)

        latency = time.time() - start_time

        cited_evidence = [
            {
                "chunk_id": r["chunk_id"],
                "paper_id": r["paper_id"],
                "section": r["section"],
                "page_start": r["page_start"],
                "score": r["score"],
                "preview": r["preview"]
            }
            for r in results
        ]

        rows.append({
            "question_id": question_id,
            "paper_id": paper_id,
            "task_type": task_type,
            "system": "single_pass_rag",
            "question": question,
            "gold_answer": gold_answer,
            "system_answer": answer,
            "cited_evidence": json.dumps(cited_evidence, ensure_ascii=False),
            "latency_seconds": round(latency, 2),
            "num_tool_calls": 0,
            "correctness_score": "",
            "evidence_quality_score": "",
            "grounding_score": "",
            "navigation_score": "",
            "not_answerable_score": "",
            "failure_type": "",
            "notes": ""
        })

        print(answer[:500])
        print(f"Latency: {latency:.2f}s")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved baseline outputs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
