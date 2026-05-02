# Paper Reading Assistant

## Project Title
Paper Reading Assistant: Comparing Single-Pass RAG with Agentic Retrieval for Academic PDF Question Answering

## Project Category
Solo Agent Application with a RAG baseline.

## Overview
This project builds a lightweight paper reading assistant for academic PDF question answering. The goal is to help students locate relevant evidence in research papers and answer structured questions about methods, baselines, experiments, limitations, and evidence locations.

The project compares two systems:
1. Single-Pass RAG Baseline: retrieves relevant chunks once and answers directly.
2. Plan-First Agentic Retrieval: first creates a retrieval plan, performs retrieval with tool traces, and then answers using evidence.

## Corpus
The corpus contains four academic papers related to RAG, tool use, and agentic reasoning:
- Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
- ReAct: Synergizing Reasoning and Acting in Language Models
- Toolformer: Language Models Can Teach Themselves to Use Tools

PDFs are stored locally under data/pdfs/ and are not included in the repository.

## Pipeline
1. Parse PDFs with PyMuPDF.
2. Clean extracted text and remove references/noisy chunks.
3. Split papers into overlapping chunks.
4. Retrieve relevant chunks using TF-IDF and cosine similarity.
5. Run the single-pass RAG baseline.
6. Run the plan-first agentic retrieval system.
7. Evaluate outputs on a golden set of 24 questions.

## Evaluation
The evaluation set contains 24 questions across six categories:
- method
- baseline
- experiment
- limitation
- evidence location
- not-answerable

Each system answers all 24 questions, producing 48 total outputs.

Metrics:
- correctness
- evidence quality
- grounding
- navigation usefulness
- not-answerable handling
- latency
- tool-call count

## Main Results
The plan-first agent improved correctness, evidence quality, and navigation usefulness compared with the single-pass RAG baseline, while maintaining the same grounding score.

| System | Correctness | Evidence Quality | Grounding | Navigation | Avg Latency |
|---|---:|---:|---:|---:|---:|
| Single-pass RAG | 1.75 / 2 | 1.75 / 2 | 2.00 / 2 | 1.75 / 2 | 3.41s |
| Plan-first agent | 1.92 / 2 | 1.92 / 2 | 2.00 / 2 | 1.92 / 2 | 5.89s |

The agentic workflow improved evidence quality and navigation, but introduced additional latency.

## How to Run
Install dependencies:

    python -m pip install -r requirements.txt

Create a .env file with your DeepSeek API key:

    DEEPSEEK_API_KEY=your_api_key_here

Parse PDFs:

    python src/parse_pdfs.py

Chunk documents:

    python src/chunk_documents.py

Test retrieval:

    python src/run_retrieval_test.py

Run single-pass RAG baseline:

    python src/rag_baseline.py

Run plan-first agent:

    python src/paper_agent.py

Merge outputs:

    python src/merge_outputs.py

Analyze results:

    python src/analyze_results.py

## Project Structure
paper-reading-assistant/
- README.md
- requirements.txt
- src/
- data/
- outputs/
- report/

## Key Files
- data/evaluation_questions.csv: the 24-question golden evaluation set
- data/chunks.json: cleaned and chunked paper text
- outputs/baseline_outputs.csv: single-pass RAG outputs
- outputs/agent_outputs.csv: plan-first agent outputs
- outputs/evaluation_results_full_scored.csv: manually scored evaluation results
- outputs/evaluation_summary_full.csv: overall performance summary
- outputs/evaluation_summary_by_task.csv: task-level performance summary
- outputs/failure_summary.csv: failure type summary
- report/final_report_draft.md: full project report draft
- report/results_section.md: generated results section
- report/failure_analysis.md: generated failure analysis

## Limitations
This project uses a small corpus of four papers and a manually scored evaluation set of 24 questions. The retriever uses TF-IDF rather than dense embeddings, which makes the system simple and transparent but may miss semantically relevant evidence. The agent is intentionally lightweight and constrained.

## References
See the final report for full references.
