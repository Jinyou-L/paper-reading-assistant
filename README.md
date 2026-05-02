# Paper Reading Assistant

## Project Title

**Paper Reading Assistant: Comparing Single-Pass RAG with Agentic Retrieval for Academic PDF Question Answering**

## Project Category

Solo Agent Application with a RAG baseline.

## Overview

This project builds and evaluates a lightweight paper reading assistant for evidence-grounded academic PDF question answering.

The goal is not only to generate answers about papers, but to help users locate and verify the evidence behind those answers. I frame academic paper QA as an **evidence navigation problem**, not just a summarization problem.

The main comparison is between:

1. **Single-Pass RAG Baseline**: retrieves relevant chunks once and answers directly from those chunks.
2. **Plan-First Agentic Retrieval**: first creates a retrieval plan, retrieves evidence using the planned query and a fallback query, and then answers with evidence and tool traces.

The project also includes additional experiments on retrieval quality and a deeper checker-agent workflow.

## Corpus

The corpus contains four public academic papers related to RAG, tool use, and agentic reasoning:

- *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- *SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*
- *ReAct: Synergizing Reasoning and Acting in Language Models*
- *Toolformer: Language Models Can Teach Themselves to Use Tools*

PDFs are stored locally under `data/pdfs/` and are **not included** in this repository.

The processed corpus is included as:

- `data/chunks.json`
- `data/parsed/`

## Pipeline

The main pipeline is:

1. Parse PDFs with PyMuPDF.
2. Clean extracted text and remove noisy/reference chunks.
3. Split papers into overlapping chunks with page/section metadata.
4. Retrieve relevant chunks using TF-IDF.
5. Run the Single-Pass RAG baseline.
6. Run the Plan-First Agentic Retrieval system.
7. Evaluate outputs on a 24-question golden set.
8. Run additional retrieval and checker-agent experiments.

## Main Evaluation

The evaluation set contains 24 questions across six categories:

- method
- baseline
- experiment
- limitation
- evidence location
- not-answerable

Each of the two main systems answers all 24 questions, producing 48 total outputs.

The outputs are manually scored using a 0–2 rubric on:

- correctness
- evidence quality
- grounding
- navigation usefulness
- not-answerable handling
- latency
- tool-call count

## Main Results

The Plan-First Agent improved correctness, evidence quality, and navigation usefulness compared with the Single-Pass RAG baseline, while maintaining the same grounding score.

| System | Correctness | Evidence Quality | Grounding | Navigation | Avg Latency |
|---|---:|---:|---:|---:|---:|
| Single-Pass RAG | 1.75 / 2 | 1.75 / 2 | 2.00 / 2 | 1.75 / 2 | 3.41s |
| Plan-First Agent | 1.92 / 2 | 1.92 / 2 | 2.00 / 2 | 1.92 / 2 | 5.89s |

The main finding is that the Plan-First Agent provides the best quality-latency tradeoff in this project. It improves evidence quality and navigation, but introduces additional latency.

## Additional Experiments

In addition to the main comparison, this repository includes two deeper extensions.

### 1. Retrieval Comparison

I compared TF-IDF and BM25 retrieval to test whether retrieval failures were caused by the retrieval method itself.

| Retrieval Method | Avg Relevance | Direct Evidence Rate |
|---|---:|---:|
| TF-IDF | 1.67 / 2 | 58.3% |
| BM25 | 1.67 / 2 | 58.3% |

The result suggests that switching between keyword retrieval methods alone did not solve missed direct evidence.

### 2. Page-Level Retrieval Metrics

I also added page-level retrieval metrics using manually labeled gold evidence pages.

For the 20 answerable questions:

| Retrieval Method | Hit@5 | Recall@5 | MRR |
|---|---:|---:|---:|
| BM25 | 0.95 | 0.708 | 0.692 |
| TF-IDF | 0.95 | 0.804 | 0.673 |

Both retrievers usually found at least one correct evidence page in the top five retrieved chunks, but ranking and evidence coverage still varied.

### 3. Plan + Checker Agent

I implemented a deeper multi-step agent with evidence sufficiency checking and retry retrieval.

Workflow:

```text
plan → retrieve → judge evidence sufficiency → retry if insufficient → answer
```

The checker agent triggered retry retrieval for 13 out of 24 questions.

| System | Correctness | Evidence Quality | Grounding | Navigation | Avg Latency |
|---|---:|---:|---:|---:|---:|
| Single-Pass RAG | 1.75 | 1.75 | 2.00 | 1.75 | 3.41s |
| Plan-First Agent | 1.92 | 1.92 | 2.00 | 1.92 | 5.89s |
| Plan + Checker Agent | 1.79 | 1.79 | 2.00 | 1.79 | 9.79s |

The checker-agent extension made the workflow more inspectable and added error recovery, but it did not outperform the simpler Plan-First Agent. This suggests that more agent steps do not automatically improve quality.

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file with your DeepSeek API key:

```bash
DEEPSEEK_API_KEY=your_api_key_here
```

Parse PDFs:

```bash
python src/parse_pdfs.py
```

Chunk documents:

```bash
python src/chunk_documents.py
```

Test retrieval:

```bash
python src/run_retrieval_test.py
```

Run Single-Pass RAG baseline:

```bash
python src/rag_baseline.py
```

Run Plan-First Agent:

```bash
python src/paper_agent.py
```

Merge main outputs:

```bash
python src/merge_outputs.py
```

Analyze main results:

```bash
python src/analyze_results.py
```

Run retrieval comparison:

```bash
python src/retrieval_compare.py
python src/analyze_retrieval_comparison.py
```

Run page-level retrieval metrics:

```bash
python src/evaluate_retrieval_metrics.py
```

Run Plan + Checker Agent:

```bash
python src/paper_agent_checker.py
python src/merge_three_systems.py
```

## Project Structure

```text
paper-reading-assistant/
├── README.md
├── requirements.txt
├── app.py
├── src/
│   ├── parse_pdfs.py
│   ├── chunk_documents.py
│   ├── retrieval.py
│   ├── run_retrieval_test.py
│   ├── rag_baseline.py
│   ├── paper_agent.py
│   ├── paper_agent_checker.py
│   ├── retrieval_compare.py
│   ├── evaluate_retrieval_metrics.py
│   ├── merge_outputs.py
│   ├── merge_three_systems.py
│   ├── analyze_results.py
│   └── analyze_retrieval_comparison.py
├── data/
│   ├── chunks.json
│   ├── evaluation_questions.csv
│   ├── evaluation_questions_with_gold_pages.csv
│   └── parsed/
├── outputs/
│   ├── evaluation_results_full_scored.csv
│   ├── evaluation_summary_full.csv
│   ├── evaluation_summary_by_task.csv
│   ├── failure_summary.csv
│   ├── retrieval_comparison_summary.csv
│   ├── retrieval_metrics_summary.csv
│   ├── agent_checker_outputs_scored.csv
│   ├── evaluation_results_three_systems_scored.csv
│   ├── evaluation_summary_three_systems.csv
│   └── failure_summary_three_systems.csv
└── report/
    ├── final_report_draft.md
    ├── results_section.md
    └── failure_analysis.md
```

## Key Files

### Report

- `report/final_report_draft.md`: full final project report
- `report/results_section.md`: generated results section
- `report/failure_analysis.md`: generated failure analysis

### Data

- `data/evaluation_questions.csv`: 24-question golden evaluation set
- `data/evaluation_questions_with_gold_pages.csv`: evaluation questions with gold evidence pages
- `data/chunks.json`: cleaned and chunked paper text

### Main Evaluation Outputs

- `outputs/baseline_outputs.csv`: Single-Pass RAG outputs
- `outputs/agent_outputs.csv`: Plan-First Agent outputs
- `outputs/evaluation_results_full_scored.csv`: manually scored main evaluation
- `outputs/evaluation_summary_full.csv`: overall main evaluation summary
- `outputs/evaluation_summary_by_task.csv`: task-level main evaluation summary
- `outputs/failure_summary.csv`: main failure summary

### Additional Experiment Outputs

- `outputs/retrieval_comparison_outputs_scored.csv`: manually scored retrieval comparison
- `outputs/retrieval_comparison_summary.csv`: TF-IDF vs BM25 retrieval summary
- `outputs/retrieval_metrics_by_method.csv`: page-level retrieval metrics by question
- `outputs/retrieval_metrics_summary.csv`: Hit@5, Recall@5, and MRR summary
- `outputs/agent_checker_outputs_scored.csv`: scored Plan + Checker Agent outputs
- `outputs/evaluation_results_three_systems_scored.csv`: combined three-system evaluation
- `outputs/evaluation_summary_three_systems.csv`: three-system summary
- `outputs/failure_summary_three_systems.csv`: three-system failure summary

## Limitations

This project uses a small controlled corpus of four papers and a manually scored evaluation set of 24 questions. The small corpus was chosen intentionally to support careful manual evaluation, but it limits generalizability.

The main retriever uses TF-IDF, and additional retrieval experiments compare BM25. These keyword-based methods may miss semantic matches. Future work should test dense retrieval, hybrid retrieval, and reranking.

PDF parsing remains a limitation because tables, figures, equations, and two-column layouts can produce noisy or misordered chunks.

The Plan-First Agent improves evidence quality and navigation, but it is slower than Single-Pass RAG. The deeper Plan + Checker Agent adds error recovery, but did not outperform the simpler Plan-First Agent.

## References

See `report/final_report_draft.md` for full references.
