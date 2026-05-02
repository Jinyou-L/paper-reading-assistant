# Paper Reading Assistant: Comparing Single-Pass RAG with Agentic Retrieval for Academic PDF Question Answering

## 1. Introduction and Motivation

This project builds and evaluates a lightweight paper reading assistant for academic PDF question answering. The goal is not only to generate answers about papers, but to help users locate and verify the evidence behind those answers.

When students read academic papers, they often need to answer questions such as:

- What is the main method proposed in the paper?
- Which baselines are compared?
- What datasets or tasks are used in the experiments?
- What limitations are stated or implied?
- Which page or section supports a specific claim?

A basic chatbot may answer these questions, but it often hides where the answer came from. For academic reading, this is a serious limitation because the user needs to verify the answer in the original paper.

I frame this project as an **evidence navigation problem**, not just a summarization problem. The main research question is:

> Does a plan-first agentic retrieval workflow improve correctness, evidence quality, and navigation usefulness compared with single-pass RAG?

To answer this question, I compare a **Single-Pass RAG baseline** with a **Plan-First Agentic Retrieval system** on a controlled academic PDF corpus. I also run additional experiments on retrieval methods and a deeper checker-agent workflow to understand where the systems succeed and fail.

---

## 2. Project Scope

This project is a **solo agent application with a RAG baseline**. The scope is intentionally focused.

The project does not attempt to build a large-scale academic search engine. Instead, it uses a small controlled corpus of related papers so that the retrieved evidence and generated answers can be manually evaluated carefully.

The main contribution is not simply that the assistant can answer questions. The main contribution is a controlled comparison between retrieval workflows:

1. A simple Single-Pass RAG baseline.
2. A Plan-First Agentic Retrieval system.
3. Additional extensions that test retrieval methods and error recovery.

This scope fits a three-week solo project because it balances system implementation, evaluation, failure analysis, and reflection.

---

## 3. Corpus and Data Processing

### 3.1 Corpus

The corpus contains four public academic papers related to retrieval-augmented generation, language model agents, and tool use:

1. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  
   Lewis et al., 2020

2. **ReAct: Synergizing Reasoning and Acting in Language Models**  
   Yao et al., 2022 / 2023

3. **SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**  
   Asai et al., 2023

4. **Toolformer: Language Models Can Teach Themselves to Use Tools**  
   Schick et al., 2023

These papers were selected because they are closely related to the project topic. They cover RAG, agentic reasoning, tool use, and self-reflective retrieval.

The corpus is intentionally small and controlled. This made it possible to manually inspect evidence quality, score model outputs, and analyze failures.

### 3.2 PDF Parsing and Cleaning

The PDFs were parsed into text using PyMuPDF. The parsed text was then cleaned before retrieval.

The cleaning process included:

- removing obvious noisy text;
- filtering references and bibliography-like chunks;
- reducing retrieval pollution from reference sections;
- preserving page-level and section-level metadata.

This step was important because academic PDFs often contain two-column layouts, figures, tables, formulas, and references that can produce noisy extracted text.

### 3.3 Chunking

After parsing and cleaning, the documents were split into overlapping text chunks. Each chunk contains metadata such as:

- paper ID;
- page number;
- section label;
- chunk ID;
- chunk text.

The final corpus contains:

> **123 cleaned chunks**

These chunks were stored in `data/chunks.json`.

---

## 4. System Design

The project compares two main systems.

---

### 4.1 Single-Pass RAG Baseline

The Single-Pass RAG system is the baseline. It follows the standard retrieval-then-generation workflow:

```text
User Question
    ↓
Retrieve top-k chunks once
    ↓
Pass retrieved chunks to LLM
    ↓
Generate answer using evidence

### 4.2 Plan-First Agentic Retrieval

The Plan-First Agent is the proposed enhanced system. Instead of directly retrieving chunks, it first creates a retrieval plan.

```text
User Question
    ↓
Generate search plan
    ↓
Retrieve using planned query
    ↓
Fallback retrieval using original question
    ↓
Merge evidence
    ↓
Generate answer with evidence and tool trace
```

The agent uses several tools or modules:

- `plan_search`
- `search_sections`
- `fallback_search`
- `read_chunks`
- final evidence-grounded answer generation

The goal is to make retrieval more deliberate and inspectable. The system exposes a tool trace so that the user can see how the answer was produced.

This design is intentionally constrained. I did not build an open-ended autonomous agent that can call arbitrary tools. Instead, I used a limited retrieval workflow so that behavior is easier to debug and evaluate.

## 5. Design Decisions

### 5.1 Why Single-Pass RAG as the Baseline?

Single-Pass RAG is the simplest reasonable baseline for this task. It retrieves relevant chunks once and generates an answer directly from them.

Using it as a baseline makes the evaluation meaningful. Instead of only showing that the assistant works, I can ask whether the agentic workflow actually improves output quality.

### 5.2 Why Plan-First Agentic Retrieval?

Academic paper questions often require targeted evidence search. For example, a question about baselines may require experiment tables, while a question about limitations may require the introduction, discussion, or conclusion.

The Plan-First Agent makes this search process explicit. It first decides what information is needed, then retrieves evidence based on that plan.

### 5.3 Why TF-IDF Retrieval?

I used TF-IDF retrieval because it is lightweight, transparent, and easy to debug within a short solo project timeline.

Dense retrieval or hybrid retrieval could improve semantic matching, but TF-IDF made it easier to understand why retrieval succeeded or failed. This was useful for failure analysis.

### 5.4 Why Evidence-Only Prompting?

Both systems use evidence-only prompts. The model is instructed to answer only from retrieved evidence and to say when the evidence is insufficient.

This reduces unsupported claims and makes the evaluation more focused on retrieval and evidence quality.

## 6. Evaluation Design

### 6.1 Golden Test Set

I created a golden test set of **24 questions** across six task types:

| Task Type | Number of Questions |
|---|---:|
| Method extraction | 4 |
| Baseline identification | 4 |
| Experiment / dataset | 4 |
| Limitation extraction | 4 |
| Evidence location | 4 |
| Not-answerable | 4 |

Each question was answered by both the Single-Pass RAG baseline and the Plan-First Agent.

This produced:

> **24 questions × 2 systems = 48 system outputs**

### 6.2 Evaluation Metrics

Each output was manually scored using a **0–2 rubric**.

| Metric | Scale | Description |
|---|---:|---|
| Correctness | 0–2 | Whether the answer is correct |
| Evidence quality | 0–2 | Whether the evidence directly supports the answer |
| Grounding | 0–2 | Whether the answer avoids unsupported claims |
| Navigation usefulness | 0–2 | Whether the system helps locate useful pages/chunks |
| Not-answerable handling | 0–2 | Whether the system correctly refuses unsupported questions |
| Latency | seconds | Average response time |
| Tool calls | count | Number of tool/retrieval steps |

The scoring scale is:

| Score | Meaning |
|---:|---|
| 0 | Poor / incorrect / unsupported |
| 1 | Partially correct or partially supported |
| 2 | Strong / correct / well supported |

## 7. Main Results

### 7.1 Overall Results

The main result is that the **Plan-First Agent provides the best quality-latency tradeoff**. It improves correctness, evidence quality, and navigation over Single-Pass RAG, while remaining much faster than the later checker-agent extension.

| System | Correctness | Evidence Quality | Grounding | Navigation | Not-answerable | Avg Latency | Avg Tool Calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single-Pass RAG | 1.75 | 1.75 | 2.00 | 1.75 | 2.00 | 3.41s | 0.00 |
| Plan-First Agent | 1.92 | 1.92 | 2.00 | 1.92 | 2.00 | 5.89s | 4.00 |

The Plan-First Agent modestly improved correctness, evidence quality, and navigation usefulness compared with the Single-Pass RAG baseline.

Correctness improved from **1.75** to **1.92**. Evidence quality improved from **1.75** to **1.92**. Navigation usefulness improved from **1.75** to **1.92**.

Both systems achieved a grounding score of **2.00**, which suggests that evidence-only prompting helped reduce unsupported claims.

### 7.2 Latency Tradeoff

The improvement came with additional latency.

| System | Avg Latency |
|---|---:|
| Single-Pass RAG | 3.41s |
| Plan-First Agent | 5.89s |

The Plan-First Agent was about **73% slower** than Single-Pass RAG.

This shows a clear tradeoff:

> The agent improves evidence quality and navigation usefulness, but it is slower.

The Plan-First Agent is most useful for evidence-heavy questions, such as baselines, limitations, and evidence-location questions. For simpler method-definition questions, Single-Pass RAG was often sufficient.

## 8. Additional Experiment 1: Retrieval Method Comparison

The main evaluation measures final answer quality. To better understand retrieval quality separately from generation quality, I also compared two keyword retrieval methods:

1. TF-IDF retrieval
2. BM25 retrieval

The motivation was to ask:

> Were retrieval failures caused by the retrieval method itself?

### 8.1 Manual Retrieval Relevance

I first manually scored retrieved evidence relevance for TF-IDF and BM25.

Both methods had similar overall retrieval relevance.

| Retrieval Method | Avg Relevance | Direct Evidence Rate |
|---|---:|---:|
| TF-IDF | 1.67 / 2 | 58.3% |
| BM25 | 1.67 / 2 | 58.3% |

This suggests that simply switching from TF-IDF to BM25 did not solve the missed-evidence problem.

### 8.2 Interpretation

TF-IDF and BM25 failed in slightly different ways. TF-IDF sometimes retrieved broader related sections, while BM25 sometimes ranked exact keyword matches higher but still surfaced less direct chunks.

The result suggests that future work should test:

- dense retrieval;
- hybrid retrieval;
- reranking;
- better query reformulation.

## 9. Additional Experiment 2: Page-Level Retrieval Metrics

To further separate retrieval quality from generation quality, I added page-level retrieval metrics.

I manually labeled gold evidence pages for the answerable questions. The four not-answerable questions were excluded from retrieval metric calculation.

This produced retrieval metrics for **20 answerable questions**.

The metrics were:

| Metric | Meaning |
|---|---|
| Hit@5 | Whether any gold evidence page appears in the top-5 retrieved chunks |
| Recall@5 | Fraction of gold evidence pages found in top-5 |
| MRR | Reciprocal rank of the first retrieved gold evidence page |

### 9.1 Retrieval Metrics Results

| Retrieval Method | n | Hit@5 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 20 | 0.95 | 0.708 | 0.692 |
| TF-IDF | 20 | 0.95 | 0.804 | 0.673 |

Both TF-IDF and BM25 achieved **Hit@5 = 0.95**. This means that for 95% of the answerable questions, the top-5 retrieved chunks included at least one correct evidence page.

However, the methods differed slightly:

- TF-IDF had higher Recall@5, meaning it covered more gold evidence pages on average.
- BM25 had slightly higher MRR, meaning it sometimes ranked the first correct evidence page slightly earlier.

### 9.2 Interpretation

This result shows that retrieval was usually able to find at least one correct evidence page. However, ranking and evidence coverage still varied.

This helps explain why generation quality was not perfect. Even when retrieval found a correct page, the most direct chunk was not always ranked first, and the answer generator sometimes used partial evidence.

This also explains why the manual direct-evidence rate is lower than Hit@5. Hit@5 only checks whether at least one gold evidence page appears in the top five retrieved chunks, while the manual direct-evidence score is stricter and asks whether the retrieved text is specific and directly useful enough for answering the question.

## 10. Additional Experiment 3: Plan + Checker Agent

I also implemented a deeper multi-step agent with error recovery.

The original Plan-First Agent followed this workflow:

```text
plan → planned retrieval → fallback retrieval → answer
```

The new Plan + Checker Agent added an evidence sufficiency checker:

```text
plan → retrieve → judge evidence sufficiency
→ retry retrieval if insufficient
→ answer
```

The checker agent used the following modules:

- `search_sections(query)`
- `read_chunks(chunk_ids)`
- `judge_evidence_sufficiency(question, evidence)`
- `reformulate_query(question, missing information)`
- `final_answer(question, evidence)`

### 10.1 Checker Agent Behavior

The checker judged the retrieved evidence as:

| Sufficiency Label | Count |
|---|---:|
| Sufficient | 11 |
| Partially sufficient | 10 |
| Insufficient | 3 |

The agent triggered retry retrieval for:

> **13 out of 24 questions**

This shows that the error-recovery mechanism worked. It did not blindly retry every question. It selectively retried questions where evidence was judged partially sufficient or insufficient.

### 10.2 Three-System Comparison

| System | Correctness | Evidence Quality | Grounding | Navigation | Avg Latency | Avg Tool Calls |
|---|---:|---:|---:|---:|---:|---:|
| Single-Pass RAG | 1.75 | 1.75 | 2.00 | 1.75 | 3.41s | 0.00 |
| Plan-First Agent | 1.92 | 1.92 | 2.00 | 1.92 | 5.89s | 4.00 |
| Plan + Checker Agent | 1.79 | 1.79 | 2.00 | 1.79 | 9.79s | 5.54 |

### 10.3 Interpretation

The Plan + Checker Agent made the workflow more inspectable and added error recovery, but it did not outperform the simpler Plan-First Agent.

It increased average latency to **9.79 seconds** and sometimes became overly cautious.

This was an important finding:

> **More agent steps do not automatically improve quality.**

In the current setup, the best tradeoff was the Plan-First Agent, not the more complex checker agent.

Therefore, I treat the Plan-First Agent as the main proposed system, while the Plan + Checker Agent is analyzed as an additional experiment showing that more agentic complexity does not automatically improve quality.

## 11. Failure Analysis

### 11.1 Incomplete Evidence

The most common failure pattern was incomplete evidence retrieval.

Sometimes the system found a generally relevant chunk, but not the most direct supporting evidence. This affected baseline, limitation, and experiment questions more than simple method questions.

### 11.2 Over-Cautious Answers

The Single-Pass RAG baseline sometimes said the evidence was insufficient even when partial evidence existed.

This behavior is safer than hallucinating, but it can reduce answer completeness.

### 11.3 PDF Parsing Noise

PDF parsing created some noisy or mis-ordered chunks, especially for:

- tables;
- figures;
- formulas;
- two-column layouts;
- reference sections.

This affected retrieval because noisy chunks could appear relevant but not contain direct evidence.

### 11.4 Retrieval Ranking Errors

Even when the correct page appeared in the top-5 results, the most direct evidence was not always ranked first.

This suggests that future work should explore reranking, dense retrieval, or hybrid retrieval.

### 11.5 Checker-Agent Overhead

The Plan + Checker Agent added a useful error recovery mechanism, but it also increased latency significantly. It did not improve quality over the simpler Plan-First Agent.

This shows that agentic complexity must be justified by measurable gains.

## 12. Iteration and Reflection

The project went through several iterations.

### 12.1 Basic Parsing and RAG

The first version parsed PDFs and ran basic retrieval. This proved that the pipeline could work, but the retrieved evidence was noisy.

### 12.2 Reference Filtering

References initially polluted retrieval results. Some retrieved chunks came from bibliography sections rather than from the paper's argument or experiments.

I added reference filtering and blacklist-style cleaning to remove these chunks.

### 12.3 Overly Aggressive Cleaning

At one point, the cleaning strategy was too strict and removed valid SELF-RAG chunks. I revised the filtering logic to avoid deleting useful content.

### 12.4 Query Expansion and Metadata

I added lightweight query expansion and used page/section metadata to improve retrieval quality.

### 12.5 Pilot Evaluation

Before running the full evaluation, I used a 6-question pilot set. This helped validate the pipeline, output format, and scoring rubric.

### 12.6 Full Evaluation

I then scaled to the full 24-question evaluation set and scored 48 outputs.

### 12.7 Additional Extensions

After the main comparison, I ran additional experiments:

1. TF-IDF vs BM25 retrieval comparison.
2. Page-level retrieval metrics using Hit@5, Recall@5, and MRR.
3. Plan + Checker Agent with evidence sufficiency checking and retry retrieval.

These extensions helped me better understand where the system failed and whether additional complexity improved the results.

The main lesson was:

> **The best system is not always the most complex one.**

In this project, the Plan-First Agent gave the best balance between quality, latency, and inspectability.

## 13. Ethics, Risks, and Limitations

### 13.1 Student Over-Reliance

Students may over-rely on generated answers instead of reading the paper. This could weaken learning and critical reading.

To reduce this risk, the system exposes:

- evidence chunks;
- page IDs;
- uncertainty statements;
- tool traces.

The goal is to support reading, not replace it.

### 13.2 Misleading Citations

A retrieved chunk may look trustworthy even when it only partially supports the answer. This could mislead users.

This is why evidence quality and grounding were evaluated separately.

### 13.3 PDF Parsing Bias

The assistant works better on clean text sections than on complex PDF regions such as tables, equations, figures, or two-column layouts.

This means the system may underperform on papers where key evidence is in tables or figures.

### 13.4 Small Corpus

The corpus contains only four papers. This is enough for a controlled class project but not enough to claim broad generalization.

The small corpus was chosen intentionally to support careful manual evaluation.

### 13.5 Retrieval Limitations

The main retrieval methods were TF-IDF and BM25. These keyword-based methods may miss semantically relevant evidence when the query uses wording different from the paper.

Future work should test dense retrieval, hybrid retrieval, and reranking.

### 13.6 Latency and Usability

The Plan-First Agent and Plan + Checker Agent are slower than Single-Pass RAG. In real use, users may prefer faster answers for simple questions and more careful agentic retrieval for evidence-heavy questions.

## 14. Future Work

Future work could improve this project in several directions.

### 14.1 Dense and Hybrid Retrieval

Future versions should compare TF-IDF, BM25, dense embeddings, and hybrid retrieval.

### 14.2 Reranking

Since some failures came from the best evidence not being ranked first, reranking could improve evidence selection.

### 14.3 Better Evidence Verification

The checker agent showed that evidence sufficiency checking is useful, but the current implementation did not improve overall quality. A stronger verifier or reranker could make retry retrieval more effective.

### 14.4 Larger Corpus

A larger corpus could test whether the approach generalizes beyond four papers.

### 14.5 User Study

A small user study could evaluate whether students actually find the evidence and tool traces helpful when reading papers.

### 14.6 Better PDF Parsing

Improved PDF parsing could better handle tables, figures, equations, and two-column layouts.

## 15. Conclusion

This project built and evaluated a paper reading assistant for evidence-grounded academic PDF question answering.

The main comparison showed that the Plan-First Agent improved correctness, evidence quality, and navigation usefulness compared with the Single-Pass RAG baseline. However, it increased latency.

The additional experiments showed that TF-IDF and BM25 had similar retrieval quality overall, and that page-level retrieval metrics can help separate retrieval quality from generation quality.

The Plan + Checker Agent introduced real error recovery by judging evidence sufficiency and retrying retrieval, but it did not outperform the simpler Plan-First Agent and increased latency substantially.

Overall, the project suggests that agentic retrieval is useful when the task requires locating and verifying evidence. However, more complex agent workflows are not automatically better. In the current system, the Plan-First Agent provides the best tradeoff between answer quality, evidence navigation, latency, and inspectability.

## 16. References

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR.

Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. arXiv preprint.

Schick, T., Dwivedi-Yu, J., Dessi, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv preprint.

PyMuPDF documentation. https://pymupdf.readthedocs.io/

scikit-learn documentation. https://scikit-learn.org/

rank-bm25 documentation / package. https://pypi.org/project/rank-bm25/

DeepSeek API documentation. https://api-docs.deepseek.com/

pandas documentation. https://pandas.pydata.org/

This project also used LLM assistance for debugging, code organization, evaluation planning, and report drafting.
