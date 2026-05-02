import streamlit as st
import json
import time
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ── path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from retrieval import ChunkRetriever

load_dotenv()

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Paper Reading Assistant",
    page_icon="📄",
    layout="wide",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* white background */
.stApp {
    background-color: #fafaf8;
    color: #1a1a1a;
}

/* sidebar */
[data-testid="stSidebar"] {
    background-color: #f2f1ed;
    border-right: 1px solid #ddddd8;
}

/* title */
h1 {
    font-family: 'Source Serif 4', serif !important;
    font-size: 1.7rem !important;
    letter-spacing: -0.02em;
    color: #111111;
    border-bottom: 2px solid #1a5ccc;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #444450;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 1.5rem;
}

/* answer box */
.answer-box {
    background: #ffffff;
    border: 1px solid #ddddd8;
    border-left: 4px solid #1a5ccc;
    border-radius: 4px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
    font-size: 0.95rem;
    line-height: 1.8;
    white-space: pre-wrap;
    color: #1a1a1a;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* evidence box */
.evidence-box {
    background: #f7f7f4;
    border: 1px solid #ddddd8;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.83rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #3a3a4a;
}

/* tool trace box */
.trace-box {
    background: #f2f1ed;
    border: 1px solid #d8d8d2;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.8rem;
    font-family: 'IBM Plex Mono', monospace;
    color: #505060;
    white-space: pre-wrap;
}

/* metric badges */
.metric-row {
    display: flex;
    gap: 0.8rem;
    margin: 0.8rem 0;
    flex-wrap: wrap;
}
.metric-badge {
    background: #eeeee9;
    border: 1px solid #d8d8d2;
    border-radius: 3px;
    padding: 0.25rem 0.65rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #555560;
}
.metric-badge span {
    color: #1a5ccc;
    font-weight: 600;
}

/* mode radio */
.stRadio > label {
    color: #333340 !important;
}

/* button */
.stButton > button {
    background: #1a5ccc;
    color: #ffffff;
    border: none;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    padding: 0.5rem 1.5rem;
    letter-spacing: 0.04em;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: #1448aa;
}

/* text input */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #d0d0cc !important;
    color: #1a1a1a !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

/* selectbox */
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #d0d0cc !important;
    color: #1a1a1a !important;
}

/* spinner */
.stSpinner > div {
    border-top-color: #1a5ccc !important;
}

/* divider */
hr {
    border-color: #ddddd8;
}
</style>
""", unsafe_allow_html=True)

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "deepseek-chat"
TOP_K = 5
TOP_K_FOLLOWUP = 3

PAPER_LABELS = {
    "rag_2020":        "RAG (Lewis et al., 2020)",
    "react_2022":      "ReAct (Yao et al., 2022)",
    "self_rag_2023":   "SELF-RAG (Asai et al., 2023)",
    "toolformer_2023": "Toolformer (Schick et al., 2023)",
}

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_retriever():
    return ChunkRetriever()

def get_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.error("DEEPSEEK_API_KEY not found in .env file.")
        st.stop()
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def format_context(results):
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"[Evidence {i}]\nchunk_id: {r['chunk_id']}\n"
            f"paper_id: {r['paper_id']}\nsection: {r['section']}\n"
            f"page: {r['page_start']}\ntext:\n{r['text']}"
        )
    return "\n\n".join(blocks)

def call_llm(client, messages):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
    )
    return resp.choices[0].message.content

# ── RAG baseline ──────────────────────────────────────────────────────────────
def run_rag(client, retriever, question, paper_id):
    results = retriever.search(query=question, paper_id=paper_id, top_k=TOP_K)
    context = format_context(results)
    prompt = f"""You are answering a question about an academic paper.
Use ONLY the provided evidence. Do not use outside knowledge.
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
..."""
    t0 = time.time()
    answer = call_llm(client, [
        {"role": "system", "content": "You are a careful academic reading assistant. Answer only from provided evidence."},
        {"role": "user",   "content": prompt},
    ])
    latency = round(time.time() - t0, 2)
    return answer, results, latency, []

# ── Plan-first agent ──────────────────────────────────────────────────────────
def run_agent(client, retriever, question, paper_id, task_type="method"):
    t0 = time.time()
    tool_trace = []

    # Step 1: generate plan
    plan_prompt = f"""You are a careful academic paper reading assistant.
Your job is to create a short retrieval plan before answering.
Paper ID: {paper_id}
Task type: {task_type}
Question: {question}

Write a short search plan in this exact format:
Search Plan:
1. What kind of information is needed?
2. Which paper sections are likely relevant?
3. What search query should be used?
Search Query:
..."""
    plan_text = call_llm(client, [
        {"role": "system", "content": "Create concise retrieval plans."},
        {"role": "user",   "content": plan_prompt},
    ])

    # extract search query
    search_query = question
    marker = "Search Query:"
    if marker.lower() in plan_text.lower():
        for line in plan_text.splitlines():
            if line.strip().lower().startswith(marker.lower()):
                q = line.split(":", 1)[-1].strip()
                if q:
                    search_query = q
                    break

    tool_trace.append({"step": "plan_search", "plan": plan_text, "extracted_query": search_query})

    # Step 2: retrieve
    initial  = retriever.search(query=search_query, paper_id=paper_id, top_k=TOP_K)
    followup = retriever.search(query=question,      paper_id=paper_id, top_k=TOP_K_FOLLOWUP)
    seen, merged = set(), []
    for r in initial + followup:
        if r["chunk_id"] not in seen:
            merged.append(r); seen.add(r["chunk_id"])
    final_results = merged[:TOP_K]

    tool_trace.append({"step": "search_sections", "query": search_query,
                       "chunks": [r["chunk_id"] for r in initial]})
    tool_trace.append({"step": "search_sections (followup)", "query": question,
                       "chunks": [r["chunk_id"] for r in followup]})
    tool_trace.append({"step": "read_chunks",
                       "chunks": [r["chunk_id"] for r in final_results]})

    # Step 3: answer
    context = format_context(final_results)
    answer_prompt = f"""You are answering a question about an academic paper.
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
..."""
    answer = call_llm(client, [
        {"role": "system", "content": "You are a careful academic reading assistant. Answer only from provided evidence."},
        {"role": "user",   "content": answer_prompt},
    ])
    latency = round(time.time() - t0, 2)
    return answer, final_results, latency, tool_trace

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙ Settings")
    paper_id = st.selectbox(
        "Select paper",
        options=list(PAPER_LABELS.keys()),
        format_func=lambda k: PAPER_LABELS[k],
    )
    mode = st.radio(
        "System mode",
        options=["Single-Pass RAG", "Plan-First Agent"],
        index=0,
    )
    st.markdown("---")
    st.markdown("""
<div style='font-size:0.78rem; color:#555566; line-height:1.6;'>
<b>Single-Pass RAG</b><br>
Retrieve top-k chunks once → answer directly.<br><br>
<b>Plan-First Agent</b><br>
Generate search plan → retrieve → answer with tool trace.
</div>
""", unsafe_allow_html=True)

# ── main area ─────────────────────────────────────────────────────────────────
st.markdown("# 📄 Paper Reading Assistant")
st.markdown(
    f"**Paper:** {PAPER_LABELS.get(paper_id, paper_id)} &nbsp;|&nbsp; "
    f"**Mode:** {mode}",
    unsafe_allow_html=True,
)

question = st.text_area(
    "Ask a question about the paper",
    placeholder="e.g. What is the main method proposed? What baselines are compared?",
    height=90,
)

ask_btn = st.button("Ask →")

if ask_btn:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        retriever = load_retriever()
        client    = get_client()

        with st.spinner("Thinking..."):
            if mode == "Single-Pass RAG":
                answer, results, latency, trace = run_rag(client, retriever, question, paper_id)
                num_calls = 0
            else:
                answer, results, latency, trace = run_agent(client, retriever, question, paper_id)
                num_calls = 4

        # ── metrics bar
        st.markdown(f"""
<div class="metric-row">
  <div class="metric-badge">mode <span>{mode}</span></div>
  <div class="metric-badge">latency <span>{latency}s</span></div>
  <div class="metric-badge">chunks retrieved <span>{len(results)}</span></div>
  <div class="metric-badge">tool calls <span>{num_calls}</span></div>
</div>
""", unsafe_allow_html=True)

        # ── answer
        st.markdown("### Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        # ── retrieved chunks
        with st.expander("📎 Retrieved Chunks", expanded=False):
            for i, r in enumerate(results, 1):
                st.markdown(f"""
<div class="evidence-box">
<b>Rank {i}</b> &nbsp;·&nbsp; {r['chunk_id']} &nbsp;·&nbsp; section: {r['section']} &nbsp;·&nbsp; page: {r['page_start']} &nbsp;·&nbsp; score: {r['score']:.4f}<br><br>
{r['text'][:600]}{'…' if len(r['text']) > 600 else ''}
</div>
""", unsafe_allow_html=True)

        # ── tool trace (agent only)
        if trace:
            with st.expander("🔍 Tool Trace", expanded=True):
                for step in trace:
                    label = step.get("step", "step")
                    if label == "plan_search":
                        st.markdown(f"""
<div class="trace-box">
<b>① plan_search</b><br>
extracted query: <span style='color:#4f8ef7'>{step['extracted_query']}</span><br><br>
{step['plan']}
</div>
""", unsafe_allow_html=True)
                    elif "search_sections" in label:
                        chunks_str = ", ".join(step.get("chunks", []))
                        st.markdown(f"""
<div class="trace-box">
<b>② {label}</b><br>
query: <span style='color:#4f8ef7'>{step['query']}</span><br>
returned: {chunks_str}
</div>
""", unsafe_allow_html=True)
                    elif label == "read_chunks":
                        chunks_str = ", ".join(step.get("chunks", []))
                        st.markdown(f"""
<div class="trace-box">
<b>③ read_chunks</b><br>
{chunks_str}
</div>
""", unsafe_allow_html=True)