import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CHUNKS_PATH = Path("data/chunks.json")
QUESTIONS_PATH = Path("data/evaluation_questions.csv")
OUTPUT_PATH = Path("outputs/retrieval_comparison_outputs.csv")

TOP_K = 5


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return text.split()


def load_data():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    questions = pd.read_csv(QUESTIONS_PATH)
    return chunks, questions


class TfidfRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.texts = [
            f"{c.get('paper_id', '')} {c.get('section', '')} page {c.get('page_start', '')} {c.get('text', '')}"
            for c in chunks
        ]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=1
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(self, query, paper_id=None, top_k=5):
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        results = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            if paper_id and chunk["paper_id"] != paper_id:
                continue
            results.append((idx, float(score)))

        results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        return [format_result(self.chunks[idx], score) for idx, score in results]


class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized = [
            tokenize(f"{c.get('paper_id', '')} {c.get('section', '')} {c.get('text', '')}")
            for c in chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query, paper_id=None, top_k=5):
        q_tokens = tokenize(query)
        scores = self.bm25.get_scores(q_tokens)

        results = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            if paper_id and chunk["paper_id"] != paper_id:
                continue
            results.append((idx, float(score)))

        results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        return [format_result(self.chunks[idx], score) for idx, score in results]


class DenseRetriever:
    def __init__(self, chunks):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.texts = [
            f"{c.get('paper_id', '')} {c.get('section', '')} page {c.get('page_start', '')} {c.get('text', '')}"
            for c in chunks
        ]

        print("Encoding chunks for dense retrieval...")
        self.embeddings = self.model.encode(
            self.texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def search(self, query, paper_id=None, top_k=5):
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, q_emb)

        results = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            if paper_id and chunk["paper_id"] != paper_id:
                continue
            results.append((idx, float(score)))

        results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
        return [format_result(self.chunks[idx], score) for idx, score in results]


def format_result(chunk, score):
    return {
        "chunk_id": chunk["chunk_id"],
        "paper_id": chunk["paper_id"],
        "section": chunk.get("section", "Unknown"),
        "page_start": chunk.get("page_start", ""),
        "score": round(float(score), 4),
        "preview": chunk["text"][:500]
    }


def main():
    chunks, questions = load_data()

    retrievers = {
        "tfidf": TfidfRetriever(chunks),
        "bm25": BM25Retriever(chunks),
    }

    # Dense retrieval is useful but optional. If the model download fails, skip it.
    try:
        retrievers["dense_miniLM"] = DenseRetriever(chunks)
    except Exception as e:
        print("Dense retriever failed to initialize. Continuing with TF-IDF and BM25.")
        print("Error:", e)

    rows = []

    for _, q in questions.iterrows():
        question_id = q["question_id"]
        paper_id = q["paper_id"]
        task_type = q["task_type"]
        question = q["question"]

        print("=" * 100)
        print(question_id, paper_id, task_type)
        print(question)

        for method_name, retriever in retrievers.items():
            results = retriever.search(
                query=question,
                paper_id=paper_id,
                top_k=TOP_K
            )

            rows.append({
                "question_id": question_id,
                "paper_id": paper_id,
                "task_type": task_type,
                "question": question,
                "retrieval_method": method_name,
                "top_k": TOP_K,
                "retrieved_evidence_json": json.dumps(results, ensure_ascii=False),
                "top1_chunk_id": results[0]["chunk_id"] if results else "",
                "top1_page": results[0]["page_start"] if results else "",
                "top1_preview": results[0]["preview"] if results else "",
                "retrieval_relevance_score": "",
                "direct_evidence_found": "",
                "notes": ""
            })

            if results:
                print(f"[{method_name}] top1:", results[0]["chunk_id"], "page", results[0]["page_start"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved retrieval comparison outputs to {OUTPUT_PATH}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
