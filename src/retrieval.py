import json
from pathlib import Path
from typing import List, Optional, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CHUNKS_PATH = Path("data/chunks.json")


SECTION_BOOSTS = {
    "Abstract": 1.25,
    "Introduction": 1.15,
    "Method": 1.25,
    "Methods": 1.25,
    "Approach": 1.25,
    "Experiments": 1.15,
    "Results": 1.10,
    "Limitations": 1.20,
    "Conclusion": 1.10,
}


def expand_query(query: str) -> str:
    """
    Add lightweight query expansion terms for common paper-reading tasks.
    This helps TF-IDF retrieval find method/experiment/limitation sections.
    """
    q = query.lower()
    extra_terms = []

    if any(word in q for word in ["method", "approach", "proposed", "main idea"]):
        extra_terms.extend(["method", "approach", "model", "framework", "proposed", "we introduce"])

    if any(word in q for word in ["baseline", "compare", "compared", "against"]):
        extra_terms.extend(["baseline", "baselines", "compare", "comparison", "outperform", "experiments"])

    if any(word in q for word in ["experiment", "dataset", "evaluation", "setting"]):
        extra_terms.extend(["experiment", "experiments", "dataset", "datasets", "evaluation", "benchmark", "results"])

    if any(word in q for word in ["limitation", "limitations", "weakness", "future work"]):
        extra_terms.extend(["limitations", "limitation", "future work", "discussion", "fail", "cannot"])

    if any(word in q for word in ["retrieve", "retrieval", "when to retrieve"]):
        extra_terms.extend(["retrieve", "retrieval", "retriever", "on demand", "passages", "relevant"])

    if any(word in q for word in ["tool", "tools", "api"]):
        extra_terms.extend(["tool", "tools", "api", "calculator", "search engine", "qa system", "translation", "calendar"])

    if any(word in q for word in ["reasoning", "acting", "react"]):
        extra_terms.extend(["reasoning", "acting", "actions", "thoughts", "observations", "interleaved"])

    return query + " " + " ".join(extra_terms)


class ChunkRetriever:
    def __init__(self, chunks_path: Path = CHUNKS_PATH):
        if not chunks_path.exists():
            raise FileNotFoundError(f"Could not find {chunks_path}. Run chunk_documents.py first.")

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.texts = [
            f"{c.get('paper_id', '')} {c.get('section', '')} page {c.get('page_start', '')} {c.get('text', '')}"
            for c in self.chunks
        ]

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=1
        )

        self.matrix = self.vectorizer.fit_transform(self.texts)

    def search(
        self,
        query: str,
        paper_id: Optional[str] = None,
        top_k: int = 5,
        min_score: float = 0.001
    ) -> List[Dict]:
        if not query.strip():
            return []

        expanded_query = expand_query(query)
        query_vec = self.vectorizer.transform([expanded_query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        results = []

        for idx, base_score in enumerate(scores):
            chunk = self.chunks[idx]

            if paper_id is not None and chunk["paper_id"] != paper_id:
                continue

            section = chunk.get("section", "Unknown")
            boost = SECTION_BOOSTS.get(section, 1.0)

            # Slight boost for early pages because abstract/introduction often define the method.
            page = chunk.get("page_start") or 999
            page_boost = 1.08 if page <= 3 else 1.0

            final_score = float(base_score * boost * page_boost)

            if final_score < min_score:
                continue

            results.append({
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "section": chunk["section"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "score": final_score,
                "text": chunk["text"],
                "preview": chunk["text"][:500]
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def read_chunk(self, chunk_id: str) -> Optional[Dict]:
        for chunk in self.chunks:
            if chunk["chunk_id"] == chunk_id:
                return chunk
        return None


def main():
    retriever = ChunkRetriever()

    test_queries = [
        ("rag_2020", "What is the main method proposed in this paper?"),
        ("self_rag_2023", "How does SELF-RAG decide when to retrieve?"),
        ("react_2022", "How does ReAct combine reasoning and acting?"),
        ("toolformer_2023", "What tools does Toolformer use?")
    ]

    for paper_id, query in test_queries:
        print("=" * 100)
        print(f"Paper: {paper_id}")
        print(f"Query: {query}")
        print("-" * 100)

        results = retriever.search(query=query, paper_id=paper_id, top_k=3)

        if not results:
            print("No results found.")
            continue

        for i, r in enumerate(results, start=1):
            print(f"\nRank {i}")
            print(f"chunk_id: {r['chunk_id']}")
            print(f"section: {r['section']}")
            print(f"page: {r['page_start']}")
            print(f"score: {r['score']:.4f}")
            print(f"preview: {r['preview'][:500]}")


if __name__ == "__main__":
    main()
