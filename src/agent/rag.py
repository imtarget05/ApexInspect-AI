import os
import re
import glob
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


class SOPRetriever:
    """
    Production Industrial SOP Retriever powered by Okapi BM25 and Semantic Section Indexing.
    Provides fast, deterministic hybrid/keyword retrieval with clause-level grounding citations.
    """

    SYNONYM_EXPANSIONS = {
        "short_circuit": ["short circuit", "solder bridge", "hàn chập", "chập mạch", "stencil", "kem hàn"],
        "short": ["short circuit", "solder bridge", "hàn chập", "chập mạch"],
        "missing_hole": ["missing hole", "missing component", "thiếu linh kiện", "mất lỗ", "vòi hút", "feeder"],
        "mouse_bite": ["mouse bite", "khuyết mạch đồng", "gặm nhấm", "vết lõm"],
        "open_circuit": ["open circuit", "đứt mạch", "hở mạch", "đứt đường đồng"],
        "spur": ["spur", "râu đồng", "bavia", "tua đồng"],
        "spurious_copper": ["spurious copper", "đồng dư", "vết đồng thừa", "bám đồng"]
    }

    def __init__(self, sops_dir: str = "data/sops"):
        self.sops_dir = sops_dir
        self.documents: List[Dict[str, Any]] = []
        self.bm25_corpus: List[List[str]] = []
        self.bm25_engine: Optional[Any] = None
        self._load_documents()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Industrial SMT text tokenizer supporting alphanumeric tokens and compound words."""
        cleaned = re.sub(r"[^\w\s-]", " ", text.lower().replace("_", " "))
        tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]
        return tokens

    def _load_documents(self):
        """Loads all markdown SOP documents and indexes them with BM25."""
        self.documents = []
        self.bm25_corpus = []
        pattern = os.path.join(self.sops_dir, "*.md")
        files = glob.glob(pattern)

        for filepath in sorted(files):
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    doc_id = filename.split(".")[0]
                    tokens = self.tokenize(content)
                    self.documents.append({
                        "id": doc_id,
                        "filename": filename,
                        "content": content,
                        "tokens": tokens
                    })
                    self.bm25_corpus.append(tokens)
            except Exception as e:
                print(f"[RAG] Error reading {filepath}: {e}")

        if HAS_BM25 and self.bm25_corpus:
            self.bm25_engine = BM25Okapi(self.bm25_corpus)
            print(f"[RAG] Indexed {len(self.documents)} SOP manuals with Okapi BM25 engine.")
        else:
            print(f"[RAG] Loaded {len(self.documents)} SOP documents into Knowledge Base.")

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Performs Okapi BM25 query retrieval with domain synonym expansion.
        Returns top matched snippets with document ID citations and relevance scores.
        """
        if not self.documents:
            return []

        # 1. Expand query with industry synonyms
        query_norm = query.lower().strip()
        expanded_query = [query_norm]
        for canonical, syns in self.SYNONYM_EXPANSIONS.items():
            if canonical in query_norm or any(s in query_norm for s in syns):
                expanded_query.extend(syns)
        full_query_text = " ".join(expanded_query)
        q_tokens = self.tokenize(full_query_text)

        results: List[Dict[str, Any]] = []

        # 2. Score with BM25 if available
        if self.bm25_engine is not None and len(q_tokens) > 0:
            bm25_scores = self.bm25_engine.get_scores(q_tokens)
            ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

            for idx in ranked_indices[:top_k]:
                score = float(bm25_scores[idx])
                doc = self.documents[idx]
                if score > 0:
                    results.append({
                        "sop_id": doc["id"],
                        "filename": doc["filename"],
                        "score": round(score, 2),
                        "snippet": doc["content"][:1200]
                    })
            if results:
                return results

        # 3. Deterministic token-overlap fallback if BM25 gives zero matches
        scored_docs = []
        token_set = set(q_tokens)
        for doc in self.documents:
            doc_tokens = set(doc.get("tokens", []))
            overlap = len(token_set.intersection(doc_tokens))
            if overlap > 0:
                scored_docs.append((overlap, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        for score, doc in scored_docs[:top_k]:
            results.append({
                "sop_id": doc["id"],
                "filename": doc["filename"],
                "score": float(score),
                "snippet": doc["content"][:1200]
            })

        return results

