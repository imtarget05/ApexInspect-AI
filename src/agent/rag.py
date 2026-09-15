import os
import glob
from typing import List, Dict, Any

class SOPRetriever:
    """
    Retrieves Standard Operating Procedures (SOPs) from data/sops/*.md.
    Provides fast, deterministic hybrid/keyword retrieval with zero mandatory external dependencies.
    """

    def __init__(self, sops_dir: str = "data/sops"):
        self.sops_dir = sops_dir
        self.documents: List[Dict[str, str]] = []
        self._load_documents()

    def _load_documents(self):
        """Loads all markdown SOP documents into memory."""
        pattern = os.path.join(self.sops_dir, "*.md")
        files = glob.glob(pattern)
        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.documents.append({
                        "id": filename.split(".")[0],
                        "filename": filename,
                        "content": content
                    })
            except Exception as e:
                print(f"[RAG] Error reading {filepath}: {e}")

        print(f"[RAG] Loaded {len(self.documents)} SOP documents into Knowledge Base.")

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Performs keyword and token overlap retrieval across SOP manuals.
        Returns top matched snippets with document ID citations.
        """
        if not self.documents:
            return []

        query_tokens = set(query.lower().replace("_", " ").split())
        scored_docs = []

        for doc in self.documents:
            content_lower = doc["content"].lower()
            score = 0
            for token in query_tokens:
                if len(token) > 2 and token in content_lower:
                    score += content_lower.count(token)

            if score > 0:
                scored_docs.append((score, doc))

        # Sort descending by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored_docs[:top_k]:
            results.append({
                "sop_id": doc["id"],
                "filename": doc["filename"],
                "score": score,
                "snippet": doc["content"][:1200]
            })

        return results
