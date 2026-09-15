"""SOP Research Sub-Agent – Research Agent querying standard operating procedure manuals."""

from typing import Dict, Any, List
from ..rag import SOPRetriever


class SOPResearchAgent:
    """Specialist sub-agent for factory Standard Operating Procedure (SOP) retrieval."""

    def __init__(self, sops_dir: str = "data/sops"):
        self.retriever = SOPRetriever(sops_dir=sops_dir)

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 1: Retrieves matching SOP documents from knowledge base."""
        defect_class = state["defect_class"]
        sop_results = self.retriever.search(defect_class, top_k=2)
        sop_context = "\n\n".join([f"[{r['sop_id']}]:\n{r['snippet']}" for r in sop_results])
        citations = [r['sop_id'] for r in sop_results]
        return {
            "sop_citations": citations,
            "sop_context": sop_context,
        }
