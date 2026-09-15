"""Intervention Governance Sub-Agent – Action & Governance Agent determining factory actions & HITL gates."""

from typing import Dict, Any


class InterventionGovernanceAgent:
    """Specialist sub-agent for factory intervention decisions and Human-in-the-Loop governance."""

    def __init__(self):
        pass

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Node 3: Formulates safe factory intervention action proposal."""
        defect_class = state["defect_class"]
        consecutive_count = state["consecutive_count"]

        action = "HALT_LINE" if consecutive_count >= 3 and defect_class in ["short_circuit", "missing_component", "short"] else "ROUTE_REWORK"
        return {
            "proposed_action": action,
            "requires_hitl": True,
            "approval_status": "PENDING"
        }
