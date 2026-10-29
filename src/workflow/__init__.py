"""
Package src.workflow (Nhánh của Phạm Đức Trọng — Điều phối Đồ thị LangGraph).
"""

from src.workflow.edges import route_after_judge
from src.workflow.orchestrator import (
    LegalMultiAgentOrchestrator,
    run_legal_multi_agent_workflow,
)
from src.workflow.state import LegalWorkflowState

__all__ = [
    "LegalWorkflowState",
    "route_after_judge",
    "LegalMultiAgentOrchestrator",
    "run_legal_multi_agent_workflow",
]
