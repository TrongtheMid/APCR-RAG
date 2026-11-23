"""
Facade tương thích ngược cho src.workflow (Nhánh của Phạm Đức Trọng).
"""

from src.workflow import (
    LegalMultiAgentOrchestrator,
    LegalWorkflowState,
    contextualize_followup_query,
    get_orchestrator,
    route_after_consultant,
    route_after_judge,
    run_legal_multi_agent_workflow,
    stream_legal_multi_agent_workflow,
)

__all__ = [
    "LegalWorkflowState",
    "route_after_consultant",
    "route_after_judge",
    "contextualize_followup_query",
    "LegalMultiAgentOrchestrator",
    "get_orchestrator",
    "run_legal_multi_agent_workflow",
    "stream_legal_multi_agent_workflow",
]
