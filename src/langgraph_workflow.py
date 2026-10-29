"""
Facade tương thích ngược cho src.workflow (Nhánh của Phạm Đức Trọng).
"""

from src.workflow import (
    LegalMultiAgentOrchestrator,
    LegalWorkflowState,
    route_after_judge,
    run_legal_multi_agent_workflow,
)

__all__ = [
    "LegalWorkflowState",
    "route_after_judge",
    "LegalMultiAgentOrchestrator",
    "run_legal_multi_agent_workflow",
]
