"""
Package src.workflow (Nhánh của Phạm Đức Trọng — Điều phối Đồ thị LangGraph).
"""

from src.workflow.edges import route_after_consultant, route_after_judge
from src.workflow.intent_router import build_conversational_reply, classify_chat_intent
from src.workflow.orchestrator import (
    LegalMultiAgentOrchestrator,
    contextualize_followup_query,
    get_orchestrator,
    run_legal_multi_agent_workflow,
    stream_legal_multi_agent_workflow,
)
from src.workflow.state import LegalWorkflowState

__all__ = [
    "LegalWorkflowState",
    "route_after_consultant",
    "route_after_judge",
    "classify_chat_intent",
    "build_conversational_reply",
    "contextualize_followup_query",
    "LegalMultiAgentOrchestrator",
    "get_orchestrator",
    "run_legal_multi_agent_workflow",
    "stream_legal_multi_agent_workflow",
]
