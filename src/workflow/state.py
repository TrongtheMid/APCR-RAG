"""
NHÁNH PHẠM ĐỨC TRỌNG — Định nghĩa Trạng thái Đồ thị LangGraph (LegalWorkflowState).
Hỗ trợ đầy đủ vòng lặp Self-Revision, lịch sử hội thoại đa lượt (chat_history),
nhánh hỏi ngược làm rõ (clarification) và nhật ký thực thi từng nút (workflow_trace).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class LegalWorkflowState(TypedDict, total=False):
    """Trạng thái xuyên suốt đồ thị LangGraph của hệ thống Đa tác tử APCR-RAG."""

    query: str
    effective_query: str
    chat_history: List[Dict[str, str]]
    phan_nhom_filter: Optional[str]
    linh_vuc_filter: Optional[str]
    giai_doan_filter: Optional[str]
    effective_date: Optional[str]
    only_active: bool
    simulate_initial_error: bool

    # Kết quả từ AGENT 01 — RETRIEVER
    routing_info: Dict[str, Any]
    retrieved_items: List[Dict[str, Any]]
    expired_alerts: List[Dict[str, Any]]
    primary_procedure: Optional[Dict[str, Any]]

    # Kết quả từ AGENT 02 — CONSULTANT (hỗ trợ nhánh Clarification)
    irac_draft: Optional[Dict[str, Any]]
    needs_clarification: bool
    clarification_question: Optional[str]

    # Kết quả từ AGENT 03 — JUDGE
    verification_result: Optional[Dict[str, Any]]
    critique_feedback: Optional[str]

    # Quản lý vòng lặp phản biện (tối đa 2 vòng)
    revision_count: int
    max_revisions: int
    revision_history: List[Dict[str, Any]]

    # Theo dõi tiến trình Streaming & Audit Trace
    workflow_trace: List[Dict[str, Any]]
    execution_engine: str

    # Đầu ra cuối cùng cho giao diện Streamlit
    graph_data: Optional[Dict[str, Any]]
    final_status: str
    total_latency_ms: float
