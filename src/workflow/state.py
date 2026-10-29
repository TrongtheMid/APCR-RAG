"""
NHÁNH PHẠM ĐỨC TRỌNG — Định nghĩa Trạng thái Đồ thị LangGraph (LegalWorkflowState).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class LegalWorkflowState(TypedDict, total=False):
    """Trạng thái xuyên suốt đồ thị LangGraph của hệ thống Đa tác tử APCR-RAG."""
    query: str
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

    # Kết quả từ AGENT 02 — CONSULTANT
    irac_draft: Optional[Dict[str, Any]]

    # Kết quả từ AGENT 03 — JUDGE
    verification_result: Optional[Dict[str, Any]]
    critique_feedback: Optional[str]

    # Quản lý vòng lặp phản biện (tối đa 2 vòng)
    revision_count: int
    max_revisions: int
    revision_history: List[Dict[str, Any]]

    # Đầu ra cuối cùng cho giao diện Streamlit
    graph_data: Optional[Dict[str, Any]]
    final_status: str
    total_latency_ms: float
