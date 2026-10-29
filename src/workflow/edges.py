"""
NHÁNH PHẠM ĐỨC TRỌNG — Các Cạnh Điều kiện (Conditional Edges) trên Đồ thị LangGraph.
"""

from __future__ import annotations

from typing import Literal

from src.workflow.state import LegalWorkflowState


def route_after_judge(
    state: LegalWorkflowState,
    default_max_revisions: int = 2,
) -> Literal["accept", "revise", "fallback"]:
    """
    Quyết định luồng đi sau khi Judge Agent chấm điểm:
    - 'accept': Bản nháp đạt chuẩn (PASS) hoặc đã từ chối an toàn -> Kết thúc (END).
    - 'revise': Phát hiện lỗi (FAIL) và còn lượt sửa (revision_count <= max_revisions) -> Quay lại Consultant.
    - 'fallback': Vượt quá số vòng tối đa (2 vòng) -> Chuyển sang nút cảnh báo an toàn rồi kết thúc.
    """
    ver = state.get("verification_result") or {}
    if ver.get("passed", False):
        return "accept"

    if state.get("revision_count", 0) <= state.get("max_revisions", default_max_revisions):
        return "revise"

    return "fallback"
