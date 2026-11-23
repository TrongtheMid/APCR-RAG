"""
NHÁNH PHẠM ĐỨC TRỌNG — Các Cạnh Điều kiện (Conditional Edges) trên Đồ thị LangGraph:
- route_after_consultant: Hỗ trợ rẽ nhánh hỏi ngược làm rõ (clarify) hoặc đi tới Judge.
- route_after_judge: Điều phối accept (PASS) / revise (Self-Revision Loop) / fallback.
"""

from __future__ import annotations

from typing import Literal

from src.workflow.state import LegalWorkflowState


def route_after_consultant(
    state: LegalWorkflowState,
) -> Literal["clarify", "judge"]:
    """
    Quyết định luồng đi sau Consultant Agent:
    - 'clarify': Nếu thiếu dữ kiện quan trọng và Consultant bật cờ `needs_clarification`,
      dừng sớm để hỏi ngược làm rõ với người dân.
    - 'judge': Chuyển bản nháp IRAC sang Judge Agent thẩm định 2 lớp.
    """
    if state.get("needs_clarification", False) and state.get("clarification_question"):
        return "clarify"
    return "judge"


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
