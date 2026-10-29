"""
Lược đồ Kết quả Thẩm định Chéo 2 Lớp (JudgeVerificationResult) của Judge / Critic Agent.
"""

from typing import List
from pydantic import BaseModel, Field


class JudgeVerificationResult(BaseModel):
    """Kết quả thẩm định chéo 2 lớp (Lớp 1: Code Tất định + Lớp 2: LLM Phản biện ngữ nghĩa)."""
    passed: bool = Field(..., description="True nếu vượt qua cả Lớp 1 và Lớp 2 (PASS), False nếu cần sửa lại (FAIL)")
    layer1_passed: bool = Field(..., description="Kết quả kiểm tra tất định bằng Python (Exact Match số liệu & mã)")
    layer1_errors: List[str] = Field(
        default_factory=list,
        description="Danh sách lỗi sai lệch số liệu (lệ phí, thời hạn, mã thủ tục, văn bản hết hiệu lực) ở Lớp 1"
    )
    layer2_passed: bool = Field(..., description="Kết quả thẩm định điều kiện bắt buộc & phân cấp thẩm quyền ở Lớp 2")
    layer2_errors: List[str] = Field(
        default_factory=list,
        description="Danh sách lỗi bỏ sót điều kiện tiên quyết hoặc nhầm lẫn thẩm quyền ở Lớp 2"
    )
    critique_feedback: str = Field(
        default="",
        description="Chỉ dẫn sửa lỗi chi tiết gửi ngược lại cho Consultant Agent trong vòng lặp Self-Revision"
    )
