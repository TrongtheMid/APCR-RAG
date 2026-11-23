"""
Lược đồ Kết quả Thẩm định Chéo 2 Lớp (JudgeVerificationResult) của Judge / Critic Agent.
Hỗ trợ cả chỉ số đối soát Regex văn xuôi (Lớp 1) và chỉ số độ phủ mệnh đề Claim-level (Lớp 2).
"""

from typing import List
from pydantic import BaseModel, Field


class JudgeVerificationResult(BaseModel):
    """Kết quả thẩm định chéo 2 lớp (Lớp 1: Code Tất định + Lớp 2: Phản biện Ngữ nghĩa Claim-level)."""

    passed: bool = Field(
        ..., description="True nếu vượt qua cả Lớp 1 và Lớp 2 (PASS), False nếu cần sửa lại (FAIL)"
    )
    layer1_passed: bool = Field(
        ..., description="Kết quả kiểm tra tất định bằng Python (Exact Match metadata & Regex văn xuôi)"
    )
    layer1_errors: List[str] = Field(
        default_factory=list,
        description="Danh sách lỗi sai lệch số liệu (lệ phí, thời hạn, mã thủ tục, mâu thuẫn văn xuôi) ở Lớp 1",
    )
    layer2_passed: bool = Field(
        ..., description="Kết quả thẩm định điều kiện bắt buộc Claim-level & phân cấp thẩm quyền ở Lớp 2"
    )
    layer2_errors: List[str] = Field(
        default_factory=list,
        description="Danh sách lỗi bỏ sót điều kiện tiên quyết, mệnh đề cấm hoặc nhầm lẫn thẩm quyền ở Lớp 2",
    )
    critique_feedback: str = Field(
        default="",
        description="Chỉ dẫn sửa lỗi chi tiết gửi ngược lại cho Consultant Agent trong vòng lặp Self-Revision",
    )

    # Các trường chẩn đoán mở rộng (tương thích ngược nhờ default values)
    layer1_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Điểm đối soát tất định Lớp 1 (0.0 - 1.0)"
    )
    layer2_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Điểm thẩm định ngữ nghĩa & mệnh đề Lớp 2 (0.0 - 1.0)"
    )
    prose_extracted_fees: List[int] = Field(
        default_factory=list,
        description="Danh sách số tiền lệ phí (VNĐ) bóc tách bằng Regex từ phần văn xuôi IRAC",
    )
    prose_extracted_days: List[int] = Field(
        default_factory=list,
        description="Danh sách số ngày thời hạn bóc tách bằng Regex từ phần văn xuôi IRAC",
    )
    claim_coverage_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Tỷ lệ mệnh đề điều kiện bắt buộc được phủ trong bản nháp"
    )
    verified_claims_count: int = Field(
        default=0, ge=0, description="Số mệnh đề điều kiện bắt buộc đã được xác minh"
    )
    total_claims_count: int = Field(
        default=0, ge=0, description="Tổng số mệnh đề điều kiện bắt buộc trích xuất từ JSON gốc"
    )
