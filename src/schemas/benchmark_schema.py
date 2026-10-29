"""
Lược đồ Câu hỏi Kiểm thử Thực nghiệm (BenchmarkQuestion) trong Bộ 50 câu hỏi Benchmark.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class BenchmarkQuestion(BaseModel):
    """Cấu trúc một ca kiểm thử trong Bộ 50 câu hỏi Benchmark."""
    question_id: str = Field(..., description="Mã câu hỏi (VD: BM-HC-01, BM-HS-26)")
    question: str = Field(..., description="Nội dung câu hỏi thực tế của người dân")
    phan_nhom: Literal["Thủ tục hành chính", "Tố tụng hình sự"]
    linh_vuc: str
    difficulty: Literal["Tra cứu đơn", "Tình huống nhiều điều kiện", "Gài bẫy"]
    expected_ma_thu_tuc: str = Field(..., description="Mã thủ tục chuẩn (Ground Truth)")
    expected_law_ids: List[str] = Field(default_factory=list)
    expected_muc_phi_vnd: int
    expected_so_ngay_toi_da: int
    ground_truth_summary: str = Field(..., description="Đáp án chuẩn tóm tắt để chấm Faithfulness/RAGAS")
    trap_note: Optional[str] = Field(default=None, description="Ghi chú điểm gài bẫy (nếu thuộc nhóm Gài bẫy)")
