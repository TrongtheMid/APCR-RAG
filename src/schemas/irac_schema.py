"""
Lược đồ Cấu trúc Đầu ra Lập luận Pháp lý IRAC (Issue - Rule - Application - Conclusion)
và Siêu dữ liệu Trích dẫn (CitationMetadata) của Consultant Agent.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class CitationMetadata(BaseModel):
    """Các trường dữ liệu định lượng trích xuất từ bản nháp để Judge Agent Lớp 1 đối soát tất định."""
    ma_thu_tuc: str = Field(..., description="Mã thủ tục chính được viện dẫn (VD: TTHS-BL-121)")
    law_ids: List[str] = Field(default_factory=list, description="Danh sách mã law_id được trích dẫn")
    form_ids: List[str] = Field(default_factory=list, description="Danh sách mã form_id biểu mẫu bắt buộc")
    muc_phi_vnd: int = Field(..., ge=0, description="Mức lệ phí (VNĐ) nêu trong bản tư vấn")
    so_ngay_toi_da: int = Field(..., ge=0, description="Số ngày giải quyết tối đa nêu trong bản tư vấn")
    cap_tham_quyen: List[str] = Field(default_factory=list, description="Cơ quan có thẩm quyền được tư vấn")
    trang_thai_hieu_luc: str = Field(
        default="ĐANG HIỆU LỰC", description="Trạng thái hiệu lực của căn cứ pháp lý được áp dụng"
    )


class IRACOutput(BaseModel):
    """Cấu trúc lập luận pháp lý chuẩn IRAC của Consultant Agent."""
    is_refusal: bool = Field(
        default=False,
        description="True nếu hệ thống chủ động từ chối trả lời do thiếu căn cứ pháp lý hoặc ngoài phạm vi"
    )
    refusal_reason: Optional[str] = Field(
        default=None,
        description="Lý do từ chối trả lời chuẩn mực (nếu is_refusal = True)"
    )
    issue: str = Field(
        ..., description="[I - ISSUE] Xác định vấn đề pháp lý, tên thủ tục và giai đoạn/phân nhóm cần giải quyết"
    )
    rule: str = Field(
        ..., description="[R - RULE] Trích dẫn nguyên văn căn cứ pháp lý (Số hiệu văn bản, Điều, Khoản, Điểm, Mã định danh)"
    )
    application: str = Field(
        ...,
        description=(
            "[A - APPLICATION] Áp dụng chi tiết gồm: (1) Cơ quan tiếp nhận, (2) Trình tự thực hiện, "
            "(3) Thành phần hồ sơ kèm mã biểu mẫu & điều kiện bắt buộc, (4) Thời hạn giải quyết, (5) Mức lệ phí"
        )
    )
    conclusion: str = Field(
        ..., description="[C - CONCLUSION] Kết luận tóm tắt và cảnh báo pháp lý quan trọng cho người dân"
    )
    citations: Optional[CitationMetadata] = Field(
        default=None,
        description="Siêu dữ liệu định lượng phục vụ kiểm chứng tự động Lớp 1 của Judge Agent"
    )
    confidence_score: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Độ tin cậy của câu trả lời tư vấn (0.0 - 1.0)"
    )
