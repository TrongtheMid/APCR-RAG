"""
Định nghĩa Pydantic Models cho:
1. Lược đồ 6 trường JSON chuẩn hóa kèm siêu dữ liệu thời gian (Temporal Metadata).
2. Cấu trúc đầu ra lập luận pháp lý IRAC của Consultant Agent (Structured Output).
3. Cấu trúc kết quả thẩm định chéo 2 lớp của Judge / Critic Agent.
4. Cấu trúc bộ 50 câu hỏi kiểm thử (Benchmark Dataset) và Cảnh báo hiểu nhầm.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# 1. LƯỢC ĐỒ 6 TRƯỜNG JSON CHUẨN HÓA (SCHEMA DỮ LIỆU THỦ TỤC & TỐ TỤNG)
# ==============================================================================

class MaVaTenThuTuc(BaseModel):
    """Trường 1: Mã định danh, tên thủ tục, phân nhóm và phân cấp thẩm quyền."""
    ma_thu_tuc: str = Field(..., description="Mã định danh duy nhất của thủ tục (VD: TTHS-BL-121, TTHC-HT-01)")
    ten_thu_tuc: str = Field(..., description="Tên chính thức của thủ tục hành chính hoặc tố tụng hình sự")
    phan_nhom: Literal["Thủ tục hành chính", "Tố tụng hình sự"] = Field(
        ..., description="Phân nhóm trụ cột của thủ tục"
    )
    linh_vuc: str = Field(..., description="Lĩnh vực chuyên môn (VD: Hộ tịch, Cư trú & Căn cước, Biện pháp ngăn chặn...)")
    cap_tham_quyen: List[str] = Field(
        default_factory=list,
        description="Danh sách cơ quan có thẩm quyền tiếp nhận/giải quyết (VD: UBND cấp xã, Cơ quan CSĐT, Tòa án...)"
    )
    giai_doan_to_tung: List[str] = Field(
        default_factory=list,
        description="Giai đoạn tố tụng tương ứng đối với Tố tụng hình sự (Điều tra, Truy tố, Xét xử, Thi hành án)"
    )


class GiayToBatBuoc(BaseModel):
    """Biểu mẫu / Giấy tờ thành phần trong bộ hồ sơ."""
    form_id: str = Field(..., description="Mã định danh biểu mẫu/giấy tờ (VD: DON-BL-01, TK-KS-01)")
    ten_giay_to: str = Field(..., description="Tên đầy đủ và yêu cầu của giấy tờ trong hồ sơ")


class ThanhPhanHoSo(BaseModel):
    """Trường 3: Thành phần hồ sơ và điều kiện bắt buộc."""
    giay_to_bat_buoc: List[GiayToBatBuoc] = Field(
        default_factory=list,
        description="Danh mục các giấy tờ, biểu mẫu bắt buộc phải nộp/xuất trình"
    )
    dieu_kien_bat_buoc: str = Field(
        ..., description="Điều kiện pháp lý tiên quyết để thực hiện thủ tục (VD: ít nhất 02 người thân thích bảo lĩnh)"
    )


class ThoiHanGiaiQuyet(BaseModel):
    """Trường 4: Thời hạn giải quyết theo luật định."""
    so_ngay_toi_da: int = Field(..., ge=0, description="Số ngày giải quyết tối đa (con số nguyên để kiểm chứng tất định)")
    don_vi_tinh: str = Field(..., description="Đơn vị tính thời gian (VD: ngày làm việc, ngày, giờ)")
    chi_tiet: str = Field(..., description="Giải thích chi tiết về thời hạn trong các trường hợp cụ thể")


class LePhi(BaseModel):
    """Trường 5: Mức lệ phí thực hiện thủ tục."""
    muc_phi_vnd: int = Field(..., ge=0, description="Mức lệ phí tính bằng VNĐ (0 nếu miễn phí hoàn toàn)")
    mo_ta: str = Field(..., description="Mô tả chi tiết mức thu lệ phí hoặc các trường hợp miễn/giảm")


class CanCuPhapLy(BaseModel):
    """Trường 6: Căn cứ pháp lý kèm Siêu dữ liệu Thời gian (Temporal Metadata)."""
    law_id: str = Field(..., description="Mã định danh điều luật (VD: bltths2015-d121, lct2020-d20)")
    so_hieu_van_ban: str = Field(..., description="Tên và số hiệu văn bản quy phạm pháp luật")
    dieu_khoan_diem: str = Field(..., description="Trích dẫn cụ thể Điều, Khoản, Điểm")
    ngay_hieu_luc: str = Field(..., description="Ngày bắt đầu có hiệu lực (định dạng YYYY-MM-DD)")
    trang_thai_hieu_luc: Literal["ĐANG HIỆU LỰC", "HẾT HIỆU LỰC"] = Field(
        ..., description="Trạng thái hiệu lực pháp lý hiện tại"
    )
    van_ban_thay_the: Optional[str] = Field(
        default=None,
        description="Số hiệu văn bản thay thế nếu điều khoản này đã HẾT HIỆU LỰC"
    )


class ProcedureRecord(BaseModel):
    """Lược đồ 6 Trường Chuẩn hóa cho một Thủ tục Hành chính hoặc Tố tụng Hình sự."""
    ma_va_ten_thu_tuc: MaVaTenThuTuc
    trinh_tu_thuc_hien: List[str] = Field(
        ..., min_length=1, description="Trường 2: Các bước thực hiện tuần tự"
    )
    thanh_phan_ho_so: ThanhPhanHoSo
    thoi_han_giai_quyet: ThoiHanGiaiQuyet
    le_phi: LePhi
    can_cu_phap_ly: List[CanCuPhapLy] = Field(
        ..., min_length=1, description="Danh sách căn cứ pháp lý và trạng thái hiệu lực"
    )


# ==============================================================================
# 2. CẤU TRÚC ĐẦU RA IRAC CỦA CONSULTANT AGENT (STRUCTURED OUTPUT)
# ==============================================================================

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
    """Cấu trúc lập luận pháp lý chuẩn IRAC (Issue - Rule - Application - Conclusion) của Consultant Agent."""
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


# ==============================================================================
# 3. CẤU TRÚC KẾT QUẢ THẨM ĐỊNH 2 LỚP CỦA JUDGE AGENT
# ==============================================================================

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


# ==============================================================================
# 4. CẤU TRÚC CÂU HỎI BENCHMARK (50 CÂU) & CẢNH BÁO HIỂU NHẦM PHÁP LÝ
# ==============================================================================

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
