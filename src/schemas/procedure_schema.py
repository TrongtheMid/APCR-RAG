"""
Lược đồ 6 Trường Chuẩn hóa (ProcedureRecord) kết hợp Siêu dữ liệu Thời gian (Temporal Metadata)
cho Thủ tục Hành chính Công và Thủ tục Tố tụng Hình sự.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


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
        description="Danh sách cơ quan có thẩm quyền tiếp nhận/giải quyết"
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
        ..., description="Điều kiện pháp lý tiên quyết để thực hiện thủ tục"
    )


class ThoiHanGiaiQuyet(BaseModel):
    """Trường 4: Thời hạn giải quyết theo luật định."""
    so_ngay_toi_da: int = Field(..., ge=0, description="Số ngày giải quyết tối đa (số nguyên phục vụ kiểm chứng tất định)")
    don_vi_tinh: str = Field(..., description="Đơn vị tính thời gian (VD: ngày làm việc, ngày)")
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
    ngay_hieu_luc: str = Field(..., description="Ngày bắt đầu có hiệu lực (YYYY-MM-DD)")
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
