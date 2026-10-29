"""
NHÁNH NGUYỄN HỮU HẢI LONG — Module Metadata Filtering:
Lọc chính xác hồ sơ thủ tục theo:
- Phân nhóm (Thủ tục hành chính / Tố tụng hình sự)
- Lĩnh vực chuyên môn
- Cấp cơ quan có thẩm quyền (UBND xã / UBND huyện / Công an xã / Cơ quan CSĐT / VKS / Tòa án)
- Giai đoạn tố tụng (Điều tra / Truy tố / Xét xử / Thi hành án)
- Trạng thái hiệu lực (ĐANG HIỆU LỰC / HẾT HIỆU LỰC) và Ngày hiệu lực (effective_date)
"""

from __future__ import annotations

from typing import Any, Dict, Optional


def passes_metadata_filter(
    proc: Dict[str, Any],
    phan_nhom: Optional[str] = None,
    linh_vuc: Optional[str] = None,
    cap_tham_quyen: Optional[str] = None,
    giai_doan_to_tung: Optional[str] = None,
    only_active: bool = True,
    effective_date: Optional[str] = None,
) -> bool:
    """Kiểm tra hồ sơ thủ tục có thỏa mãn bộ lọc Metadata hay không."""
    info = proc["ma_va_ten_thu_tuc"]
    laws = proc["can_cu_phap_ly"]

    if phan_nhom and phan_nhom not in ("Tất cả", ""):
        if info["phan_nhom"] != phan_nhom:
            return False

    if linh_vuc and linh_vuc not in ("Tất cả", ""):
        if linh_vuc.lower() not in info["linh_vuc"].lower():
            return False

    if cap_tham_quyen and cap_tham_quyen not in ("Tất cả", ""):
        authorities = " ".join(info.get("cap_tham_quyen", [])).lower()
        if cap_tham_quyen.lower() not in authorities:
            return False

    if giai_doan_to_tung and giai_doan_to_tung not in ("Tất cả", ""):
        stages = " ".join(info.get("giai_doan_to_tung", [])).lower()
        if giai_doan_to_tung.lower() not in stages:
            return False

    if only_active:
        has_active = any(l["trang_thai_hieu_luc"] == "ĐANG HIỆU LỰC" for l in laws)
        if not has_active:
            return False

    if effective_date:
        valid_by_date = any(l["ngay_hieu_luc"] <= effective_date for l in laws)
        if not valid_by_date:
            return False

    return True
