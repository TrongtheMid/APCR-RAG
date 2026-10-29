"""
NHÁNH PHẠM ĐỨC TRỌNG — Module Judge Lớp 1 (Deterministic Code Verification):
Đối soát chính xác tuyệt đối (Exact Match) bằng Python các mã định danh (ma_thu_tuc, law_id, form_id),
con số lệ phí (muc_phi_vnd), số ngày giải quyết tối đa (so_ngay_toi_da) và trạng thái hiệu lực (trang_thai_hieu_luc)
giữa bản nháp của Consultant Agent với đối tượng JSON 6 trường gốc.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.schemas import IRACOutput


def verify_layer1_deterministic(
    draft: IRACOutput,
    ground_truth_proc: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    LỚP 1 — KIỂM TRA TẤT ĐỊNH BẰNG CODE PYTHON (EXACT MATCH):
    Đối soát từng trường số liệu định lượng và mã định danh trong draft.citations với JSON 6 trường gốc.
    """
    if draft.is_refusal:
        return True, []

    errors: List[str] = []
    info = ground_truth_proc["ma_va_ten_thu_tuc"]
    ho_so = ground_truth_proc["thanh_phan_ho_so"]
    thoi_han = ground_truth_proc["thoi_han_giai_quyet"]
    le_phi = ground_truth_proc["le_phi"]
    laws = ground_truth_proc["can_cu_phap_ly"]

    gt_ma_tt = info["ma_thu_tuc"]
    gt_fee = int(le_phi["muc_phi_vnd"])
    gt_days = int(thoi_han["so_ngay_toi_da"])
    gt_law_ids = {c["law_id"] for c in laws}
    gt_form_ids = {f["form_id"] for f in ho_so.get("giay_to_bat_buoc", [])}
    gt_status = laws[0]["trang_thai_hieu_luc"]

    if draft.citations is None:
        errors.append(
            "[Lớp 1 - Thiếu Citations] Bản nháp không đính kèm cấu trúc CitationMetadata để đối soát số liệu."
        )
        return False, errors

    cit = draft.citations

    if cit.ma_thu_tuc != gt_ma_tt:
        errors.append(
            f"[Lớp 1 - Sai Mã Thủ Tục] Bản nháp trích dẫn mã '{cit.ma_thu_tuc}', "
            f"nhưng hồ sơ gốc là '{gt_ma_tt}'."
        )

    if int(cit.muc_phi_vnd) != gt_fee:
        errors.append(
            f"[Lớp 1 - Ảo Giác Lệ Phí] Bản nháp nêu mức lệ phí là {cit.muc_phi_vnd:,} VNĐ, "
            f"sai lệch so với JSON gốc là {gt_fee:,} VNĐ ({le_phi['mo_ta']})."
        )

    if int(cit.so_ngay_toi_da) != gt_days:
        errors.append(
            f"[Lớp 1 - Ảo Giác Thời Hạn] Bản nháp nêu thời hạn tối đa là {cit.so_ngay_toi_da} ngày, "
            f"sai lệch so với JSON gốc là {gt_days} ({thoi_han['don_vi_tinh']})."
        )

    cited_laws = set(cit.law_ids)
    if not cited_laws or not cited_laws.intersection(gt_law_ids):
        errors.append(
            f"[Lớp 1 - Sai Mã Điều Luật] Danh sách law_id trích dẫn {cit.law_ids} "
            f"không khớp với căn cứ pháp lý chuẩn {sorted(gt_law_ids)}."
        )

    cited_forms = set(cit.form_ids)
    missing_forms = gt_form_ids - cited_forms
    if missing_forms:
        errors.append(
            f"[Lớp 1 - Đứt Gãy Hồ Sơ] Bản nháp bỏ sót các mã biểu mẫu bắt buộc: {sorted(missing_forms)}."
        )

    if gt_status == "HẾT HIỆU LỰC" and "HẾT HIỆU LỰC" not in draft.rule.upper():
        errors.append(
            f"[Lớp 1 - Lỗi Hiệu Lực Thời Gian] Thủ tục '{gt_ma_tt}' đã HẾT HIỆU LỰC nhưng bản nháp không cảnh báo."
        )

    return len(errors) == 0, errors
