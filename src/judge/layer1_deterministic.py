"""
NHÁNH PHẠM ĐỨC TRỌNG — Module Judge Lớp 1 (Deterministic Code Verification + Anti-Contradiction Regex):
1. Đối soát chính xác tuyệt đối (Exact Match) các mã định danh (ma_thu_tuc, law_id, form_id),
   mức lệ phí (muc_phi_vnd), số ngày giải quyết tối đa (so_ngay_toi_da) và trạng thái hiệu lực
   trong draft.citations so với JSON 6 trường gốc.
2. Regex Extractor (Anti-Contradiction): Quét trực tiếp phần văn xuôi (issue, rule, application,
   conclusion) để bóc tách mọi số tiền (VNĐ / đồng / triệu), số ngày và mã biểu mẫu, phát hiện
   ngay trường hợp LLM điền đúng metadata nhưng viết sai hoặc bịa số liệu trong câu trả lời.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from src.schemas import IRACOutput


# Biểu thức chính quy bóc tách số tiền VNĐ / đồng / triệu đồng trong văn xuôi tiếng Việt
_MONEY_VND_RE = re.compile(
    r"(?<!\d)(\d{1,3}(?:[.,]\d{3})+|\d+)\s*(?:vnđ|vnd|đồng|đ\b)",
    flags=re.IGNORECASE | re.UNICODE,
)
_MONEY_TRIEU_RE = re.compile(
    r"(?<!\d)(\d+(?:[.,]\d+)?)\s*triệu(?:\s*(?:đồng|vnđ|vnd))?",
    flags=re.IGNORECASE | re.UNICODE,
)

# Biểu thức chính quy bóc tách số ngày thời hạn trong văn xuôi (bỏ qua ngày/tháng dạng dd/mm/yyyy)
_DAYS_RE = re.compile(
    r"(?<!/)(?<!\d)(\d{1,3})\s*(?:ngày\s+làm\s+việc|ngày\b)(?!\s*/|\s*\d|\s+tháng|\s+sinh|\s+tuyên|\s+nhận|\s+có\s+người|\s+dự\s+kiến|\s+kết\s+thúc|\s+nộp|\s+chấm\s+dứt|\s+bản\s+án)",
    flags=re.IGNORECASE | re.UNICODE,
)
_DEADLINE_EXPLICIT_RE = re.compile(
    r"(?:tối\s+đa|trong\s+thời\s+hạn|không\s+quá|thời\s+hạn(?:\s+giải\s+quyết|\s+kháng\s+cáo|\s+giám\s+định|\s+xem\s+xét)?\s*(?:là|:)?\s*(?:tối\s+đa\s*)?|chờ\s+kết\s+quả\s+sau|sau)\s*(\d{1,3})\s*(?:ngày\s+làm\s+việc|ngày\b)",
    flags=re.IGNORECASE | re.UNICODE,
)


def _parse_int_amount(raw_num: str) -> int:
    cleaned = raw_num.replace(".", "").replace(",", "").strip()
    return int(cleaned) if cleaned.isdigit() else -1


def extract_fees_from_text(text: str) -> List[int]:
    """Bóc tách toàn bộ các mức tiền (quy đổi ra VNĐ) xuất hiện trong văn bản tiếng Việt."""
    if not text:
        return []
    found: Set[int] = set()
    for match in _MONEY_VND_RE.finditer(text):
        val = _parse_int_amount(match.group(1))
        if val >= 0:
            found.add(val)

    for match in _MONEY_TRIEU_RE.finditer(text):
        raw = match.group(1).replace(",", ".")
        try:
            val = int(round(float(raw) * 1_000_000))
            if val > 0:
                found.add(val)
        except ValueError:
            continue

    return sorted(found)


def extract_days_from_text(text: str) -> List[int]:
    """Bóc tách toàn bộ các mốc thời hạn tính theo ngày xuất hiện trong văn bản tiếng Việt."""
    if not text:
        return []
    found: Set[int] = set()
    for match in _DEADLINE_EXPLICIT_RE.finditer(text):
        val = int(match.group(1))
        if 0 <= val <= 365:
            found.add(val)
    for match in _DAYS_RE.finditer(text):
        val = int(match.group(1))
        if 0 <= val <= 365:
            found.add(val)
    return sorted(found)


def _collect_ground_truth_allowed_numbers(
    ground_truth_proc: Dict[str, Any],
) -> Tuple[Set[int], Set[int]]:
    """
    Thu thập tập hợp các con số lệ phí (VNĐ) và số ngày hợp lệ xuất hiện trong JSON gốc
    (bao gồm giá trị chính và các trường hợp phụ/miễn giảm/bước trung gian trong mô tả gốc).
    """
    le_phi = ground_truth_proc["le_phi"]
    thoi_han = ground_truth_proc["thoi_han_giai_quyet"]
    ho_so = ground_truth_proc["thanh_phan_ho_so"]
    steps_text = " ".join(ground_truth_proc.get("trinh_tu_thuc_hien", []))

    gt_corpus = " ".join(
        [
            str(le_phi.get("mo_ta", "")),
            str(thoi_han.get("don_vi_tinh", "")),
            str(thoi_han.get("chi_tiet", "")),
            str(ho_so.get("dieu_kien_bat_buoc", "")),
            steps_text,
        ]
    )

    allowed_fees: Set[int] = {int(le_phi["muc_phi_vnd"])}
    allowed_fees.update(extract_fees_from_text(gt_corpus))

    allowed_days: Set[int] = {int(thoi_han["so_ngay_toi_da"])}
    allowed_days.update(extract_days_from_text(gt_corpus))

    return allowed_fees, allowed_days


def verify_layer1_deterministic_detailed(
    draft: IRACOutput,
    ground_truth_proc: Dict[str, Any],
) -> Dict[str, Any]:
    """
    LỚP 1 — KIỂM TRA TẤT ĐỊNH BẰNG CODE PYTHON (EXACT MATCH + ANTI-CONTRADICTION REGEX):
    Trả về chi tiết điểm số, danh sách lỗi và các số liệu bóc tách từ văn xuôi.
    """
    if draft.is_refusal:
        return {
            "passed": True,
            "errors": [],
            "score": 1.0,
            "prose_extracted_fees": [],
            "prose_extracted_days": [],
        }

    errors: List[str] = []
    checks_total = 0
    checks_passed = 0

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

    prose_body = f"{draft.application}\n{draft.conclusion}"
    full_prose = f"{draft.issue}\n{draft.rule}\n{prose_body}"
    prose_fees = extract_fees_from_text(prose_body)
    prose_days = extract_days_from_text(prose_body)

    if draft.citations is None:
        errors.append(
            "[Lớp 1 - Thiếu Citations] Bản nháp không đính kèm cấu trúc CitationMetadata để đối soát số liệu."
        )
        return {
            "passed": False,
            "errors": errors,
            "score": 0.0,
            "prose_extracted_fees": prose_fees,
            "prose_extracted_days": prose_days,
        }

    cit = draft.citations

    # --- A. ĐỐI SOÁT METADATA EXACT MATCH (6 tiêu chí) ---
    checks_total += 1
    if cit.ma_thu_tuc != gt_ma_tt:
        errors.append(
            f"[Lớp 1 - Sai Mã Thủ Tục] Bản nháp trích dẫn mã '{cit.ma_thu_tuc}', "
            f"nhưng hồ sơ gốc là '{gt_ma_tt}'."
        )
    else:
        checks_passed += 1

    checks_total += 1
    if int(cit.muc_phi_vnd) != gt_fee:
        errors.append(
            f"[Lớp 1 - Ảo Giác Lệ Phí] Bản nháp nêu mức lệ phí là {cit.muc_phi_vnd:,} VNĐ, "
            f"sai lệch so với JSON gốc là {gt_fee:,} VNĐ ({le_phi['mo_ta']})."
        )
    else:
        checks_passed += 1

    checks_total += 1
    if int(cit.so_ngay_toi_da) != gt_days:
        errors.append(
            f"[Lớp 1 - Ảo Giác Thời Hạn] Bản nháp nêu thời hạn tối đa là {cit.so_ngay_toi_da} ngày, "
            f"sai lệch so với JSON gốc là {gt_days} ({thoi_han['don_vi_tinh']})."
        )
    else:
        checks_passed += 1

    checks_total += 1
    cited_laws = set(cit.law_ids)
    if not cited_laws or not cited_laws.intersection(gt_law_ids):
        errors.append(
            f"[Lớp 1 - Sai Mã Điều Luật] Danh sách law_id trích dẫn {cit.law_ids} "
            f"không khớp với căn cứ pháp lý chuẩn {sorted(gt_law_ids)}."
        )
    else:
        checks_passed += 1

    checks_total += 1
    cited_forms = set(cit.form_ids)
    missing_forms = gt_form_ids - cited_forms
    if missing_forms:
        errors.append(
            f"[Lớp 1 - Đứt Gãy Hồ Sơ] Bản nháp bỏ sót các mã biểu mẫu bắt buộc: {sorted(missing_forms)}."
        )
    else:
        checks_passed += 1

    checks_total += 1
    if gt_status == "HẾT HIỆU LỰC" and "HẾT HIỆU LỰC" not in draft.rule.upper():
        errors.append(
            f"[Lớp 1 - Lỗi Hiệu Lực Thời Gian] Thủ tục '{gt_ma_tt}' đã HẾT HIỆU LỰC nhưng bản nháp không cảnh báo."
        )
    else:
        checks_passed += 1

    # --- B. ANTI-CONTRADICTION REGEX: ĐỐI SOÁT CHÉO VĂN XUÔI VS METADATA & JSON GỐC (3 tiêu chí) ---
    allowed_fees, allowed_days = _collect_ground_truth_allowed_numbers(ground_truth_proc)

    # B1. Kiểm tra mâu thuẫn mức lệ phí trong văn xuôi (chặn LLM ghi đúng citations nhưng bịa tiền trong text)
    checks_total += 1
    ungrounded_fees = [f for f in prose_fees if f not in allowed_fees]
    prose_lower = prose_body.lower()
    fee_mentioned_correctly = (
        (gt_fee in prose_fees)
        or (gt_fee == 0 and ("miễn phí" in prose_lower or "miễn lệ phí" in prose_lower or "miễn án phí" in prose_lower or "miễn chi phí" in prose_lower or "0 vnđ" in prose_lower or "0 đ" in prose_lower))
    )
    if ungrounded_fees:
        fmt_bad = ", ".join(f"{f:,} VNĐ" for f in ungrounded_fees)
        errors.append(
            f"[Lớp 1 - Mâu Thuẫn Văn Xuôi / Ảo Giác Lệ Phí trong Text] Phần văn bản tư vấn chứa mức tiền "
            f"không có trong JSON gốc ({fmt_bad}), trong khi lệ phí chuẩn là {gt_fee:,} VNĐ."
        )
    elif not fee_mentioned_correctly:
        errors.append(
            f"[Lớp 1 - Thiếu Lệ Phí trong Văn Xuôi] Citations ghi {gt_fee:,} VNĐ nhưng phần trình bày "
            f"Application/Conclusion không nêu rõ mức lệ phí chuẩn này."
        )
    else:
        checks_passed += 1

    # B2. Kiểm tra mâu thuẫn số ngày thời hạn trong văn xuôi
    checks_total += 1
    ungrounded_days = [d for d in prose_days if d not in allowed_days]
    days_mentioned_correctly = (gt_days in prose_days) or (
        gt_days == 1 and ("trong ngày" in prose_lower or "01 ngày" in prose_lower or "1 ngày" in prose_lower)
    )
    if ungrounded_days:
        errors.append(
            f"[Lớp 1 - Mâu Thuẫn Văn Xuôi / Ảo Giác Thời Hạn trong Text] Phần văn bản tư vấn nêu mốc "
            f"thời hạn {ungrounded_days} ngày không tồn tại trong JSON gốc (chuẩn tối đa: {gt_days} ngày)."
        )
    elif not days_mentioned_correctly:
        errors.append(
            f"[Lớp 1 - Thiếu Thời Hạn trong Văn Xuôi] Citations ghi tối đa {gt_days} ngày nhưng văn bản "
            f"Application/Conclusion chưa nêu rõ mốc thời hạn này."
        )
    else:
        checks_passed += 1

    # B3. Kiểm tra mã biểu mẫu có thực sự xuất hiện trong văn xuôi (tránh chỉ khai báo trên citations)
    checks_total += 1
    full_prose_upper = full_prose.upper()
    missing_prose_forms = sorted(
        fid for fid in gt_form_ids if fid.upper() not in full_prose_upper
    )
    if missing_prose_forms:
        errors.append(
            f"[Lớp 1 - Thiếu Mã Biểu Mẫu trong Văn Xuôi] Phần trình bày chưa liệt kê rõ các mã biểu mẫu "
            f"bắt buộc: {missing_prose_forms}."
        )
    else:
        checks_passed += 1

    score = round(checks_passed / max(1, checks_total), 4)
    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "score": score,
        "prose_extracted_fees": prose_fees,
        "prose_extracted_days": prose_days,
    }


def verify_layer1_deterministic(
    draft: IRACOutput,
    ground_truth_proc: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """API tương thích ngược trả về (passed, errors) cho Lớp 1."""
    detail = verify_layer1_deterministic_detailed(draft, ground_truth_proc)
    return bool(detail["passed"]), list(detail["errors"])
