"""
NHÁNH NGUYỄN HỮU HẢI LONG — Module Xử lý Ngôn ngữ & Tách từ Tiếng Việt:
- Chuẩn hóa văn bản Unicode NFC.
- Tách từ kết hợp Unigram + Bigram tiếng Việt cho chỉ mục BM25.
- Chuyển đổi Lược đồ 6 trường JSON thành văn bản đánh chỉ mục.
- Tô sáng (highlight) từ khóa khớp bằng thẻ HTML <mark>.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Tuple


def normalize_text(text: str) -> str:
    """Chuẩn hóa chữ thường và khoảng trắng phục vụ tách từ tiếng Việt."""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s\d\-/]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_vietnamese(text: str) -> List[str]:
    """Tách từ kết hợp unigram và bigram tiếng Việt để tăng độ chính xác cho BM25."""
    norm = normalize_text(text)
    words = [w for w in norm.split(" ") if len(w) > 1 or w.isdigit()]
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


def procedure_to_searchable_text(proc_dict: Dict[str, Any]) -> str:
    """Chuyển đổi cấu trúc 6 trường JSON thành chuỗi văn bản giàu ngữ nghĩa để đánh chỉ mục."""
    info = proc_dict["ma_va_ten_thu_tuc"]
    ho_so = proc_dict["thanh_phan_ho_so"]
    thoi_han = proc_dict["thoi_han_giai_quyet"]
    le_phi = proc_dict["le_phi"]
    can_cu_list = proc_dict["can_cu_phap_ly"]

    forms_str = "; ".join(
        f"{g['form_id']} {g['ten_giay_to']}" for g in ho_so.get("giay_to_bat_buoc", [])
    )
    laws_str = "; ".join(
        f"{c['law_id']} {c['so_hieu_van_ban']} {c['dieu_khoan_diem']} ({c['trang_thai_hieu_luc']})"
        for c in can_cu_list
    )
    steps_str = " ".join(proc_dict.get("trinh_tu_thuc_hien", []))

    return (
        f"Mã thủ tục: {info['ma_thu_tuc']}. Tên thủ tục: {info['ten_thu_tuc']}. "
        f"Phân nhóm: {info['phan_nhom']}. Lĩnh vực: {info['linh_vuc']}. "
        f"Cơ quan thẩm quyền: {', '.join(info.get('cap_tham_quyen', []))}. "
        f"Giai đoạn tố tụng: {', '.join(info.get('giai_doan_to_tung', []))}. "
        f"Trình tự thực hiện: {steps_str}. "
        f"Thành phần hồ sơ: {forms_str}. "
        f"Điều kiện bắt buộc: {ho_so.get('dieu_kien_bat_buoc', '')}. "
        f"Thời hạn giải quyết: tối đa {thoi_han['so_ngay_toi_da']} {thoi_han['don_vi_tinh']}. {thoi_han['chi_tiet']}. "
        f"Lệ phí: {le_phi['muc_phi_vnd']} VNĐ. {le_phi['mo_ta']}. "
        f"Căn cứ pháp lý: {laws_str}."
    )


def highlight_keywords_html(text: str, query: str) -> Tuple[str, List[str]]:
    """
    Tô sáng (highlight màu vàng) các từ khóa trong đoạn văn bản khớp với truy vấn của người dùng.
    Trả về chuỗi HTML đã bọc thẻ <mark> và danh sách các từ khóa khớp.
    """
    if not query or not query.strip():
        return text, []

    raw_terms = [
        t.strip()
        for t in re.split(r"[\s,;.?!]+", query.strip())
        if len(t.strip()) >= 2
    ]
    if not raw_terms:
        return text, []

    sorted_terms = sorted(set(raw_terms), key=len, reverse=True)
    matched_terms: List[str] = []
    highlighted = text

    for term in sorted_terms:
        pattern = re.compile(rf"({re.escape(term)})", flags=re.IGNORECASE | re.UNICODE)
        if pattern.search(highlighted):
            matched_terms.append(term)
            highlighted = pattern.sub(
                r'<mark style="background-color: #fde047; color: #1e293b; padding: 1px 4px; border-radius: 3px; font-weight: 600;">\1</mark>',
                highlighted,
            )

    return highlighted, matched_terms
