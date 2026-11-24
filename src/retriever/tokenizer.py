"""
NHÁNH NGUYỄN HỮU HẢI LONG — Module Xử lý Ngôn ngữ & Tách từ Tiếng Việt:
- Chuẩn hóa văn bản Unicode NFC.
- Tách từ tiếng Việt bằng underthesea/pyvi (nếu đã cài) hoặc fallback Unigram+Bigram.
- Từ điển đồng nghĩa đời thường -> thuật ngữ luật (COLLOQUIAL_TO_LEGAL_DICT).
- Chuyển đổi Lược đồ 6 trường JSON thành văn bản đánh chỉ mục.
- Tô sáng (highlight) từ khóa khớp bằng thẻ HTML <mark>.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Từ điển đồng nghĩa đời thường → thuật ngữ pháp lý chuẩn
# Phục vụ chuẩn hóa truy vấn trước khi đưa vào tokenizer
# ──────────────────────────────────────────────────────────────────────────────
COLLOQUIAL_TO_LEGAL_DICT: Dict[str, str] = {
    # Biện pháp ngăn chặn
    "bảo lãnh":          "bảo lĩnh",
    "xin về":            "bảo lĩnh tại ngoại",
    "nộp tiền bảo lãnh": "đặt tiền bảo đảm",
    "ký quỹ":            "đặt tiền bảo đảm",
    "kho bạc":           "Kho bạc Nhà nước",
    # Hộ tịch
    "giấy khai sinh":    "đăng ký khai sinh",
    "giấy chứng sinh":   "giấy chứng sinh",
    "mới sinh":          "khai sinh",
    "giấy kết hôn":      "Giấy chứng nhận kết hôn",
    "giấy báo tử":       "khai tử",
    # Cư trú
    "sổ hộ khẩu":        "đăng ký thường trú",
    "nhập khẩu":         "đăng ký thường trú",
    "hộ khẩu":           "đăng ký thường trú",
    "ở trọ":             "đăng ký tạm trú",
    # Căn cước
    "cmnd":              "thẻ Căn cước",
    "chứng minh nhân dân": "thẻ Căn cước",
    "cccd":              "thẻ Căn cước",
    # Hình sự
    "báo công an":       "tố giác tội phạm",
    "trình báo":         "tố giác tội phạm",
    "mất trộm":          "trộm cắp tài sản",
    "bị đánh":           "cố ý gây thương tích",
    "chống án":          "kháng cáo",
    "xin hoãn đi tù":    "hoãn chấp hành án phạt tù",
    "làm sổ đỏ":         "Giấy chứng nhận quyền sử dụng đất",
    "sang tên":          "đăng ký biến động đất đai",
    "bằng lái":          "giấy phép lái xe",
    "passport":          "hộ chiếu",
    "mở công ty":        "đăng ký doanh nghiệp",
    "thất nghiệp":       "trợ cấp thất nghiệp",
    "rút bhxh":          "bảo hiểm xã hội một lần",
    "phiếu số 1":        "Phiếu lý lịch tư pháp số 1",
    "phiếu số 2":        "Phiếu lý lịch tư pháp số 2",
}


# ──────────────────────────────────────────────────────────────────────────────
# Tích hợp NLP tiếng Việt thực chiến (underthesea / pyvi)
# Fallback sang Unigram+Bigram nội bộ nếu chưa cài
# ──────────────────────────────────────────────────────────────────────────────
_NLP_BACKEND: str = "builtin"

try:
    from underthesea import word_tokenize as _underthesea_tokenize  # type: ignore
    _NLP_BACKEND = "underthesea"
except ImportError:
    try:
        from pyvi import ViTokenizer as _pyvi_tokenizer  # type: ignore
        _NLP_BACKEND = "pyvi"
    except ImportError:
        pass


def normalize_text(text: str) -> str:
    """Chuẩn hóa chữ thường và khoảng trắng phục vụ tách từ tiếng Việt."""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s\d\-/]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def apply_colloquial_normalization(text: str) -> str:
    """Áp dụng từ điển đồng nghĩa đời thường → thuật ngữ pháp lý trước khi tokenize."""
    result = text
    for colloquial, legal in COLLOQUIAL_TO_LEGAL_DICT.items():
        result = re.sub(re.escape(colloquial), legal, result, flags=re.IGNORECASE)
    return result


def vietnamese_legal_tokenize(text: str) -> List[str]:
    """
    Tách từ tiếng Việt chuyên biệt cho văn bản pháp lý.
    Ưu tiên: underthesea → pyvi → fallback Unigram+Bigram nội bộ.
    Giữ nguyên từ ghép pháp lý quan trọng (bảo lĩnh, tạm giam, hộ tịch...).
    """
    normalized = apply_colloquial_normalization(normalize_text(text))

    if _NLP_BACKEND == "underthesea":
        try:
            tokens_raw = _underthesea_tokenize(normalized, format="text")
            tokens = [t.replace(" ", "_") for t in tokens_raw.split() if len(t) >= 1]
            # Bổ sung unigram cho từ ghép để tăng recall BM25
            extra = []
            for t in tokens:
                if "_" in t:
                    extra.extend(t.split("_"))
            return tokens + extra
        except Exception:
            pass

    if _NLP_BACKEND == "pyvi":
        try:
            tokens_raw = _pyvi_tokenizer.tokenize(normalized)
            tokens = [t for t in tokens_raw.split() if len(t) >= 1]
            extra = [t.replace("_", " ") for t in tokens if "_" in t]
            return tokens + extra
        except Exception:
            pass

    # Fallback: Unigram + Bigram nội bộ (zero-dependency)
    words = [w for w in normalized.split(" ") if len(w) > 1 or w.isdigit()]
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


def tokenize_vietnamese(text: str) -> List[str]:
    """Alias tương thích ngược cho vietnamese_legal_tokenize."""
    return vietnamese_legal_tokenize(text)


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
