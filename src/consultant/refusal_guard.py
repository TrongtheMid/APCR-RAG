"""
NHÁNH VŨ GIA ANH ĐỨC — Module Deterministic Refusal Guard:
Kiểm tra ngưỡng điểm truy xuất, từ khóa ngoài phạm vi và mức độ khớp chủ đề thực chất
(Query-Procedure Topic Relevance) để chủ động từ chối trả lời khi không đủ căn cứ pháp lý xác thực.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from src.consultant.prompts import REFUSAL_TEMPLATE_MESSAGE
from src.schemas import IRACOutput

OUT_OF_DOMAIN_KEYWORDS = [
    "thời tiết hôm nay",
    "công thức nấu ăn",
    "kết quả bóng đá",
    "giá vàng hôm nay",
    "viết code game",
    "dự báo thời tiết",
    "bói toán",
    "xổ số",
    "xin visa",
    "đăng ký nhãn hiệu",
    "bản quyền tác giả",
    "phá sản doanh nghiệp",
    "chia tài sản ly hôn",
]

# Các từ giao tiếp / hành chính chung không mang tính phân biệt thủ tục cụ thể
_GENERIC_FILLER_TOKENS = {
    "tôi", "muốn", "xin", "cấp", "lại", "mới", "đổi", "làm", "thủ", "tục",
    "giấy", "phép", "chứng", "nhận", "đơn", "hồ", "sơ", "ở", "đâu", "bao",
    "nhiêu", "tiền", "lệ", "phí", "mấy", "ngày", "lâu", "cần", "những", "gì",
    "như", "thế", "nào", "có", "được", "không", "cho", "hỏi", "về", "việc",
    "đi", "nộp", "đăng", "ký", "cơ", "quan", "nhà", "nước", "quy", "định",
    "luật", "hiện", "nay", "năm", "tháng", "tại", "và", "hoặc", "của",
    "dịch", "vụ", "hỗ", "trợ", "danh", "sách", "mục", "giúp", "biết", "hiểu",
    "mình", "bạn", "anh", "chị", "em", "vậy", "thì", "sao", "nhỉ", "nhé",
    "nếu", "tổng", "hợp", "các", "mà", "là", "này", "kia", "đây", "đó",
}


def _extract_distinctive_tokens(query: str) -> List[str]:
    q_words = re.findall(r"[^\W\d_]{2,}", query.lower(), flags=re.UNICODE)
    return [w for w in q_words if w not in _GENERIC_FILLER_TOKENS]


def _has_topic_relevance(query: str, best_proc: Dict[str, Any]) -> bool:
    """
    Kiểm tra xem các từ khóa đặc trưng trong câu hỏi có thực sự xuất hiện trong
    hồ sơ thủ tục đứng đầu hay không (ngăn chặn việc chỉ trùng từ chung 'xin cấp lại giấy phép').
    """
    distinctive = _extract_distinctive_tokens(query)
    if not distinctive:
        return False

    info = best_proc["ma_va_ten_thu_tuc"]
    ho_so = best_proc["thanh_phan_ho_so"]
    laws_text = " ".join(
        f"{c['so_hieu_van_ban']} {c['dieu_khoan_diem']}" for c in best_proc.get("can_cu_phap_ly", [])
    )
    forms_text = " ".join(
        g["ten_giay_to"] for g in ho_so.get("giay_to_bat_buoc", [])
    )
    proc_text = (
        f"{info['ten_thu_tuc']} {info['linh_vuc']} {' '.join(info.get('cap_tham_quyen', []))} "
        f"{ho_so.get('dieu_kien_bat_buoc', '')} {forms_text} {laws_text}"
    ).lower()

    matched = sum(1 for w in distinctive if w in proc_text)
    return (matched / len(distinctive)) >= 0.34


def check_deterministic_refusal(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    min_score_threshold: float,
) -> Optional[IRACOutput]:
    """
    Kiểm tra điều kiện Từ chối trả lời chủ động (Deterministic Refusal):
    1. Danh sách kết quả truy xuất rỗng.
    2. Điểm Hybrid Search cao nhất thấp hơn ngưỡng an toàn min_score_threshold.
    3. Truy vấn hoàn toàn không liên quan hoặc không khớp chủ đề thực chất với hồ sơ Top-1.
    """
    if not retrieved_items:
        return IRACOutput(
            is_refusal=True,
            refusal_reason="Không tìm thấy hồ sơ thủ tục hoặc căn cứ pháp lý nào khớp với yêu cầu.",
            direct_advice=REFUSAL_TEMPLATE_MESSAGE,
            issue="Yêu cầu nằm ngoài phạm vi kho dữ liệu pháp lý chuẩn hóa.",
            rule="Nguyên tắc An toàn Pháp lý — Không tư vấn khi thiếu căn cứ văn bản quy phạm pháp luật.",
            application=REFUSAL_TEMPLATE_MESSAGE,
            conclusion="Hệ thống từ chối trả lời để ngăn ngừa rủi ro ảo giác pháp lý (Hallucination).",
            citations=None,
            confidence_score=0.0,
        )

    best_item = retrieved_items[0]
    best_score = float(best_item.get("hybrid_score", 0.0))
    best_proc = best_item["procedure"]
    q_lower = query.lower()

    is_out_of_domain = any(k in q_lower for k in OUT_OF_DOMAIN_KEYWORDS)
    has_no_distinctive_terms = len(_extract_distinctive_tokens(query)) == 0
    is_topic_mismatch = has_no_distinctive_terms or (
        (best_score < 0.45) and (not _has_topic_relevance(query, best_proc))
    )

    if best_score < min_score_threshold or is_out_of_domain or is_topic_mismatch:
        return IRACOutput(
            is_refusal=True,
            refusal_reason=(
                f"Điểm tương đồng truy xuất ({best_score:.2f}) dưới ngưỡng an toàn "
                f"hoặc chủ đề câu hỏi chưa có trong kho 40 thủ tục Hành chính & Tố tụng Hình sự."
            ),
            direct_advice=REFUSAL_TEMPLATE_MESSAGE,
            issue="Câu hỏi chưa đủ căn cứ pháp lý xác thực trong hệ thống.",
            rule="Cơ chế Từ chối Trả lời Chủ động (Deterministic Refusal).",
            application=REFUSAL_TEMPLATE_MESSAGE,
            conclusion="Vui lòng đặt câu hỏi cụ thể về các thủ tục hành chính công hoặc tố tụng hình sự trong danh mục hỗ trợ.",
            citations=None,
            confidence_score=round(best_score, 2),
        )

    return None
