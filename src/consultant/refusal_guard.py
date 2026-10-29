"""
NHÁNH VŨ GIA ANH ĐỨC — Module Deterministic Refusal Guard:
Kiểm tra ngưỡng điểm truy xuất và từ khóa ngoài phạm vi để chủ động từ chối trả lời
khi không đủ căn cứ pháp lý xác thực.
"""

from __future__ import annotations

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
]


def check_deterministic_refusal(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    min_score_threshold: float,
) -> Optional[IRACOutput]:
    """
    Kiểm tra điều kiện Từ chối trả lời chủ động (Deterministic Refusal):
    1. Danh sách kết quả truy xuất rỗng.
    2. Điểm Hybrid Search cao nhất thấp hơn ngưỡng an toàn min_score_threshold.
    3. Truy vấn hoàn toàn không liên quan đến pháp luật/thủ tục hành chính/tố tụng hình sự.
    """
    if not retrieved_items:
        return IRACOutput(
            is_refusal=True,
            refusal_reason="Không tìm thấy hồ sơ thủ tục hoặc căn cứ pháp lý nào khớp với yêu cầu.",
            issue="Yêu cầu nằm ngoài phạm vi kho dữ liệu pháp lý chuẩn hóa.",
            rule="Nguyên tắc An toàn Pháp lý — Không tư vấn khi thiếu căn cứ văn bản quy phạm pháp luật.",
            application=REFUSAL_TEMPLATE_MESSAGE,
            conclusion="Hệ thống từ chối trả lời để ngăn ngừa rủi ro ảo giác pháp lý (Hallucination).",
            citations=None,
            confidence_score=0.0,
        )

    best_item = retrieved_items[0]
    best_score = float(best_item.get("hybrid_score", 0.0))
    q_lower = query.lower()

    if best_score < min_score_threshold or any(k in q_lower for k in OUT_OF_DOMAIN_KEYWORDS):
        return IRACOutput(
            is_refusal=True,
            refusal_reason=(
                f"Điểm tương đồng truy xuất cao nhất ({best_score:.2f}) thấp hơn ngưỡng an toàn "
                f"({min_score_threshold:.2f}) hoặc nội dung câu hỏi nằm ngoài phạm vi Thủ tục Hành chính & Tố tụng Hình sự."
            ),
            issue="Câu hỏi chưa đủ căn cứ pháp lý xác thực trong hệ thống.",
            rule="Cơ chế Từ chối Trả lời Chủ động (Deterministic Refusal).",
            application=REFUSAL_TEMPLATE_MESSAGE,
            conclusion="Vui lòng đặt câu hỏi cụ thể về thủ tục hành chính công hoặc tố tụng hình sự.",
            citations=None,
            confidence_score=round(best_score, 2),
        )

    return None
