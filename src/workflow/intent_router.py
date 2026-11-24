"""
NHÁNH PHẠM ĐỨC TRỌNG — Bộ Phân loại Ý định Hội thoại (Intent Router) & Phản hồi Tự nhiên:
Giúp hệ thống phân biệt giữa:
  - 'greeting': Chào hỏi, cảm ơn, hỏi thăm, hỏi danh tính trợ lý.
  - 'services': Hỏi hệ thống hỗ trợ những dịch vụ / thủ tục / lĩnh vực nào.
  - 'followup_freeform': Yêu cầu tóm tắt, giải thích lại, làm rõ nối tiếp mạch hội thoại trước đó.
  - 'legal_qa': Câu hỏi pháp lý / thủ tục cụ thể (đi qua đầy đủ Retriever -> Consultant -> Judge 2 lớp).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional

from src.core.llm_factory import invoke_text_with_fallback

ChatIntent = Literal["greeting", "services", "followup_freeform", "legal_qa"]

_SERVICE_INQUIRY_PATTERNS = (
    "dịch vụ mà bạn hỗ trợ",
    "dịch vụ bạn hỗ trợ",
    "các dịch vụ",
    "những dịch vụ",
    "dịch vụ nào",
    "hỗ trợ những gì",
    "hỗ trợ những thủ tục",
    "hỗ trợ thủ tục nào",
    "hỗ trợ gì",
    "làm được gì",
    "giúp được gì",
    "tư vấn được gì",
    "tư vấn những gì",
    "tư vấn những thủ tục",
    "những thủ tục nào",
    "danh sách thủ tục",
    "danh mục thủ tục",
    "lĩnh vực nào",
    "bao nhiêu thủ tục",
    "phạm vi hỗ trợ",
    "chức năng của bạn",
    "bạn biết những gì",
)

_GREETING_RE = re.compile(
    r"(?:^|\b)(xin\s+chào|chào\s+bạn|chào\s+bot|chào\s+ad|chào|hello|hi|hey|"
    r"cảm\s+ơn|cám\s+ơn|thanks|thank\s+you|tạm\s+biệt|bye|"
    r"bạn\s+là\s+ai|bạn\s+tên\s+gì|giới\s+thiệu\s+về\s+bạn|"
    r"bạn\s+khỏe\s+không|ok|oke|okay|dạ|vâng|tuyệt\s+vời|hay\s+quá|rất\s+hữu\s+ích)(?:\b|$)",
    flags=re.IGNORECASE | re.UNICODE,
)

_FREEFORM_FOLLOWUP_TRIGGERS = (
    "tóm tắt",
    "ngắn gọn",
    "dễ hiểu hơn",
    "giải thích lại",
    "giải thích thêm",
    "nói rõ hơn",
    "cụ thể hơn",
    "chú ý gì nhất",
    "quan trọng nhất",
    "nhắc lại",
)


def classify_chat_intent(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    matched_expansions: Optional[List[str]] = None,
) -> ChatIntent:
    """Phân loại ý định tin nhắn để điều hướng giữa hội thoại tự nhiên và luồng RAG pháp lý."""
    q_clean = query.strip()
    q_lower = q_clean.lower()

    # Nếu câu hỏi nhắc rõ thuật ngữ thủ tục trong từ điển pháp lý -> đi vào luồng RAG chuẩn
    if matched_expansions:
        return "legal_qa"

    if any(pat in q_lower for pat in _SERVICE_INQUIRY_PATTERNS):
        return "services"

    if len(q_lower.split()) <= 12 and _GREETING_RE.search(q_lower):
        return "greeting"

    if chat_history and any(t.get("procedure_name") or t.get("direct_advice") for t in chat_history):
        if len(q_lower.split()) <= 20 and any(trig in q_lower for trig in _FREEFORM_FOLLOWUP_TRIGGERS):
            return "followup_freeform"

    return "legal_qa"


def _format_history_snippet(chat_history: Optional[List[Dict[str, str]]]) -> str:
    if not chat_history:
        return "Chưa có lịch sử trước đó."
    lines: List[str] = []
    for i, turn in enumerate(chat_history[-4:], start=1):
        u = (turn.get("query") or "").strip()
        p = (turn.get("procedure_name") or "").strip()
        a = (turn.get("direct_advice") or "").strip()
        a_short = (a[:350] + "...") if len(a) > 350 else a
        lines.append(f"Lượt {i} - Người dùng: {u}")
        if p:
            lines.append(f"  (Thủ tục liên quan: {turn.get('procedure_code', '')} - {p})")
        if a_short:
            lines.append(f"  Trợ lý đã trả lời: {a_short}")
    return "\n".join(lines)


def _build_services_catalog_reply(
    procedures: List[Dict[str, Any]],
    phan_nhom_filter: Optional[str] = None,
) -> str:
    grouped: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for proc in procedures:
        info = proc["ma_va_ten_thu_tuc"]
        grouped[info["phan_nhom"]][info["linh_vuc"]].append(info["ten_thu_tuc"])

    total_procs = len(procedures)
    lines: List[str] = [
        f"Chào bạn! Mình là **Trợ lý Pháp lý APCR-RAG**, hiện đang hỗ trợ tư vấn tự động trên kho **{total_procs} thủ tục pháp lý chuẩn hóa** thuộc 2 nhóm dịch vụ chính:\n"
    ]

    ordered_groups = ["Thủ tục hành chính", "Tố tụng hình sự"]
    for grp in ordered_groups:
        if grp not in grouped:
            continue
        icon = "🏛️" if grp == "Thủ tục hành chính" else "⚖️"
        grp_count = sum(len(v) for v in grouped[grp].values())
        active_note = " *(đang lọc tại trang này)*" if phan_nhom_filter == grp else ""
        lines.append(f"### {icon} {grp} ({grp_count} thủ tục){active_note}")
        for lv, names in grouped[grp].items():
            preview = "; ".join(n.replace("Thủ tục ", "") for n in names[:3])
            if len(names) > 3:
                preview += f" *(và {len(names) - 3} thủ tục khác)*"
            lines.append(f"- **{lv}** ({len(names)} thủ tục): {preview}.")
        lines.append("")

    lines.append(
        "**Ngoài hỏi — đáp từng tình huống cụ thể**, bạn còn có thể:\n"
        "- Hỏi tiếp nhiều lượt tự nhiên (ví dụ: *\"Cần chuẩn bị giấy tờ gì?\"*, *\"Lệ phí bao nhiêu?\"*, *\"Tóm tắt ngắn gọn hơn đi\"*).\n"
        "- Nhờ kiểm tra quy định cũ hết hiệu lực (Sổ hộ khẩu giấy, CMND 9 số) hoặc phát hiện bẫy lừa đảo (phí tố giác tội phạm, bảo lĩnh 1 người).\n\n"
        "👉 Bạn đang quan tâm đến thủ tục hoặc vấn đề cụ thể nào? Cứ nhắn tự nhiên cho mình nhé!"
    )
    return "\n".join(lines)


def build_conversational_reply(
    intent: ChatIntent,
    query: str,
    procedures: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    phan_nhom_filter: Optional[str] = None,
    use_llm: bool = True,
) -> str:
    """Tạo câu trả lời tự nhiên cho các lượt hội thoại xã giao, hỏi dịch vụ hoặc tóm tắt nối tiếp."""
    if intent == "services":
        return _build_services_catalog_reply(procedures, phan_nhom_filter=phan_nhom_filter)

    if intent == "greeting":
        q_lower = query.lower()
        if use_llm:
            history_txt = _format_history_snippet(chat_history)
            prompt = (
                "Bạn là Trợ lý Pháp lý APCR-RAG (thân thiện, lịch sự, xưng 'mình' và gọi người dùng là 'bạn').\n"
                "Hệ thống hỗ trợ tư vấn 40 thủ tục Hành chính công (Hộ tịch, Cư trú & Căn cước, Giao thông, Hộ chiếu, "
                "Đất đai & Xây dựng, Kinh doanh & Thuế, Lao động & BHXH) và Tố tụng Hình sự (Tố giác, Bảo lĩnh, Thăm gặp, Kháng cáo, Xóa án tích, Lý lịch tư pháp).\n"
                "Hãy trả lời tin nhắn chào hỏi / xã giao dưới đây thật tự nhiên, ngắn gọn (2-4 câu) bằng tiếng Việt.\n\n"
                f"NGỮ CẢNH HỘI THOẠI TRƯỚC ĐÓ:\n{history_txt}\n\n"
                f"TIN NHẮN CỦA NGƯỜI DÙNG: {query}"
            )
            llm_reply = invoke_text_with_fallback(prompt, temperature=0.3)
            if llm_reply:
                return llm_reply

        if any(k in q_lower for k in ("cảm ơn", "cám ơn", "thank")):
            return (
                "Không có chi bạn nhé! Rất vui được hỗ trợ bạn. "
                "Nếu bạn còn thắc mắc thêm về hồ sơ, lệ phí hay thủ tục pháp lý nào khác, cứ nhắn cho mình bất cứ lúc nào."
            )
        if any(k in q_lower for k in ("tạm biệt", "bye")):
            return "Chào tạm biệt bạn! Chúc bạn thực hiện thủ tục thuận lợi và suôn sẻ."
        if any(k in q_lower for k in ("bạn là ai", "tên gì", "giới thiệu")):
            return (
                "Mình là **Trợ lý Pháp lý APCR-RAG** — chatbot tư vấn về **Thủ tục Hành chính công** "
                "và **Tố tụng Hình sự**, có cơ chế đối soát 2 lớp để đảm bảo đúng lệ phí, thời hạn và căn cứ luật. "
                "Bạn đang cần tìm hiểu thủ tục nào để mình hỗ trợ ngay?"
            )
        return (
            "Xin chào bạn! Mình là **Trợ lý Pháp lý APCR-RAG**. "
            "Bạn có thể hỏi mình bất cứ thắc mắc nào về thủ tục hành chính (khai sinh, kết hôn, căn cước, "
            "bằng lái xe, hộ chiếu, sổ đỏ, BHXH...) hoặc tố tụng hình sự (tố giác tội phạm, bảo lĩnh, kháng cáo, lý lịch tư pháp...). "
            "Hôm nay bạn cần mình hỗ trợ vấn đề gì?"
        )

    # intent == "followup_freeform"
    last_turn = (chat_history or [{}])[-1]
    prev_advice = (last_turn.get("direct_advice") or "").strip()
    prev_proc = (last_turn.get("procedure_name") or "thủ tục vừa trao đổi").strip()

    if use_llm:
        history_txt = _format_history_snippet(chat_history)
        prompt = (
            "Bạn là Trợ lý Pháp lý APCR-RAG đang trò chuyện tiếp với người dùng.\n"
            "Hãy đáp ứng đúng yêu cầu nối tiếp của người dùng (ví dụ: tóm tắt ngắn gọn, giải thích dễ hiểu hơn, nêu điểm cần chú ý nhất) "
            "dựa TUYỆT ĐỐI vào thông tin thực tế trong lịch sử hội thoại bên dưới (không tự bịa thêm số tiền lệ phí hoặc số ngày khác).\n\n"
            f"LỊCH SỬ HỘI THOẠI:\n{history_txt}\n\n"
            f"YÊU CẦU MỚI CỦA NGƯỜI DÙNG: {query}"
        )
        llm_reply = invoke_text_with_fallback(prompt, temperature=0.2)
        if llm_reply:
            return llm_reply

    if prev_advice:
        return f"Tóm tắt nhanh lại nội dung về **{prev_proc}** để bạn dễ theo dõi:\n\n{prev_advice}"
    return (
        "Bạn có thể nói rõ hơn thủ tục hoặc khía cạnh bạn muốn tìm hiểu "
        "(ví dụ: giấy tờ cần chuẩn bị, nơi nộp, thời hạn hay lệ phí) để mình hỗ trợ chính xác nhất nhé!"
    )
