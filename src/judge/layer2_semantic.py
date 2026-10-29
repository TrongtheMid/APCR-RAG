"""
NHÁNH PHẠM ĐỨC TRỌNG — Module Judge Lớp 2 (Semantic & Legal Condition Critic):
Thẩm định bản nháp của Consultant có nêu đầy đủ điều kiện pháp lý tiên quyết
(ít nhất 02 người thân thích khi bảo lĩnh, miễn phí khi tố giác tội phạm, cấm ủy quyền LLTP số 2...)
và đúng cấp cơ quan có thẩm quyền theo giai đoạn tố tụng hay không.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.core.llm_factory import get_chat_llm
from src.schemas import IRACOutput


def try_llm_critic(
    query: str,
    draft: IRACOutput,
    ground_truth_proc: Dict[str, Any],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[str]:
    """Gọi LLM Judge kiểm tra bổ sung nếu có GOOGLE_API_KEY."""
    llm = get_chat_llm(temperature=0.0, model_name=model_name, api_key=api_key)
    if llm is None:
        return []
    try:
        prompt = (
            "Bạn là Tác tử Thẩm định Pháp lý (JUDGE AGENT - Lớp 2).\n"
            "Hãy kiểm tra xem BẢN NHÁP TƯ VẤN có bị mâu thuẫn về thẩm quyền hoặc bỏ sót điều kiện bắt buộc "
            "so với HỒ SƠ JSON GỐC hay không.\n"
            "Nếu hoàn toàn chính xác, chỉ trả lời đúng một từ: PASS.\n"
            "Nếu có lỗi sai nghiêm trọng, hãy mô tả ngắn gọn lỗi đó trong 1 câu.\n\n"
            f"CÂU HỎI: {query}\n"
            f"ĐIỀU KIỆN BẮT BUỘC GỐC: {ground_truth_proc['thanh_phan_ho_so']['dieu_kien_bat_buoc']}\n"
            f"CƠ QUAN THẨM QUYỀN GỐC: {ground_truth_proc['ma_va_ten_thu_tuc']['cap_tham_quyen']}\n"
            f"BẢN NHÁP APPLICATION: {draft.application}"
        )
        resp = llm.invoke(prompt)
        content = str(resp.content).strip()
        if content.upper().startswith("PASS"):
            return []
        return [f"[Lớp 2 - LLM Critic] {content}"]
    except Exception:
        return []


def verify_layer2_semantic_critic(
    query: str,
    draft: IRACOutput,
    ground_truth_proc: Dict[str, Any],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    LỚP 2 — THẨM ĐỊNH NGỮ NGHĨA, ĐIỀU KIỆN BẮT BUỘC & PHÂN CẤP THẨM QUYỀN.
    """
    if draft.is_refusal:
        return True, []

    errors: List[str] = []
    info = ground_truth_proc["ma_va_ten_thu_tuc"]
    ma_tt = info["ma_thu_tuc"]
    gt_authorities = info.get("cap_tham_quyen", [])

    combined_draft_text = f"{draft.issue}\n{draft.rule}\n{draft.application}\n{draft.conclusion}".lower()

    if gt_authorities:
        has_valid_auth = any(auth.lower() in combined_draft_text for auth in gt_authorities)
        if not has_valid_auth:
            errors.append(
                f"[Lớp 2 - Nhầm Lẫn Thẩm Quyền] Bản nháp chưa nêu đúng cơ quan có thẩm quyền chuẩn: "
                f"{', '.join(gt_authorities)}."
            )

    if ma_tt == "TTHS-BL-121" and "02 người" not in combined_draft_text and "2 người" not in combined_draft_text:
        errors.append(
            "[Lớp 2 - Thiếu Điều Kiện Tiên Quyết] Thủ tục bảo lĩnh (TTHS-BL-121) bắt buộc phải nêu rõ điều kiện "
            "'ít nhất 02 người thân thích' cùng đứng ra nhận bảo lĩnh."
        )

    if ma_tt == "TTHS-TG-145" and "miễn" not in combined_draft_text and "0 vnđ" not in combined_draft_text:
        errors.append(
            "[Lớp 2 - Thiếu Cảnh Báo Miễn Phí] Thủ tục tố giác tội phạm (TTHS-TG-145) phải khẳng định rõ "
            "hoàn toàn miễn phí (0 VNĐ) để phòng chống lừa đảo."
        )

    if ma_tt == "TTHS-LLTP-045" and "số 2" in query.lower() and "ủy quyền" not in combined_draft_text:
        errors.append(
            "[Lớp 2 - Thiếu Điều Kiện Cấm Ủy Quyền] Câu hỏi liên quan đến Phiếu lý lịch tư pháp số 2, "
            "bắt buộc phải cảnh báo quy định KHÔNG được ủy quyền cho người khác làm thủ tục."
        )

    if len(draft.application.strip()) < 120:
        errors.append(
            "[Lớp 2 - Sơ Sài Nội Dung Application] Phần Application trong cấu trúc IRAC quá ngắn, "
            "chưa trình bày đầy đủ trình tự các bước và điều kiện bắt buộc."
        )

    if not errors:
        llm_errors = try_llm_critic(
            query=query,
            draft=draft,
            ground_truth_proc=ground_truth_proc,
            model_name=model_name,
            api_key=api_key,
        )
        errors.extend(llm_errors)

    return len(errors) == 0, errors
