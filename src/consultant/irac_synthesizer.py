"""
NHÁNH VŨ GIA ANH ĐỨC — Module IRAC Synthesizer:
- Kết nối LLM thật qua `llm_factory` (Google Gemini & Local Ollama Qwen2.5/Vistral).
- Hỗ trợ Structured Output (`IRACOutput`) với cơ chế JSON fallback có kiểm soát
  cho các provider/model không hỗ trợ `.with_structured_output()` trực tiếp.
- Hỗ trợ ngữ cảnh hội thoại đa lượt (`chat_history`) và vòng lặp Tự hiệu chỉnh (`Self-Revision`)
  theo `critique_feedback` từ Judge Agent.
- Tự động fallback sang bộ tổng hợp tất định từ JSON 6 trường khi không có LLM hoặc lỗi parse.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.consultant.conversational_advisor import build_direct_advice
from src.consultant.prompts import (
    IRAC_SYSTEM_PROMPT,
    JSON_FALLBACK_INSTRUCTION,
    SELF_REVISION_PROMPT_TEMPLATE,
    format_chat_history_for_prompt,
)
from src.core.llm_factory import get_chat_llm
from src.schemas import CitationMetadata, IRACOutput


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Trích xuất đối tượng JSON hợp lệ từ chuỗi phản hồi LLM (kể cả khi bọc trong ```json ... ```).
    Trả về None nếu không parse được JSON object hợp lệ.
    """
    if not raw_text:
        return None
    cleaned = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        return None


def _validate_irac_candidate(raw_obj: Any) -> Optional[IRACOutput]:
    """
    Kiểm tra nghiêm ngặt đầu ra LLM theo schema `IRACOutput`.
    Không âm thầm chấp nhận dữ liệu sai schema hoặc thiếu trường bắt buộc.
    """
    if raw_obj is None:
        return None
    try:
        if isinstance(raw_obj, IRACOutput):
            parsed = raw_obj
        elif isinstance(raw_obj, dict):
            parsed = IRACOutput.model_validate(raw_obj)
        else:
            return None
    except ValidationError:
        return None

    if not parsed.is_refusal:
        required_text_fields = (parsed.issue, parsed.rule, parsed.application, parsed.conclusion)
        if any(not str(field or "").strip() for field in required_text_fields):
            return None
        if parsed.citations is None:
            return None

    return parsed


def _invoke_structured_irac(llm: Any, user_message: str) -> Optional[IRACOutput]:
    """
    Gọi LLM với ưu tiên `.with_structured_output(IRACOutput)`.
    Nếu provider không hỗ trợ structured output trực tiếp hoặc lỗi gọi tool/schema,
    thực hiện fallback có kiểm soát sang JSON mode và validate nghiêm ngặt bằng Pydantic.
    """
    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(IRACOutput)
            raw_res = structured_llm.invoke(user_message)
            validated = _validate_irac_candidate(raw_res)
            if validated is not None:
                return validated
        except Exception:
            pass

    try:
        json_prompt = f"{user_message}\n\n{JSON_FALLBACK_INSTRUCTION}"
        resp = llm.invoke(json_prompt)
        content = getattr(resp, "content", resp)
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", part)) if isinstance(part, dict) else str(part)
                for part in content
            )
        parsed_dict = _extract_json_object(str(content or ""))
        return _validate_irac_candidate(parsed_dict)
    except Exception:
        return None


def _build_llm_user_message(
    query: str,
    primary_proc: Dict[str, Any],
    expired_alerts: List[Dict[str, Any]],
    critique_feedback: Optional[str],
    chat_history: Optional[List[Dict[str, str]]] = None,
    previous_draft: Optional[IRACOutput | Dict[str, Any]] = None,
) -> str:
    context_json = json.dumps(primary_proc, ensure_ascii=False, indent=2)
    expired_json = (
        json.dumps(expired_alerts, ensure_ascii=False, indent=2)
        if expired_alerts
        else "Không có"
    )
    history_text = format_chat_history_for_prompt(chat_history)

    revision_block = ""
    if critique_feedback:
        prev_block = ""
        if previous_draft is not None:
            try:
                prev_dict = (
                    previous_draft.model_dump()
                    if isinstance(previous_draft, IRACOutput)
                    else dict(previous_draft)
                )
                prev_block = (
                    "\nBẢN NHÁP TRƯỚC ĐÓ CẦN SỬA:\n"
                    f"{json.dumps(prev_dict, ensure_ascii=False, indent=2)}"
                )
            except Exception:
                prev_block = ""

        revision_block = "\n\n" + SELF_REVISION_PROMPT_TEMPLATE.format(
            critique_feedback=critique_feedback,
            previous_draft_block=prev_block,
        )

    return (
        f"{IRAC_SYSTEM_PROMPT}\n\n"
        f"NGỮ CẢNH HỘI THOẠI GẦN ĐÂY (CHỈ DÙNG ĐỂ HIỂU MẠCH HỎI, KHÔNG THAY THẾ JSON GỐC):\n{history_text}\n\n"
        f"CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÂN: {query}\n\n"
        f"DỮ LIỆU HỒ SƠ THỦ TỤC 6 TRƯỜNG (JSON GỐC — NGUỒN SỰ THẬT DUY NHẤT):\n{context_json}\n\n"
        f"CẢNH BÁO QUY ĐỊNH CŨ HẾT HIỆU LỰC (NẾU CÓ):\n{expired_json}"
        f"{revision_block}"
    )


def try_generate_with_gemini(
    query: str,
    primary_proc: Dict[str, Any],
    expired_alerts: List[Dict[str, Any]],
    critique_feedback: Optional[str],
    hybrid_score: float,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    previous_draft: Optional[IRACOutput | Dict[str, Any]] = None,
) -> Optional[IRACOutput]:
    """
    Gọi LLM thật (Google Gemini hoặc Local Ollama qua `llm_factory`) với Structured Output (`IRACOutput`).
    Giữ tên hàm `try_generate_with_gemini` tương thích ngược với caller cũ và hỗ trợ thêm provider/Ollama/chat_history.
    """
    llm = get_chat_llm(
        temperature=0.1,
        model_name=model_name,
        api_key=api_key,
        provider=provider,
        base_url=base_url,
    )
    if llm is None:
        return None

    user_message = _build_llm_user_message(
        query=query,
        primary_proc=primary_proc,
        expired_alerts=expired_alerts,
        critique_feedback=critique_feedback,
        chat_history=chat_history,
        previous_draft=previous_draft,
    )
    result = _invoke_structured_irac(llm, user_message)
    if result is None:
        return None

    if not result.direct_advice:
        result.direct_advice = build_direct_advice(query, primary_proc, expired_alerts)
    result.confidence_score = round(min(0.99, max(0.75, hybrid_score)), 2)
    return result


try_generate_with_llm = try_generate_with_gemini


def synthesize_structured_irac(
    query: str,
    primary_proc: Dict[str, Any],
    expired_alerts: List[Dict[str, Any]],
    critique_feedback: Optional[str],
    revision_count: int,
    hybrid_score: float,
    simulate_initial_error: bool = False,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> IRACOutput:
    """
    Tổng hợp lời tư vấn tự nhiên (`direct_advice`) kết hợp lập luận chuẩn IRAC
    trực tiếp từ đối tượng JSON 6 trường, bảo đảm cả tính thân thiện cho người dân
    lẫn độ chính xác tuyệt đối khi đi qua Judge Agent 2 lớp.
    """
    info = primary_proc["ma_va_ten_thu_tuc"]
    steps = primary_proc["trinh_tu_thuc_hien"]
    ho_so = primary_proc["thanh_phan_ho_so"]
    thoi_han = primary_proc["thoi_han_giai_quyet"]
    le_phi = primary_proc["le_phi"]
    laws = primary_proc["can_cu_phap_ly"]

    ma_tt = info["ma_thu_tuc"]
    ten_tt = info["ten_thu_tuc"]
    phan_nhom = info["phan_nhom"]
    linh_vuc = info["linh_vuc"]
    cap_tq = info.get("cap_tham_quyen", [])
    giai_doan = info.get("giai_doan_to_tung", [])

    law_ids = [c["law_id"] for c in laws]
    form_ids = [f["form_id"] for f in ho_so.get("giay_to_bat_buoc", [])]
    muc_phi = int(le_phi["muc_phi_vnd"])
    so_ngay = int(thoi_han["so_ngay_toi_da"])
    primary_status = laws[0]["trang_thai_hieu_luc"]

    # Giả lập lỗi ở bản nháp đầu tiên (revision_count == 0) để demo Judge Agent 2 lớp
    if simulate_initial_error and revision_count == 0:
        fake_fee = muc_phi + 500000
        fake_days = so_ngay + 10
        return IRACOutput(
            is_refusal=False,
            refusal_reason=None,
            direct_advice=(
                f"[BẢN NHÁP LẦN 1 — CHƯA QUA KIỂM CHỨNG]: Chào bạn, về vấn đề \"{query}\", "
                f"bạn có thể liên hệ cơ quan hành chính gần nhất để làm {ten_tt}. "
                f"Dự kiến thời gian chờ khoảng {fake_days} ngày và cần chuẩn bị khoản lệ phí {fake_fee:,} VNĐ."
            ),
            issue=(
                f"[BẢN NHÁP LẦN 1 - CHƯA QUA KIỂM CHỨNG] Giải quyết {ten_tt} (Mã: {ma_tt}) "
                f"thuộc nhóm {phan_nhom}."
            ),
            rule=(
                f"Căn cứ văn bản pháp luật liên quan đến {linh_vuc} "
                f"(Mã điều luật: {', '.join(law_ids)})."
            ),
            application=(
                f"1. Cơ quan tiếp nhận: UBND hoặc Cơ quan công an gần nhất.\n"
                f"2. Thời hạn giải quyết: Tối đa {fake_days} ngày làm việc.\n"
                f"3. Mức lệ phí: {fake_fee:,} VNĐ.\n"
                f"4. Hồ sơ: Nộp đơn đề nghị theo mẫu."
            ),
            conclusion=(
                f"Người dân chuẩn bị lệ phí {fake_fee:,} VNĐ và chờ kết quả sau {fake_days} ngày."
            ),
            citations=CitationMetadata(
                ma_thu_tuc=ma_tt,
                law_ids=law_ids,
                form_ids=[],
                muc_phi_vnd=fake_fee,
                so_ngay_toi_da=fake_days,
                cap_tham_quyen=["Cơ quan hành chính chung"],
                trang_thai_hieu_luc=primary_status,
            ),
            confidence_score=0.55,
        )

    direct_advice = build_direct_advice(
        query=query,
        primary_proc=primary_proc,
        expired_alerts=expired_alerts,
        revision_count=revision_count,
    )

    stage_note = (
        f" thuộc giai đoạn **{', '.join(giai_doan)}**"
        if giai_doan
        else ""
    )
    revision_badge = (
        f" *(đã được hiệu chỉnh chuẩn xác tại Vòng #{revision_count} theo phản biện của Judge Agent)*"
        if revision_count > 0
        else ""
    )
    history_note = (
        " *(kết hợp ngữ cảnh hội thoại trước đó)*"
        if chat_history
        else ""
    )

    issue_text = (
        f"Với câu hỏi *\"{query}\"*, chuyên viên xác định vấn đề của bạn liên quan trực tiếp đến "
        f"**{ten_tt}** (`{ma_tt}`) thuộc nhóm **{phan_nhom} — {linh_vuc}**{stage_note}{history_note}{revision_badge}."
    )

    rule_lines = []
    for c in laws:
        replace_note = (
            f" — hiện đã được thay thế bởi **{c['van_ban_thay_the']}**"
            if c.get("van_ban_thay_the")
            else ""
        )
        rule_lines.append(
            f"- Tình huống của bạn được điều chỉnh bởi **{c['so_hieu_van_ban']}** ({c['dieu_khoan_diem']}, "
            f"mã `{c['law_id']}`), hiệu lực từ `{c['ngay_hieu_luc']}` — trạng thái **{c['trang_thai_hieu_luc']}**{replace_note}."
        )

    for exp in expired_alerts:
        exp_law = exp["can_cu_phap_ly"][0]
        rule_lines.append(
            f"- ⚠️ **Lưu ý về quy định cũ đã hết hiệu lực**: `{exp['ma_va_ten_thu_tuc']['ma_thu_tuc']}` — "
            f"{exp_law['so_hieu_van_ban']} ({exp_law['dieu_khoan_diem']}) hiện đã **{exp_law['trang_thai_hieu_luc']}** "
            f"và được thay thế bằng **{exp_law.get('van_ban_thay_the')}**."
        )

    rule_text = "\n".join(rule_lines)

    forms_formatted = "\n".join(
        f"  + {g['ten_giay_to']} (mã biểu mẫu: `{g['form_id']}`)"
        for g in ho_so.get("giay_to_bat_buoc", [])
    )
    steps_formatted = "\n".join(f"  + {s}" for s in steps)
    fee_display = (
        "0 VNĐ (Hoàn toàn miễn lệ phí)"
        if muc_phi == 0
        else f"{muc_phi:,} VNĐ".replace(",", ".")
    )
    stage_explain = (
        f" *(tương ứng với giai đoạn: {', '.join(giai_doan)})*"
        if giai_doan
        else ""
    )

    application_text = (
        f"Áp dụng quy định vào trường hợp của bạn, bạn cần thực hiện theo hướng dẫn chi tiết sau:\n\n"
        f"**1. Nơi tiếp nhận & giải quyết hồ sơ:** Bạn nộp tại **{', '.join(cap_tq)}**{stage_explain}.\n\n"
        f"**2. Các bước bạn cần làm:**\n{steps_formatted}\n\n"
        f"**3. Giấy tờ bạn cần chuẩn bị (kèm mã biểu mẫu):**\n{forms_formatted}\n\n"
        f"**4. Điều kiện bắt buộc cần lưu ý:**\n"
        f"  + ⚠️ {ho_so['dieu_kien_bat_buoc']}\n\n"
        f"**5. Thời gian chờ giải quyết:** Tối đa **{so_ngay}** {thoi_han['don_vi_tinh']}. {thoi_han['chi_tiet']}\n\n"
        f"**6. Lệ phí theo quy định:** **{fee_display}**. {le_phi['mo_ta']}"
    )

    trap_warnings: List[str] = []
    q_lower = query.lower()
    if "hộ khẩu" in q_lower or "hk07" in q_lower:
        trap_warnings.append(
            "Từ ngày 01/01/2023, Sổ hộ khẩu giấy và Sổ tạm trú giấy đã hết hiệu lực hoàn toàn; "
            "cán bộ tiếp nhận không được phép yêu cầu bạn xuất trình Sổ hộ khẩu giấy."
        )
    if "cmnd" in q_lower or "9 số" in q_lower or "chứng minh nhân dân" in q_lower:
        trap_warnings.append(
            "Từ ngày 01/01/2025, mọi CMND (9 số và 12 số) đều đã hết giá trị sử dụng theo Khoản 2 Điều 46 Luật Căn cước 2023, "
            "dù hạn 15 năm in trên thẻ cũ vẫn còn."
        )
    if ma_tt == "TTHS-BL-121" and ("1 người" in q_lower or "một mình" in q_lower):
        trap_warnings.append(
            "Một người đơn lẻ không đủ điều kiện bảo lĩnh; bạn cần vận động thêm người nhà để có ít nhất 02 người thân thích cùng ký đơn cam đoan."
        )
    if ma_tt == "TTHS-TG-145" and ("lệ phí" in q_lower or "triệu" in q_lower or "phí" in q_lower):
        trap_warnings.append(
            "Thủ tục tố giác tội phạm được miễn phí hoàn toàn (0 VNĐ); tuyệt đối không chuyển tiền cho bất kỳ ai tự xưng công an qua mạng."
        )
    if ma_tt == "TTHS-LLTP-045" and ("số 2" in q_lower and "ủy quyền" in q_lower):
        trap_warnings.append(
            "Riêng Phiếu lý lịch tư pháp số 2 tuyệt đối không được ủy quyền làm thay (chỉ Phiếu số 1 mới được ủy quyền)."
        )

    conclusion_text = (
        f"Tóm lại, bạn chuẩn bị đầy đủ hồ sơ gồm các biểu mẫu ({', '.join(form_ids)}) và nộp tại **{', '.join(cap_tq)}** "
        f"để được giải quyết **{ten_tt}** (`{ma_tt}`) trong tối đa **{so_ngay} ngày**, mức lệ phí **{fee_display}**."
    )
    if trap_warnings:
        conclusion_text += "\n\n**Lời khuyên giúp bạn tránh hiểu nhầm:**\n- " + "\n- ".join(trap_warnings)

    citations = CitationMetadata(
        ma_thu_tuc=ma_tt,
        law_ids=law_ids,
        form_ids=form_ids,
        muc_phi_vnd=muc_phi,
        so_ngay_toi_da=so_ngay,
        cap_tham_quyen=cap_tq,
        trang_thai_hieu_luc=primary_status,
    )

    conf = min(0.98, max(0.82, hybrid_score))
    return IRACOutput(
        is_refusal=False,
        refusal_reason=None,
        direct_advice=direct_advice,
        issue=issue_text,
        rule=rule_text,
        application=application_text,
        conclusion=conclusion_text,
        citations=citations,
        confidence_score=round(conf, 2),
    )
