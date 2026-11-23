"""
NHÁNH VŨ GIA ANH ĐỨC — Module IRAC Synthesizer:
- Kết nối Gemini Structured Output (nếu có GOOGLE_API_KEY).
- Tổng hợp lời tư vấn tự nhiên (`direct_advice`) như chuyên viên pháp lý trả lời trực tiếp,
  kết hợp lập luận chuẩn IRAC (`issue`, `rule`, `application`, `conclusion`) từ Lược đồ 6 trường JSON
  để đáp ứng đầy đủ kiểm chứng 2 lớp của Judge Agent.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.consultant.conversational_advisor import build_direct_advice
from src.consultant.prompts import IRAC_SYSTEM_PROMPT
from src.core.llm_factory import get_chat_llm
from src.schemas import CitationMetadata, IRACOutput


def try_generate_with_gemini(
    query: str,
    primary_proc: Dict[str, Any],
    expired_alerts: List[Dict[str, Any]],
    critique_feedback: Optional[str],
    hybrid_score: float,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[IRACOutput]:
    """Gọi Google Gemini với Structured Output (IRACOutput) nếu có API Key hợp lệ."""
    llm = get_chat_llm(temperature=0.1, model_name=model_name, api_key=api_key)
    if llm is None:
        return None
    try:
        structured_llm = llm.with_structured_output(IRACOutput)
        context_json = json.dumps(primary_proc, ensure_ascii=False, indent=2)
        expired_json = (
            json.dumps(expired_alerts, ensure_ascii=False, indent=2)
            if expired_alerts
            else "Không có"
        )
        revision_prompt = (
            f"\n\n[YÊU CẦU TỰ SỬA LỖI (SELF-REVISION) TỪ JUDGE AGENT]:\n{critique_feedback}\n"
            "Hãy sửa lại toàn bộ số liệu lệ phí, thời hạn, mã trích dẫn và bổ sung đầy đủ điều kiện bắt buộc!"
            if critique_feedback
            else ""
        )

        user_message = (
            f"{IRAC_SYSTEM_PROMPT}\n\n"
            f"CÂU HỎI CỦA NGƯỜI DÂN: {query}\n\n"
            f"DỮ LIỆU HỒ SƠ THỦ TỤC 6 TRƯỜNG (JSON GỐC):\n{context_json}\n\n"
            f"CẢNH BÁO QUY ĐỊNH CŨ HẾT HIỆU LỰC (NẾU CÓ):\n{expired_json}"
            f"{revision_prompt}"
        )
        result: IRACOutput = structured_llm.invoke(user_message)
        if not result.direct_advice:
            result.direct_advice = build_direct_advice(query, primary_proc, expired_alerts)
        result.confidence_score = round(min(0.99, max(0.75, hybrid_score)), 2)
        return result
    except Exception:
        return None


def synthesize_structured_irac(
    query: str,
    primary_proc: Dict[str, Any],
    expired_alerts: List[Dict[str, Any]],
    critique_feedback: Optional[str],
    revision_count: int,
    hybrid_score: float,
    simulate_initial_error: bool = False,
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

    issue_text = (
        f"Với câu hỏi *\"{query}\"*, chuyên viên xác định vấn đề của bạn liên quan trực tiếp đến "
        f"**{ten_tt}** (`{ma_tt}`) thuộc nhóm **{phan_nhom} — {linh_vuc}**{stage_note}{revision_badge}."
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
