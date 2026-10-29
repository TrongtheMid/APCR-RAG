"""
NHÁNH VŨ GIA ANH ĐỨC — Module IRAC Synthesizer:
- Kết nối Gemini Structured Output (nếu có GOOGLE_API_KEY).
- Tổng hợp lập luận chuẩn IRAC trực tiếp từ Lược đồ 6 trường JSON kèm xử lý Self-Revision.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

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
    Tổng hợp câu trả lời chuẩn IRAC trực tiếp từ đối tượng JSON 6 trường,
    đảm bảo đầy đủ Issue - Rule - Application - Conclusion và CitationMetadata.
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

    stage_str = f" | Giai đoạn tố tụng: {', '.join(giai_doan)}" if giai_doan else ""
    revision_badge = (
        f" (Đã hiệu chỉnh chuẩn xác tại Vòng phản biện #{revision_count} theo yêu cầu của Judge Agent)"
        if revision_count > 0
        else ""
    )
    issue_text = (
        f"Xác định thủ tục pháp lý cần thực hiện: **{ten_tt}** (Mã định danh: `{ma_tt}`){revision_badge}.\n"
        f"- **Phân nhóm**: {phan_nhom} — **Lĩnh vực**: {linh_vuc}{stage_str}.\n"
        f"- **Câu hỏi tình huống**: \"{query}\""
    )

    rule_lines = []
    for c in laws:
        replace_note = (
            f" *(Đã được thay thế bởi: {c['van_ban_thay_the']})*"
            if c.get("van_ban_thay_the")
            else ""
        )
        rule_lines.append(
            f"- **{c['so_hieu_van_ban']}** — **{c['dieu_khoan_diem']}** "
            f"(Mã định danh: `{c['law_id']}`, Ngày hiệu lực: `{c['ngay_hieu_luc']}`, "
            f"Trạng thái: **{c['trang_thai_hieu_luc']}**){replace_note}."
        )

    for exp in expired_alerts:
        exp_law = exp["can_cu_phap_ly"][0]
        rule_lines.append(
            f"- ⚠️ **CẢNH BÁO VĂN BẢN HẾT HIỆU LỰC**: `{exp['ma_va_ten_thu_tuc']['ma_thu_tuc']}` — "
            f"{exp_law['so_hieu_van_ban']} ({exp_law['dieu_khoan_diem']}) hiện ở trạng thái "
            f"**{exp_law['trang_thai_hieu_luc']}**. Văn bản thay thế: **{exp_law.get('van_ban_thay_the')}**."
        )

    rule_text = "\n".join(rule_lines)

    forms_formatted = "\n".join(
        f"  + `{g['form_id']}`: {g['ten_giay_to']}"
        for g in ho_so.get("giay_to_bat_buoc", [])
    )
    steps_formatted = "\n".join(f"  + {s}" for s in steps)
    fee_display = (
        "0 VNĐ (Miễn lệ phí hoàn toàn)"
        if muc_phi == 0
        else f"{muc_phi:,} VNĐ".replace(",", ".")
    )

    application_text = (
        f"**1. Cơ quan có thẩm quyền tiếp nhận & giải quyết**:\n"
        f"  + {', '.join(cap_tq)}"
        + (f" *(Phân theo giai đoạn tố tụng: {', '.join(giai_doan)})*" if giai_doan else "")
        + "\n\n"
        f"**2. Trình tự thực hiện chuẩn**:\n{steps_formatted}\n\n"
        f"**3. Thành phần hồ sơ bắt buộc (kèm mã biểu mẫu)**:\n{forms_formatted}\n\n"
        f"**4. Điều kiện pháp lý bắt buộc (Tiên quyết)**:\n"
        f"  + ⚠️ {ho_so['dieu_kien_bat_buoc']}\n\n"
        f"**5. Thời hạn giải quyết theo luật định**:\n"
        f"  + Tối đa **{so_ngay}** {thoi_han['don_vi_tinh']}. {thoi_han['chi_tiet']}\n\n"
        f"**6. Mức lệ phí nhà nước**:\n"
        f"  + **{fee_display}**. {le_phi['mo_ta']}"
    )

    trap_warnings: List[str] = []
    q_lower = query.lower()
    if "hộ khẩu" in q_lower or "hk07" in q_lower:
        trap_warnings.append(
            "Kể từ ngày 01/01/2023, Sổ hộ khẩu giấy và Sổ tạm trú giấy đã hết hiệu lực hoàn toàn theo Luật Cư trú 2020; "
            "cơ quan nhà nước không cấp mới và không được yêu cầu công dân xuất trình Sổ hộ khẩu giấy."
        )
    if "cmnd" in q_lower or "9 số" in q_lower or "chứng minh nhân dân" in q_lower:
        trap_warnings.append(
            "Kể từ ngày 01/01/2025, mọi Chứng minh nhân dân (9 số và 12 số) đều hết giá trị sử dụng theo Khoản 2 Điều 46 Luật Căn cước 2023, "
            "bất kể thời hạn 15 năm in trên thẻ vẫn còn."
        )
    if ma_tt == "TTHS-BL-121" and ("1 người" in q_lower or "một mình" in q_lower):
        trap_warnings.append(
            "Một cá nhân đơn lẻ KHÔNG đủ điều kiện nhận bảo lĩnh; pháp luật bắt buộc phải có ít nhất 02 người thân thích cùng đứng ra bảo lĩnh."
        )
    if ma_tt == "TTHS-TG-145" and ("lệ phí" in q_lower or "triệu" in q_lower or "phí" in q_lower):
        trap_warnings.append(
            "Thủ tục trình báo, tố giác tội phạm được MIỄN PHÍ HOÀN TOÀN (0 VNĐ). Cảnh giác tuyệt đối với các đối tượng mạo danh công an/an ninh mạng yêu cầu chuyển phí thụ lý hồ sơ."
        )
    if ma_tt == "TTHS-LLTP-045" and ("số 2" in q_lower and "ủy quyền" in q_lower):
        trap_warnings.append(
            "Phiếu lý lịch tư pháp số 2 tuyệt đối KHÔNG được ủy quyền cho người khác làm hộ theo Khoản 2 Điều 46 Luật Lý lịch tư pháp."
        )

    conclusion_text = (
        f"Để thực hiện **{ten_tt}** (`{ma_tt}`), người dân nộp hồ sơ tại **{', '.join(cap_tq)}** "
        f"với đầy đủ biểu mẫu ({', '.join(form_ids)}), thời hạn giải quyết tối đa **{so_ngay} ngày** "
        f"và mức lệ phí **{fee_display}**."
    )
    if trap_warnings:
        conclusion_text += "\n\n**CẢNH BÁO TRÁNH HIỂU NHẦM PHÁP LÝ**:\n- " + "\n- ".join(trap_warnings)

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
        issue=issue_text,
        rule=rule_text,
        application=application_text,
        conclusion=conclusion_text,
        citations=citations,
        confidence_score=round(conf, 2),
    )
