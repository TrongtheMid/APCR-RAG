"""
NHÁNH CỦA VŨ GIA ANH ĐỨC:
- Thiết kế System Prompt IRAC Đa lĩnh vực (Thủ tục Hành chính & Tố tụng Hình sự).
- Cài đặt Cơ chế Từ chối Trả lời Chủ động (Deterministic Refusal).
- Xây dựng Luồng Tự sửa lỗi (Self-Revision Loop) dựa trên Critique Feedback từ Judge Agent.
- Hàm trích xuất dữ liệu Đồ thị Quan hệ 6 trường (Nodes & Edges) phục vụ trực quan hóa giao diện.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.schema import CitationMetadata, IRACOutput


# ==============================================================================
# SYSTEM PROMPT CHUẨN LẬP LUẬN PHÁP LÝ IRAC & SELF-REVISION
# ==============================================================================

IRAC_SYSTEM_PROMPT = """Bạn là Tác tử Tư vấn Pháp lý Chuyên sâu (CONSULTANT AGENT) trong Hệ thống Tư vấn về Thủ tục Hành chính và Tố tụng Hình sự.
Nhiệm vụ của bạn là phân tích tình huống của người dân dựa TUYỆT ĐỐI vào Hồ sơ Thủ tục JSON 6 trường được cung cấp và trình bày theo khuôn mẫu lập luận pháp lý chuẩn IRAC:

1. [I - ISSUE (Vấn đề pháp lý & Thủ tục cần giải quyết)]:
   - Xác định rõ tên thủ tục hành chính hoặc thủ tục tố tụng hình sự cần thực hiện kèm mã định danh (ma_thu_tuc).
   - Xác định phân nhóm (Thủ tục hành chính / Tố tụng hình sự) và giai đoạn tố tụng (nếu có).

2. [R - RULE (Quy định pháp luật áp dụng)]:
   - Trích dẫn chính xác Số hiệu văn bản, Điều, Khoản, Điểm và mã định danh (law_id).
   - Nêu rõ trạng thái hiệu lực pháp lý (ĐANG HIỆU LỰC / HẾT HIỆU LỰC). Nếu người dân hỏi quy định cũ đã hết hiệu lực (như Sổ hộ khẩu giấy, CMND 9 số), phải cảnh báo rõ văn bản thay thế.

3. [A - APPLICATION (Áp dụng chi tiết vào tình huống)]:
   - Cơ quan có thẩm quyền tiếp nhận & giải quyết (phân biệt rõ cấp Xã / Huyện / Tỉnh hoặc Công an xã / Cơ quan CSĐT / Viện kiểm sát / Tòa án theo giai đoạn).
   - Trình tự thực hiện từng bước.
   - Danh mục thành phần hồ sơ bắt buộc kèm mã biểu mẫu (form_id) và ĐIỀU KIỆN BẮT BUỘC (ví dụ: ít nhất 02 người thân thích bảo lĩnh, không được ủy quyền Phiếu LLTP số 2...).
   - Thời hạn giải quyết tối đa (so_ngay_toi_da + đơn vị tính).
   - Mức lệ phí chính xác bằng số (muc_phi_vnd VNĐ).

4. [C - CONCLUSION (Kết luận & Cảnh báo quan trọng)]:
   - Tóm tắt hành động người dân cần thực hiện ngay và lưu ý phòng tránh hiểu nhầm / lừa đảo.

QUY TẮC AN TOÀN BẮT BUỘC:
- Tuyệt đối KHÔNG tự suy diễn mức lệ phí, số ngày giải quyết hoặc tên biểu mẫu ngoài JSON gốc.
- Nếu có chỉ dẫn sửa lỗi (CRITIQUE FEEDBACK) từ Judge Agent, bạn BẮT BUỘC phải khắc phục triệt để tất cả các lỗi được chỉ ra.
"""

REFUSAL_TEMPLATE_MESSAGE = (
    "Hệ thống chủ động từ chối đưa ra lời khuyên pháp lý suy đoán (Deterministic Refusal) "
    "do câu hỏi nằm ngoài phạm vi Cơ sở dữ liệu Thủ tục Hành chính & Tố tụng Hình sự đã được kiểm chứng, "
    "hoặc độ tương đồng ngữ nghĩa dưới ngưỡng an toàn. Vui lòng cung cấp thêm thông tin cụ thể về thủ tục "
    "(Hộ tịch, Cư trú, Căn cước, Lao động, BHXH, Tố giác tội phạm, Bảo lĩnh, Xóa án tích, Kháng cáo) "
    "hoặc liên hệ Tổng đài Dịch vụ công Quốc gia 18001096 / Trợ giúp pháp lý Nhà nước."
)


class ConsultantAgent:
    """
    AGENT 02 — CONSULTANT AGENT (Tác tử Tư vấn theo Khuôn mẫu IRAC):
    - Kiểm tra ngưỡng an toàn để Từ chối trả lời chủ động (Deterministic Refusal).
    - Sinh câu trả lời cấu trúc IRAC kèm CitationMetadata.
    - Tự hiệu chỉnh (Self-Revision) khi nhận phản hồi từ Judge Agent.
    """

    def __init__(self, min_score_threshold: Optional[float] = None) -> None:
        self.min_score_threshold = (
            min_score_threshold
            if min_score_threshold is not None
            else float(os.getenv("MIN_RETRIEVAL_SCORE", "0.18"))
        )
        self.api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    def _has_valid_api_key(self) -> bool:
        return bool(self.api_key and self.api_key != "your_google_gemini_api_key_here")

    def check_deterministic_refusal(
        self, query: str, retrieved_items: List[Dict[str, Any]]
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

        # Kiểm tra câu hỏi ngoài lề rõ rệt (ví dụ: thời tiết, nấu ăn, bóng đá, chứng khoán...)
        out_of_domain_keywords = [
            "thời tiết hôm nay", "công thức nấu ăn", "kết quả bóng đá", "giá vàng hôm nay",
            "viết code game", "dự báo thời tiết", "bói toán", "xổ số",
        ]
        q_lower = query.lower()
        if best_score < self.min_score_threshold or any(k in q_lower for k in out_of_domain_keywords):
            return IRACOutput(
                is_refusal=True,
                refusal_reason=(
                    f"Điểm tương đồng truy xuất cao nhất ({best_score:.2f}) thấp hơn ngưỡng an toàn "
                    f"({self.min_score_threshold:.2f}) hoặc nội dung câu hỏi nằm ngoài phạm vi Thủ tục Hành chính & Tố tụng Hình sự."
                ),
                issue="Câu hỏi chưa đủ căn cứ pháp lý xác thực trong hệ thống.",
                rule="Cơ chế Từ chối Trả lời Chủ động (Deterministic Refusal).",
                application=REFUSAL_TEMPLATE_MESSAGE,
                conclusion="Vui lòng đặt câu hỏi cụ thể về thủ tục hành chính công hoặc tố tụng hình sự.",
                citations=None,
                confidence_score=round(best_score, 2),
            )

        return None

    def generate_irac_consultation(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        expired_alerts: Optional[List[Dict[str, Any]]] = None,
        critique_feedback: Optional[str] = None,
        revision_count: int = 0,
        simulate_initial_error: bool = False,
    ) -> IRACOutput:
        """
        Sinh bản nháp tư vấn theo cấu trúc IRAC từ hồ sơ JSON 6 trường.
        Nếu có critique_feedback từ Judge Agent (revision_count > 0), thực hiện Self-Revision.
        Nếu simulate_initial_error=True ở vòng đầu (revision_count == 0), giả lập một bản nháp có lỗi
        để minh họa trực quan cơ chế Judge Agent 2 lớp bắt lỗi và Consultant tự sửa ở vòng lặp sau.
        """
        refusal = self.check_deterministic_refusal(query, retrieved_items)
        if refusal is not None:
            return refusal

        primary_entry = retrieved_items[0]
        primary_proc: Dict[str, Any] = primary_entry["procedure"]
        hybrid_score = float(primary_entry.get("hybrid_score", 0.92))

        # Nếu có cấu hình Gemini API Key và không bật chế độ giả lập lỗi cố định, thử gọi LLM Structured Output
        if self._has_valid_api_key() and not (simulate_initial_error and revision_count == 0):
            llm_output = self._try_generate_with_gemini(
                query=query,
                primary_proc=primary_proc,
                expired_alerts=expired_alerts or [],
                critique_feedback=critique_feedback,
                hybrid_score=hybrid_score,
            )
            if llm_output is not None:
                return llm_output

        # Chế độ tổng hợp IRAC chuẩn xác từ Lược đồ 6 trường JSON (Deterministic IRAC Synthesizer)
        return self._synthesize_structured_irac(
            query=query,
            primary_proc=primary_proc,
            expired_alerts=expired_alerts or [],
            critique_feedback=critique_feedback,
            revision_count=revision_count,
            hybrid_score=hybrid_score,
            simulate_initial_error=simulate_initial_error,
        )

    def _try_generate_with_gemini(
        self,
        query: str,
        primary_proc: Dict[str, Any],
        expired_alerts: List[Dict[str, Any]],
        critique_feedback: Optional[str],
        hybrid_score: float,
    ) -> Optional[IRACOutput]:
        """Gọi Google Gemini với Structured Output (IRACOutput) nếu có API Key hợp lệ."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.1,
            )
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

    def _synthesize_structured_irac(
        self,
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

        # Nếu bật chế độ giả lập lỗi ở bản nháp đầu tiên (revision_count == 0) để demo Judge Agent 2 lớp
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
                    form_ids=[],  # Cố tình thiếu form_id và sai lệ phí/thời hạn để Judge Lớp 1 + Lớp 2 bắt lỗi
                    muc_phi_vnd=fake_fee,
                    so_ngay_toi_da=fake_days,
                    cap_tham_quyen=["Cơ quan hành chính chung"],
                    trang_thai_hieu_luc=primary_status,
                ),
                confidence_score=0.55,
            )

        # 1. Xây dựng [I - ISSUE]
        stage_str = (
            f" | Giai đoạn tố tụng: {', '.join(giai_doan)}"
            if giai_doan
            else ""
        )
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

        # 2. Xây dựng [R - RULE]
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

        # 3. Xây dựng [A - APPLICATION]
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

        # 4. Xây dựng [C - CONCLUSION]
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


def extract_procedure_graph_data(proc_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    HẠNG MỤC 2.5 — Trích xuất Dữ liệu Đồ thị Quan hệ 6 trường (Nodes & Edges)
    để hiển thị trực quan bên cột phải màn hình Hỏi — Đáp (Streamlit-Agraph / Interactive Viewer).
    """
    info = proc_dict["ma_va_ten_thu_tuc"]
    ho_so = proc_dict["thanh_phan_ho_so"]
    thoi_han = proc_dict["thoi_han_giai_quyet"]
    le_phi = proc_dict["le_phi"]
    laws = proc_dict["can_cu_phap_ly"]

    root_id = info["ma_thu_tuc"]
    nodes: List[Dict[str, Any]] = [
        {
            "id": root_id,
            "label": f"{root_id}\n{info['ten_thu_tuc'][:42]}...",
            "full_title": info["ten_thu_tuc"],
            "group": "procedure",
            "color": "#2563eb",
            "size": 30,
            "details": f"Phân nhóm: {info['phan_nhom']} | Lĩnh vực: {info['linh_vuc']}",
        }
    ]
    edges: List[Dict[str, str]] = []

    # 1. Nodes Cơ quan thẩm quyền
    for idx, auth in enumerate(info.get("cap_tham_quyen", [])):
        node_id = f"{root_id}_AUTH_{idx + 1}"
        nodes.append(
            {
                "id": node_id,
                "label": f"Cơ quan:\n{auth}",
                "full_title": f"Cơ quan thẩm quyền: {auth}",
                "group": "authority",
                "color": "#059669",
                "size": 22,
                "details": f"Giai đoạn: {', '.join(info.get('giai_doan_to_tung', [])) or 'Hành chính công'}",
            }
        )
        edges.append({"source": root_id, "target": node_id, "label": "THẨM_QUYỀN"})

    # 2. Nodes Thành phần hồ sơ (Biểu mẫu)
    for form in ho_so.get("giay_to_bat_buoc", []):
        node_id = form["form_id"]
        nodes.append(
            {
                "id": node_id,
                "label": f"Biểu mẫu:\n{form['form_id']}",
                "full_title": form["ten_giay_to"],
                "group": "form",
                "color": "#d97706",
                "size": 20,
                "details": form["ten_giay_to"],
            }
        )
        edges.append({"source": root_id, "target": node_id, "label": "YÊU_CẦU_HỒ_SƠ"})

    # 3. Node Thời hạn giải quyết
    time_id = f"{root_id}_TIME"
    nodes.append(
        {
            "id": time_id,
            "label": f"Thời hạn:\nTối đa {thoi_han['so_ngay_toi_da']} ngày",
            "full_title": f"Tối đa {thoi_han['so_ngay_toi_da']} {thoi_han['don_vi_tinh']}",
            "group": "deadline",
            "color": "#7c3aed",
            "size": 21,
            "details": thoi_han["chi_tiet"],
        }
    )
    edges.append({"source": root_id, "target": time_id, "label": "THỜI_HẠN"})

    # 4. Node Lệ phí
    fee_id = f"{root_id}_FEE"
    fee_val = le_phi["muc_phi_vnd"]
    fee_label = "Miễn phí (0đ)" if fee_val == 0 else f"{fee_val:,} VNĐ"
    nodes.append(
        {
            "id": fee_id,
            "label": f"Lệ phí:\n{fee_label}",
            "full_title": f"Mức phí: {fee_label}",
            "group": "fee",
            "color": "#db2777",
            "size": 21,
            "details": le_phi["mo_ta"],
        }
    )
    edges.append({"source": root_id, "target": fee_id, "label": "LỆ_PHÍ"})

    # 5. Nodes Căn cứ pháp lý
    for law in laws:
        node_id = law["law_id"]
        is_active = law["trang_thai_hieu_luc"] == "ĐANG HIỆU LỰC"
        nodes.append(
            {
                "id": node_id,
                "label": f"Điều luật:\n{law['law_id']}\n({law['trang_thai_hieu_luc']})",
                "full_title": f"{law['so_hieu_van_ban']} — {law['dieu_khoan_diem']}",
                "group": "law",
                "color": "#0284c7" if is_active else "#dc2626",
                "size": 22,
                "details": f"Hiệu lực: {law['ngay_hieu_luc']} | Trạng thái: {law['trang_thai_hieu_luc']}",
            }
        )
        edges.append({"source": root_id, "target": node_id, "label": "CĂN_CỨ_PHÁP_LÝ"})

    return {"nodes": nodes, "edges": edges, "procedure": proc_dict}
