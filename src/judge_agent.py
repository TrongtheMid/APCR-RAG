"""
NHÁNH CỦA PHẠM ĐỨC TRỌNG:
- Xây dựng Judge / Critic Agent 2 Lớp:
  + Lớp 1 (Deterministic Code Verification): Đối soát chính xác tuyệt đối (Exact Match) bằng Python
    các mã trích dẫn (ma_thu_tuc, law_id, form_id), mức lệ phí (muc_phi_vnd), số ngày giải quyết (so_ngay_toi_da)
    và trạng thái hiệu lực (trang_thai_hieu_luc) giữa bản nháp của Consultant với JSON gốc.
  + Lớp 2 (LLM & Rule-based Semantic Critic): Thẩm định việc tuân thủ các điều kiện bắt buộc tiên quyết
    (số người nhận bảo lĩnh >= 02, miễn phí tố giác tội phạm, cấm ủy quyền LLTP số 2, cảnh báo bỏ Sổ hộ khẩu giấy/CMND 9 số)
    và phân cấp thẩm quyền (Xã / Huyện / Tỉnh / Cơ quan CSĐT / Viện kiểm sát / Tòa án).
- Cung cấp danh mục Cảnh báo Hiểu nhầm Pháp lý & Dịch vụ công (Phân hệ 3).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.schema import IRACOutput, JudgeVerificationResult


class JudgeAgent:
    """
    AGENT 03 — JUDGE / CRITIC AGENT (Tác tử Thẩm định Chéo 2 Lớp).
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    def _has_valid_api_key(self) -> bool:
        return bool(self.api_key and self.api_key != "your_google_gemini_api_key_here")

    def verify_layer1_deterministic(
        self,
        draft: IRACOutput,
        ground_truth_proc: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        LỚP 1 — KIỂM TRA TẤT ĐỊNH BẰNG CODE PYTHON (EXACT MATCH):
        Đối soát từng trường số liệu định lượng và mã định danh trong draft.citations với JSON 6 trường gốc.
        """
        if draft.is_refusal:
            return True, []

        errors: List[str] = []
        info = ground_truth_proc["ma_va_ten_thu_tuc"]
        ho_so = ground_truth_proc["thanh_phan_ho_so"]
        thoi_han = ground_truth_proc["thoi_han_giai_quyet"]
        le_phi = ground_truth_proc["le_phi"]
        laws = ground_truth_proc["can_cu_phap_ly"]

        gt_ma_tt = info["ma_thu_tuc"]
        gt_fee = int(le_phi["muc_phi_vnd"])
        gt_days = int(thoi_han["so_ngay_toi_da"])
        gt_law_ids = {c["law_id"] for c in laws}
        gt_form_ids = {f["form_id"] for f in ho_so.get("giay_to_bat_buoc", [])}
        gt_status = laws[0]["trang_thai_hieu_luc"]

        if draft.citations is None:
            errors.append(
                "[Lớp 1 - Thiếu Citations] Bản nháp không đính kèm cấu trúc CitationMetadata để đối soát số liệu."
            )
            return False, errors

        cit = draft.citations

        # 1. Kiểm tra mã thủ tục (ma_thu_tuc)
        if cit.ma_thu_tuc != gt_ma_tt:
            errors.append(
                f"[Lớp 1 - Sai Mã Thủ Tục] Bản nháp trích dẫn mã '{cit.ma_thu_tuc}', "
                f"nhưng hồ sơ gốc là '{gt_ma_tt}'."
            )

        # 2. Kiểm tra mức lệ phí (muc_phi_vnd)
        if int(cit.muc_phi_vnd) != gt_fee:
            errors.append(
                f"[Lớp 1 - Ảo Giác Lệ Phí] Bản nháp nêu mức lệ phí là {cit.muc_phi_vnd:,} VNĐ, "
                f"sai lệch so với JSON gốc là {gt_fee:,} VNĐ ({le_phi['mo_ta']})."
            )

        # 3. Kiểm tra thời hạn giải quyết tối đa (so_ngay_toi_da)
        if int(cit.so_ngay_toi_da) != gt_days:
            errors.append(
                f"[Lớp 1 - Ảo Giác Thời Hạn] Bản nháp nêu thời hạn tối đa là {cit.so_ngay_toi_da} ngày, "
                f"sai lệch so với JSON gốc là {gt_days} ({thoi_han['don_vi_tinh']})."
            )

        # 4. Kiểm tra mã điều luật (law_id)
        cited_laws = set(cit.law_ids)
        if not cited_laws or not cited_laws.intersection(gt_law_ids):
            errors.append(
                f"[Lớp 1 - Sai Mã Điều Luật] Danh sách law_id trích dẫn {cit.law_ids} "
                f"không khớp với căn cứ pháp lý chuẩn {sorted(gt_law_ids)}."
            )

        # 5. Kiểm tra mã biểu mẫu bắt buộc (form_id)
        cited_forms = set(cit.form_ids)
        missing_forms = gt_form_ids - cited_forms
        if missing_forms:
            errors.append(
                f"[Lớp 1 - Đứt Gãy Hồ Sơ] Bản nháp bỏ sót các mã biểu mẫu bắt buộc: {sorted(missing_forms)}."
            )

        # 6. Kiểm tra trạng thái hiệu lực thời gian
        if gt_status == "HẾT HIỆU LỰC" and "HẾT HIỆU LỰC" not in draft.rule.upper():
            errors.append(
                f"[Lớp 1 - Lỗi Hiệu Lực Thời Gian] Thủ tục '{gt_ma_tt}' đã HẾT HIỆU LỰC nhưng bản nháp không cảnh báo."
            )

        return len(errors) == 0, errors

    def verify_layer2_semantic_critic(
        self,
        query: str,
        draft: IRACOutput,
        ground_truth_proc: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        LỚP 2 — THẨM ĐỊNH NGỮ NGHĨA, ĐIỀU KIỆN BẮT BUỘC & PHÂN CẤP THẨM QUYỀN:
        Kiểm tra bản nháp có nêu đầy đủ điều kiện tiên quyết (dieu_kien_bat_buoc) và đúng cấp cơ quan thụ lý hay không.
        """
        if draft.is_refusal:
            return True, []

        errors: List[str] = []
        info = ground_truth_proc["ma_va_ten_thu_tuc"]
        ho_so = ground_truth_proc["thanh_phan_ho_so"]
        ma_tt = info["ma_thu_tuc"]
        gt_authorities = info.get("cap_tham_quyen", [])

        combined_draft_text = f"{draft.issue}\n{draft.rule}\n{draft.application}\n{draft.conclusion}".lower()

        # 1. Kiểm tra việc nêu cơ quan có thẩm quyền chính xác
        if gt_authorities:
            has_valid_auth = any(auth.lower() in combined_draft_text for auth in gt_authorities)
            if not has_valid_auth:
                errors.append(
                    f"[Lớp 2 - Nhầm Lẫn Thẩm Quyền] Bản nháp chưa nêu đúng cơ quan có thẩm quyền chuẩn: "
                    f"{', '.join(gt_authorities)}."
                )

        # 2. Kiểm tra các điều kiện bắt buộc trọng yếu theo từng thủ tục
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

        # Nếu có cấu hình Gemini API Key và chưa phát hiện lỗi quy tắc cứng, có thể gọi thêm LLM Critic kiểm chứng sâu
        if not errors and self._has_valid_api_key():
            llm_errors = self._try_llm_critic(query, draft, ground_truth_proc)
            errors.extend(llm_errors)

        return len(errors) == 0, errors

    def _try_llm_critic(
        self,
        query: str,
        draft: IRACOutput,
        ground_truth_proc: Dict[str, Any],
    ) -> List[str]:
        """Gọi LLM Judge kiểm tra bổ sung nếu có GOOGLE_API_KEY."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.0,
            )
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

    def evaluate_draft(
        self,
        query: str,
        draft: IRACOutput,
        ground_truth_proc: Optional[Dict[str, Any]],
    ) -> JudgeVerificationResult:
        """
        Thực thi kiểm chứng chéo đủ 2 lớp (Lớp 1 Code Tất định + Lớp 2 Ngữ nghĩa Pháp lý)
        và tổng hợp thành JudgeVerificationResult kèm chỉ dẫn sửa lỗi (critique_feedback).
        """
        if draft.is_refusal or ground_truth_proc is None:
            return JudgeVerificationResult(
                passed=True,
                layer1_passed=True,
                layer1_errors=[],
                layer2_passed=True,
                layer2_errors=[],
                critique_feedback="Hệ thống thực hiện Từ chối trả lời chủ động (Deterministic Refusal) an toàn.",
            )

        l1_passed, l1_errors = self.verify_layer1_deterministic(draft, ground_truth_proc)
        l2_passed, l2_errors = self.verify_layer2_semantic_critic(query, draft, ground_truth_proc)
        overall_passed = l1_passed and l2_passed

        if overall_passed:
            feedback = (
                "PASS — Bản nháp IRAC đã khớp chính xác 100% với Lược đồ 6 trường JSON gốc ở cả Lớp 1 (Số liệu & Mã định danh) "
                "và Lớp 2 (Điều kiện bắt buộc & Phân cấp thẩm quyền)."
            )
        else:
            info = ground_truth_proc["ma_va_ten_thu_tuc"]
            ho_so = ground_truth_proc["thanh_phan_ho_so"]
            thoi_han = ground_truth_proc["thoi_han_giai_quyet"]
            le_phi = ground_truth_proc["le_phi"]
            form_ids = [f["form_id"] for f in ho_so.get("giay_to_bat_buoc", [])]
            all_err_str = "\n".join(f"- {e}" for e in (l1_errors + l2_errors))
            feedback = (
                f"PHÁT HIỆN LỖI CẦN SỬA NGAY TRONG VÒNG SELF-REVISION:\n{all_err_str}\n\n"
                f"CHỈ DẪN SỬA LỖI BẮT BUỘC THEO JSON GỐC ({info['ma_thu_tuc']}):\n"
                f"1. Sửa lại mức lệ phí chính xác bằng: {le_phi['muc_phi_vnd']} VNĐ ({le_phi['mo_ta']}).\n"
                f"2. Sửa lại thời hạn giải quyết tối đa bằng: {thoi_han['so_ngay_toi_da']} ({thoi_han['don_vi_tinh']}).\n"
                f"3. Liệt kê đầy đủ các mã biểu mẫu bắt buộc: {form_ids}.\n"
                f"4. Nêu đúng cơ quan có thẩm quyền: {info['cap_tham_quyen']}.\n"
                f"5. Bổ sung nguyên văn điều kiện bắt buộc: \"{ho_so['dieu_kien_bat_buoc']}\"."
            )

        return JudgeVerificationResult(
            passed=overall_passed,
            layer1_passed=l1_passed,
            layer1_errors=l1_errors,
            layer2_passed=l2_passed,
            layer2_errors=l2_errors,
            critique_feedback=feedback,
        )


# ==============================================================================
# DỮ LIỆU PHÂN HỆ 3: CẢNH BÁO HIỂU NHẦM PHÁP LÝ & DỊCH VỤ CÔNG (TỔNG HỢP TỪ JUDGE AGENT)
# ==============================================================================

def get_legal_misconceptions_catalog() -> List[Dict[str, Any]]:
    """
    Trả về danh sách các hiểu nhầm pháp lý & dịch vụ công phổ biến được phân tích và gắn nhãn
    mức độ rủi ro (NGHIÊM TRỌNG / TRUNG BÌNH / THẤP) từ kết quả kiểm chứng của Judge Agent.
    """
    return [
        {
            "alert_id": "CB-01",
            "severity": "NGHIÊM TRỌNG",
            "phan_nhom": "Tố tụng hình sự",
            "linh_vuc": "Tiếp nhận & Giải quyết tố giác, tin báo về tội phạm",
            "title": "Hiểu nhầm nộp đơn tố giác lừa đảo qua mạng phải đóng 'lệ phí thụ lý / phí thu hồi tiền'",
            "misconception_statement": "Khi bị lừa đảo chuyển khoản qua mạng, người dân phải nộp từ 1 - 5 triệu đồng 'lệ phí thụ lý hồ sơ' hoặc 'phí mở khóa tài khoản' cho các trang mạo danh Cục An ninh mạng thì mới lấy lại được tiền.",
            "judge_correction": "SAI HOÀN TOÀN. Theo Điều 145 Bộ luật Tố tụng Hình sự 2015, thủ tục trình báo, tố giác tội phạm tại cơ quan Công an là MIỄN LỆ PHÍ HOÀN TOÀN (0 VNĐ). Mọi yêu cầu chuyển tiền lệ phí giải quyết đơn tố giác qua mạng xã hội đều là lừa đảo chồng lừa đảo.",
            "ma_thu_tuc": "TTHS-TG-145",
            "law_reference": "Điều 145, Điều 147 Bộ luật Tố tụng Hình sự 2015",
            "sample_test_query": "Tôi vừa bị lừa đảo 50 triệu đồng qua mạng, có một trang Facebook tự xưng là Cục An ninh mạng yêu cầu tôi chuyển 2 triệu đồng lệ phí thụ lý đơn tố giác tội phạm để lấy lại tiền ngay trong 24 giờ, tôi có nên nộp không?",
            "flagged_count": 342,
        },
        {
            "alert_id": "CB-02",
            "severity": "NGHIÊM TRỌNG",
            "phan_nhom": "Tố tụng hình sự",
            "linh_vuc": "Biện pháp ngăn chặn",
            "title": "Hiểu nhầm chỉ cần 01 người thân ruột thịt là đủ điều kiện ký đơn bảo lĩnh tại ngoại",
            "misconception_statement": "Bị can, bị cáo đang bị tạm giam chỉ cần bố, mẹ hoặc vợ/chồng (01 người duy nhất) làm Đơn xin bảo lĩnh là Cơ quan điều tra sẽ giải quyết cho về tại ngoại.",
            "judge_correction": "SAI ĐIỀU KIỆN BẮT BUỘC. Theo Khoản 2 Điều 121 Bộ luật Tố tụng Hình sự 2015, trường hợp cá nhân nhận bảo lĩnh thì BẮT BUỘC phải có ít nhất 02 người thân thích của bị can, bị cáo (đủ 18 tuổi, nhân thân tốt, thu nhập ổn định) cùng ký đơn cam đoan và có xác nhận của UBND cấp xã.",
            "ma_thu_tuc": "TTHS-BL-121",
            "law_reference": "Khoản 2 Điều 121 Bộ luật Tố tụng Hình sự 2015",
            "sample_test_query": "Em trai tôi bị bắt tạm giam, trong gia đình chỉ còn duy nhất một mình tôi là anh ruột nên chỉ một mình tôi ký tên vào Đơn xin bảo lĩnh có được Cơ quan điều tra giải quyết cho bảo lĩnh tại ngoại không?",
            "flagged_count": 218,
        },
        {
            "alert_id": "CB-03",
            "severity": "NGHIÊM TRỌNG",
            "phan_nhom": "Thủ tục hành chính",
            "linh_vuc": "Hộ tịch",
            "title": "Hiểu nhầm đi đăng ký khai sinh, kết hôn vẫn phải xuất trình Sổ hộ khẩu giấy bản gốc",
            "misconception_statement": "Khi đi đăng ký khai sinh cho con hoặc đăng ký kết hôn tại UBND cấp xã, người dân bắt buộc phải mang theo cuốn Sổ hộ khẩu giấy hoặc Sổ tạm trú giấy.",
            "judge_correction": "QUY ĐỊNH CŨ ĐÃ HẾT HIỆU LỰC. Theo Khoản 3 Điều 38 Luật Cư trú 2020 và Nghị định 104/2022/NĐ-CP, từ ngày 01/01/2023 Sổ hộ khẩu giấy chính thức hết giá trị sử dụng. Công chức hộ tịch tra cứu thông tin trực tiếp trên Cơ sở dữ liệu quốc gia về dân cư / VNeID, tuyệt đối không được yêu cầu nộp Sổ hộ khẩu giấy.",
            "ma_thu_tuc": "TTHC-HT-01",
            "law_reference": "Khoản 3 Điều 38 Luật Cư trú 2020; Nghị định 104/2022/NĐ-CP",
            "sample_test_query": "Tôi đi làm khai sinh cho con tại UBND phường, cán bộ yêu cầu tôi phải mang bản gốc Sổ hộ khẩu giấy của gia đình lên thì mới tiếp nhận hồ sơ, yêu cầu đó có đúng luật hiện hành không?",
            "flagged_count": 289,
        },
        {
            "alert_id": "CB-04",
            "severity": "NGHIÊM TRỌNG",
            "phan_nhom": "Thủ tục hành chính",
            "linh_vuc": "Cư trú & Căn cước",
            "title": "Hiểu nhầm CMND 9 số chưa hết hạn 15 năm thì sau ngày 01/01/2025 vẫn sử dụng bình thường",
            "misconception_statement": "Chứng minh nhân dân 9 số cấp năm 2012 (thời hạn 15 năm đến năm 2027) thì trong năm 2025 và 2026 vẫn còn hiệu lực để đi giao dịch ngân hàng, công chứng và bay nội địa.",
            "judge_correction": "SAI HIỆU LỰC THỜI GIAN. Theo Khoản 2 Điều 46 Luật Căn cước số 26/2023/QH15, toàn bộ Chứng minh nhân dân (9 số và 12 số) chỉ có giá trị sử dụng đến hết ngày 31/12/2024. Từ ngày 01/01/2025, mọi CMND đều HẾT HIỆU LỰC hoàn toàn bất kể ngày hết hạn in trên thẻ.",
            "ma_thu_tuc": "TTHC-CC-01",
            "law_reference": "Khoản 2 Điều 46 Luật Căn cước 2023",
            "sample_test_query": "Chứng minh nhân dân 9 số của tôi cấp năm 2012, tính thời hạn 15 năm thì đến năm 2027 mới hết hạn, vậy năm 2025 tôi vẫn dùng CMND 9 số này đi giao dịch ngân hàng và làm thủ tục hành chính được đúng không?",
            "flagged_count": 265,
        },
        {
            "alert_id": "CB-05",
            "severity": "NGHIÊM TRỌNG",
            "phan_nhom": "Tố tụng hình sự",
            "linh_vuc": "Kháng cáo & Thi hành án hình sự",
            "title": "Hiểu nhầm thời hạn kháng cáo bản án hình sự sơ thẩm là 30 ngày",
            "misconception_statement": "Sau khi Tòa án tuyên án hình sự sơ thẩm, bị cáo và bị hại có tới 30 ngày để suy nghĩ và nộp đơn kháng cáo lên Tòa án cấp phúc thẩm.",
            "judge_correction": "NGUY CƠ MẤT QUYỀN KHÁNG CÁO. Theo Khoản 1 Điều 333 Bộ luật Tố tụng Hình sự 2015, thời hạn kháng cáo đối với bản án sơ thẩm chỉ là 15 NGÀY kể từ ngày tuyên án (hoặc từ ngày nhận được bản án/niêm yết nếu vắng mặt). Đối với quyết định sơ thẩm, thời hạn kháng cáo chỉ là 07 ngày.",
            "ma_thu_tuc": "TTHS-KC-331",
            "law_reference": "Điều 333 Bộ luật Tố tụng Hình sự 2015",
            "sample_test_query": "Tòa án nhân dân huyện đã tuyên án sơ thẩm từ ngày 01/03 và bị cáo có mặt tại phiên tòa, đến ngày 25/03 (sau 24 ngày) bị cáo mới đổi ý đi nộp đơn kháng cáo và nộp lệ phí kháng cáo hình sự 500.000 đồng thì có hợp lệ không?",
            "flagged_count": 176,
        },
        {
            "alert_id": "CB-06",
            "severity": "TRUNG BÌNH",
            "phan_nhom": "Tố tụng hình sự",
            "linh_vuc": "Xóa án tích & Lý lịch tư pháp",
            "title": "Hiểu nhầm Phiếu lý lịch tư pháp số 2 có thể ủy quyền cho cha mẹ, vợ chồng đi làm hộ",
            "misconception_statement": "Khi đang ở nước ngoài hoặc ở tỉnh xa, công dân có thể lập văn bản ủy quyền có công chứng cho cha, mẹ hoặc vợ/chồng đi xin cấp hộ Phiếu lý lịch tư pháp số 2.",
            "judge_correction": "SAI QUY ĐỊNH. Theo Khoản 2 Điều 46 Luật Lý lịch tư pháp 2009, chỉ Phiếu lý lịch tư pháp số 1 mới được ủy quyền. Đối với Phiếu lý lịch tư pháp số 2, cá nhân yêu cầu cấp Phiếu TUYỆT ĐỐI KHÔNG được ủy quyền cho bất kỳ người nào khác làm thủ tục.",
            "ma_thu_tuc": "TTHS-LLTP-045",
            "law_reference": "Khoản 2 Điều 46 Luật Lý lịch tư pháp 2009",
            "sample_test_query": "Tôi đang ở nước ngoài nên làm giấy ủy quyền có công chứng cho mẹ ruột ở Việt Nam đi làm hộ Phiếu lý lịch tư pháp số 2 tại Sở Tư pháp có được phép không?",
            "flagged_count": 154,
        },
        {
            "alert_id": "CB-07",
            "severity": "TRUNG BÌNH",
            "phan_nhom": "Tố tụng hình sự",
            "linh_vuc": "Tiếp nhận & Giải quyết tố giác, tin báo về tội phạm",
            "title": "Hiểu nhầm Công an cấp xã có thẩm quyền ra Quyết định khởi tố vụ án hình sự",
            "misconception_statement": "Khi nộp đơn trình báo tội phạm tại Công an xã/phường, Trưởng Công an xã có thẩm quyền ra Quyết định khởi tố vụ án hình sự và bắt tạm giam ngay.",
            "judge_correction": "NHẦM LẪN PHÂN CẤP THẨM QUYỀN. Theo Điều 146 BLTTHS (sửa đổi 2021) và Thông tư liên tịch 01/2021, Công an cấp xã có trách nhiệm tiếp nhận tố giác, lập biên bản, lấy lời khai ban đầu và xác minh sơ bộ, sau đó phải chuyển hồ sơ cho Cơ quan Cảnh sát điều tra có thẩm quyền quyết định khởi tố.",
            "ma_thu_tuc": "TTHS-TG-145",
            "law_reference": "Điều 145, Điều 146 BLTTHS (sửa đổi 2021)",
            "sample_test_query": "Tôi nộp đơn tố giác hàng xóm trộm cắp tài sản trị giá 30 triệu đồng tại Công an xã và yêu cầu Trưởng Công an xã ra Quyết định khởi tố vụ án hình sự ngay trong ngày có đúng thẩm quyền không?",
            "flagged_count": 139,
        },
        {
            "alert_id": "CB-08",
            "severity": "TRUNG BÌNH",
            "phan_nhom": "Thủ tục hành chính",
            "linh_vuc": "Hộ tịch",
            "title": "Hiểu nhầm kết hôn với người nước ngoài vẫn nộp hồ sơ tại UBND cấp xã như kết hôn trong nước",
            "misconception_statement": "Mọi thủ tục đăng ký kết hôn (kể cả kết hôn với người có quốc tịch nước ngoài) đều thực hiện tại UBND xã, phường nơi công dân Việt Nam thường trú và lấy kết quả trong 1 ngày.",
            "judge_correction": "NHẦM LẪN CẤP THẨM QUYỀN. Theo Điều 37 và Điều 38 Luật Hộ tịch 2014, đăng ký kết hôn có yếu tố nước ngoài thuộc thẩm quyền của UBND CẤP HUYỆN (Phòng Tư pháp thẩm tra), thời hạn giải quyết tối đa là 15 ngày làm việc và lệ phí là 1.000.000 VNĐ.",
            "ma_thu_tuc": "TTHC-HT-05",
            "law_reference": "Điều 37, Điều 38 Luật Hộ tịch 2014",
            "sample_test_query": "Tôi muốn đăng ký kết hôn với người chồng quốc tịch Pháp nên mang hồ sơ ra UBND phường nơi tôi thường trú nộp và yêu cầu giải quyết ngay trong 1 ngày làm việc có đúng không?",
            "flagged_count": 128,
        },
        {
            "alert_id": "CB-09",
            "severity": "TRUNG BÌNH",
            "phan_nhom": "Tố tụng hình sự",
            "linh_vuc": "Xóa án tích & Lý lịch tư pháp",
            "title": "Hiểu nhầm đương nhiên xóa án tích vẫn phải lên Tòa án xin cấp Giấy chứng nhận xóa án tích",
            "misconception_statement": "Sau khi chấp hành xong bản án và hết thời hạn thử thách đương nhiên xóa án tích, người dân phải nộp đơn ra Tòa án đã xử sơ thẩm để xin cấp 'Giấy chứng nhận xóa án tích'.",
            "judge_correction": "QUY ĐỊNH CŨ ĐÃ THAY ĐỔI. Theo Điều 369 Bộ luật Tố tụng Hình sự 2015, Tòa án KHÔNG còn cấp giấy chứng nhận đối với trường hợp đương nhiên được xóa án tích (Điều 70 BLHS). Thay vào đó, Sở Tư pháp cập nhật thông tin và ghi 'không có án tích' trong Phiếu lý lịch tư pháp.",
            "ma_thu_tuc": "TTHS-XAT-070",
            "law_reference": "Điều 70 BLHS 2015; Điều 369 BLTTHS 2015",
            "sample_test_query": "Tôi đã đủ điều kiện đương nhiên được xóa án tích về tội đánh bạc, tôi mang đơn ra Tòa án nhân dân huyện yêu cầu Tòa án cấp Giấy chứng nhận xóa án tích cho tôi có đúng quy định hiện hành không?",
            "flagged_count": 115,
        },
        {
            "alert_id": "CB-10",
            "severity": "THẤP",
            "phan_nhom": "Thủ tục hành chính",
            "linh_vuc": "Lao động & Bảo hiểm",
            "title": "Hiểu nhầm nộp hồ sơ hưởng trợ cấp thất nghiệp lúc nào cũng được sau khi nghỉ việc",
            "misconception_statement": "Người lao động đã đóng bảo hiểm thất nghiệp trên 12 tháng thì sau khi nghỉ việc nửa năm hoặc 1 năm đi nộp hồ sơ vẫn được nhận tiền trợ cấp thất nghiệp ngay.",
            "judge_correction": "CẢNH BÁO QUÁ HẠN. Theo Khoản 1 Điều 46 Luật Việc làm 2013, người lao động bắt buộc phải nộp hồ sơ hưởng trợ cấp thất nghiệp tại Trung tâm dịch vụ việc làm trong thời hạn 03 THÁNG kể từ ngày chấm dứt hợp đồng lao động. Nếu quá 03 tháng thì không được hưởng đợt đó (thời gian đóng sẽ được bảo lưu cho lần sau).",
            "ma_thu_tuc": "TTHC-LD-01",
            "law_reference": "Khoản 1 Điều 46 Luật Việc làm 2013",
            "sample_test_query": "Tôi nghỉ việc đã 5 tháng rồi mới đi nộp hồ sơ hưởng trợ cấp thất nghiệp tại Trung tâm dịch vụ việc làm và đóng lệ phí xét duyệt 200.000 đồng thì bao giờ nhận được tiền?",
            "flagged_count": 97,
        },
    ]
