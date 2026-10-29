"""
NHÁNH PHẠM ĐỨC TRỌNG — Lớp Điều phối JudgeAgent (AGENT 03):
Tổng hợp kết quả thẩm định chéo 2 lớp (Lớp 1 Code Tất định + Lớp 2 Phản biện Ngữ nghĩa)
và tạo chỉ dẫn sửa lỗi (critique_feedback) gửi về cho Consultant Agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.core.config import GEMINI_MODEL, GOOGLE_API_KEY, has_valid_api_key
from src.judge.layer1_deterministic import verify_layer1_deterministic
from src.judge.layer2_semantic import verify_layer2_semantic_critic
from src.schemas import IRACOutput, JudgeVerificationResult


class JudgeAgent:
    """
    AGENT 03 — JUDGE / CRITIC AGENT (Tác tử Thẩm định Chéo 2 Lớp).
    """

    def __init__(self) -> None:
        self.api_key = GOOGLE_API_KEY
        self.model_name = GEMINI_MODEL

    def _has_valid_api_key(self) -> bool:
        return has_valid_api_key(self.api_key)

    def verify_layer1_deterministic(
        self,
        draft: IRACOutput,
        ground_truth_proc: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        return verify_layer1_deterministic(draft, ground_truth_proc)

    def verify_layer2_semantic_critic(
        self,
        query: str,
        draft: IRACOutput,
        ground_truth_proc: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        return verify_layer2_semantic_critic(
            query=query,
            draft=draft,
            ground_truth_proc=ground_truth_proc,
            model_name=self.model_name,
            api_key=self.api_key,
        )

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
