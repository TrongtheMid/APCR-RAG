"""
NHÁNH PHẠM ĐỨC TRỌNG — Lớp Điều phối JudgeAgent (AGENT 03):
Tổng hợp kết quả thẩm định chéo 2 lớp:
- Lớp 1: Code Tất định (Exact Match metadata + Anti-Contradiction Regex trên văn xuôi).
- Lớp 2: Phản biện Ngữ nghĩa Tổng quát cấp mệnh đề (Claim-Level Semantic Critic).
Và tạo chỉ dẫn sửa lỗi (`critique_feedback`) gửi ngược về Consultant Agent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.core.config import GEMINI_MODEL, GOOGLE_API_KEY, has_valid_api_key
from src.judge.layer1_deterministic import (
    verify_layer1_deterministic,
    verify_layer1_deterministic_detailed,
)
from src.judge.layer2_semantic import (
    verify_layer2_semantic_critic,
    verify_layer2_semantic_critic_detailed,
)
from src.schemas import IRACOutput, JudgeVerificationResult


class JudgeAgent:
    """
    AGENT 03 — JUDGE / CRITIC AGENT (Tác tử Thẩm định Chéo 2 Lớp nâng cấp).
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
        Thực thi kiểm chứng chéo đủ 2 lớp (Lớp 1 Code Tất định + Anti-Contradiction Regex,
        Lớp 2 Claim-Level Semantic Critic) và tổng hợp thành JudgeVerificationResult.
        """
        if draft.is_refusal or ground_truth_proc is None:
            return JudgeVerificationResult(
                passed=True,
                layer1_passed=True,
                layer1_errors=[],
                layer2_passed=True,
                layer2_errors=[],
                critique_feedback="Hệ thống thực hiện Từ chối trả lời chủ động (Deterministic Refusal) an toàn.",
                layer1_score=1.0,
                layer2_score=1.0,
                claim_coverage_score=1.0,
            )

        l1_detail = verify_layer1_deterministic_detailed(draft, ground_truth_proc)
        l2_detail = verify_layer2_semantic_critic_detailed(
            query=query,
            draft=draft,
            ground_truth_proc=ground_truth_proc,
            model_name=self.model_name,
            api_key=self.api_key,
        )

        l1_passed = bool(l1_detail["passed"])
        l1_errors = list(l1_detail["errors"])
        l2_passed = bool(l2_detail["passed"])
        l2_errors = list(l2_detail["errors"])
        overall_passed = l1_passed and l2_passed

        if overall_passed:
            feedback = (
                f"PASS — Bản nháp IRAC đạt chuẩn ở cả Lớp 1 (Exact Match & Regex văn xuôi: "
                f"{l1_detail['score'] * 100:.0f}%) và Lớp 2 (Claim-Level Coverage: "
                f"{l2_detail['verified_claims_count']}/{l2_detail['total_claims_count']} mệnh đề — "
                f"{l2_detail['claim_coverage_score'] * 100:.0f}%)."
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
                f"1. Sửa lại mức lệ phí chính xác (cả citations và văn xuôi) bằng: {le_phi['muc_phi_vnd']} VNĐ ({le_phi['mo_ta']}).\n"
                f"2. Sửa lại thời hạn giải quyết tối đa (cả citations và văn xuôi) bằng: {thoi_han['so_ngay_toi_da']} ({thoi_han['don_vi_tinh']}).\n"
                f"3. Liệt kê đầy đủ các mã biểu mẫu bắt buộc trong văn bản: {form_ids}.\n"
                f"4. Nêu đúng cơ quan có thẩm quyền: {info['cap_tham_quyen']}.\n"
                f"5. Bổ sung đầy đủ các mệnh đề điều kiện bắt buộc: \"{ho_so['dieu_kien_bat_buoc']}\"."
            )

        return JudgeVerificationResult(
            passed=overall_passed,
            layer1_passed=l1_passed,
            layer1_errors=l1_errors,
            layer2_passed=l2_passed,
            layer2_errors=l2_errors,
            critique_feedback=feedback,
            layer1_score=float(l1_detail["score"]),
            layer2_score=float(l2_detail["score"]),
            prose_extracted_fees=list(l1_detail["prose_extracted_fees"]),
            prose_extracted_days=list(l1_detail["prose_extracted_days"]),
            claim_coverage_score=float(l2_detail["claim_coverage_score"]),
            verified_claims_count=int(l2_detail["verified_claims_count"]),
            total_claims_count=int(l2_detail["total_claims_count"]),
        )
