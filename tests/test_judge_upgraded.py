"""
Kiểm thử tự động cho Judge Agent 2 Lớp nâng cấp (Nhánh Phạm Đức Trọng):
- Lớp 1: Exact Match + Anti-Contradiction Regex (bắt mâu thuẫn giữa văn xuôi IRAC và citations/JSON).
- Lớp 2: General Claim-Level Semantic Critic (độ phủ mệnh đề, cực tính cấm/miễn phí, giai đoạn tố tụng).
"""

from __future__ import annotations

import unittest

from src.consultant import ConsultantAgent
from src.judge import (
    JudgeAgent,
    evaluate_claims_coverage,
    extract_days_from_text,
    extract_fees_from_text,
    verify_layer1_deterministic_detailed,
    verify_layer2_semantic_critic_detailed,
)
from src.retriever import get_retriever_agent
from src.schemas import CitationMetadata, IRACOutput


class TestJudgeUpgraded(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = get_retriever_agent()
        cls.consultant = ConsultantAgent()
        cls.judge = JudgeAgent()

    def test_regex_extractors(self) -> None:
        text = "Lệ phí nhà nước là 200.000 VNĐ (trường hợp khác 2 triệu đồng), giải quyết tối đa 10 ngày làm việc kể từ 01/01/2025."
        fees = extract_fees_from_text(text)
        days = extract_days_from_text(text)
        self.assertIn(200000, fees)
        self.assertIn(2000000, fees)
        self.assertEqual(days, [10])

    def test_layer1_catches_prose_fee_contradiction_even_when_citations_correct(self) -> None:
        """
        Kịch bản nguy hiểm với LLM thật: citations.muc_phi_vnd = 0 (đúng JSON),
        nhưng phần văn xuôi application lại bịa 'lệ phí 500.000 VNĐ'.
        Lớp 1 Anti-Contradiction Regex PHẢI bắt được lỗi này.
        """
        proc = self.retriever.procedures_by_id["TTHS-TG-145"]
        ret = self.retriever.retrieve_hybrid("Trình báo tố giác tội phạm lừa đảo qua mạng", phan_nhom="Tố tụng hình sự")
        genuine_draft = self.consultant.generate_irac_consultation(
            query="Trình báo tố giác tội phạm lừa đảo qua mạng",
            retrieved_items=ret["results"],
        )
        # Cố tình chèn mức tiền bịa vào văn xuôi nhưng giữ nguyên citations.muc_phi_vnd == 0
        tampered = genuine_draft.model_copy(
            update={
                "conclusion": genuine_draft.conclusion + " Người dân cần đóng thêm lệ phí 500.000 VNĐ khi nộp đơn."
            }
        )
        detail = verify_layer1_deterministic_detailed(tampered, proc)
        self.assertFalse(detail["passed"])
        self.assertTrue(any("Mâu Thuẫn Văn Xuôi" in e and "500,000" in e for e in detail["errors"]))
        self.assertIn(500000, detail["prose_extracted_fees"])

    def test_layer1_catches_prose_days_contradiction_even_when_citations_correct(self) -> None:
        proc = self.retriever.procedures_by_id["TTHS-BL-121"]
        ret = self.retriever.retrieve_hybrid("Thủ tục nhận bảo lĩnh tại ngoại", phan_nhom="Tố tụng hình sự")
        genuine_draft = self.consultant.generate_irac_consultation(
            query="Thủ tục nhận bảo lĩnh tại ngoại",
            retrieved_items=ret["results"],
        )
        tampered = genuine_draft.model_copy(
            update={
                "conclusion": genuine_draft.conclusion + " Thời hạn giải quyết kéo dài tối đa 45 ngày làm việc."
            }
        )
        detail = verify_layer1_deterministic_detailed(tampered, proc)
        self.assertFalse(detail["passed"])
        self.assertTrue(any("Ảo Giác Thời Hạn trong Text" in e for e in detail["errors"]))
        self.assertIn(45, detail["prose_extracted_days"])

    def test_layer1_catches_missing_form_id_in_prose(self) -> None:
        proc = self.retriever.procedures_by_id["TTHC-HT-05"]
        form_ids = [g["form_id"] for g in proc["thanh_phan_ho_so"]["giay_to_bat_buoc"]]
        law_ids = [c["law_id"] for c in proc["can_cu_phap_ly"]]
        draft = IRACOutput(
            is_refusal=False,
            issue="Đăng ký kết hôn có yếu tố nước ngoài (TTHC-HT-05).",
            rule=f"Căn cứ {law_ids[0]} ĐANG HIỆU LỰC.",
            application=(
                "Cơ quan có thẩm quyền: UBND cấp huyện (Phòng Tư pháp). "
                "Thời hạn giải quyết tối đa 15 ngày làm việc. Mức lệ phí 1.000.000 VNĐ. "
                "Điều kiện: Thẩm quyền thuộc UBND cấp huyện, không thuộc UBND cấp xã, giấy tờ nước ngoài phải hợp pháp hóa lãnh sự."
            ),
            conclusion="Nộp hồ sơ tại UBND cấp huyện, lệ phí 1.000.000 VNĐ, tối đa 15 ngày.",
            citations=CitationMetadata(
                ma_thu_tuc="TTHC-HT-05",
                law_ids=law_ids,
                form_ids=form_ids,  # citations có đủ nhưng văn xuôi cố tình không ghi mã biểu mẫu
                muc_phi_vnd=1000000,
                so_ngay_toi_da=15,
                cap_tham_quyen=["UBND cấp huyện"],
                trang_thai_hieu_luc="ĐANG HIỆU LỰC",
            ),
            confidence_score=0.9,
        )
        detail = verify_layer1_deterministic_detailed(draft, proc)
        self.assertFalse(detail["passed"])
        self.assertTrue(any("Thiếu Mã Biểu Mẫu trong Văn Xuôi" in e for e in detail["errors"]))

    def test_layer2_general_claim_level_verification_detects_missing_prohibition(self) -> None:
        """Kiểm chứng Lớp 2 tổng quát trên TTHC-HT-01 (không còn hardcode theo 3 mã cũ)."""
        proc = self.retriever.procedures_by_id["TTHC-HT-01"]
        # Bản nháp bỏ sót mệnh đề phủ định "KHÔNG yêu cầu Sổ hộ khẩu giấy"
        weak_draft = IRACOutput(
            is_refusal=False,
            issue="Thủ tục đăng ký khai sinh TTHC-HT-01.",
            rule="Luật Hộ tịch 2014 ĐANG HIỆU LỰC.",
            application=(
                "Cơ quan có thẩm quyền: UBND cấp xã nơi cư trú của cha hoặc mẹ. "
                "Trình tự: Cha mẹ nộp tờ khai đăng ký khai sinh và giấy chứng sinh tại bộ phận một cửa UBND cấp xã. "
                "Thời hạn giải quyết trong 01 ngày làm việc, mức lệ phí 0 VNĐ (miễn lệ phí)."
            ),
            conclusion="Nộp hồ sơ tại UBND cấp xã trong ngày làm việc, miễn phí.",
            citations=None,
            confidence_score=0.85,
        )
        detail = verify_layer2_semantic_critic_detailed("Đăng ký khai sinh", weak_draft, proc)
        self.assertFalse(detail["passed"])
        self.assertTrue(any("Mệnh Đề Cấm" in e or "Claim Coverage" in e for e in detail["errors"]))

    def test_layer2_passes_genuine_draft_with_full_claim_coverage(self) -> None:
        proc = self.retriever.procedures_by_id["TTHS-BL-121"]
        ret = self.retriever.retrieve_hybrid("Xin bảo lĩnh tại ngoại cho bị can cần mấy người?", phan_nhom="Tố tụng hình sự")
        draft = self.consultant.generate_irac_consultation(
            query="Xin bảo lĩnh tại ngoại cho bị can cần mấy người?",
            retrieved_items=ret["results"],
        )
        ver = self.judge.evaluate_draft("Xin bảo lĩnh tại ngoại cho bị can cần mấy người?", draft, proc)
        self.assertTrue(ver.passed)
        self.assertEqual(ver.layer1_score, 1.0)
        self.assertGreaterEqual(ver.claim_coverage_score, 0.9)
        self.assertGreater(ver.total_claims_count, 0)

    def test_claim_coverage_helper(self) -> None:
        cond = "Thời hạn tối đa là 15 ngày. Tuyệt đối KHÔNG được ủy quyền cho người khác."
        good = evaluate_claims_coverage("Thời hạn giải quyết tối đa 15 ngày và tuyệt đối không được ủy quyền cho người khác.", cond)
        bad = evaluate_claims_coverage("Thời hạn giải quyết tối đa 15 ngày và có thể nhờ người khác đi làm hộ.", cond)
        self.assertEqual(good["coverage_score"], 1.0)
        self.assertLess(bad["coverage_score"], 1.0)
        self.assertTrue(len(bad["polarity_errors"]) >= 1)


if __name__ == "__main__":
    unittest.main()
