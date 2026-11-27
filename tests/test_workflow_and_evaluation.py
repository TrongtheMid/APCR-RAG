"""
Kiểm thử tự động cho LangGraph Workflow (Streaming, Multi-turn, Self-Revision)
và Benchmark Runner (Ablation Study 4 cấu hình, Confusion Matrix 2x2).
"""

from __future__ import annotations

import unittest

from src.consultant import ConsultantAgent
from src.evaluation import (
    compute_routing_confusion_matrix,
    evaluate_self_revision_rate,
    load_benchmark_questions,
    run_ablation_study,
)
from src.judge import JudgeAgent
from src.workflow import (
    LegalMultiAgentOrchestrator,
    contextualize_followup_query,
)


class TestWorkflowAndEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.orch = LegalMultiAgentOrchestrator(
            consultant=ConsultantAgent(use_llm=False),
            judge=JudgeAgent(use_llm=False),
        )

    def test_streaming_workflow_emits_ordered_agent_events(self) -> None:
        events = list(
            self.orch.stream(
                query="Đăng ký kết hôn với người nước ngoài nộp ở đâu?",
                phan_nhom_filter="Thủ tục hành chính",
            )
        )
        nodes = [e["node"] for e in events]
        self.assertEqual(nodes[0], "retriever")
        self.assertEqual(nodes[1], "consultant")
        self.assertEqual(nodes[2], "judge")
        self.assertEqual(nodes[-1], "__end__")

        final_state = events[-1]["state"]
        self.assertEqual(final_state["final_status"], "VERIFIED_PASS")
        self.assertEqual(final_state["primary_procedure"]["ma_va_ten_thu_tuc"]["ma_thu_tuc"], "TTHC-HT-05")
        self.assertGreaterEqual(len(final_state["workflow_trace"]), 3)

    def test_self_revision_loop_recovers_from_simulated_error(self) -> None:
        state = self.orch.invoke(
            query="Chỉ một mình tôi đứng ra làm đơn bảo lĩnh cho em trai có được không?",
            phan_nhom_filter="Tố tụng hình sự",
            simulate_initial_error=True,
        )
        self.assertEqual(state["final_status"], "REVISED_PASS_AFTER_1_LOOP")
        self.assertEqual(len(state["revision_history"]), 2)
        self.assertFalse(state["revision_history"][0]["verification"]["passed"])
        self.assertTrue(state["revision_history"][1]["verification"]["passed"])

    def test_multiturn_followup_contextualization(self) -> None:
        history = [
            {
                "query": "Muốn xin bảo lĩnh tại ngoại cho em trai thì cần điều kiện gì?",
                "procedure_code": "TTHS-BL-121",
                "procedure_name": "Thủ tục nhận bảo lĩnh cho bị can, bị cáo (xin bảo lãnh tại ngoại)",
            }
        ]
        eff = contextualize_followup_query("Lệ phí bao nhiêu?", history)
        self.assertIn("bảo lĩnh", eff.lower())

        state = self.orch.invoke(
            query="Lệ phí bao nhiêu?",
            phan_nhom_filter="Tố tụng hình sự",
            chat_history=history,
        )
        self.assertEqual(state["primary_procedure"]["ma_va_ten_thu_tuc"]["ma_thu_tuc"], "TTHS-BL-121")
        self.assertEqual(state["irac_draft"]["citations"]["muc_phi_vnd"], 0)

    def test_deterministic_refusal_for_out_of_domain_query(self) -> None:
        state = self.orch.invoke("Dự báo thời tiết hôm nay ở Hà Nội thế nào?")
        self.assertEqual(state["final_status"], "REFUSED_SAFELY")
        self.assertTrue(state["irac_draft"]["is_refusal"])

    def test_conversational_direct_advice_tone_for_cmnd_and_situations(self) -> None:
        res_cmnd = self.orch.invoke(
            "CMND 9 số cấp năm 2012 thì năm 2025 còn dùng được không?",
            phan_nhom_filter="Thủ tục hành chính",
        )
        advice = res_cmnd["irac_draft"]["direct_advice"]
        self.assertTrue(advice.startswith("Chào bạn"))
        self.assertIn("KHÔNG", advice)
        self.assertIn("01/01/2025", advice)
        self.assertEqual(res_cmnd["final_status"], "VERIFIED_PASS")

        res_bail = self.orch.invoke(
            "Chỉ một mình tôi đứng ra làm đơn bảo lĩnh cho em trai có được không?",
            phan_nhom_filter="Tố tụng hình sự",
        )
        self.assertIn("CHƯA đủ điều kiện", res_bail["irac_draft"]["direct_advice"])
        self.assertIn("02 người thân thích", res_bail["irac_draft"]["direct_advice"])
        self.assertEqual(res_bail["final_status"], "VERIFIED_PASS")

    def test_expanded_dvc_procedures_and_multiturn_topic_switch(self) -> None:
        r_gplx = self.orch.invoke(
            "tôi muốn xin cấp lại giấy phép lái xe",
            phan_nhom_filter="Thủ tục hành chính",
        )
        p_gplx = r_gplx["primary_procedure"]["ma_va_ten_thu_tuc"]
        self.assertEqual(p_gplx["ma_thu_tuc"], "TTHC-GT-01")
        self.assertEqual(r_gplx["final_status"], "VERIFIED_PASS")
        self.assertIn("135.000 VNĐ", r_gplx["irac_draft"]["direct_advice"])

        history = [
            {
                "query": "tôi muốn xin cấp lại giấy phép lái xe",
                "procedure_code": p_gplx["ma_thu_tuc"],
                "procedure_name": p_gplx["ten_thu_tuc"],
            }
        ]
        # Hỏi nối tiếp về giấy tờ -> giữ nguyên TTHC-GT-01
        r_docs = self.orch.invoke(
            "Cần chuẩn bị những giấy tờ gì?",
            phan_nhom_filter="Thủ tục hành chính",
            chat_history=history,
        )
        self.assertEqual(r_docs["primary_procedure"]["ma_va_ten_thu_tuc"]["ma_thu_tuc"], "TTHC-GT-01")
        self.assertIn("DON-GPLX-01", r_docs["irac_draft"]["direct_advice"])

        # Đổi chủ đề sang làm hộ chiếu -> tự động chuyển sang TTHC-HC-01
        r_hc = self.orch.invoke(
            "Thế thủ tục làm hộ chiếu gắn chíp mất mấy ngày?",
            phan_nhom_filter="Thủ tục hành chính",
            chat_history=history,
        )
        self.assertEqual(r_hc["primary_procedure"]["ma_va_ten_thu_tuc"]["ma_thu_tuc"], "TTHC-HC-01")

        # Câu hỏi ngoài danh mục 40 thủ tục -> Từ chối an toàn
        r_visa = self.orch.invoke(
            "Tôi muốn xin visa du lịch Mỹ",
            phan_nhom_filter="Thủ tục hành chính",
            chat_history=history,
        )
        self.assertEqual(r_visa["final_status"], "REFUSED_SAFELY")
        self.assertTrue(r_visa["irac_draft"]["is_refusal"])

    def test_benchmark_300_legal_ground_truth_integrity(self) -> None:
        import json
        from src.core.config import DEFAULT_DATA_PATH
        from src.schemas.benchmark_schema import BenchmarkQuestion

        with open(DEFAULT_DATA_PATH, "r", encoding="utf-8") as f:
            procs = json.load(f)
        proc_map = {p["ma_va_ten_thu_tuc"]["ma_thu_tuc"]: p for p in procs}
        active_codes = {
            code
            for code, p in proc_map.items()
            if p["can_cu_phap_ly"][0]["trang_thai_hieu_luc"] == "ĐANG HIỆU LỰC"
        }

        questions = load_benchmark_questions()
        self.assertEqual(len(questions), 300)

        seen_ids = set()
        seen_texts = set()
        covered_codes = set()
        hc_count = hs_count = 0

        for raw_q in questions:
            bq = BenchmarkQuestion.model_validate(raw_q)
            self.assertNotIn(bq.question_id, seen_ids)
            seen_ids.add(bq.question_id)

            norm_q = bq.question.strip().lower()
            self.assertNotIn(norm_q, seen_texts)
            seen_texts.add(norm_q)

            self.assertIn(bq.expected_ma_thu_tuc, proc_map)
            gt_proc = proc_map[bq.expected_ma_thu_tuc]
            gt_info = gt_proc["ma_va_ten_thu_tuc"]
            gt_laws = [c["law_id"] for c in gt_proc["can_cu_phap_ly"]]

            self.assertEqual(bq.phan_nhom, gt_info["phan_nhom"])
            self.assertEqual(bq.linh_vuc, gt_info["linh_vuc"])
            self.assertEqual(bq.expected_law_ids, gt_laws)
            self.assertEqual(bq.expected_muc_phi_vnd, int(gt_proc["le_phi"]["muc_phi_vnd"]))
            self.assertEqual(bq.expected_so_ngay_toi_da, int(gt_proc["thoi_han_giai_quyet"]["so_ngay_toi_da"]))
            if bq.difficulty == "Gài bẫy":
                self.assertTrue(bool(bq.trap_note and len(bq.trap_note.strip()) >= 10))

            covered_codes.add(bq.expected_ma_thu_tuc)
            if bq.phan_nhom == "Thủ tục hành chính":
                hc_count += 1
            else:
                hs_count += 1

        self.assertEqual(hc_count, 150)
        self.assertEqual(hs_count, 150)
        self.assertEqual(covered_codes, active_codes)

    def test_ablation_study_and_confusion_matrix(self) -> None:
        questions = load_benchmark_questions()
        self.assertEqual(len(questions), 300)

        cm = compute_routing_confusion_matrix(questions)
        mx = cm["matrix"]
        total_cm = (
            mx["actual_hanh_chinh_pred_hanh_chinh"]
            + mx["actual_hanh_chinh_pred_hinh_su"]
            + mx["actual_hinh_su_pred_hanh_chinh"]
            + mx["actual_hinh_su_pred_hinh_su"]
        )
        self.assertEqual(total_cm, 300)
        self.assertGreaterEqual(cm["accuracy_pct"], 98.0)

        stress = evaluate_self_revision_rate(questions[:20])
        self.assertEqual(stress["total_stress_cases"], 20)
        self.assertEqual(stress["detected_at_round1"], 20)
        self.assertEqual(stress["recovered_to_pass"], 20)
        self.assertEqual(stress["self_revision_correction_rate_pct"], 100.0)

        ab = run_ablation_study()
        self.assertEqual(len(ab["ablation_table"]), 4)
        # Full APCR-RAG phải đạt 100% P@1 & Faithfulness trên bộ 300 câu, vượt trội Baseline và cấu hình tắt Judge
        baseline_row = ab["ablation_table"][0]
        no_judge_row = ab["ablation_table"][2]
        full_row = ab["ablation_table"][3]
        self.assertEqual(full_row["precision_at_1_pct"], 100.0)
        self.assertEqual(full_row["faithfulness_pct"], 100.0)
        self.assertGreater(full_row["faithfulness_pct"], baseline_row["faithfulness_pct"])
        self.assertGreater(full_row["faithfulness_pct"], no_judge_row["faithfulness_pct"])


if __name__ == "__main__":
    unittest.main()
