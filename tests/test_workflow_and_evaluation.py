"""
Kiểm thử tự động cho LangGraph Workflow (Streaming, Multi-turn, Self-Revision)
và Benchmark Runner (Ablation Study 4 cấu hình, Confusion Matrix 2x2).
"""

from __future__ import annotations

import unittest

from src.evaluation import (
    compute_routing_confusion_matrix,
    evaluate_self_revision_rate,
    load_benchmark_questions,
    run_ablation_study,
)
from src.workflow import (
    contextualize_followup_query,
    run_legal_multi_agent_workflow,
    stream_legal_multi_agent_workflow,
)


class TestWorkflowAndEvaluation(unittest.TestCase):
    def test_streaming_workflow_emits_ordered_agent_events(self) -> None:
        events = list(
            stream_legal_multi_agent_workflow(
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
        state = run_legal_multi_agent_workflow(
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

        state = run_legal_multi_agent_workflow(
            query="Lệ phí bao nhiêu?",
            phan_nhom_filter="Tố tụng hình sự",
            chat_history=history,
        )
        self.assertEqual(state["primary_procedure"]["ma_va_ten_thu_tuc"]["ma_thu_tuc"], "TTHS-BL-121")
        self.assertEqual(state["irac_draft"]["citations"]["muc_phi_vnd"], 0)

    def test_deterministic_refusal_for_out_of_domain_query(self) -> None:
        state = run_legal_multi_agent_workflow("Dự báo thời tiết hôm nay ở Hà Nội thế nào?")
        self.assertEqual(state["final_status"], "REFUSED_SAFELY")
        self.assertTrue(state["irac_draft"]["is_refusal"])

    def test_conversational_direct_advice_tone_for_cmnd_and_situations(self) -> None:
        res_cmnd = run_legal_multi_agent_workflow(
            "CMND 9 số cấp năm 2012 thì năm 2025 còn dùng được không?",
            phan_nhom_filter="Thủ tục hành chính",
        )
        advice = res_cmnd["irac_draft"]["direct_advice"]
        self.assertTrue(advice.startswith("Chào bạn"))
        self.assertIn("KHÔNG", advice)
        self.assertIn("01/01/2025", advice)
        self.assertEqual(res_cmnd["final_status"], "VERIFIED_PASS")

        res_bail = run_legal_multi_agent_workflow(
            "Chỉ một mình tôi đứng ra làm đơn bảo lĩnh cho em trai có được không?",
            phan_nhom_filter="Tố tụng hình sự",
        )
        self.assertIn("CHƯA đủ điều kiện", res_bail["irac_draft"]["direct_advice"])
        self.assertIn("02 người thân thích", res_bail["irac_draft"]["direct_advice"])
        self.assertEqual(res_bail["final_status"], "VERIFIED_PASS")

    def test_ablation_study_and_confusion_matrix(self) -> None:
        questions = load_benchmark_questions()
        self.assertEqual(len(questions), 50)

        cm = compute_routing_confusion_matrix(questions)
        mx = cm["matrix"]
        total_cm = (
            mx["actual_hanh_chinh_pred_hanh_chinh"]
            + mx["actual_hanh_chinh_pred_hinh_su"]
            + mx["actual_hinh_su_pred_hanh_chinh"]
            + mx["actual_hinh_su_pred_hinh_su"]
        )
        self.assertEqual(total_cm, 50)
        self.assertGreaterEqual(cm["accuracy_pct"], 90.0)

        stress = evaluate_self_revision_rate(questions[:10])
        self.assertEqual(stress["total_stress_cases"], 10)
        self.assertEqual(stress["detected_at_round1"], 10)
        self.assertEqual(stress["recovered_to_pass"], 10)
        self.assertEqual(stress["self_revision_correction_rate_pct"], 100.0)

        ab = run_ablation_study()
        self.assertEqual(len(ab["ablation_table"]), 4)
        # Full APCR-RAG phải có Faithfulness cao hơn Single-Agent Baseline và cấu hình không có Judge
        baseline_row = ab["ablation_table"][0]
        no_judge_row = ab["ablation_table"][2]
        full_row = ab["ablation_table"][3]
        self.assertGreater(full_row["faithfulness_pct"], baseline_row["faithfulness_pct"])
        self.assertGreater(full_row["faithfulness_pct"], no_judge_row["faithfulness_pct"])


if __name__ == "__main__":
    unittest.main()
