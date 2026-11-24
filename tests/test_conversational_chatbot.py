"""
Kiểm thử tự động cho Chế độ Chatbot Hội thoại Tự nhiên (Intent Router + Conversational Replies):
- Chào hỏi / xã giao ('xin chào', 'cảm ơn bạn').
- Hỏi danh sách dịch vụ ('đâu là các dịch vụ mà bạn hỗ trợ').
- Hỏi nối tiếp tự do ('tóm tắt ngắn gọn hơn').
- Bộ lọc refusal_guard chặn câu hỏi chung chung không có thuật ngữ pháp lý.
"""

from __future__ import annotations

import unittest

from src.consultant.refusal_guard import check_deterministic_refusal
from src.core.llm_factory import extract_response_text
from src.retriever import get_retriever_agent
from src.workflow import classify_chat_intent, run_legal_multi_agent_workflow


class TestConversationalChatbot(unittest.TestCase):
    def test_intent_classification(self) -> None:
        self.assertEqual(classify_chat_intent("xin chào"), "greeting")
        self.assertEqual(classify_chat_intent("Cảm ơn bạn nhiều nhé!"), "greeting")
        self.assertEqual(classify_chat_intent("đâu là các dịch vụ mà bạn hỗ trợ"), "services")
        self.assertEqual(classify_chat_intent("Bạn tư vấn được những thủ tục nào?"), "services")
        self.assertEqual(
            classify_chat_intent(
                "Tóm tắt ngắn gọn hơn giúp tôi",
                chat_history=[{"query": "Xin cấp lại GPLX", "procedure_name": "Cấp lại GPLX"}],
            ),
            "followup_freeform",
        )
        self.assertEqual(
            classify_chat_intent(
                "Đăng ký khai sinh cần giấy tờ gì?",
                matched_expansions=["Thủ tục đăng ký khai sinh"],
            ),
            "legal_qa",
        )

    def test_greeting_and_services_workflow_do_not_misclassify_procedure(self) -> None:
        r_greet = run_legal_multi_agent_workflow("xin chào")
        self.assertEqual(r_greet["final_status"], "CONVERSATIONAL_REPLY")
        self.assertIsNone(r_greet.get("primary_procedure"))
        self.assertTrue(len(r_greet["irac_draft"]["direct_advice"]) > 20)

        r_srv = run_legal_multi_agent_workflow("đâu là các dịch vụ mà bạn hỗ trợ")
        self.assertEqual(r_srv["final_status"], "CONVERSATIONAL_REPLY")
        self.assertIsNone(r_srv.get("primary_procedure"))
        advice = r_srv["irac_draft"]["direct_advice"]
        self.assertIn("Thủ tục hành chính", advice)
        self.assertIn("Tố tụng hình sự", advice)
        self.assertIn("40 thủ tục", advice)

    def test_followup_freeform_keeps_previous_procedure_context(self) -> None:
        history = [
            {
                "query": "Tôi muốn xin cấp lại giấy phép lái xe",
                "procedure_code": "TTHC-GT-01",
                "procedure_name": "Thủ tục cấp đổi, cấp lại Giấy phép lái xe",
                "direct_advice": "Nộp tại Sở Giao thông vận tải hoặc trực tuyến Cổng DVC Quốc gia, lệ phí 135.000 VNĐ, tối đa 5 ngày làm việc.",
            }
        ]
        r_sum = run_legal_multi_agent_workflow("Tóm tắt ngắn gọn lại giúp mình", chat_history=history)
        self.assertEqual(r_sum["final_status"], "CONVERSATIONAL_REPLY")
        self.assertEqual(r_sum["primary_procedure"]["ma_va_ten_thu_tuc"]["ma_thu_tuc"], "TTHC-GT-01")
        self.assertTrue(len(r_sum["irac_draft"]["direct_advice"]) > 20)

    def test_generic_service_query_blocked_by_refusal_guard_directly(self) -> None:
        retriever = get_retriever_agent()
        ret = retriever.retrieve_hybrid("đâu là các dịch vụ mà bạn hỗ trợ", top_k=3)
        refusal = check_deterministic_refusal(
            "đâu là các dịch vụ mà bạn hỗ trợ",
            ret["results"],
            min_score_threshold=0.18,
        )
        self.assertIsNotNone(refusal)
        self.assertTrue(refusal.is_refusal)

    def test_extract_response_text_handles_list_blocks(self) -> None:
        class _FakeResp:
            content = [{"type": "text", "text": "Xin chào!"}, {"type": "text", "text": "Mình có thể giúp gì?"}]

        self.assertEqual(extract_response_text(_FakeResp()), "Xin chào!\nMình có thể giúp gì?")


if __name__ == "__main__":
    unittest.main()
