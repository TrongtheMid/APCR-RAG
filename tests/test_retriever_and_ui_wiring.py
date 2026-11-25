"""
Kiểm thử cho các nâng cấp kết nối toàn hệ thống:
- `vietnamese_legal_tokenize` với backend `pyvi` (tách từ ghép & không sinh token chứa khoảng trắng).
- `rerank_candidates` bảo toàn ngưỡng từ chối `< 0.18` cho câu ngoài miền.
- `extract_text_from_scanned_pdf_with_gemini` (mock Multimodal Gemini).
- `orchestrator.invoke` kích hoạt `NEEDS_CLARIFICATION` kèm `clarification_candidates`.
- `get_related_procedures` & `generate_procedure_docx_bytes` sẵn sàng cho UI.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.consultant import generate_procedure_docx_bytes, get_related_procedures
from src.ingestion.pdf_extractor import extract_text_from_scanned_pdf_with_gemini
from src.retriever import RetrieverAgent, rerank_candidates, vietnamese_legal_tokenize
from src.workflow import LegalMultiAgentOrchestrator


class TestRetrieverAndUIWiring(unittest.TestCase):
    def test_pyvi_tokenizer_produces_clean_tokens_without_spaces(self) -> None:
        tokens = vietnamese_legal_tokenize("Thủ tục đăng ký khai sinh và cấp Phiếu lý lịch tư pháp")
        self.assertTrue(len(tokens) > 5)
        self.assertTrue(all(" " not in t for t in tokens))
        self.assertTrue(any("_" in t for t in tokens))

    def test_reranker_preserves_low_scores_and_boosts_valid_domain_match(self) -> None:
        retriever = RetrieverAgent()
        out_of_domain = retriever.retrieve_hybrid("Tôi muốn xin visa du lịch Mỹ", top_k=3, phan_nhom="Thủ tục hành chính")
        self.assertLess(out_of_domain["results"][0]["hybrid_score"], 0.18)
        self.assertIn("rerank_score", out_of_domain["results"][0])

        in_domain = retriever.retrieve_hybrid(
            "Đăng ký kết hôn với người nước ngoài nộp ở đâu?",
            top_k=3,
            phan_nhom="Thủ tục hành chính",
        )
        self.assertEqual(in_domain["results"][0]["ma_thu_tuc"], "TTHC-HT-05")
        self.assertGreaterEqual(in_domain["results"][0]["rerank_score"], 0.4)

        # Kiểm tra rerank_candidates cô lập điểm < 0.20 không bị nâng lên >= 0.18
        fake_cands = [
            {
                "procedure": in_domain["results"][0]["procedure"],
                "ma_thu_tuc": "FAKE",
                "ten_thu_tuc": "Fake",
                "hybrid_score": 0.12,
                "bm25_score": 0.12,
                "vector_score": 0.0,
            }
        ]
        reranked = rerank_candidates("Đăng ký kết hôn với người nước ngoài", fake_cands, top_k=1)
        self.assertEqual(reranked[0]["hybrid_score"], 0.12)

    def test_scanned_pdf_multimodal_ocr_fallback_mocked(self) -> None:
        self.assertEqual(extract_text_from_scanned_pdf_with_gemini(b"not-a-pdf"), "")

        fake_llm = SimpleNamespace(invoke=lambda msgs: SimpleNamespace(content="Điều 1. Thủ tục hành chính mẫu qua OCR."))
        with patch("src.ingestion.pdf_extractor.has_valid_api_key", return_value=True), patch(
            "src.ingestion.pdf_extractor.get_llm_settings",
            return_value={"provider": "gemini", "google_api_key": "valid_dummy_key"},
        ), patch("src.ingestion.pdf_extractor.get_chat_llm", return_value=fake_llm):
            txt = extract_text_from_scanned_pdf_with_gemini(b"%PDF-1.4\n%fake scanned bytes")
        self.assertIn("Điều 1", txt)

    def test_orchestrator_returns_clarification_candidates_and_docx_ready(self) -> None:
        orch = LegalMultiAgentOrchestrator()
        state = orch.invoke("Đăng ký kết hôn thì nộp ở đâu?")
        self.assertEqual(state["final_status"], "NEEDS_CLARIFICATION")
        self.assertTrue(state["needs_clarification"])
        self.assertTrue(len(state.get("clarification_candidates") or []) >= 2)

        procs = json.loads(Path("data/data_procedures.json").read_text(encoding="utf-8"))
        related = get_related_procedures("TTHC-HT-05", procs, max_hops=1)
        self.assertTrue(len(related) >= 1)
        docx_bytes = generate_procedure_docx_bytes(procs[0], user_data={"ho_ten": "Nguyễn Văn A"})
        self.assertTrue(docx_bytes.startswith(b"PK"))


if __name__ == "__main__":
    unittest.main()
