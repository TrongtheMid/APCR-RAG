"""
Kiểm thử tự động cho các nâng cấp nhánh Vũ Gia Anh Đức:
1. LLM Factory (Gemini, Local Ollama, chế độ auto/offline, fallback an toàn).
2. Consultant Prompts, JSON fallback có kiểm soát, Self-Revision và lịch sử hội thoại.
3. Clarifier (hỏi ngược làm rõ) & tích hợp đầu-cuối trên LangGraph Workflow.
4. Đồ thị tri thức liên thủ tục với NetworkX (không trùng lặp node/edge, truy vấn liên thông).
5. Sinh biểu mẫu DOCX điền sẵn (kiểm tra đầu vào, không bịa dữ kiện, bytes & file).
6. Baseline Single-Agent dùng chung cấu hình provider/model với Consultant.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.baseline import BaselineSingleAgentRAG
from src.consultant import (
    ClarificationRule,
    ConsultantAgent,
    ProcedureFormGenerator,
    analyze_clarification_need,
    build_inter_procedure_nx_graph,
    extract_inter_procedure_graph_data,
    extract_procedure_graph_data,
    format_chat_history_for_prompt,
    generate_procedure_docx_bytes,
    generate_procedure_docx_file,
    get_related_procedures,
    get_supported_form_fields,
    register_clarification_rule,
    try_generate_with_gemini,
    validate_form_inputs,
)
from src.consultant.clarifier import _CLARIFICATION_RULES
from src.core.config import DEFAULT_DATA_PATH
from src.core.llm_factory import get_chat_llm, is_ollama_available, resolve_llm_config
from src.evaluation import load_benchmark_questions
from src.retriever import get_retriever_agent
from src.schemas import CitationMetadata, IRACOutput
from src.workflow import run_legal_multi_agent_workflow


class TestLLMFactory(unittest.TestCase):
    def test_offline_fallback_without_keys_or_ollama(self) -> None:
        env = {
            "LLM_PROVIDER": "auto",
            "GOOGLE_API_KEY": "",
            "OLLAMA_ENABLED": "false",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = resolve_llm_config()
            self.assertEqual(cfg["provider"], "offline")
            self.assertIsNone(get_chat_llm())

    def test_explicit_offline_provider_returns_none(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-key-123"}, clear=False):
            cfg = resolve_llm_config(provider="none")
            self.assertEqual(cfg["provider"], "offline")
            self.assertIsNone(get_chat_llm(provider="none"))

    def test_ollama_unavailable_returns_none_safely(self) -> None:
        with patch("src.core.llm_factory.urllib.request.urlopen", side_effect=OSError("connection refused")):
            self.assertFalse(is_ollama_available(model_name="qwen2.5:7b"))
            self.assertIsNone(get_chat_llm(provider="ollama", model_name="qwen2.5:7b"))

    def test_ollama_available_initializes_chat_ollama(self) -> None:
        fake_tags = json.dumps({"models": [{"name": "qwen2.5:7b"}]}).encode("utf-8")
        fake_resp = MagicMock()
        fake_resp.status = 200
        fake_resp.read.return_value = fake_tags
        fake_resp.__enter__.return_value = fake_resp

        fake_module = types.ModuleType("langchain_ollama")
        mock_chat_ollama = MagicMock(return_value="OLLAMA_CLIENT")
        fake_module.ChatOllama = mock_chat_ollama  # type: ignore[attr-defined]

        with (
            patch("src.core.llm_factory.urllib.request.urlopen", return_value=fake_resp),
            patch.dict(sys.modules, {"langchain_ollama": fake_module}),
        ):
            self.assertTrue(is_ollama_available(model_name="qwen2.5:7b"))
            llm = get_chat_llm(temperature=0.2, provider="ollama", model_name="qwen2.5:7b")
            self.assertEqual(llm, "OLLAMA_CLIENT")
            mock_chat_ollama.assert_called_once_with(
                model="qwen2.5:7b",
                base_url="http://localhost:11434",
                temperature=0.2,
            )

    def test_gemini_provider_with_valid_key(self) -> None:
        fake_module = types.ModuleType("langchain_google_genai")
        mock_gemini = MagicMock(return_value="GEMINI_CLIENT")
        fake_module.ChatGoogleGenerativeAI = mock_gemini  # type: ignore[attr-defined]

        with patch.dict(sys.modules, {"langchain_google_genai": fake_module}):
            llm = get_chat_llm(
                temperature=0.1,
                model_name="gemini-1.5-flash",
                api_key="AIzaSyTestDummyKey",
                provider="gemini",
            )
            self.assertEqual(llm, "GEMINI_CLIENT")
            mock_gemini.assert_called_once()


class TestIRACSynthesizerAndPrompts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(DEFAULT_DATA_PATH, "r", encoding="utf-8") as f:
            cls.procedures = json.load(f)
        cls.proc_by_id = {
            p["ma_va_ten_thu_tuc"]["ma_thu_tuc"]: p for p in cls.procedures
        }

    def test_format_chat_history_bounds_turns(self) -> None:
        history = [{"query": f"Câu hỏi {i}", "procedure_code": "TTHC-HT-01"} for i in range(10)]
        formatted = format_chat_history_for_prompt(history, max_turns=3)
        self.assertNotIn("Câu hỏi 0", formatted)
        self.assertIn("Câu hỏi 9", formatted)

    def test_structured_output_json_fallback_when_native_unsupported(self) -> None:
        proc = self.proc_by_id["TTHC-HT-02"]
        valid_payload = {
            "is_refusal": False,
            "refusal_reason": None,
            "direct_advice": "Chào bạn, kết hôn trong nước nộp tại UBND cấp xã, miễn phí 0 VNĐ.",
            "issue": "Đăng ký kết hôn trong nước (TTHC-HT-02).",
            "rule": "Luật Hộ tịch 60/2014/QH13 (lht2014-d17) ĐANG HIỆU LỰC.",
            "application": (
                "Nộp tại UBND cấp xã với biểu mẫu TK-KH-01, GXNTTHN-02, CCCD-KH-03. "
                "Thời hạn giải quyết tối đa 1 ngày làm việc, lệ phí 0 VNĐ (miễn phí). "
                "Nam đủ 20 tuổi, nữ đủ 18 tuổi và cả hai bên cùng có mặt ký nhận."
            ),
            "conclusion": "Chuẩn bị TK-KH-01, GXNTTHN-02, CCCD-KH-03 nộp tại UBND cấp xã, tối đa 1 ngày, miễn phí.",
            "citations": {
                "ma_thu_tuc": "TTHC-HT-02",
                "law_ids": ["lht2014-d17", "lhn2014-d8"],
                "form_ids": ["TK-KH-01", "GXNTTHN-02", "CCCD-KH-03"],
                "muc_phi_vnd": 0,
                "so_ngay_toi_da": 1,
                "cap_tham_quyen": ["UBND cấp xã"],
                "trang_thai_hieu_luc": "ĐANG HIỆU LỰC",
            },
            "confidence_score": 0.95,
        }

        class OllamaLikeLLM:
            def __init__(self) -> None:
                self.prompts = []

            def with_structured_output(self, schema):
                raise NotImplementedError("Model does not support tool calling")

            def invoke(self, prompt: str):
                self.prompts.append(prompt)
                return types.SimpleNamespace(
                    content=f"```json\n{json.dumps(valid_payload, ensure_ascii=False)}\n```"
                )

        fake_llm = OllamaLikeLLM()
        with patch("src.consultant.irac_synthesizer.get_chat_llm", return_value=fake_llm):
            out = try_generate_with_gemini(
                query="Đăng ký kết hôn trong nước nộp ở đâu?",
                primary_proc=proc,
                expired_alerts=[],
                critique_feedback="Sửa lại mức lệ phí về 0 VNĐ.",
                hybrid_score=0.91,
                provider="ollama",
                model_name="qwen2.5:7b",
                chat_history=[{"query": "Hỏi hộ tịch", "procedure_code": "TTHC-HT-02"}],
            )
        self.assertIsInstance(out, IRACOutput)
        assert out is not None
        self.assertEqual(out.citations.ma_thu_tuc, "TTHC-HT-02")  # type: ignore[union-attr]
        self.assertIn("Sửa lại mức lệ phí về 0 VNĐ.", fake_llm.prompts[0])
        self.assertIn("Hỏi hộ tịch", fake_llm.prompts[0])

    def test_invalid_llm_schema_falls_back_to_deterministic_irac(self) -> None:
        proc = self.proc_by_id["TTHC-HT-02"]

        class BrokenLLM:
            def with_structured_output(self, schema):
                raise RuntimeError("broken")

            def invoke(self, prompt: str):
                return types.SimpleNamespace(content='{"issue": "missing everything else"}')

        agent = ConsultantAgent(provider="ollama", model_name="qwen2.5:7b")
        with (
            patch("src.consultant.agent.resolve_llm_config", return_value={"provider": "ollama", "model_name": "qwen2.5:7b", "api_key": None, "base_url": "http://localhost:11434"}),
            patch("src.consultant.irac_synthesizer.get_chat_llm", return_value=BrokenLLM()),
        ):
            out = agent.generate_irac_consultation(
                query="Đăng ký kết hôn trong nước cần giấy tờ gì?",
                retrieved_items=[{"procedure": proc, "hybrid_score": 0.9}],
            )
        self.assertIsInstance(out, IRACOutput)
        self.assertIsNotNone(out.citations)
        self.assertEqual(out.citations.ma_thu_tuc, "TTHC-HT-02")  # type: ignore[union-attr]


class TestClarifierAndMultiturnWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = get_retriever_agent()
        cls.consultant = ConsultantAgent()

    def test_clarifier_triggers_on_ambiguous_marriage_and_resolves_turn2(self) -> None:
        q1 = "Tôi muốn đăng ký kết hôn thì nộp hồ sơ ở đâu và mất phí bao nhiêu?"
        wf1 = run_legal_multi_agent_workflow(q1, phan_nhom_filter="Thủ tục hành chính")
        self.assertEqual(wf1["final_status"], "NEEDS_CLARIFICATION")
        self.assertTrue(wf1["needs_clarification"])
        self.assertIn("trong nước", (wf1["clarification_question"] or "").lower())
        self.assertIn("nước ngoài", (wf1["clarification_question"] or "").lower())

        history = [
            {
                "query": q1,
                "procedure_code": "TTHC-HT-02",
                "procedure_name": "Thủ tục đăng ký kết hôn trong nước",
            }
        ]
        wf2 = run_legal_multi_agent_workflow(
            "Chúng tôi đăng ký kết hôn trong nước tại Việt Nam thì nộp ở đâu?",
            phan_nhom_filter="Thủ tục hành chính",
            chat_history=history,
        )
        self.assertEqual(wf2["final_status"], "VERIFIED_PASS")
        self.assertFalse(wf2["needs_clarification"])
        self.assertEqual(wf2["primary_procedure"]["ma_va_ten_thu_tuc"]["ma_thu_tuc"], "TTHC-HT-02")

    def test_clarifier_triggers_on_lltp_delegation_and_criminal_stage_and_expungement(self) -> None:
        # 1. Phiếu LLTP số 1 vs số 2 khi hỏi ủy quyền mà chưa nói rõ số phiếu
        ret_lltp = self.retriever.retrieve_hybrid(
            "Tôi nhờ mẹ đi làm hộ Phiếu lý lịch tư pháp có được không?",
            phan_nhom="Tố tụng hình sự",
        )
        c_lltp = self.consultant.evaluate_clarification(
            "Tôi nhờ mẹ đi làm hộ Phiếu lý lịch tư pháp có được không?",
            ret_lltp["results"],
        )
        self.assertTrue(c_lltp.needs_clarification)
        self.assertEqual(c_lltp.rule_id, "lltp_type_branch")

        # Nếu đã nêu rõ Phiếu số 2 -> không hỏi lại
        c_lltp_2 = self.consultant.evaluate_clarification(
            "Tôi nhờ mẹ đi làm hộ Phiếu lý lịch tư pháp số 2 có được không?",
            ret_lltp["results"],
        )
        self.assertFalse(c_lltp_2.needs_clarification)

        # 2. Bảo lĩnh hỏi nơi nộp mà chưa nêu giai đoạn tố tụng
        ret_bail = self.retriever.retrieve_hybrid(
            "Gia đình tôi muốn nộp đơn xin bảo lĩnh thì nộp ở cơ quan nào?",
            phan_nhom="Tố tụng hình sự",
        )
        c_bail = self.consultant.evaluate_clarification(
            "Gia đình tôi muốn nộp đơn xin bảo lĩnh thì nộp ở cơ quan nào?",
            ret_bail["results"],
        )
        self.assertTrue(c_bail.needs_clarification)
        self.assertEqual(c_bail.rule_id, "criminal_stage_authority_branch")

        # 3. Xóa án tích hỏi nơi nộp mà chưa nêu tội danh / loại xóa án tích
        ret_xat = self.retriever.retrieve_hybrid(
            "Làm thủ tục xóa án tích thì nộp hồ sơ ở đâu?",
            phan_nhom="Tố tụng hình sự",
        )
        c_xat = self.consultant.evaluate_clarification(
            "Làm thủ tục xóa án tích thì nộp hồ sơ ở đâu?",
            ret_xat["results"],
        )
        self.assertTrue(c_xat.needs_clarification)
        self.assertEqual(c_xat.rule_id, "expungement_offense_branch")

    def test_refusal_runs_before_clarification(self) -> None:
        res = self.consultant.consult_with_history(
            query="Xin visa đi du lịch Nhật Bản cần giấy tờ gì?",
            retrieved_items=[],
        )
        self.assertFalse(res["needs_clarification"])
        self.assertTrue(res["irac_output"].is_refusal)

    def test_benchmark_50_questions_do_not_false_trigger_clarification(self) -> None:
        questions = load_benchmark_questions()
        false_positives = []
        for q in questions:
            ret = self.retriever.retrieve_hybrid(
                q["question"],
                top_k=3,
                phan_nhom=q["phan_nhom"],
                only_active=True,
            )
            clar = analyze_clarification_need(q["question"], ret["results"])
            if clar.needs_clarification:
                false_positives.append((q["question_id"], clar.rule_id, q["question"]))
        self.assertEqual(false_positives, [], f"False clarification triggers: {false_positives}")

    def test_register_custom_clarification_rule_extensible(self) -> None:
        initial_count = len(_CLARIFICATION_RULES)
        custom_rule = ClarificationRule(
            rule_id="test_custom_rule",
            description="Test extensibility",
            evaluator=lambda q, items, hist: None,
        )
        try:
            register_clarification_rule(custom_rule)
            self.assertEqual(len(_CLARIFICATION_RULES), initial_count + 1)
        finally:
            if custom_rule in _CLARIFICATION_RULES:
                _CLARIFICATION_RULES.remove(custom_rule)


class TestInterProcedureKnowledgeGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(DEFAULT_DATA_PATH, "r", encoding="utf-8") as f:
            cls.procedures = json.load(f)

    def test_single_procedure_graph_backward_compatible(self) -> None:
        single = extract_procedure_graph_data(self.procedures[0])
        self.assertIn("nodes", single)
        self.assertIn("edges", single)
        self.assertIn("procedure", single)
        node_ids = [n["id"] for n in single["nodes"]]
        self.assertEqual(len(node_ids), len(set(node_ids)))

    def test_inter_procedure_nx_graph_no_duplicates_and_related_queries(self) -> None:
        nx_graph = build_inter_procedure_nx_graph(self.procedures)
        self.assertGreaterEqual(nx_graph.number_of_nodes(), len(self.procedures))
        self.assertTrue(nx_graph.has_edge("TTHS-BL-121", "TTHS-DT-122"))
        self.assertTrue(nx_graph.has_edge("TTHC-HT-06", "TTHC-CT-01"))
        self.assertTrue(nx_graph.has_edge("TTHC-CC-OLD-02", "TTHC-CC-01"))

        related = get_related_procedures("TTHS-BL-121", self.procedures, max_hops=2)
        related_ids = {r["ma_thu_tuc"] for r in related}
        self.assertIn("TTHS-DT-122", related_ids)
        self.assertIn("TTHC-CT-04", related_ids)

        sub = extract_inter_procedure_graph_data(
            self.procedures,
            focus_ma_thu_tuc="TTHC-HT-06",
            max_hops=1,
        )
        sub_node_ids = [n["id"] for n in sub["nodes"]]
        self.assertEqual(len(sub_node_ids), len(set(sub_node_ids)))
        sub_edge_keys = [(e["source"], e["target"], e["label"]) for e in sub["edges"]]
        self.assertEqual(len(sub_edge_keys), len(set(sub_edge_keys)))

    def test_only_present_procedures_create_declared_edges(self) -> None:
        subset = [p for p in self.procedures if p["ma_va_ten_thu_tuc"]["ma_thu_tuc"] == "TTHS-BL-121"]
        nx_sub = build_inter_procedure_nx_graph(subset)
        self.assertFalse(nx_sub.has_node("TTHS-DT-122"))
        self.assertFalse(nx_sub.has_edge("TTHS-BL-121", "TTHS-DT-122"))


class TestDocxFormGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open(DEFAULT_DATA_PATH, "r", encoding="utf-8") as f:
            cls.procedures = json.load(f)
        cls.proc_by_id = {
            p["ma_va_ten_thu_tuc"]["ma_thu_tuc"]: p for p in cls.procedures
        }

    def test_input_validation_errors(self) -> None:
        proc = self.proc_by_id["TTHC-HT-02"]
        with self.assertRaises(ValueError):
            validate_form_inputs({"ma_va_ten_thu_tuc": {"ma_thu_tuc": "X"}})
        with self.assertRaises(TypeError):
            validate_form_inputs(proc, user_data=["invalid"])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            validate_form_inputs(proc, user_data={}, form_id="NON-EXISTENT-FORM")

        fields = get_supported_form_fields(proc, form_id="TK-KH-01")
        self.assertEqual(fields["ma_thu_tuc"], "TTHC-HT-02")
        self.assertEqual(fields["selected_form"]["form_id"], "TK-KH-01")

    def test_docx_generation_bytes_and_file(self) -> None:
        proc = self.proc_by_id["TTHC-HT-02"]
        user_data = {
            "ho_ten": "Nguyễn Văn An",
            "cccd": "001099001122",
            "tai_lieu_kem_theo": ["TK-KH-01"],
        }

        try:
            import docx  # type: ignore # noqa: F401
            has_docx = True
        except ImportError:
            has_docx = False

        if has_docx:
            raw_bytes = generate_procedure_docx_bytes(proc, user_data=user_data, form_id="TK-KH-01")
            self.assertTrue(raw_bytes.startswith(b"PK"))

            from docx import Document  # type: ignore

            loaded = Document(io.BytesIO(raw_bytes))
            all_text = "\n".join(p.text for p in loaded.paragraphs)
            table_text = "\n".join(
                cell.text for tbl in loaded.tables for row in tbl.rows for cell in row.cells
            )
            full_text = f"{all_text}\n{table_text}"
            self.assertIn("Nguyễn Văn An", full_text)
            self.assertIn("001099001122", full_text)
            self.assertIn("TTHC-HT-02", full_text)
            self.assertIn("TK-KH-01", full_text)
            self.assertIn("[x] [TK-KH-01]", full_text)
            self.assertIn("[ ] [GXNTTHN-02]", full_text)
            self.assertIn("....", full_text)

            with tempfile.TemporaryDirectory() as tmpdir:
                out_file = generate_procedure_docx_file(
                    proc,
                    output_path=Path(tmpdir) / "to_khai_ket_hon",
                    user_data=user_data,
                    form_id="TK-KH-01",
                )
                self.assertTrue(out_file.exists())
                self.assertEqual(out_file.suffix, ".docx")
                self.assertGreater(out_file.stat().st_size, 0)
        else:
            # Kiểm tra báo lỗi rõ ràng khi chưa cài python-docx, và kiểm tra logic sinh DOCX qua mock module
            with self.assertRaises(ImportError) as ctx:
                generate_procedure_docx_bytes(proc, user_data=user_data, form_id="TK-KH-01")
            self.assertIn("python-docx", str(ctx.exception))

            captured_texts: list[str] = []

            class _FakeRun:
                def __init__(self, text: str) -> None:
                    self.text = text
                    self.bold = False
                    self.italic = False
                    self.font = types.SimpleNamespace(size=None)

            class _FakeParagraph:
                def __init__(self, text: str = "") -> None:
                    self.alignment = None
                    self.paragraph_format = types.SimpleNamespace(space_before=None)
                    if text:
                        captured_texts.append(text)

                def add_run(self, text: str) -> _FakeRun:
                    captured_texts.append(text)
                    return _FakeRun(text)

            class _FakeCell:
                def __init__(self) -> None:
                    self._text = ""

                @property
                def text(self) -> str:
                    return self._text

                @text.setter
                def text(self, val: str) -> None:
                    self._text = val
                    captured_texts.append(val)

            class _FakeTable:
                def __init__(self, rows: int, cols: int) -> None:
                    self.style = None
                    self.rows = [
                        types.SimpleNamespace(cells=[_FakeCell() for _ in range(cols)])
                        for _ in range(rows)
                    ]

            class _FakeDocument:
                def __init__(self) -> None:
                    self.styles = {
                        "Normal": types.SimpleNamespace(font=types.SimpleNamespace(name="", size=None))
                    }

                def add_paragraph(self, text: str = "") -> _FakeParagraph:
                    return _FakeParagraph(text)

                def add_table(self, rows: int, cols: int) -> _FakeTable:
                    return _FakeTable(rows, cols)

                def save(self, buf: io.BytesIO) -> None:
                    buf.write(b"PK\x03\x04FAKE_DOCX_CONTENT")

            fake_docx = types.ModuleType("docx")
            fake_docx.Document = _FakeDocument  # type: ignore[attr-defined]
            fake_enum_text = types.ModuleType("docx.enum.text")
            fake_enum_text.WD_ALIGN_PARAGRAPH = types.SimpleNamespace(CENTER=1, RIGHT=2)  # type: ignore[attr-defined]
            fake_enum = types.ModuleType("docx.enum")
            fake_shared = types.ModuleType("docx.shared")
            fake_shared.Pt = lambda v: v  # type: ignore[attr-defined]

            with patch.dict(
                sys.modules,
                {
                    "docx": fake_docx,
                    "docx.enum": fake_enum,
                    "docx.enum.text": fake_enum_text,
                    "docx.shared": fake_shared,
                },
            ):
                gen = ProcedureFormGenerator()
                raw_bytes = gen.generate_bytes(proc, user_data=user_data, form_id="TK-KH-01")
                self.assertTrue(raw_bytes.startswith(b"PK"))
                joined = "\n".join(captured_texts)
                self.assertIn("Nguyễn Văn An", joined)
                self.assertIn("001099001122", joined)
                self.assertIn("TTHC-HT-02", joined)
                self.assertIn("TK-KH-01", joined)
                self.assertIn("[x] [TK-KH-01]", joined)
                self.assertIn("[ ] [GXNTTHN-02]", joined)
                self.assertIn("....", joined)

                with tempfile.TemporaryDirectory() as tmpdir:
                    out_file = gen.generate_file(
                        proc,
                        output_path=Path(tmpdir) / "to_khai_ket_hon",
                        user_data=user_data,
                        form_id="TK-KH-01",
                    )
                    self.assertTrue(out_file.exists())
                    self.assertEqual(out_file.suffix, ".docx")
                    self.assertGreater(out_file.stat().st_size, 0)


class TestFairBaseline(unittest.TestCase):
    def test_baseline_offline_fallback(self) -> None:
        baseline = BaselineSingleAgentRAG(provider="none")
        res = baseline.answer_single_step("Đăng ký khai sinh mất bao nhiêu tiền?")
        self.assertEqual(res["llm_provider"], "offline")
        self.assertIsNone(res["llm_model"])
        self.assertIn("Single-Agent Baseline", res["answer"])
        self.assertEqual(len(res["retrieved_chunks"]), 2)

    def test_baseline_uses_shared_llm_factory_provider_and_model(self) -> None:
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = types.SimpleNamespace(
            content="Trả lời một bước từ Ollama Qwen2.5."
        )
        baseline = BaselineSingleAgentRAG(provider="ollama", model_name="qwen2.5:7b")

        with (
            patch(
                "src.baseline.single_agent_rag.resolve_llm_config",
                return_value={
                    "provider": "ollama",
                    "model_name": "qwen2.5:7b",
                    "api_key": None,
                    "base_url": "http://localhost:11434",
                },
            ),
            patch("src.baseline.single_agent_rag.get_chat_llm", return_value=fake_llm),
        ):
            res = baseline.answer_single_step("Bảo lĩnh cần mấy người?")

        self.assertEqual(res["llm_provider"], "ollama")
        self.assertEqual(res["llm_model"], "qwen2.5:7b")
        self.assertEqual(res["answer"], "Trả lời một bước từ Ollama Qwen2.5.")


if __name__ == "__main__":
    unittest.main()
