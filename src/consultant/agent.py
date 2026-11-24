"""
NHÁNH VŨ GIA ANH ĐỨC — Lớp Điều phối ConsultantAgent (AGENT 02):
Kết nối Deterministic Refusal Guard, Clarification (hỏi ngược làm rõ),
LLM Factory (Gemini & Local Ollama), IRAC Synthesizer và hỗ trợ ngữ cảnh hội thoại đa lượt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.consultant.clarifier import (
    ClarificationResult,
    analyze_clarification_need,
    build_clarification_irac,
)
from src.consultant.irac_synthesizer import (
    synthesize_structured_irac,
    try_generate_with_gemini,
)
from src.consultant.refusal_guard import check_deterministic_refusal
from src.core.config import MIN_RETRIEVAL_SCORE
from src.core.llm_factory import resolve_llm_config
from src.schemas import IRACOutput


class ConsultantAgent:
    """
    AGENT 02 — CONSULTANT AGENT (Tác tử Tư vấn theo Khuôn mẫu IRAC):
    - Kiểm tra ngưỡng an toàn để Từ chối trả lời chủ động (Deterministic Refusal).
    - Nhận diện thiếu dữ kiện phân định thủ tục/thẩm quyền để Hỏi ngược làm rõ (Clarification).
    - Hỗ trợ lịch sử hội thoại đa lượt (`chat_history`) với giới hạn ngữ cảnh an toàn.
    - Sinh câu trả lời cấu trúc IRAC kèm CitationMetadata (qua Gemini, Ollama hoặc fallback tất định).
    - Tự hiệu chỉnh (Self-Revision) khi nhận phản hồi từ Judge Agent.
    """

    def __init__(
        self,
        min_score_threshold: Optional[float] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_llm: bool = True,
    ) -> None:
        self.min_score_threshold = (
            min_score_threshold
            if min_score_threshold is not None
            else MIN_RETRIEVAL_SCORE
        )
        self.use_llm = use_llm
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    def _resolve_llm_config(self) -> Dict[str, Any]:
        if not self.use_llm:
            return {
                "provider": "offline",
                "model_name": None,
                "api_key": None,
                "base_url": None,
            }
        return resolve_llm_config(
            provider=self.provider,
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _has_valid_api_key(self) -> bool:
        """Giữ phương thức tương thích ngược: True nếu có LLM khả dụng (Gemini API key hoặc Ollama)."""
        cfg = self._resolve_llm_config()
        return cfg["provider"] != "offline"

    def check_deterministic_refusal(
        self, query: str, retrieved_items: List[Dict[str, Any]]
    ) -> Optional[IRACOutput]:
        return check_deterministic_refusal(
            query=query,
            retrieved_items=retrieved_items,
            min_score_threshold=self.min_score_threshold,
        )

    def evaluate_clarification(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> ClarificationResult:
        """
        Kiểm tra xem câu hỏi có cần hỏi ngược làm rõ trước khi tư vấn hay không.
        Luôn chạy Refusal Guard trước để các câu hỏi ngoài phạm vi được từ chối an toàn.
        """
        refusal = self.check_deterministic_refusal(query, retrieved_items)
        if refusal is not None:
            return ClarificationResult(needs_clarification=False)

        return analyze_clarification_need(
            query=query,
            retrieved_items=retrieved_items,
            chat_history=chat_history,
        )

    def consult_with_history(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        expired_alerts: Optional[List[Dict[str, Any]]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        critique_feedback: Optional[str] = None,
        revision_count: int = 0,
        simulate_initial_error: bool = False,
        enable_clarification: bool = True,
    ) -> Dict[str, Any]:
        """
        API đa lượt đầy đủ cho ConsultantAgent:
        1. Chạy Refusal Guard trước tiên.
        2. Nếu `enable_clarification=True` và không trong vòng Self-Revision, kiểm tra thiếu dữ kiện.
        3. Sinh bản nháp IRAC có xét ngữ cảnh `chat_history`.
        """
        refusal = self.check_deterministic_refusal(query, retrieved_items)
        if refusal is not None:
            return {
                "needs_clarification": False,
                "clarification_question": None,
                "clarification": ClarificationResult(needs_clarification=False),
                "irac_output": refusal,
            }

        if (
            enable_clarification
            and revision_count == 0
            and not critique_feedback
            and not simulate_initial_error
        ):
            clarification = analyze_clarification_need(
                query=query,
                retrieved_items=retrieved_items,
                chat_history=chat_history,
            )
            if clarification.needs_clarification:
                return {
                    "needs_clarification": True,
                    "clarification_question": clarification.clarification_question,
                    "clarification": clarification,
                    "irac_output": build_clarification_irac(clarification, query),
                }

        irac_output = self.generate_irac_consultation(
            query=query,
            retrieved_items=retrieved_items,
            expired_alerts=expired_alerts,
            critique_feedback=critique_feedback,
            revision_count=revision_count,
            simulate_initial_error=simulate_initial_error,
            chat_history=chat_history,
        )
        return {
            "needs_clarification": False,
            "clarification_question": None,
            "clarification": ClarificationResult(needs_clarification=False),
            "irac_output": irac_output,
        }

    def generate_irac_consultation(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        expired_alerts: Optional[List[Dict[str, Any]]] = None,
        critique_feedback: Optional[str] = None,
        revision_count: int = 0,
        simulate_initial_error: bool = False,
        chat_history: Optional[List[Dict[str, str]]] = None,
        allow_clarification: bool = False,
        previous_draft: Optional[IRACOutput | Dict[str, Any]] = None,
    ) -> IRACOutput:
        refusal = self.check_deterministic_refusal(query, retrieved_items)
        if refusal is not None:
            return refusal

        if (
            allow_clarification
            and revision_count == 0
            and not critique_feedback
            and not simulate_initial_error
        ):
            clarification = analyze_clarification_need(
                query=query,
                retrieved_items=retrieved_items,
                chat_history=chat_history,
            )
            if clarification.needs_clarification:
                return build_clarification_irac(clarification, query)

        primary_entry = retrieved_items[0]
        primary_proc: Dict[str, Any] = primary_entry["procedure"]
        hybrid_score = float(primary_entry.get("hybrid_score", 0.92))

        cfg = self._resolve_llm_config()
        if cfg["provider"] != "offline" and not (simulate_initial_error and revision_count == 0):
            llm_output = try_generate_with_gemini(
                query=query,
                primary_proc=primary_proc,
                expired_alerts=expired_alerts or [],
                critique_feedback=critique_feedback,
                hybrid_score=hybrid_score,
                model_name=cfg["model_name"],
                api_key=cfg["api_key"],
                provider=cfg["provider"],
                base_url=cfg["base_url"],
                chat_history=chat_history,
                previous_draft=previous_draft,
            )
            if llm_output is not None:
                return llm_output

        return synthesize_structured_irac(
            query=query,
            primary_proc=primary_proc,
            expired_alerts=expired_alerts or [],
            critique_feedback=critique_feedback,
            revision_count=revision_count,
            hybrid_score=hybrid_score,
            simulate_initial_error=simulate_initial_error,
            chat_history=chat_history,
        )
