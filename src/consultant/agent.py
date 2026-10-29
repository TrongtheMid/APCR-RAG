"""
NHÁNH VŨ GIA ANH ĐỨC — Lớp Điều phối ConsultantAgent (AGENT 02):
Kết nối Deterministic Refusal Guard, Gemini Structured Output và IRAC Synthesizer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.consultant.irac_synthesizer import (
    synthesize_structured_irac,
    try_generate_with_gemini,
)
from src.consultant.refusal_guard import check_deterministic_refusal
from src.core.config import GEMINI_MODEL, GOOGLE_API_KEY, MIN_RETRIEVAL_SCORE, has_valid_api_key
from src.schemas import IRACOutput


class ConsultantAgent:
    """
    AGENT 02 — CONSULTANT AGENT (Tác tử Tư vấn theo Khuôn mẫu IRAC):
    - Kiểm tra ngưỡng an toàn để Từ chối trả lời chủ động (Deterministic Refusal).
    - Sinh câu trả lời cấu trúc IRAC kèm CitationMetadata.
    - Tự hiệu chỉnh (Self-Revision) khi nhận phản hồi từ Judge Agent.
    """

    def __init__(self, min_score_threshold: Optional[float] = None) -> None:
        self.min_score_threshold = (
            min_score_threshold
            if min_score_threshold is not None
            else MIN_RETRIEVAL_SCORE
        )
        self.api_key = GOOGLE_API_KEY
        self.model_name = GEMINI_MODEL

    def _has_valid_api_key(self) -> bool:
        return has_valid_api_key(self.api_key)

    def check_deterministic_refusal(
        self, query: str, retrieved_items: List[Dict[str, Any]]
    ) -> Optional[IRACOutput]:
        return check_deterministic_refusal(
            query=query,
            retrieved_items=retrieved_items,
            min_score_threshold=self.min_score_threshold,
        )

    def generate_irac_consultation(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        expired_alerts: Optional[List[Dict[str, Any]]] = None,
        critique_feedback: Optional[str] = None,
        revision_count: int = 0,
        simulate_initial_error: bool = False,
    ) -> IRACOutput:
        refusal = self.check_deterministic_refusal(query, retrieved_items)
        if refusal is not None:
            return refusal

        primary_entry = retrieved_items[0]
        primary_proc: Dict[str, Any] = primary_entry["procedure"]
        hybrid_score = float(primary_entry.get("hybrid_score", 0.92))

        if self._has_valid_api_key() and not (simulate_initial_error and revision_count == 0):
            llm_output = try_generate_with_gemini(
                query=query,
                primary_proc=primary_proc,
                expired_alerts=expired_alerts or [],
                critique_feedback=critique_feedback,
                hybrid_score=hybrid_score,
                model_name=self.model_name,
                api_key=self.api_key,
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
        )
