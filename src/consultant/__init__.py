"""
Package src.consultant (Nhánh của Vũ Gia Anh Đức):
Bao gồm System Prompts IRAC, Refusal Guard, IRAC Synthesizer, Graph Builder và ConsultantAgent.
"""

from src.consultant.agent import ConsultantAgent
from src.consultant.graph_builder import extract_procedure_graph_data
from src.consultant.irac_synthesizer import (
    synthesize_structured_irac,
    try_generate_with_gemini,
)
from src.consultant.prompts import IRAC_SYSTEM_PROMPT, REFUSAL_TEMPLATE_MESSAGE
from src.consultant.refusal_guard import check_deterministic_refusal

__all__ = [
    "IRAC_SYSTEM_PROMPT",
    "REFUSAL_TEMPLATE_MESSAGE",
    "check_deterministic_refusal",
    "try_generate_with_gemini",
    "synthesize_structured_irac",
    "extract_procedure_graph_data",
    "ConsultantAgent",
]
