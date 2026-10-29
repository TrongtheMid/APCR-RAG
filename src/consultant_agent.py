"""
Facade tương thích ngược cho src.consultant (Nhánh của Vũ Gia Anh Đức).
"""

from src.consultant import (
    IRAC_SYSTEM_PROMPT,
    REFUSAL_TEMPLATE_MESSAGE,
    ConsultantAgent,
    check_deterministic_refusal,
    extract_procedure_graph_data,
    synthesize_structured_irac,
    try_generate_with_gemini,
)

__all__ = [
    "IRAC_SYSTEM_PROMPT",
    "REFUSAL_TEMPLATE_MESSAGE",
    "ConsultantAgent",
    "check_deterministic_refusal",
    "extract_procedure_graph_data",
    "synthesize_structured_irac",
    "try_generate_with_gemini",
]
