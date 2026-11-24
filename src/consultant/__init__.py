"""
Package src.consultant (Nhánh của Vũ Gia Anh Đức):
Bao gồm System Prompts IRAC, Refusal Guard, Clarification, IRAC Synthesizer,
Graph Builder (NetworkX liên thủ tục), Form Generator (DOCX) và ConsultantAgent.
"""

from src.consultant.agent import ConsultantAgent
from src.consultant.clarifier import (
    ClarificationResult,
    ClarificationRule,
    analyze_clarification_need,
    build_clarification_irac,
    register_clarification_rule,
)
from src.consultant.conversational_advisor import build_direct_advice
from src.consultant.form_generator import (
    ProcedureFormGenerator,
    generate_procedure_docx_bytes,
    generate_procedure_docx_file,
    get_supported_form_fields,
    validate_form_inputs,
)
from src.consultant.graph_builder import (
    INTER_PROCEDURE_RELATIONS,
    build_inter_procedure_nx_graph,
    extract_inter_procedure_graph_data,
    extract_procedure_graph_data,
    get_related_procedures,
)
from src.consultant.irac_synthesizer import (
    synthesize_structured_irac,
    try_generate_with_gemini,
    try_generate_with_llm,
)
from src.consultant.prompts import (
    IRAC_SYSTEM_PROMPT,
    REFUSAL_TEMPLATE_MESSAGE,
    format_chat_history_for_prompt,
)
from src.consultant.refusal_guard import check_deterministic_refusal

__all__ = [
    "IRAC_SYSTEM_PROMPT",
    "REFUSAL_TEMPLATE_MESSAGE",
    "format_chat_history_for_prompt",
    "build_direct_advice",
    "check_deterministic_refusal",
    "ClarificationResult",
    "ClarificationRule",
    "analyze_clarification_need",
    "build_clarification_irac",
    "register_clarification_rule",
    "try_generate_with_gemini",
    "try_generate_with_llm",
    "synthesize_structured_irac",
    "INTER_PROCEDURE_RELATIONS",
    "extract_procedure_graph_data",
    "build_inter_procedure_nx_graph",
    "extract_inter_procedure_graph_data",
    "get_related_procedures",
    "ProcedureFormGenerator",
    "validate_form_inputs",
    "get_supported_form_fields",
    "generate_procedure_docx_bytes",
    "generate_procedure_docx_file",
    "ConsultantAgent",
]
