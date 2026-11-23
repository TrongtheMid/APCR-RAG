"""
Facade tương thích ngược cho src.judge (Nhánh của Phạm Đức Trọng).
"""

from src.judge import (
    JudgeAgent,
    evaluate_claims_coverage,
    extract_days_from_text,
    extract_fees_from_text,
    extract_mandatory_claims,
    get_legal_misconceptions_catalog,
    try_llm_critic,
    verify_layer1_deterministic,
    verify_layer1_deterministic_detailed,
    verify_layer2_semantic_critic,
    verify_layer2_semantic_critic_detailed,
)

__all__ = [
    "JudgeAgent",
    "extract_fees_from_text",
    "extract_days_from_text",
    "extract_mandatory_claims",
    "evaluate_claims_coverage",
    "get_legal_misconceptions_catalog",
    "try_llm_critic",
    "verify_layer1_deterministic",
    "verify_layer1_deterministic_detailed",
    "verify_layer2_semantic_critic",
    "verify_layer2_semantic_critic_detailed",
]
