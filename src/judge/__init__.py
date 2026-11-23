"""
Package src.judge (Nhánh của Phạm Đức Trọng):
Bao gồm Lớp 1 Code Tất định + Anti-Contradiction Regex, Lớp 2 Claim-Level Semantic Critic,
Danh mục Cảnh báo Hiểu nhầm và JudgeAgent.
"""

from src.judge.agent import JudgeAgent
from src.judge.layer1_deterministic import (
    extract_days_from_text,
    extract_fees_from_text,
    verify_layer1_deterministic,
    verify_layer1_deterministic_detailed,
)
from src.judge.layer2_semantic import (
    evaluate_claims_coverage,
    extract_mandatory_claims,
    try_llm_critic,
    verify_layer2_semantic_critic,
    verify_layer2_semantic_critic_detailed,
)
from src.judge.misconceptions_catalog import get_legal_misconceptions_catalog

__all__ = [
    "extract_fees_from_text",
    "extract_days_from_text",
    "verify_layer1_deterministic",
    "verify_layer1_deterministic_detailed",
    "extract_mandatory_claims",
    "evaluate_claims_coverage",
    "verify_layer2_semantic_critic",
    "verify_layer2_semantic_critic_detailed",
    "try_llm_critic",
    "get_legal_misconceptions_catalog",
    "JudgeAgent",
]
