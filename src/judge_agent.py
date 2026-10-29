"""
Facade tương thích ngược cho src.judge (Nhánh của Phạm Đức Trọng).
"""

from src.judge import (
    JudgeAgent,
    get_legal_misconceptions_catalog,
    try_llm_critic,
    verify_layer1_deterministic,
    verify_layer2_semantic_critic,
)

__all__ = [
    "JudgeAgent",
    "get_legal_misconceptions_catalog",
    "try_llm_critic",
    "verify_layer1_deterministic",
    "verify_layer2_semantic_critic",
]
