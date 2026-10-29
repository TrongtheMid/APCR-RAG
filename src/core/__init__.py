"""
Tầng Core: Quản lý cấu hình môi trường, đường dẫn dữ liệu và khởi tạo LLM dùng chung.
"""

from src.core.config import (
    BENCHMARK_DATA_PATH,
    BM25_WEIGHT,
    CHROMA_COLLECTION_NAME,
    DEFAULT_DATA_PATH,
    EMBEDDING_MODEL_NAME,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    MAX_REVISION_LOOPS,
    MIN_RETRIEVAL_SCORE,
    USE_LIGHTWEIGHT_EMBEDDING,
    VECTOR_WEIGHT,
    has_valid_api_key,
)
from src.core.llm_factory import get_chat_llm

__all__ = [
    "DEFAULT_DATA_PATH",
    "BENCHMARK_DATA_PATH",
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "CHROMA_COLLECTION_NAME",
    "EMBEDDING_MODEL_NAME",
    "USE_LIGHTWEIGHT_EMBEDDING",
    "BM25_WEIGHT",
    "VECTOR_WEIGHT",
    "MIN_RETRIEVAL_SCORE",
    "MAX_REVISION_LOOPS",
    "has_valid_api_key",
    "get_chat_llm",
]
