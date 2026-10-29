"""
Quản lý tập trung toàn bộ biến môi trường (.env), hằng số hệ thống và đường dẫn dữ liệu
cho Hệ thống APCR-RAG (Administrative & Penal Criminal RAG).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "data_procedures.json"
BENCHMARK_DATA_PATH = PROJECT_ROOT / "data" / "benchmark_50_questions.json"

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "legal_procedures_bge_m3")
EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
USE_LIGHTWEIGHT_EMBEDDING: bool = os.getenv("USE_LIGHTWEIGHT_EMBEDDING", "true").lower() == "true"

BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.65"))
VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.35"))
MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.18"))
MAX_REVISION_LOOPS: int = int(os.getenv("MAX_REVISION_LOOPS", "2"))


def has_valid_api_key(api_key: str | None = None) -> bool:
    """Kiểm tra xem GOOGLE_API_KEY có được cấu hình hợp lệ hay không."""
    key = (api_key if api_key is not None else GOOGLE_API_KEY).strip()
    return bool(key and key != "your_google_gemini_api_key_here")
