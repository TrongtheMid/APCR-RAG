"""
Package src.retriever (Nhánh của Nguyễn Hữu Hải Long):
Bao gồm Tokenizer, Query Rewriter, Metadata Filter, BM25 Engine, Vector Store và RetrieverAgent.
"""

from src.core.config import DEFAULT_DATA_PATH
from src.retriever.agent import RetrieverAgent, get_retriever_agent
from src.retriever.bm25_engine import BM25Okapi, search_direct_bm25
from src.retriever.metadata_filter import passes_metadata_filter
from src.retriever.query_rewriter import COLLOQUIAL_TO_LEGAL_MAP, rewrite_query_and_route
from src.retriever.tokenizer import (
    highlight_keywords_html,
    normalize_text,
    procedure_to_searchable_text,
    tokenize_vietnamese,
)
from src.retriever.vector_store import (
    LightweightLegalEmbeddingFunction,
    init_chroma_collection,
    query_chroma_scores,
)

__all__ = [
    "DEFAULT_DATA_PATH",
    "COLLOQUIAL_TO_LEGAL_MAP",
    "normalize_text",
    "tokenize_vietnamese",
    "procedure_to_searchable_text",
    "highlight_keywords_html",
    "rewrite_query_and_route",
    "passes_metadata_filter",
    "BM25Okapi",
    "search_direct_bm25",
    "LightweightLegalEmbeddingFunction",
    "init_chroma_collection",
    "query_chroma_scores",
    "RetrieverAgent",
    "get_retriever_agent",
]
