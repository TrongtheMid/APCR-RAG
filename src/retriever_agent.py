"""
Facade tương thích ngược cho src.retriever (Nhánh của Nguyễn Hữu Hải Long).
"""

from src.retriever import (
    COLLOQUIAL_TO_LEGAL_MAP,
    DEFAULT_DATA_PATH,
    BM25Okapi,
    LightweightLegalEmbeddingFunction,
    RetrieverAgent,
    get_retriever_agent,
    highlight_keywords_html,
    normalize_text,
    passes_metadata_filter,
    procedure_to_searchable_text,
    rewrite_query_and_route,
    search_direct_bm25,
    tokenize_vietnamese,
)

__all__ = [
    "DEFAULT_DATA_PATH",
    "COLLOQUIAL_TO_LEGAL_MAP",
    "BM25Okapi",
    "LightweightLegalEmbeddingFunction",
    "RetrieverAgent",
    "get_retriever_agent",
    "highlight_keywords_html",
    "normalize_text",
    "passes_metadata_filter",
    "procedure_to_searchable_text",
    "rewrite_query_and_route",
    "search_direct_bm25",
    "tokenize_vietnamese",
]
