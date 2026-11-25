"""
Facade tương thích ngược cho src.retriever (Nhánh của Nguyễn Hữu Hải Long).
Export toàn bộ public API của package retriever để code legacy/test import trực tiếp.
"""

from src.retriever import (
    COLLOQUIAL_TO_LEGAL_DICT,
    COLLOQUIAL_TO_LEGAL_MAP,
    DEFAULT_DATA_PATH,
    BM25Okapi,
    LightweightLegalEmbeddingFunction,
    RetrieverAgent,
    LightweightLegalReranker,
    get_retriever_agent,
    highlight_keywords_html,
    init_chroma_collection,
    normalize_text,
    passes_metadata_filter,
    procedure_to_searchable_text,
    query_chroma_scores,
    rerank_candidates,
    reset_retriever_agent,
    rewrite_query_and_route,
    search_direct_bm25,
    tokenize_vietnamese,
    vietnamese_legal_tokenize,
)

__all__ = [
    "DEFAULT_DATA_PATH",
    "COLLOQUIAL_TO_LEGAL_MAP",
    "COLLOQUIAL_TO_LEGAL_DICT",
    "BM25Okapi",
    "LightweightLegalEmbeddingFunction",
    "RetrieverAgent",
    "LightweightLegalReranker",
    "rerank_candidates",
    "get_retriever_agent",
    "reset_retriever_agent",
    "highlight_keywords_html",
    "init_chroma_collection",
    "normalize_text",
    "passes_metadata_filter",
    "procedure_to_searchable_text",
    "query_chroma_scores",
    "rewrite_query_and_route",
    "search_direct_bm25",
    "tokenize_vietnamese",
    "vietnamese_legal_tokenize",
]
