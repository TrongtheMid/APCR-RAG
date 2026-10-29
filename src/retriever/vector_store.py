"""
NHÁNH NGUYỄN HỮU HẢI LONG — Module Vector Store (ChromaDB & Embeddings):
- Cung cấp hàm nhúng LightweightLegalEmbeddingFunction (384 chiều) tốc độ cao.
- Hỗ trợ chuyển sang SentenceTransformer (BAAI/bge-m3) khi USE_LIGHTWEIGHT_EMBEDDING=false.
- Khởi tạo và truy vấn ChromaDB Vector Store.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List

from src.core.config import (
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    USE_LIGHTWEIGHT_EMBEDDING,
)
from src.retriever.tokenizer import tokenize_vietnamese


class LightweightLegalEmbeddingFunction:
    """
    Hàm nhúng Vector nội bộ tốc độ cao (Hashing + Subword N-gram 384 chiều) tương thích ChromaDB.
    Được sử dụng mặc định khi USE_LIGHTWEIGHT_EMBEDDING=true hoặc chưa tải mô hình BAAI/bge-m3.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in input:
            vec = [0.0] * self.dim
            tokens = tokenize_vietnamese(text)
            for token in tokens:
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
                weight = 2.0 if "_" in token else 1.0
                vec[idx] += sign * weight
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            embeddings.append([v / norm for v in vec])
        return embeddings


def init_chroma_collection(
    raw_procedures: List[Dict[str, Any]],
    corpus_texts: List[str],
) -> Any:
    """Khởi tạo ChromaDB Vector Store và nạp danh sách thủ tục."""
    try:
        import chromadb

        embedding_fn: Any = LightweightLegalEmbeddingFunction(dim=384)
        if not USE_LIGHTWEIGHT_EMBEDDING:
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )

                embedding_fn = SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL_NAME
                )
            except Exception:
                embedding_fn = LightweightLegalEmbeddingFunction(dim=384)

        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        ids = [p["ma_va_ten_thu_tuc"]["ma_thu_tuc"] for p in raw_procedures]
        metadatas = []
        for p in raw_procedures:
            info = p["ma_va_ten_thu_tuc"]
            primary_law = p["can_cu_phap_ly"][0]
            metadatas.append(
                {
                    "ma_thu_tuc": info["ma_thu_tuc"],
                    "ten_thu_tuc": info["ten_thu_tuc"],
                    "phan_nhom": info["phan_nhom"],
                    "linh_vuc": info["linh_vuc"],
                    "trang_thai_hieu_luc": primary_law["trang_thai_hieu_luc"],
                    "ngay_hieu_luc": primary_law["ngay_hieu_luc"],
                    "muc_phi_vnd": int(p["le_phi"]["muc_phi_vnd"]),
                    "so_ngay_toi_da": int(p["thoi_han_giai_quyet"]["so_ngay_toi_da"]),
                }
            )

        collection.upsert(ids=ids, documents=corpus_texts, metadatas=metadatas)
        return collection
    except Exception:
        return None


def query_chroma_scores(
    collection: Any,
    query_text: str,
    n_results: int,
) -> Dict[str, float]:
    """Truy vấn độ tương đồng Cosine từ ChromaDB."""
    vector_scores_map: Dict[str, float] = {}
    if collection is None:
        return vector_scores_map
    try:
        chroma_res = collection.query(query_texts=[query_text], n_results=n_results)
        ids_list = chroma_res.get("ids", [[]])[0]
        distances_list = chroma_res.get("distances", [[]])[0]
        for doc_id, dist in zip(ids_list, distances_list):
            vector_scores_map[doc_id] = max(0.0, 1.0 - float(dist))
    except Exception:
        pass
    return vector_scores_map
