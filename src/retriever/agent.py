"""
NHÁNH NGUYỄN HỮU HẢI LONG — Lớp Điều phối RetrieverAgent:
Kết hợp Query Rewriting, Metadata Filtering, Rank-BM25 và ChromaDB Vector Search.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import BM25_WEIGHT, DEFAULT_DATA_PATH, VECTOR_WEIGHT
from src.retriever.bm25_engine import BM25Okapi
from src.retriever.metadata_filter import passes_metadata_filter
from src.retriever.query_rewriter import rewrite_query_and_route
from src.retriever.tokenizer import procedure_to_searchable_text, tokenize_vietnamese
from src.retriever.vector_store import init_chroma_collection, query_chroma_scores
from src.schemas import ProcedureRecord


class RetrieverAgent:
    """
    AGENT 01 — RETRIEVER AGENT (Tác tử Tra cứu Hồ sơ Đa tầng):
    - Query Rewriting & Domain Routing
    - Metadata Filtering (Phân nhóm, Lĩnh vực, Cấp thẩm quyền, Giai đoạn tố tụng, Hiệu lực thời gian)
    - Hybrid Search (Rank-BM25 + ChromaDB Vector Store)
    """

    def __init__(
        self,
        data_path: Optional[Path | str] = None,
        bm25_weight: float = BM25_WEIGHT,
        vector_weight: float = VECTOR_WEIGHT,
    ) -> None:
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        self.raw_procedures: List[Dict[str, Any]] = []
        self.validated_records: List[ProcedureRecord] = []
        self.procedures_by_id: Dict[str, Dict[str, Any]] = {}
        self.corpus_texts: List[str] = []
        self.tokenized_corpus: List[List[str]] = []

        self._load_and_validate_data()
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chroma_collection = init_chroma_collection(self.raw_procedures, self.corpus_texts)

    def _load_and_validate_data(self) -> None:
        """Đọc dữ liệu từ data_procedures.json và kiểm tra tính hợp lệ qua Pydantic Schema."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy kho dữ liệu chuẩn hóa tại: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        for item in raw_list:
            record = ProcedureRecord.model_validate(item)
            self.validated_records.append(record)
            proc_dict = record.model_dump()
            ma_tt = proc_dict["ma_va_ten_thu_tuc"]["ma_thu_tuc"]
            self.raw_procedures.append(proc_dict)
            self.procedures_by_id[ma_tt] = proc_dict
            full_text = procedure_to_searchable_text(proc_dict)
            self.corpus_texts.append(full_text)
            self.tokenized_corpus.append(tokenize_vietnamese(full_text))

    def rewrite_query_and_route(
        self, query: str, preferred_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        return rewrite_query_and_route(query, preferred_domain=preferred_domain)

    def _passes_metadata_filter(
        self,
        proc: Dict[str, Any],
        phan_nhom: Optional[str] = None,
        linh_vuc: Optional[str] = None,
        cap_tham_quyen: Optional[str] = None,
        giai_doan_to_tung: Optional[str] = None,
        only_active: bool = True,
        effective_date: Optional[str] = None,
    ) -> bool:
        return passes_metadata_filter(
            proc,
            phan_nhom=phan_nhom,
            linh_vuc=linh_vuc,
            cap_tham_quyen=cap_tham_quyen,
            giai_doan_to_tung=giai_doan_to_tung,
            only_active=only_active,
            effective_date=effective_date,
        )

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 3,
        phan_nhom: Optional[str] = None,
        linh_vuc: Optional[str] = None,
        cap_tham_quyen: Optional[str] = None,
        giai_doan_to_tung: Optional[str] = None,
        only_active: bool = True,
        effective_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Thực hiện Hybrid Search kết hợp điểm số giữa Rank-BM25 và ChromaDB Vector Store."""
        start_time = time.perf_counter()
        routing_info = rewrite_query_and_route(query, preferred_domain=phan_nhom)
        rewritten_query = routing_info["rewritten_query"]

        query_tokens = tokenize_vietnamese(rewritten_query)
        bm25_raw_scores = self.bm25.get_scores(query_tokens)
        max_bm25 = (
            max(bm25_raw_scores)
            if len(bm25_raw_scores) > 0 and max(bm25_raw_scores) > 0
            else 1.0
        )
        bm25_norm_scores = [float(s / max_bm25) for s in bm25_raw_scores]

        vector_scores_map = query_chroma_scores(
            self.chroma_collection,
            rewritten_query,
            n_results=len(self.raw_procedures),
        )

        scored_results: List[Dict[str, Any]] = []
        for idx, proc in enumerate(self.raw_procedures):
            if not passes_metadata_filter(
                proc,
                phan_nhom=phan_nhom,
                linh_vuc=linh_vuc,
                cap_tham_quyen=cap_tham_quyen,
                giai_doan_to_tung=giai_doan_to_tung,
                only_active=only_active,
                effective_date=effective_date,
            ):
                continue

            ma_tt = proc["ma_va_ten_thu_tuc"]["ma_thu_tuc"]
            bm25_s = bm25_norm_scores[idx]
            vec_s = vector_scores_map.get(ma_tt, bm25_s)
            hybrid_score = (self.bm25_weight * bm25_s) + (self.vector_weight * vec_s)

            if ma_tt.lower() in rewritten_query.lower():
                hybrid_score = min(1.0, hybrid_score + 0.25)

            scored_results.append(
                {
                    "procedure": proc,
                    "ma_thu_tuc": ma_tt,
                    "ten_thu_tuc": proc["ma_va_ten_thu_tuc"]["ten_thu_tuc"],
                    "hybrid_score": round(hybrid_score, 4),
                    "bm25_score": round(bm25_s, 4),
                    "vector_score": round(vec_s, 4),
                }
            )

        scored_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_results = scored_results[:top_k]

        q_lower = query.lower()
        expired_alerts: List[Dict[str, Any]] = []
        if "hộ khẩu" in q_lower or "hk07" in q_lower:
            if "TTHC-CT-OLD-01" in self.procedures_by_id:
                expired_alerts.append(self.procedures_by_id["TTHC-CT-OLD-01"])
        if "cmnd" in q_lower or "chứng minh nhân dân" in q_lower or "9 số" in q_lower:
            if "TTHC-CC-OLD-02" in self.procedures_by_id:
                expired_alerts.append(self.procedures_by_id["TTHC-CC-OLD-02"])

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "routing_info": routing_info,
            "results": top_results,
            "expired_alerts": expired_alerts,
            "latency_ms": latency_ms,
        }


_DEFAULT_RETRIEVER: Optional[RetrieverAgent] = None


def get_retriever_agent() -> RetrieverAgent:
    """Lấy thực thể RetrieverAgent dùng chung (Singleton)."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = RetrieverAgent()
    return _DEFAULT_RETRIEVER
