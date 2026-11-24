"""
NHÁNH NGUYỄN HỮU HẢI LONG — Lớp Điều phối RetrieverAgent:
Kết hợp Query Rewriting, Metadata Filtering, Rank-BM25 và ChromaDB Vector Search
thông qua thuật toán Reciprocal Rank Fusion (RRF) và phát hiện hết hiệu lực động.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import BM25_WEIGHT, DEFAULT_DATA_PATH, VECTOR_WEIGHT
from src.retriever.bm25_engine import BM25Okapi
from src.retriever.metadata_filter import passes_metadata_filter
from src.retriever.query_rewriter import rewrite_query_and_route
from src.retriever.tokenizer import procedure_to_searchable_text, tokenize_vietnamese
from src.retriever.vector_store import init_chroma_collection, query_chroma_scores
from src.schemas import ProcedureRecord


# ──────────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion (RRF)
# ──────────────────────────────────────────────────────────────────────────────
_RRF_K = 60  # Hằng số giảm phạt vị trí (tiêu chuẩn: 60)


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    """Điểm RRF cho vị trí rank (1-indexed): 1 / (k + rank)."""
    return 1.0 / (k + rank)


def _reciprocal_rank_fusion(
    bm25_ranked: List[Tuple[str, float]],   # [(ma_tt, bm25_score), ...] đã sort giảm dần
    vec_ranked: List[Tuple[str, float]],    # [(ma_tt, vec_score), ...] đã sort giảm dần
    bm25_weight: float = BM25_WEIGHT,
    vector_weight: float = VECTOR_WEIGHT,
) -> Dict[str, float]:
    """
    Kết hợp hai danh sách xếp hạng bằng RRF có trọng số.
    Chỉ xếp hạng các ứng viên có điểm dương (> 0) để tránh nhiễu đồng hạng 0.0.
    Trả về {ma_thu_tuc: rrf_score}.
    """
    rrf_scores: Dict[str, float] = {}

    bm25_pos_rank = 0
    for ma_tt, score in bm25_ranked:
        if score <= 0.0:
            continue
        bm25_pos_rank += 1
        rrf_scores[ma_tt] = rrf_scores.get(ma_tt, 0.0) + bm25_weight * _rrf_score(bm25_pos_rank)

    vec_pos_rank = 0
    for ma_tt, score in vec_ranked:
        if score <= 0.0:
            continue
        vec_pos_rank += 1
        rrf_scores[ma_tt] = rrf_scores.get(ma_tt, 0.0) + vector_weight * _rrf_score(vec_pos_rank)

    return rrf_scores


class RetrieverAgent:
    """
    AGENT 01 — RETRIEVER AGENT (Tác tử Tra cứu Hồ sơ Đa tầng):
    - Query Rewriting & Domain Routing (kết hợp COLLOQUIAL_TO_LEGAL_MAP + LLM fallback)
    - Metadata Filtering (Phân nhóm, Lĩnh vực, Cấp thẩm quyền, Giai đoạn tố tụng, Hiệu lực thời gian)
    - Hybrid Search: Rank-BM25 + ChromaDB Vector Store kết hợp qua Reciprocal Rank Fusion (RRF)
    - Phát hiện văn bản hết hiệu lực động (không hardcode mã thủ tục)
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

        # Chỉ mục phụ: thủ tục đã hết hiệu lực (phục vụ cảnh báo động)
        self.expired_procedures: List[Dict[str, Any]] = []

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

            # Phân loại chỉ mục hết hiệu lực (động, không hardcode mã)
            has_any_expired = any(
                l["trang_thai_hieu_luc"] == "HẾT HIỆU LỰC"
                for l in proc_dict["can_cu_phap_ly"]
            )
            if has_any_expired:
                self.expired_procedures.append(proc_dict)

            full_text = procedure_to_searchable_text(proc_dict)
            self.corpus_texts.append(full_text)
            self.tokenized_corpus.append(tokenize_vietnamese(full_text))

    # ── Delegate methods ────────────────────────────────────────────────────────

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

    # ── Expired Alert Detection (Dynamic) ──────────────────────────────────────

    def _get_relevant_expired_alerts(
        self, query: str, detected_domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Phát hiện văn bản hết hiệu lực có liên quan đến câu hỏi theo cách động.
        Không hardcode mã thủ tục — dùng BM25 trên corpus tất cả thủ tục HẾT HIỆU LỰC.
        """
        if not self.expired_procedures:
            return []

        q_lower = query.lower()
        relevant: List[Dict[str, Any]] = []

        for proc in self.expired_procedures:
            info = proc["ma_va_ten_thu_tuc"]
            ten = info["ten_thu_tuc"].lower()
            linh_vuc = info["linh_vuc"].lower()
            dieu_kien = proc["thanh_phan_ho_so"].get("dieu_kien_bat_buoc", "").lower()

            # Khớp ngữ nghĩa nhẹ: lĩnh vực hoặc từ khóa tên thủ tục trong câu hỏi
            match_score = 0
            for word in q_lower.split():
                if len(word) >= 3:
                    if word in ten or word in linh_vuc or word in dieu_kien:
                        match_score += 1

            # Khớp domain: chỉ trả về cảnh báo cùng phân nhóm nếu được lọc
            if detected_domain and detected_domain != info["phan_nhom"]:
                continue

            if match_score >= 1:
                relevant.append(proc)

        return relevant

    # ── Core Retrieval ──────────────────────────────────────────────────────────

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
        """
        Thực hiện Hybrid Search kết hợp Rank-BM25 và ChromaDB Vector Store
        thông qua thuật toán Reciprocal Rank Fusion (RRF) có trọng số.
        """
        start_time = time.perf_counter()
        routing_info = rewrite_query_and_route(query, preferred_domain=phan_nhom)
        rewritten_query = routing_info["rewritten_query"]
        detected_domain = routing_info["detected_domain"]

        # ── BM25 Scoring ───────────────────────────────────────────────────────
        query_tokens = tokenize_vietnamese(rewritten_query)
        bm25_raw_scores = self.bm25.get_scores(query_tokens)

        # Normalize BM25
        raw_max = (
            float(max(bm25_raw_scores))
            if len(bm25_raw_scores) > 0 and float(max(bm25_raw_scores)) > 0
            else 1.0
        )
        has_legal_expansion = bool(routing_info.get("matched_expansions"))
        norm_denom = raw_max if has_legal_expansion else max(raw_max, 14.0)
        bm25_norm_scores = [float(min(1.0, s / norm_denom)) for s in bm25_raw_scores]

        # ── Vector Scoring ─────────────────────────────────────────────────────
        vector_scores_map = query_chroma_scores(
            self.chroma_collection,
            rewritten_query,
            n_results=len(self.raw_procedures),
        )

        # ── Metadata Filter + Build Ranked Lists ───────────────────────────────
        bm25_candidates: List[Tuple[str, float]] = []
        vec_candidates: List[Tuple[str, float]] = []

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
            vec_s = vector_scores_map.get(ma_tt, 0.0)

            bm25_candidates.append((ma_tt, bm25_s))
            vec_candidates.append((ma_tt, vec_s))

        # Sort theo điểm giảm dần để xếp hạng
        bm25_ranked = sorted(bm25_candidates, key=lambda x: x[1], reverse=True)
        vec_ranked = sorted(vec_candidates, key=lambda x: x[1], reverse=True)

        # ── Reciprocal Rank Fusion (RRF) chuẩn hóa về thang [0.0, 1.0] ─────────
        rrf_scores = _reciprocal_rank_fusion(
            bm25_ranked, vec_ranked,
            bm25_weight=self.bm25_weight,
            vector_weight=self.vector_weight,
        )
        max_rrf_theoretical = (self.bm25_weight + self.vector_weight) / (_RRF_K + 1.0) or 1.0

        # Dựng danh sách kết quả kết hợp RRF rank + cường độ tín hiệu thực tế
        q_lower = rewritten_query.lower()
        bm25_score_map = dict(bm25_candidates)
        all_scored: List[Dict[str, Any]] = []
        for ma_tt, _ in bm25_candidates:
            proc = self.procedures_by_id.get(ma_tt)
            if proc is None:
                continue
            bm25_s = bm25_score_map.get(ma_tt, 0.0)
            vec_s = vector_scores_map.get(ma_tt, 0.0)
            rrf_scaled = min(1.0, rrf_scores.get(ma_tt, 0.0) / max_rrf_theoretical)
            evidence = max(bm25_s, vec_s)
            raw_weighted = self.bm25_weight * bm25_s + self.vector_weight * vec_s
            hybrid_s = 0.5 * (rrf_scaled * evidence) + 0.5 * raw_weighted

            # Điểm thưởng khớp mã thủ tục / số hiệu điều luật trong câu hỏi
            if ma_tt.lower() in q_lower:
                hybrid_s = min(1.0, hybrid_s + 0.25)

            all_scored.append(
                {
                    "procedure": proc,
                    "ma_thu_tuc": ma_tt,
                    "ten_thu_tuc": proc["ma_va_ten_thu_tuc"]["ten_thu_tuc"],
                    "hybrid_score": round(hybrid_s, 4),
                    "bm25_score": round(bm25_s, 4),
                    "vector_score": round(vec_s, 4),
                }
            )

        all_scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_results = all_scored[:top_k]

        # ── Phát hiện văn bản hết hiệu lực (Động) ─────────────────────────────
        expired_alerts = self._get_relevant_expired_alerts(query, detected_domain)

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
