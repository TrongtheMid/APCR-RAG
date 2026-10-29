"""
NHÁNH NGUYỄN HỮU HẢI LONG — Module BM25 Engine & Zero-LLM Direct Search:
- Cài đặt thuật toán Okapi BM25 (tích hợp fallback thuần Python khi chưa cài rank-bm25).
- Cung cấp hàm search_direct_bm25() phục vụ Phân hệ 4: Tra cứu văn bản & thủ tục tức thì (Zero-LLM).
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from src.retriever.metadata_filter import passes_metadata_filter
from src.retriever.tokenizer import highlight_keywords_html, tokenize_vietnamese

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:  # type: ignore[no-redef]
        """Thuật toán Okapi BM25 chuẩn (Fallback nội bộ khi chưa cài gói rank-bm25)."""

        def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
            self.corpus = corpus
            self.k1 = k1
            self.b = b
            self.corpus_size = len(corpus)
            self.doc_len = [len(doc) for doc in corpus]
            self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 1.0
            self.doc_freqs: List[Dict[str, int]] = []
            self.idf: Dict[str, float] = {}
            df: Dict[str, int] = {}

            for doc in corpus:
                frequencies: Dict[str, int] = {}
                for word in doc:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in frequencies:
                    df[word] = df.get(word, 0) + 1

            for word, freq in df.items():
                self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

        def get_scores(self, query: List[str]) -> List[float]:
            scores = [0.0] * self.corpus_size
            for q in query:
                if q not in self.idf:
                    continue
                q_idf = self.idf[q]
                for idx, doc_freq in enumerate(self.doc_freqs):
                    freq = doc_freq.get(q, 0)
                    if freq == 0:
                        continue
                    dl = self.doc_len[idx]
                    denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                    scores[idx] += q_idf * (freq * (self.k1 + 1.0)) / denom
            return scores


def search_direct_bm25(
    query: str,
    effective_date: Optional[str] = None,
    only_active: bool = False,
    phan_nhom: Optional[str] = None,
    linh_vuc: Optional[str] = None,
    top_k: int = 10,
    retriever: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    HÀM TRA CỨU VĂN BẢN & THỦ TỤC TỨC THÌ (ZERO-LLM API — PHÂN HỆ 4):
    - Tìm kiếm trực tiếp trên chỉ mục Full-text Rank-BM25 + Bộ lọc Metadata hiệu lực thời gian.
    - Không gọi LLM: Latency ~ 1 ms, Token Cost = 0.
    - Trả về danh sách Điều/Khoản/Thủ tục kèm nhãn ĐANG HIỆU LỰC / HẾT HIỆU LỰC và highlight từ khóa màu vàng.
    """
    from src.retriever.agent import get_retriever_agent

    start_time = time.perf_counter()
    agent = retriever or get_retriever_agent()

    has_query = bool(query and query.strip())
    if has_query:
        tokens = tokenize_vietnamese(query)
        raw_scores = agent.bm25.get_scores(tokens)
        max_score = max(raw_scores) if len(raw_scores) > 0 and max(raw_scores) > 0 else 1.0
        norm_scores = [float(s / max_score) for s in raw_scores]
    else:
        norm_scores = [1.0 for _ in agent.raw_procedures]

    items: List[Dict[str, Any]] = []
    for idx, proc in enumerate(agent.raw_procedures):
        if not passes_metadata_filter(
            proc,
            phan_nhom=phan_nhom,
            linh_vuc=linh_vuc,
            only_active=only_active,
            effective_date=effective_date,
        ):
            continue

        score = norm_scores[idx]
        info = proc["ma_va_ten_thu_tuc"]
        ho_so = proc["thanh_phan_ho_so"]
        thoi_han = proc["thoi_han_giai_quyet"]
        le_phi = proc["le_phi"]
        laws = proc["can_cu_phap_ly"]
        primary_law = laws[0]

        full_searchable = agent.corpus_texts[idx]
        if has_query and query.strip().lower() in full_searchable.lower():
            score = max(score, 0.85)

        if has_query and score < 0.05:
            continue

        ten_hl, matched_ten = highlight_keywords_html(info["ten_thu_tuc"], query)
        dk_hl, matched_dk = highlight_keywords_html(ho_so["dieu_kien_bat_buoc"], query)
        law_text = "; ".join(f"{l['so_hieu_van_ban']} — {l['dieu_khoan_diem']}" for l in laws)
        law_hl, matched_law = highlight_keywords_html(law_text, query)

        matched_keywords = list(dict.fromkeys(matched_ten + matched_dk + matched_law))

        items.append(
            {
                "ma_thu_tuc": info["ma_thu_tuc"],
                "ten_thu_tuc": info["ten_thu_tuc"],
                "ten_thu_tuc_highlighted": ten_hl,
                "phan_nhom": info["phan_nhom"],
                "linh_vuc": info["linh_vuc"],
                "cap_tham_quyen": info["cap_tham_quyen"],
                "giai_doan_to_tung": info["giai_doan_to_tung"],
                "trang_thai_hieu_luc": primary_law["trang_thai_hieu_luc"],
                "ngay_hieu_luc": primary_law["ngay_hieu_luc"],
                "van_ban_thay_the": primary_law.get("van_ban_thay_the"),
                "can_cu_highlighted": law_hl,
                "dieu_kien_highlighted": dk_hl,
                "muc_phi_vnd": le_phi["muc_phi_vnd"],
                "mo_ta_le_phi": le_phi["mo_ta"],
                "so_ngay_toi_da": thoi_han["so_ngay_toi_da"],
                "don_vi_tinh": thoi_han["don_vi_tinh"],
                "bm25_score": round(score, 4),
                "matched_keywords": matched_keywords,
                "raw_procedure": proc,
            }
        )

    items.sort(key=lambda x: x["bm25_score"], reverse=True)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "query": query,
        "total_matched": len(items),
        "results": items[:top_k],
        "latency_ms": latency_ms,
        "token_cost": 0,
    }
