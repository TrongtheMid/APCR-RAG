"""
NHÁNH NGUYỄN HỮU HẢI LONG — Module Reranker (Tầng Tái xếp hạng Ngữ nghĩa & Pháp lý):
- Hỗ trợ Cross-Encoder thực chiến (`BAAI/bge-reranker-v2-m3`) khi `USE_CROSS_ENCODER_RERANKER=true`.
- Mặc định dùng `LightweightLegalReranker` tốc độ cao (đối sánh tiêu đề, lĩnh vực, mã biểu mẫu, điều luật)
  để tái xếp hạng nhóm ứng viên Top-K*3 sau RRF mà không phá vỡ ngưỡng từ chối an toàn.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.core.config import RERANKER_MODEL_NAME, USE_CROSS_ENCODER_RERANKER
from src.retriever.tokenizer import normalize_text, vietnamese_legal_tokenize

_RERANK_GATE_MIN_SCORE = 0.20


def _sigmoid(val: float) -> float:
    if val >= 20.0:
        return 1.0
    if val <= -20.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-val))


def _procedure_rerank_text(proc: Dict[str, Any]) -> str:
    info = proc.get("ma_va_ten_thu_tuc") or {}
    ho_so = proc.get("thanh_phan_ho_so") or {}
    forms = ", ".join(
        f"{g.get('form_id', '')} {g.get('ten_giay_to', '')}"
        for g in ho_so.get("giay_to_bat_buoc", [])
    )
    laws = ", ".join(
        f"{c.get('so_hieu_van_ban', '')} {c.get('dieu_khoan_diem', '')}"
        for c in proc.get("can_cu_phap_ly", [])
    )
    return (
        f"{info.get('ma_thu_tuc', '')} {info.get('ten_thu_tuc', '')} | "
        f"{info.get('linh_vuc', '')} | {', '.join(info.get('cap_tham_quyen', []))} | "
        f"{forms} | {laws} | {ho_so.get('dieu_kien_bat_buoc', '')}"
    )


class LightweightLegalReranker:
    """
    Bộ tái xếp hạng nhẹ đa trường (Zero-Latency) trên thang [0.0, 1.0]:
    Ưu tiên mức độ trùng khớp cụm từ khóa ở tên thủ tục, lĩnh vực, mã hiệu và căn cứ pháp lý.
    """

    def score_pairs(self, query: str, procedures: List[Dict[str, Any]]) -> List[float]:
        if not query or not procedures:
            return [0.0 for _ in procedures]

        q_norm = normalize_text(query)
        q_tokens = {t for t in vietnamese_legal_tokenize(q_norm) if len(t) >= 2}
        q_compounds = {t for t in q_tokens if "_" in t}
        if not q_tokens:
            return [0.0 for _ in procedures]

        scores: List[float] = []
        for proc in procedures:
            info = proc.get("ma_va_ten_thu_tuc") or {}
            title_norm = normalize_text(info.get("ten_thu_tuc", ""))
            field_norm = normalize_text(info.get("linh_vuc", ""))
            code_norm = normalize_text(info.get("ma_thu_tuc", ""))
            laws_norm = normalize_text(
                " ".join(
                    f"{c.get('so_hieu_van_ban', '')} {c.get('dieu_khoan_diem', '')}"
                    for c in proc.get("can_cu_phap_ly", [])
                )
            )
            full_norm = normalize_text(_procedure_rerank_text(proc))

            title_tokens = set(vietnamese_legal_tokenize(title_norm))
            full_tokens = set(vietnamese_legal_tokenize(full_norm))

            title_overlap = len(q_tokens & title_tokens) / max(1, min(len(q_tokens), 8))
            compound_overlap = (
                len(q_compounds & title_tokens) / max(1, len(q_compounds))
                if q_compounds
                else 0.0
            )
            full_overlap = len(q_tokens & full_tokens) / max(1, min(len(q_tokens), 12))

            bonus = 0.0
            if code_norm and code_norm in q_norm:
                bonus += 0.35
            if field_norm and field_norm in q_norm:
                bonus += 0.15
            if any(tok in laws_norm for tok in q_tokens if any(ch.isdigit() for ch in tok)):
                bonus += 0.12

            raw_score = 0.45 * min(1.0, title_overlap) + 0.25 * min(1.0, compound_overlap) + 0.30 * min(1.0, full_overlap) + bonus
            scores.append(round(min(1.0, max(0.0, raw_score)), 4))

        return scores


_CROSS_ENCODER_CACHE: Any = None
_CROSS_ENCODER_LOAD_TRIED: bool = False


def _get_cross_encoder() -> Optional[Any]:
    global _CROSS_ENCODER_CACHE, _CROSS_ENCODER_LOAD_TRIED
    if not USE_CROSS_ENCODER_RERANKER:
        return None
    if _CROSS_ENCODER_LOAD_TRIED:
        return _CROSS_ENCODER_CACHE
    _CROSS_ENCODER_LOAD_TRIED = True
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        _CROSS_ENCODER_CACHE = CrossEncoder(RERANKER_MODEL_NAME)
    except Exception:
        _CROSS_ENCODER_CACHE = None
    return _CROSS_ENCODER_CACHE


def rerank_candidates(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int = 3,
    lightweight_reranker: Optional[LightweightLegalReranker] = None,
) -> List[Dict[str, Any]]:
    """
    Tái xếp hạng nhóm ứng viên hàng đầu (tối đa `top_k * 3` ứng viên) và trả về `top_k` kết quả.
    Giữ nguyên điểm thấp (< 0.20) để không ảnh hưởng tới hàng rào Deterministic Refusal.
    """
    if not candidates:
        return []

    reranker = lightweight_reranker or LightweightLegalReranker()
    pool_size = max(top_k, min(len(candidates), top_k * 3))
    pool = [dict(item) for item in candidates[:pool_size]]
    procs = [item["procedure"] for item in pool]

    cross_encoder = _get_cross_encoder()
    rerank_scores: List[float]
    if cross_encoder is not None:
        try:
            pairs = [(query, _procedure_rerank_text(proc)) for proc in procs]
            raw_logits = cross_encoder.predict(pairs)
            rerank_scores = [round(_sigmoid(float(val)), 4) for val in raw_logits]
        except Exception:
            rerank_scores = reranker.score_pairs(query, procs)
    else:
        rerank_scores = reranker.score_pairs(query, procs)

    for item, r_score in zip(pool, rerank_scores):
        item["rerank_score"] = round(float(r_score), 4)
        base_hybrid = float(item.get("hybrid_score", 0.0))
        if base_hybrid >= _RERANK_GATE_MIN_SCORE:
            item["hybrid_score"] = round(min(1.0, 0.88 * base_hybrid + 0.12 * float(r_score)), 4)

    pool.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return pool[:top_k]
