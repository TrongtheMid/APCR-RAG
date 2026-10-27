"""
NHÁNH CỦA NGUYỄN HỮU HẢI LONG:
- Quản lý dữ liệu chuẩn hóa 6 trường JSON (data/data_procedures.json).
- Retriever Agent Đa tầng: Query Rewriting & Domain Routing + Metadata Filtering + Hybrid Search (Rank-BM25 + ChromaDB).
- Hàm search_direct_bm25() (Zero-LLM API) phục vụ Phân hệ 4: Tra cứu Văn bản & Thủ tục Tức thì.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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

from src.schema import ProcedureRecord

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data_procedures.json"


# ==============================================================================
# TỪ ĐIỂN ÁNH XẠ NGÔN NGỮ ĐỜI THƯỜNG -> THUẬT NGỮ PHÁP LÝ CHUẨN (QUERY REWRITING)
# ==============================================================================

COLLOQUIAL_TO_LEGAL_MAP: List[Tuple[List[str], str, Optional[str]]] = [
    (
        ["bảo lãnh", "bảo lĩnh", "tại ngoại", "bắt tạm giam", "xin về"],
        "Thủ tục nhận bảo lĩnh cho bị can bị cáo tại ngoại Điều 121 BLTTHS TTHS-BL-121 ít nhất 02 người thân thích",
        "Tố tụng hình sự",
    ),
    (
        ["đặt tiền", "nộp tiền bảo lãnh", "tiền bảo đảm", "ký quỹ tại ngoại"],
        "Thủ tục đặt tiền để bảo đảm thay thế biện pháp tạm giam Điều 122 BLTTHS TTHS-DT-122 Kho bạc Nhà nước",
        "Tố tụng hình sự",
    ),
    (
        ["lừa đảo", "trình báo", "tố giác", "báo công an", "mất trộm", "trộm cắp", "chiếm đoạt"],
        "Thủ tục trình báo tố giác tin báo về tội phạm Điều 145 Điều 147 BLTTHS TTHS-TG-145 miễn phí",
        "Tố tụng hình sự",
    ),
    (
        ["giám định thương tích", "tỷ lệ thương tật", "tổn thương cơ thể", "định giá tài sản", "bị đánh"],
        "Thủ tục yêu cầu trưng cầu giám định tỷ lệ tổn thương cơ thể định giá tài sản Điều 205 BLTTHS TTHS-GD-205",
        "Tố tụng hình sự",
    ),
    (
        ["thăm gặp", "thăm nuôi", "gặp người bị tạm giam", "trại tạm giam", "nhà tạm giữ"],
        "Thủ tục xin thăm gặp người đang bị tạm giữ tạm giam Điều 22 Luật Thi hành tạm giữ tạm giam TTHS-TG-022",
        "Tố tụng hình sự",
    ),
    (
        ["kháng cáo", "chống án", "phúc thẩm", "quá hạn kháng cáo"],
        "Thủ tục nộp đơn kháng cáo bản án sơ thẩm hình sự Điều 331 Điều 333 BLTTHS TTHS-KC-331 thời hạn 15 ngày",
        "Tố tụng hình sự",
    ),
    (
        ["xóa án tích theo quyết định", "an ninh quốc gia", "xin tòa xóa án tích", "phá hoại hòa bình"],
        "Thủ tục nộp hồ sơ xin Tòa án xóa án tích theo quyết định Điều 71 BLHS TTHS-XAT-071 TTHS-XAT-071",
        "Tố tụng hình sự",
    ),
    (
        ["đương nhiên xóa án tích", "chứng nhận xóa án tích", "tội trộm cắp", "tội đánh bạc"],
        "Thủ tục xác nhận đương nhiên được xóa án tích Điều 70 BLHS Điều 369 BLTTHS TTHS-XAT-070 Sở Tư pháp",
        "Tố tụng hình sự",
    ),
    (
        ["lý lịch tư pháp", "phiếu số 1", "phiếu số 2", "lltp"],
        "Thủ tục yêu cầu cấp Phiếu lý lịch tư pháp số 1 và số 2 Luật Lý lịch tư pháp TTHS-LLTP-045 Sở Tư pháp",
        "Tố tụng hình sự",
    ),
    (
        ["hoãn thi hành án", "hoãn chấp hành án", "hoãn đi tù", "nuôi con dưới 36 tháng", "lao động duy nhất"],
        "Thủ tục xin hoãn chấp hành án phạt tù Điều 67 BLHS Điều 24 Luật Thi hành án hình sự TTHS-THA-067",
        "Tố tụng hình sự",
    ),
    (
        ["khai sinh", "đăng ký khai sinh", "giấy chứng sinh", "mới sinh con"],
        "Thủ tục đăng ký khai sinh thông thường Điều 15 Điều 16 Luật Hộ tịch TTHC-HT-01 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["kết hôn với người nước ngoài", "chồng nước ngoài", "vợ nước ngoài", "quốc tịch hàn", "quốc tịch pháp", "yếu tố nước ngoài"],
        "Thủ tục đăng ký kết hôn có yếu tố nước ngoài Điều 37 Luật Hộ tịch TTHC-HT-05 TTHC-HT-05 UBND cấp huyện",
        "Thủ tục hành chính",
    ),
    (
        ["kết hôn trong nước", "giấy chứng nhận kết hôn"],
        "Thủ tục đăng ký kết hôn trong nước Điều 17 Điều 18 Luật Hộ tịch TTHC-HT-02 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["độc thân", "tình trạng hôn nhân", "xác nhận độc thân"],
        "Thủ tục cấp Giấy xác nhận tình trạng hôn nhân Nghị định 123/2015/NĐ-CP TTHC-HT-03 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["khai tử", "giấy báo tử", "chứng tử", "người thân mất"],
        "Thủ tục đăng ký khai tử Điều 32 Điều 34 Luật Hộ tịch TTHC-HT-04 UBND cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["thường trú", "nhập khẩu", "sổ hộ khẩu", "hk07", "ct01"],
        "Thủ tục đăng ký thường trú Luật Cư trú 2020 Điều 20 TTHC-CT-01 Công an cấp xã bãi bỏ Sổ hộ khẩu giấy",
        "Thủ tục hành chính",
    ),
    (
        ["tạm trú", "gia hạn tạm trú", "ở trọ"],
        "Thủ tục đăng ký tạm trú Luật Cư trú 2020 Điều 27 TTHC-CT-02 Công an cấp xã",
        "Thủ tục hành chính",
    ),
    (
        ["căn cước", "cccd", "cmnd 9 số", "chứng minh nhân dân", "đổi thẻ căn cước"],
        "Thủ tục cấp đổi cấp lại thẻ Căn cước Luật Căn cước 2023 Điều 24 TTHC-CC-01 CMND hết hiệu lực 01/01/2025",
        "Thủ tục hành chính",
    ),
    (
        ["trợ cấp thất nghiệp", "bảo hiểm thất nghiệp", "thất nghiệp"],
        "Thủ tục nộp hồ sơ đăng ký và hưởng trợ cấp thất nghiệp Luật Việc làm Điều 46 TTHC-LD-01 Trung tâm dịch vụ việc làm",
        "Thủ tục hành chính",
    ),
    (
        ["giấy phép lao động", "lao động nước ngoài", "chuyên gia nước ngoài"],
        "Thủ tục cấp Giấy phép lao động cho người nước ngoài Nghị định 152/2020/NĐ-CP Nghị định 70/2023/NĐ-CP TTHC-LD-02",
        "Thủ tục hành chính",
    ),
    (
        ["bhxh một lần", "bảo hiểm xã hội 1 lần", "rút bhxh", "rút bảo hiểm xã hội"],
        "Thủ tục giải quyết hưởng chế độ Bảo hiểm xã hội một lần Điều 60 Luật BHXH TTHC-BH-01",
        "Thủ tục hành chính",
    ),
]


def normalize_text(text: str) -> str:
    """Chuẩn hóa chữ thường và khoảng trắng phục vụ tách từ tiếng Việt."""
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[^\w\s\d\-/]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_vietnamese(text: str) -> List[str]:
    """Tách từ kết hợp unigram và bigram tiếng Việt để tăng độ chính xác cho BM25."""
    norm = normalize_text(text)
    words = [w for w in norm.split(" ") if len(w) > 1 or w.isdigit()]
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


def procedure_to_searchable_text(proc_dict: Dict[str, Any]) -> str:
    """Chuyển đổi cấu trúc 6 trường JSON thành chuỗi văn bản giàu ngữ nghĩa để đánh chỉ mục."""
    info = proc_dict["ma_va_ten_thu_tuc"]
    ho_so = proc_dict["thanh_phan_ho_so"]
    thoi_han = proc_dict["thoi_han_giai_quyet"]
    le_phi = proc_dict["le_phi"]
    can_cu_list = proc_dict["can_cu_phap_ly"]

    forms_str = "; ".join(
        f"{g['form_id']} {g['ten_giay_to']}" for g in ho_so.get("giay_to_bat_buoc", [])
    )
    laws_str = "; ".join(
        f"{c['law_id']} {c['so_hieu_van_ban']} {c['dieu_khoan_diem']} ({c['trang_thai_hieu_luc']})"
        for c in can_cu_list
    )
    steps_str = " ".join(proc_dict.get("trinh_tu_thuc_hien", []))

    return (
        f"Mã thủ tục: {info['ma_thu_tuc']}. Tên thủ tục: {info['ten_thu_tuc']}. "
        f"Phân nhóm: {info['phan_nhom']}. Lĩnh vực: {info['linh_vuc']}. "
        f"Cơ quan thẩm quyền: {', '.join(info.get('cap_tham_quyen', []))}. "
        f"Giai đoạn tố tụng: {', '.join(info.get('giai_doan_to_tung', []))}. "
        f"Trình tự thực hiện: {steps_str}. "
        f"Thành phần hồ sơ: {forms_str}. "
        f"Điều kiện bắt buộc: {ho_so.get('dieu_kien_bat_buoc', '')}. "
        f"Thời hạn giải quyết: tối đa {thoi_han['so_ngay_toi_da']} {thoi_han['don_vi_tinh']}. {thoi_han['chi_tiet']}. "
        f"Lệ phí: {le_phi['muc_phi_vnd']} VNĐ. {le_phi['mo_ta']}. "
        f"Căn cứ pháp lý: {laws_str}."
    )


class LightweightLegalEmbeddingFunction:
    """
    Hàm nhúng Vector nội bộ tốc độ cao (Hashing + Subword N-gram 384 chiều) tương thích ChromaDB.
    Được sử dụng mặc định khi USE_LIGHTWEIGHT_EMBEDDING=true hoặc chưa tải mô hình BAAI/bge-m3 nặng.
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


class RetrieverAgent:
    """
    AGENT 01 — RETRIEVER AGENT (Tác tử Tra cứu Hồ sơ Đa tầng):
    - Query Rewriting & Domain Routing
    - Metadata Filtering (Phân nhóm, Lĩnh vực, Cấp thẩm quyền, Giai đoạn tố tụng, Hiệu lực thời gian)
    - Hybrid Search (Rank-BM25 + ChromaDB Vector Store)
    - Zero-LLM Direct Search (search_direct_bm25)
    """

    def __init__(
        self,
        data_path: Optional[Path | str] = None,
        bm25_weight: float = 0.65,
        vector_weight: float = 0.35,
    ) -> None:
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self.bm25_weight = float(os.getenv("BM25_WEIGHT", bm25_weight))
        self.vector_weight = float(os.getenv("VECTOR_WEIGHT", vector_weight))

        self.raw_procedures: List[Dict[str, Any]] = []
        self.validated_records: List[ProcedureRecord] = []
        self.procedures_by_id: Dict[str, Dict[str, Any]] = {}
        self.corpus_texts: List[str] = []
        self.tokenized_corpus: List[List[str]] = []

        self._load_and_validate_data()
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.chroma_collection = self._init_chroma_collection()

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

    def _init_chroma_collection(self) -> Any:
        """Khởi tạo ChromaDB Vector Store lưu trữ các thủ tục pháp lý."""
        try:
            import chromadb

            use_lightweight = os.getenv("USE_LIGHTWEIGHT_EMBEDDING", "true").lower() == "true"
            embedding_fn: Any = LightweightLegalEmbeddingFunction(dim=384)

            if not use_lightweight:
                try:
                    from chromadb.utils.embedding_functions import (
                        SentenceTransformerEmbeddingFunction,
                    )

                    model_name = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
                    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=model_name)
                except Exception:
                    embedding_fn = LightweightLegalEmbeddingFunction(dim=384)

            client = chromadb.EphemeralClient()
            collection_name = os.getenv("CHROMA_COLLECTION_NAME", "legal_procedures_bge_m3")
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

            ids = [p["ma_va_ten_thu_tuc"]["ma_thu_tuc"] for p in self.raw_procedures]
            metadatas = []
            for p in self.raw_procedures:
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

            collection.upsert(
                ids=ids,
                documents=self.corpus_texts,
                metadatas=metadatas,
            )
            return collection
        except Exception:
            return None

    def rewrite_query_and_route(
        self, query: str, preferred_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Chuyển đổi câu hỏi đời thường thành truy vấn pháp lý chuẩn (Query Rewriting)
        và phân loại miền dữ liệu (Domain Routing: Thủ tục hành chính vs Tố tụng hình sự).
        """
        q_lower = query.lower()
        expansions: List[str] = []
        detected_domain: Optional[str] = preferred_domain

        for keywords, legal_expansion, domain in COLLOQUIAL_TO_LEGAL_MAP:
            if any(kw in q_lower for kw in keywords):
                expansions.append(legal_expansion)
                if not detected_domain and domain:
                    detected_domain = domain

        if not detected_domain:
            criminal_markers = [
                "bị can", "bị cáo", "tạm giam", "tạm giữ", "tội phạm", "khởi tố",
                "điều tra", "viện kiểm sát", "xét xử", "án tích", "kháng cáo", "án phạt tù",
                "bảo lĩnh", "tố giác", "giám định", "lý lịch tư pháp",
            ]
            if any(m in q_lower for m in criminal_markers):
                detected_domain = "Tố tụng hình sự"
            else:
                detected_domain = "Thủ tục hành chính"

        rewritten_query = query
        if expansions:
            rewritten_query = f"{query} | Truy vấn pháp lý mở rộng: {' ; '.join(expansions)}"

        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "detected_domain": detected_domain,
            "matched_expansions": expansions,
        }

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
        """Kiểm tra hồ sơ thủ tục có thỏa mãn bộ lọc Metadata hay không."""
        info = proc["ma_va_ten_thu_tuc"]
        laws = proc["can_cu_phap_ly"]

        if phan_nhom and phan_nhom not in ("Tất cả", ""):
            if info["phan_nhom"] != phan_nhom:
                return False

        if linh_vuc and linh_vuc not in ("Tất cả", ""):
            if linh_vuc.lower() not in info["linh_vuc"].lower():
                return False

        if cap_tham_quyen and cap_tham_quyen not in ("Tất cả", ""):
            authorities = " ".join(info.get("cap_tham_quyen", [])).lower()
            if cap_tham_quyen.lower() not in authorities:
                return False

        if giai_doan_to_tung and giai_doan_to_tung not in ("Tất cả", ""):
            stages = " ".join(info.get("giai_doan_to_tung", [])).lower()
            if giai_doan_to_tung.lower() not in stages:
                return False

        if only_active:
            has_active = any(l["trang_thai_hieu_luc"] == "ĐANG HIỆU LỰC" for l in laws)
            if not has_active:
                return False

        if effective_date:
            # Ít nhất một văn bản căn cứ có ngày hiệu lực <= effective_date
            valid_by_date = any(l["ngay_hieu_luc"] <= effective_date for l in laws)
            if not valid_by_date:
                return False

        return True

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
        Thực hiện Hybrid Search kết hợp điểm số giữa Rank-BM25 và Vector Embedding trên ChromaDB.
        Trả về danh sách hồ sơ thủ tục phù hợp nhất kèm điểm số chi tiết.
        """
        start_time = time.perf_counter()
        routing_info = self.rewrite_query_and_route(query, preferred_domain=phan_nhom)
        rewritten_query = routing_info["rewritten_query"]

        # 1. Tính điểm BM25
        query_tokens = tokenize_vietnamese(rewritten_query)
        bm25_raw_scores = self.bm25.get_scores(query_tokens)
        max_bm25 = max(bm25_raw_scores) if len(bm25_raw_scores) > 0 and max(bm25_raw_scores) > 0 else 1.0
        bm25_norm_scores = [float(s / max_bm25) for s in bm25_raw_scores]

        # 2. Tính điểm Vector Search từ ChromaDB
        vector_scores_map: Dict[str, float] = {}
        if self.chroma_collection is not None:
            try:
                chroma_res = self.chroma_collection.query(
                    query_texts=[rewritten_query],
                    n_results=len(self.raw_procedures),
                )
                ids_list = chroma_res.get("ids", [[]])[0]
                distances_list = chroma_res.get("distances", [[]])[0]
                for doc_id, dist in zip(ids_list, distances_list):
                    sim = max(0.0, 1.0 - float(dist))
                    vector_scores_map[doc_id] = sim
            except Exception:
                pass

        # 3. Kết hợp điểm Hybrid + Metadata Filtering
        scored_results: List[Dict[str, Any]] = []
        for idx, proc in enumerate(self.raw_procedures):
            if not self._passes_metadata_filter(
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

            # Thưởng điểm nếu câu hỏi khớp trực tiếp mã thủ tục hoặc từ khóa cốt lõi trong tên thủ tục
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

        # Nếu câu hỏi liên quan đến Sổ hộ khẩu giấy hoặc CMND 9 số đã hết hiệu lực, đính kèm cảnh báo quy định cũ
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


# Singleton instance để tái sử dụng nhanh trên toàn hệ thống
_DEFAULT_RETRIEVER: Optional[RetrieverAgent] = None


def get_retriever_agent() -> RetrieverAgent:
    """Lấy thực thể RetrieverAgent dùng chung (Singleton)."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = RetrieverAgent()
    return _DEFAULT_RETRIEVER


def highlight_keywords_html(text: str, query: str) -> Tuple[str, List[str]]:
    """
    Tô sáng (highlight màu vàng) các từ khóa trong đoạn văn bản khớp với truy vấn của người dùng.
    Trả về chuỗi HTML đã bọc thẻ <mark> và danh sách các từ khóa khớp.
    """
    if not query or not query.strip():
        return text, []

    raw_terms = [
        t.strip()
        for t in re.split(r"[\s,;.?!]+", query.strip())
        if len(t.strip()) >= 2
    ]
    if not raw_terms:
        return text, []

    # Ưu tiên cụm từ dài trước
    sorted_terms = sorted(set(raw_terms), key=len, reverse=True)
    matched_terms: List[str] = []
    highlighted = text

    for term in sorted_terms:
        pattern = re.compile(rf"({re.escape(term)})", flags=re.IGNORECASE | re.UNICODE)
        if pattern.search(highlighted):
            matched_terms.append(term)
            highlighted = pattern.sub(
                r'<mark style="background-color: #fde047; color: #1e293b; padding: 1px 4px; border-radius: 3px; font-weight: 600;">\1</mark>',
                highlighted,
            )

    return highlighted, matched_terms


def search_direct_bm25(
    query: str,
    effective_date: Optional[str] = None,
    only_active: bool = False,
    phan_nhom: Optional[str] = None,
    linh_vuc: Optional[str] = None,
    top_k: int = 10,
    retriever: Optional[RetrieverAgent] = None,
) -> Dict[str, Any]:
    """
    HÀM TRA CỨU VĂN BẢN & THỦ TỤC TỨC THÌ (ZERO-LLM API — PHÂN HỆ 4):
    - Tìm kiếm trực tiếp trên chỉ mục Full-text Rank-BM25 + Bộ lọc Metadata hiệu lực thời gian.
    - Không gọi LLM: Latency ~ 0 ms, Token Cost = 0.
    - Trả về danh sách Điều/Khoản/Thủ tục kèm nhãn ĐANG HIỆU LỰC / HẾT HIỆU LỰC và highlight từ khóa màu vàng.
    """
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
        if not agent._passes_metadata_filter(
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

        # Kiểm tra khớp trực tiếp chuỗi con
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
