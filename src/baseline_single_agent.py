"""
NHÁNH CỦA VŨ GIA ANH ĐỨC (HẠNG MỤC 2.4):
Hệ thống Baseline — Single-Agent RAG Đối chứng truyền thống:
- Nối toàn bộ dữ liệu thành văn bản thô và cắt đoạn theo kích thước cố định (Fixed-size Chunking ~500 từ/ký tự),
  không bảo toàn cấu trúc Lược đồ 6 trường JSON.
- Tìm kiếm Vector đơn thuần (không có Query Rewriting, không có BM25 Hybrid, không lọc trạng thái hiệu lực).
- Sinh câu trả lời tự do trong 1 bước duy nhất (không ép chuẩn IRAC, không có Judge Agent 2 lớp, không có Self-Revision).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.retriever_agent import (
    DEFAULT_DATA_PATH,
    LightweightLegalEmbeddingFunction,
    procedure_to_searchable_text,
)


class BaselineSingleAgentRAG:
    """
    Pipeline Single-Agent RAG truyền thống dùng làm đối chứng thực nghiệm (Baseline)
    khi đánh giá bộ 50 câu hỏi kiểm thử (RAGAS Benchmark).
    """

    def __init__(
        self,
        data_path: Optional[Path | str] = None,
        chunk_word_limit: int = 120,
    ) -> None:
        self.data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        self.chunk_word_limit = chunk_word_limit
        self.chunks: List[Dict[str, Any]] = []
        self.embedder = LightweightLegalEmbeddingFunction(dim=384)
        self.chunk_embeddings: List[List[float]] = []

        self._build_fixed_chunks()

    def _build_fixed_chunks(self) -> None:
        """
        Mô phỏng phương pháp cắt đoạn cố định (Fixed-size Chunking) truyền thống:
        Gộp văn bản thô rồi cắt theo số từ cố định, làm đứt gãy danh mục giấy tờ, lệ phí,
        điều kiện bắt buộc khỏi mã thủ tục và trộn lẫn cả các quy định đã hết hiệu lực.
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        all_words: List[str] = []
        for item in raw_list:
            text = procedure_to_searchable_text(item)
            all_words.extend(text.split())

        step = max(40, self.chunk_word_limit)
        for i in range(0, len(all_words), step):
            segment_words = all_words[i : i + self.chunk_word_limit]
            chunk_text = " ".join(segment_words)
            self.chunks.append(
                {
                    "chunk_id": f"CHUNK-{len(self.chunks) + 1:03d}",
                    "text": chunk_text,
                }
            )

        if self.chunks:
            self.chunk_embeddings = self.embedder([c["text"] for c in self.chunks])

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        return sum(a * b for a, b in zip(v1, v2))

    def retrieve_naive_vector(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Tìm kiếm vector đơn thuần trên các đoạn cắt cố định (không lọc hiệu lực thời gian)."""
        if not self.chunks:
            return []
        q_vec = self.embedder([query])[0]
        scored = []
        for chunk, c_vec in zip(self.chunks, self.chunk_embeddings):
            sim = self._cosine_similarity(q_vec, c_vec)
            scored.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": round(float(sim), 4),
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def answer_single_step(self, query: str) -> Dict[str, Any]:
        """
        Sinh câu trả lời trong 1 bước duy nhất từ các đoạn chunk rời rạc
        (không kiểm chứng chéo, không phát hiện văn bản hết hiệu lực hoặc bẫy pháp lý).
        """
        start_time = time.perf_counter()
        top_chunks = self.retrieve_naive_vector(query, top_k=2)
        combined_context = "\n---\n".join(c["text"] for c in top_chunks)

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        answer_text: Optional[str] = None

        if api_key and api_key != "your_google_gemini_api_key_here":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                llm = ChatGoogleGenerativeAI(
                    model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                    google_api_key=api_key,
                    temperature=0.4,
                )
                prompt = (
                    "Hãy trả lời ngắn gọn câu hỏi pháp luật sau dựa trên đoạn văn bản cắt rời bên dưới:\n\n"
                    f"Đoạn văn bản:\n{combined_context}\n\n"
                    f"Câu hỏi: {query}"
                )
                resp = llm.invoke(prompt)
                answer_text = str(resp.content)
            except Exception:
                answer_text = None

        if not answer_text:
            answer_text = (
                f"[Single-Agent Baseline — Sinh tự do 1 bước từ đoạn cắt cố định]:\n"
                f"Dựa trên đoạn văn bản tìm thấy ({', '.join(c['chunk_id'] for c in top_chunks)}), "
                f"thông tin liên quan đến câu hỏi \"{query}\" như sau:\n\n"
                f"{combined_context[:650]}...\n\n"
                f"*(Lưu ý: Do cắt đoạn cố định và thiếu Judge Agent 2 lớp, kết quả Baseline có thể bị đứt gãy "
                f"danh mục biểu mẫu hoặc không cảnh báo được các điều khoản đã HẾT HIỆU LỰC)*."
            )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "query": query,
            "answer": answer_text,
            "retrieved_chunks": top_chunks,
            "latency_ms": latency_ms,
            "architecture": "Single-Agent RAG (Fixed Chunking + No Judge Verification)",
        }
