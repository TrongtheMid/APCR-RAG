"""
NHÁNH VŨ GIA ANH ĐỨC — Module Single-Agent RAG Baseline (Hạng mục 2.4):
Pipeline đối chứng truyền thống (cắt đoạn cố định + tìm kiếm vector thuần túy + sinh tự do 1 bước).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.baseline.fixed_chunker import build_fixed_word_chunks
from src.core.config import DEFAULT_DATA_PATH
from src.core.llm_factory import get_chat_llm
from src.retriever.vector_store import LightweightLegalEmbeddingFunction


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
        self.chunks: List[Dict[str, Any]] = build_fixed_word_chunks(
            self.data_path, chunk_word_limit=self.chunk_word_limit
        )
        self.embedder = LightweightLegalEmbeddingFunction(dim=384)
        self.chunk_embeddings: List[List[float]] = (
            self.embedder([c["text"] for c in self.chunks]) if self.chunks else []
        )

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
        """Sinh câu trả lời trong 1 bước duy nhất từ các đoạn chunk rời rạc."""
        start_time = time.perf_counter()
        top_chunks = self.retrieve_naive_vector(query, top_k=2)
        combined_context = "\n---\n".join(c["text"] for c in top_chunks)

        answer_text: Optional[str] = None
        llm = get_chat_llm(temperature=0.4)
        if llm is not None:
            try:
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


SingleAgentBaselineRAG = BaselineSingleAgentRAG
