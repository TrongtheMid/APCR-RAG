"""
NHÁNH VŨ GIA ANH ĐỨC — Module Fixed-Size Chunker (Hệ thống Baseline):
Mô phỏng phương pháp cắt đoạn theo kích thước cố định truyền thống,
làm đứt gãy cấu trúc 6 trường JSON và trộn lẫn cả quy định đã hết hiệu lực.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.retriever.tokenizer import procedure_to_searchable_text


def build_fixed_word_chunks(
    data_path: Path,
    chunk_word_limit: int = 120,
) -> List[Dict[str, Any]]:
    """
    Gộp toàn bộ dữ liệu thủ tục thành chuỗi văn bản thô rồi cắt theo số từ cố định,
    không bảo toàn biên giới giữa các thủ tục và thành phần hồ sơ.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_list = json.load(f)

    all_words: List[str] = []
    for item in raw_list:
        text = procedure_to_searchable_text(item)
        all_words.extend(text.split())

    chunks: List[Dict[str, Any]] = []
    step = max(40, chunk_word_limit)
    for i in range(0, len(all_words), step):
        segment_words = all_words[i : i + chunk_word_limit]
        chunk_text = " ".join(segment_words)
        chunks.append(
            {
                "chunk_id": f"CHUNK-{len(chunks) + 1:03d}",
                "text": chunk_text,
            }
        )
    return chunks
