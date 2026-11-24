"""
Công cụ dòng lệnh (CLI) đồng bộ văn bản quy phạm pháp luật & TTHC từ nguồn chính thống
vào hàng đợi Staging (`data/staging_procedures.json`).

Ví dụ:
    python -m src.ingestion.sync_cli --keyword "thủ tục hành chính" --limit 3
"""

from __future__ import annotations

import argparse
import json

from src.ingestion.chinhphu_client import fetch_chinhphu_detail, search_chinhphu_documents
from src.ingestion.procedure_extractor import extract_procedures_from_document
from src.ingestion.staging_store import add_candidates_to_staging
from src.retriever import get_retriever_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Đồng bộ dữ liệu pháp luật chính thống vào hàng đợi Staging")
    parser.add_argument("--keyword", type=str, default="thủ tục hành chính", help="Từ khóa tìm kiếm trên vanban.chinhphu.vn")
    parser.add_argument("--limit", type=int, default=3, help="Số lượng văn bản tối đa cần tải và bóc tách")
    parser.add_argument("--no-llm", action="store_true", help="Chỉ dùng trích xuất Regex tất định (không gọi Gemini)")
    args = parser.parse_args()

    retriever = get_retriever_agent()
    hits = search_chinhphu_documents(keyword=args.keyword, limit=args.limit)
    all_candidates = []
    for item in hits:
        doc = fetch_chinhphu_detail(item["url"], download_pdf_text=True)
        cands = extract_procedures_from_document(
            doc,
            existing_procedures=retriever.raw_procedures,
            use_llm=not args.no_llm,
        )
        all_candidates.extend(cands)

    stats = add_candidates_to_staging(
        all_candidates,
        source_summary=f"CLI sync keyword='{args.keyword}' ({len(hits)} văn bản)",
    )
    print(json.dumps({"fetched_docs": len(hits), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
