"""
Package src cho Hệ thống APCR-RAG (Administrative & Penal Criminal RAG)
Tư vấn về Thủ tục Hành chính và Tố tụng Hình sự — Học phần INT 3507 2 (Nhóm 9).

Kiến trúc Clean & Modular Architecture gồm 8 sub-package chuyên biệt:
- src.core: Cấu hình trung tâm & khởi tạo LLM / Embedding
- src.schemas: Các lược đồ Pydantic (6-Field JSON, IRAC, Judge, Benchmark)
- src.retriever: Retriever Agent & Hybrid Search (Nhánh Nguyễn Hữu Hải Long)
- src.consultant: Consultant Agent & Lập luận IRAC (Nhánh Vũ Gia Anh Đức)
- src.baseline: Single-Agent Baseline đối chứng (Nhánh Vũ Gia Anh Đức)
- src.judge: Judge Agent Kiểm chứng 2 lớp (Nhánh Phạm Đức Trọng)
- src.workflow: Điều phối LangGraph StateGraph (Nhánh Phạm Đức Trọng)
- src.evaluation: Công cụ đánh giá Benchmark 50 câu hỏi (Nhánh Phạm Đức Trọng)
"""
