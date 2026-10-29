APCR-RAG (Administrative & Penal Criminal RAG)
<img width="1443" height="1267" alt="image" src="https://github.com/user-attachments/assets/44a2d638-e50e-4bc9-b7f6-06fe0f549c4b" />


APCR-RAG/
├── data/
│   ├── data_procedures.json              # Kho 23+ thủ tục chuẩn hóa Lược đồ 6 trường + Hiệu lực
│   └── benchmark_50_questions.json       # Bộ 50 câu hỏi kiểm thử (Đơn, Tình huống, Gài bẫy)
│
├── src/
│   ├── __init__.py
│   ├── core/                             # [Tầng 0] Cấu hình chung & Khởi tạo LLM/Env
│   │   ├── __init__.py
│   │   ├── config.py                     # Quản lý biến môi trường (.env), hằng số, ngưỡng điểm
│   │   └── llm_factory.py                # Khởi tạo Gemini client dùng chung & cơ chế Fallback
│   │
│   ├── schemas/                          # [Tầng 1] Lược đồ Dữ liệu Pydantic (Tách theo miền)
│   │   ├── __init__.py
│   │   ├── procedure_schema.py           # ProcedureRecord (6 trường chuẩn hóa + Temporal Metadata)
│   │   ├── irac_schema.py                # IRACOutput & CitationMetadata
│   │   ├── judge_schema.py               # JudgeVerificationResult (Lớp 1 + Lớp 2)
│   │   └── benchmark_schema.py           # BenchmarkQuestion & MisconceptionAlert
│   │
│   ├── retriever/                        # [Tầng 2A — Nhánh Nguyễn Hữu Hải Long]
│   │   ├── __init__.py
│   │   ├── tokenizer.py                  # Chuẩn hóa tiếng Việt, tách từ unigram/bigram & highlight
│   │   ├── query_rewriter.py             # Ánh xạ câu hỏi đời thường -> thuật ngữ pháp lý + Domain Router
│   │   ├── metadata_filter.py            # Bộ lọc cấp thẩm quyền, giai đoạn tố tụng & ngày hiệu lực
│   │   ├── bm25_engine.py                # Chỉ mục Rank-BM25 & hàm search_direct_bm25 (Zero-LLM)
│   │   ├── vector_store.py               # Kết nối ChromaDB & hàm nhúng Vector (BGE-M3 / Lightweight)
│   │   └── agent.py                      # Lớp RetrieverAgent kết hợp điểm số Hybrid Search
│   │
│   ├── consultant/                       # [Tầng 2B — Nhánh Vũ Gia Anh Đức]
│   │   ├── __init__.py
│   │   ├── prompts.py                    # System Prompt chuẩn IRAC & Prompt tự sửa lỗi (Self-Revision)
│   │   ├── refusal_guard.py              # Kiểm tra ngưỡng an toàn & Từ chối chủ động (Deterministic Refusal)
│   │   ├── irac_synthesizer.py           # Bộ sinh lập luận IRAC (Gemini Structured Output + Rule Fallback)
│   │   ├── graph_builder.py              # Chuyển đổi hồ sơ 6 trường thành Nodes & Edges cho đồ thị
│   │   └── agent.py                      # Lớp ConsultantAgent
│   │
│   ├── baseline/                         # [Tầng 2C — Nhánh Vũ Gia Anh Đức (Đối chứng)]
│   │   ├── __init__.py
│   │   ├── fixed_chunker.py              # Cắt đoạn văn bản cố định (~500 từ, làm đứt gãy cấu trúc)
│   │   └── single_agent_rag.py           # Pipeline Single-Agent RAG truyền thống (sinh 1 bước)
│   │
│   ├── judge/                            # [Tầng 2D — Nhánh Phạm Đức Trọng]
│   │   ├── __init__.py
│   │   ├── layer1_deterministic.py       # Lớp 1: Code Python đối soát Exact Match (lệ phí, ngày, mã)
│   │   ├── layer2_semantic.py            # Lớp 2: LLM Critic kiểm tra điều kiện tiên quyết & thẩm quyền
│   │   ├── misconceptions_catalog.py     # Kho 10+ cảnh báo hiểu nhầm pháp lý phổ biến (Phân hệ 3)
│   │   └── agent.py                      # Lớp JudgeAgent tổng hợp 2 lớp & sinh critique_feedback
│   │
│   ├── workflow/                         # [Tầng 3 — Nhánh Phạm Đức Trọng (LangGraph)]
│   │   ├── __init__.py
│   │   ├── state.py                      # Định nghĩa LegalWorkflowState (TypedDict)
│   │   ├── nodes.py                      # Các nút thực thi: retrieve_node, consultant_node, judge_node
│   │   ├── edges.py                      # Cạnh điều kiện route_after_judge (accept / revise / fallback)
│   │   └── orchestrator.py               # Biên dịch StateGraph & hàm run_legal_multi_agent_workflow
│   │
│   └── evaluation/                       # [Tầng 4 — Đánh giá Thực nghiệm Định lượng]
│       ├── __init__.py
│       ├── metrics.py                    # Tính Recall@K, Faithfulness, Hallucination, Correction Rate
│       └── benchmark_runner.py           # Chạy 50 câu Benchmark (Multi-Agent vs Baseline) & xuất CSV/MD
│
├── ui/                                   # [Tầng 5 — Giao diện Web Đỏ — Trắng chia theo phân hệ]
│   ├── __init__.py
│   ├── styles.py                         # CSS chủ đạo tông Đỏ — Trắng
│   ├── components.py                     # Header, Nav, Banner, Sơ đồ 6 trường, Thẻ IRAC, Footer
│   └── views/
│       ├── __init__.py
│       ├── home_view.py                  # Trang chủ cổng thông tin
│       ├── qa_view.py                    # Phân hệ 1 (Hành chính) & Phân hệ 2 (Tố tụng hình sự)
│       ├── misconceptions_view.py        # Phân hệ 3 (Cảnh báo hiểu nhầm & Chấm Benchmark)
│       └── search_view.py                # Phân hệ 4 (Tra cứu văn bản tức thì Zero-LLM)
│
├── tests/                                # Kiểm thử tự động (Unit & Integration Tests)
│   ├── test_schemas.py
│   ├── test_retriever_bm25.py
│   ├── test_judge_two_layers.py
│   └── test_langgraph_loop.py
│
├── app_streamlit.py                      # Điểm khởi chạy (Entrypoint) gọn nhẹ gọi vào package ui/
├── requirements.txt
└── .env.example


<img width="917" height="712" alt="image" src="https://github.com/user-attachments/assets/b58f2bfc-7b14-47ac-a4bd-d657fb5ad278" />
