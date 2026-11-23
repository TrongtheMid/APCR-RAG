# APCR-RAG — Administrative & Penal Criminal RAG

> **Hệ thống Multi-Agent RAG Tư vấn Thủ tục Hành chính Công và Tố tụng Hình sự**  
> **Học phần:** INT 3507 2 — Các vấn đề hiện đại của Công nghệ Thông tin  
> **Giảng viên hướng dẫn:** TS. Nghiêm Nguyễn Việt Dũng  
> **Nhóm 9:** Phạm Đức Trọng · Vũ Gia Anh Đức · Nguyễn Hữu Hải Long

---

## 1. Tổng quan & Giá trị cốt lõi

**APCR-RAG** giải quyết 4 điểm nghẽn kinh điển của Single-Agent RAG truyền thống trong bài toán pháp lý (ảo giác mức lệ phí/thời hạn, đứt gãy thành phần hồ sơ do fixed-size chunking, nhầm lẫn cấp thẩm quyền và thiếu cơ chế tự kiểm chứng) bằng:

1. **Lược đồ Hồ sơ 6 Trường JSON Chuẩn hóa (`ProcedureRecord`)** kết hợp **Siêu dữ liệu Thời gian (Temporal Metadata)** (`ĐANG HIỆU LỰC` / `HẾT HIỆU LỰC`, `ngay_hieu_luc`, `van_ban_thay_the`).
2. **Kiến trúc Đa tác tử (Multi-Agent RAG)** điều phối trên **LangGraph `StateGraph`** kèm chế độ **Streaming tiến trình từng Agent** và **Hội thoại đa lượt (`chat_history`)**:
   - **Agent 1 — Retriever Agent:** Viết lại câu hỏi, định tuyến miền (Hành chính vs Tố tụng Hình sự), lọc metadata thời gian và Hybrid Search (`Rank-BM25 + ChromaDB Vector Store`).
   - **Agent 2 — Consultant Agent:** Hàng rào từ chối chủ động (`Deterministic Refusal`), lập luận 4 bước chuẩn **IRAC** (*Issue – Rule – Application – Conclusion*), tự hiệu chỉnh (`Self-Revision Loop`) và trích xuất đồ thị 6 trường.
   - **Agent 3 — Judge / Critic Agent (2 Lớp nâng cấp):**
     - **Lớp 1 (Deterministic Code + Anti-Contradiction Regex):** Đối soát tuyệt đối mã thủ tục, điều luật, biểu mẫu, lệ phí, thời hạn trên cả `CitationMetadata` **lẫn bóc tách Regex trực tiếp từ phần văn xuôi IRAC** để chặn mâu thuẫn giữa metadata và nội dung tư vấn.
     - **Lớp 2 (General Claim-Level Semantic Critic):** Tách `dieu_kien_bat_buoc` thành các mệnh đề nguyên tử (atomic claims), đối soát độ phủ mệnh đề, cực tính mệnh đề cấm/miễn phí/hết hiệu lực, phân cấp thẩm quyền và giai đoạn tố tụng trên mọi thủ tục.
3. **4 Phân hệ Web (Streamlit — Giao diện Đỏ Trắng Cổng Dịch vụ công):**
   - Hỏi — Đáp Thủ tục Hành chính Công (IRAC + Đồ thị 6 trường + Streaming Agent).
   - Hỏi — Đáp Thủ tục Tố tụng Hình sự (Timeline 5 giai đoạn tố tụng).
   - Cảnh báo Hiểu nhầm Pháp lý & Dashboard Đánh giá Benchmark 50 câu + Ablation Study.
   - Tra cứu Văn bản & Thủ tục Tức thì (Zero-LLM BM25, độ trễ ~1 ms, 0 token, tô sáng từ khóa).

---

## 2. Sơ đồ Kiến trúc Multi-Agent

```mermaid
flowchart LR
    Q([Câu hỏi người dân + Lịch sử]) --> R[1. Retriever Agent<br/>Query Rewriting · Metadata Filter · Hybrid Search]
    R --> C[2. Consultant Agent<br/>Deterministic Refusal · Lập luận IRAC]
    C -->|Thiếu dữ kiện| CL([Hỏi ngược làm rõ])
    C --> J[3. Judge Agent 2 Lớp<br/>L1: Exact Match + Anti-Contradiction Regex<br/>L2: Claim-Level Semantic Critic]
    J -->|PASS| OUT([Trả kết quả IRAC + Đồ thị 6 trường])
    J -->|FAIL & vòng <= 2| C
    J -->|Quá 2 vòng| FB([Fallback Cảnh báo An toàn])
```

---

## 3. Cấu trúc Mã nguồn (Clean & Modular Architecture)

```text
├── app_streamlit.py              # Cổng thông tin Web Streamlit 4 phân hệ + Dashboard Benchmark
├── data/
│   ├── data_procedures.json      # 23 thủ tục chuẩn hóa 6 trường (21 đang hiệu lực, 2 hết hiệu lực)
│   └── benchmark_50_questions.json # 50 câu hỏi kiểm thử (25 Hành chính, 25 Hình sự; 3 độ khó)
├── src/
│   ├── core/                     # Cấu hình trung tâm (config.py) & LLM Factory (llm_factory.py)
│   ├── schemas/                  # Pydantic Schemas (ProcedureRecord, IRACOutput, JudgeVerificationResult)
│   ├── retriever/                # [Nguyễn Hữu Hải Long] Tokenizer, Query Rewriter, Metadata Filter, BM25, Vector Store
│   ├── consultant/               # [Vũ Gia Anh Đức] IRAC Prompts, Refusal Guard, Synthesizer, Graph Builder
│   ├── baseline/                 # [Vũ Gia Anh Đức] Fixed-size Chunker & Single-Agent Baseline đối chứng
│   ├── judge/                    # [Phạm Đức Trọng] Layer 1 Regex Cross-check, Layer 2 Claim-level Critic, Misconceptions
│   ├── workflow/                 # [Phạm Đức Trọng] LangGraph StateGraph, Conditional Edges, Streaming Orchestrator
│   └── evaluation/               # [Phạm Đức Trọng] Benchmark 50 câu, Ablation Study 4 cấu hình, Confusion Matrix, RAGAS
├── tests/                        # Bộ kiểm thử tự động (unittest / pytest) cho Judge, Workflow & Evaluation
├── requirements.txt
└── .env.example
```

---

## 4. Hướng dẫn Cài đặt & Chạy Hệ thống

### 4.1. Cài đặt môi trường
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2. Cấu hình biến môi trường (Tùy chọn LLM thật)
Sao chép `.env.example` thành `.env` và điền `GOOGLE_API_KEY` (Google AI Studio). Nếu để trống, hệ thống tự động vận hành ở chế độ **Deterministic Fallback Offline** mượt mà để demo không cần kết nối mạng.

### 4.3. Khởi chạy Giao diện Web
```bash
streamlit run app_streamlit.py
```
Truy cập trình duyệt tại `http://localhost:8501`.

### 4.4. Chạy Kiểm thử Tự động & Thực nghiệm Benchmark
```bash
# Chạy bộ kiểm thử tự động (Judge Lớp 1/2, Workflow Streaming, Self-Revision, Ablation)
python -m unittest discover -s tests -v

# Chạy toàn bộ Benchmark 50 câu + Ablation Study 4 cấu hình + Confusion Matrix
python -c "import json; from src.evaluation import run_full_benchmark_evaluation; print(json.dumps(run_full_benchmark_evaluation(), indent=2, ensure_ascii=False))"
```

---

## 5. Kết quả Thực nghiệm Định lượng (Ablation Study trên 50 câu Benchmark)

| Cấu hình hệ thống | Precision@1 | Recall@3 | Faithfulness (Phí + Hạn) | Hallucination Rate | Claim Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. Single-Agent Baseline (Fixed Chunking) | 22.0% | 22.0% | 35.0% | 65.0% | 22.8% |
| 2. BM25-Only (Không Rewriter, Không Judge) | 96.0% | 98.0% | 96.0% | 4.0% | 96.0% |
| 3. Hybrid + IRAC (Không có Judge 2 Lớp) | 100.0% | 100.0% | 70.0% | 30.0% | 63.0% |
| **4. Full APCR-RAG (Hybrid + IRAC + Judge 2 Lớp)** | **100.0%** | **100.0%** | **100.0%** | **0.0%** | **100.0%** |

- **Self-Revision Stress Test (50/50 câu gài lỗi vòng 1):** Tỷ lệ Judge bắt lỗi vòng 1 đạt **100% (50/50)**, tỷ lệ tự sửa thành công đạt **100% (50/50)**.
- **Domain Routing Confusion Matrix:** Định tuyến chính xác **25/25** Thủ tục hành chính và **25/25** Tố tụng hình sự (**Accuracy = 100%, F1 = 100%**).
