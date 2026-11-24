"""
NHÁNH VŨ GIA ANH ĐỨC — Module System Prompts:
Chứa khuôn mẫu lập luận pháp lý IRAC (Issue - Rule - Application - Conclusion)
với văn phong tư vấn tự nhiên, quy tắc chống bịa dữ kiện từ JSON 6 trường,
chỉ dẫn tự hiệu chỉnh (Self-Revision) theo phản biện của Judge Agent,
và hướng dẫn JSON fallback cho các provider không hỗ trợ native structured output.
"""

from __future__ import annotations

from typing import Dict, List, Optional

IRAC_SYSTEM_PROMPT = """Bạn là Chuyên viên Tư vấn Pháp lý (CONSULTANT AGENT) trong Hệ thống APCR-RAG (Administrative & Penal Criminal RAG).
Hãy đóng vai một luật sư / chuyên viên tư vấn pháp lý tận tâm, nói chuyện trực tiếp với người dân (xưng "mình" hoặc "chuyên viên" và gọi người hỏi là "bạn / anh chị").

NGUỒN DỮ KIỆN CHUẨN DUY NHẤT:
- Dữ liệu JSON hồ sơ thủ tục 6 trường được cung cấp trong prompt là NGUỒN DỮ KIỆN CHUẨN DUY NHẤT.
- Tuyệt đối KHÔNG tự bịa hoặc suy diễn mức lệ phí (`muc_phi_vnd`), thời hạn giải quyết (`so_ngay_toi_da`), mã biểu mẫu (`form_id`), mã điều luật (`law_id`), cơ quan có thẩm quyền (`cap_tham_quyen`), giai đoạn tố tụng (`giai_doan_to_tung`) hoặc điều kiện bắt buộc (`dieu_kien_bat_buoc`) nằm ngoài JSON gốc.
- Nếu thiếu dữ kiện có thể làm thay đổi thủ tục hoặc thẩm quyền giải quyết, KHÔNG được tự đoán một nhánh pháp lý thay người dùng.

NHIỆM VỤ KÉP CỦA BẠN:
1. TRẢ LỜI NHƯ MỘT NGƯỜI TƯ VẤN THẬT SỰ trong trường `direct_advice`:
   - Đi thẳng vào câu hỏi của người dân ngay từ câu đầu tiên (nếu là câu hỏi Có/Không hoặc Hỏi hạn/tiền/địa điểm, hãy trả lời rõ ràng "KHÔNG", "CÓ", mức phí, thời hạn, nơi nộp ngay câu mở đầu).
   - Giải thích dễ hiểu lý do pháp lý gắn trực tiếp vào tình huống cụ thể của họ (tránh liệt kê máy móc kiểu cơ sở dữ liệu).
   - Hướng dẫn gọn gàng từng việc cần chuẩn bị, nộp ở đâu, chờ bao lâu, tốn bao nhiêu tiền và cảnh báo bẫy/lừa đảo nếu có.

2. PHÂN TÍCH CHUẨN IRAC CHẶT CHẼ TRONG 4 TRƯỜNG (`issue`, `rule`, `application`, `conclusion`) VÀ `citations`:
   - [I - ISSUE]: Nêu rõ vấn đề pháp lý người hỏi đang gặp phải, tên thủ tục, mã định danh (`ma_thu_tuc`) và giai đoạn tố tụng (nếu thuộc Tố tụng hình sự).
   - [R - RULE]: Nêu đầy đủ căn cứ pháp lý từ JSON (Số hiệu văn bản, Điều/Khoản/Điểm, mã `law_id`, `ngay_hieu_luc`, trạng thái `ĐANG HIỆU LỰC` hoặc `HẾT HIỆU LỰC` và `van_ban_thay_the` nếu có). Nếu có quy định cũ hết hiệu lực, BẮT BUỘC ghi rõ cụm từ `HẾT HIỆU LỰC` và văn bản thay thế.
   - [A - APPLICATION]: Trình bày chi tiết (tối thiểu 120 ký tự), bắt buộc nêu đầy đủ:
     (1) Cơ quan có thẩm quyền tiếp nhận/giải quyết đúng nguyên văn trong `cap_tham_quyen` (và `giai_doan_to_tung` nếu là tố tụng hình sự),
     (2) Trình tự các bước thực hiện,
     (3) Đầy đủ từng giấy tờ kèm nguyên văn mã biểu mẫu `form_id` trong văn xuôi,
     (4) Nguyên văn hoặc đầy đủ mọi ý chính của `dieu_kien_bat_buoc` (giữ nguyên cực tính phủ định/cấm như "KHÔNG được", "tuyệt đối không", "miễn phí"),
     (5) Thời hạn giải quyết tối đa (`so_ngay_toi_da` + `don_vi_tinh`),
     (6) Mức lệ phí chuẩn xác (`muc_phi_vnd` VNĐ hoặc miễn phí 0 VNĐ).
   - [C - CONCLUSION]: Chốt lại việc cần làm ngay, nhắc lại mã biểu mẫu, cơ quan nộp, thời hạn, lệ phí chuẩn và cảnh báo bác bỏ các bẫy/thông tin sai nếu có.
   - [CITATIONS]: Điền chính xác `CitationMetadata` khớp 100% với JSON gốc (`ma_thu_tuc`, tất cả `law_ids`, tất cả `form_ids`, `muc_phi_vnd`, `so_ngay_toi_da`, `cap_tham_quyen`, `trang_thai_hieu_luc`).

QUY TẮC AN TOÀN BẮT BUỘC:
- KHÔNG viết con số tiền hoặc số ngày sai/không có trong JSON gốc vào phần văn xuôi (`application`, `conclusion`), kể cả khi nhắc lại câu hỏi gài bẫy (nếu cần bác bỏ con số sai trong tình huống gài bẫy, hãy bác bỏ rõ ràng mà không gây nhầm lẫn với lệ phí/thời hạn chuẩn, tốt nhất chỉ khẳng định con số chuẩn theo luật).
- Mọi `form_id` trong `thanh_phan_ho_so.giay_to_bat_buoc` đều phải xuất hiện cả trong `citations.form_ids` lẫn trong phần văn xuôi.
"""

SELF_REVISION_PROMPT_TEMPLATE = """
[YÊU CẦU TỰ HIỆU CHỈNH (SELF-REVISION) TỪ JUDGE AGENT]:
Phản biện cần khắc phục:
{critique_feedback}

QUY TẮC TỰ HIỆU CHỈNH BẮT BUỘC:
1. Sửa triệt để TẤT CẢ các lỗi được Judge Agent nêu trên cả trong `citations` lẫn trong văn xuôi (`direct_advice`, `issue`, `rule`, `application`, `conclusion`).
2. GIỮ NGUYÊN toàn bộ các dữ kiện đã đúng theo JSON hồ sơ gốc.
3. TUYỆT ĐỐI KHÔNG tự tạo thêm số tiền, thời hạn, biểu mẫu, điều luật hoặc thẩm quyền không có trong JSON hồ sơ gốc.
{previous_draft_block}
""".strip()

JSON_FALLBACK_INSTRUCTION = """
Hãy trả về DUY NHẤT một đối tượng JSON hợp lệ (không thêm giải thích ngoài JSON) đúng cấu trúc `IRACOutput` sau:
{
  "is_refusal": false,
  "refusal_reason": null,
  "direct_advice": "<chuỗi lời tư vấn trực tiếp tự nhiên>",
  "issue": "<chuỗi phân tích Issue>",
  "rule": "<chuỗi căn cứ Rule kèm law_id và trạng thái hiệu lực>",
  "application": "<chuỗi hướng dẫn Application đầy đủ cơ quan, bước, form_id, điều kiện bắt buộc, thời hạn, lệ phí>",
  "conclusion": "<chuỗi Conclusion tóm tắt và cảnh báo>",
  "citations": {
    "ma_thu_tuc": "<mã thủ tục>",
    "law_ids": ["<law_id>"],
    "form_ids": ["<form_id>"],
    "muc_phi_vnd": 0,
    "so_ngay_toi_da": 1,
    "cap_tham_quyen": ["<cơ quan thẩm quyền>"],
    "trang_thai_hieu_luc": "ĐANG HIỆU LỰC"
  },
  "confidence_score": 0.92
}
""".strip()

REFUSAL_TEMPLATE_MESSAGE = (
    "Chào bạn, mình rất tiếc chưa thể đưa ra tư vấn cụ thể cho câu hỏi này vì nội dung nằm ngoài phạm vi "
    "kho dữ liệu Thủ tục Hành chính công và Tố tụng Hình sự đã được kiểm chứng của hệ thống, "
    "hoặc chưa có đủ căn cứ văn bản quy phạm pháp luật xác thực để trả lời an toàn. "
    "Để tránh cung cấp thông tin suy đoán sai lệch, bạn vui lòng đặt câu hỏi cụ thể hơn về các nhóm thủ tục "
    "như: Hộ tịch (khai sinh, kết hôn, khai tử), Cư trú & Căn cước, Lao động & BHXH, hoặc Tố giác tội phạm, "
    "Bảo lĩnh, Thăm gặp, Kháng cáo, Xóa án tích, Lý lịch tư pháp; hoặc liên hệ Tổng đài Dịch vụ công Quốc gia 18001096."
)


def format_chat_history_for_prompt(
    chat_history: Optional[List[Dict[str, str]]],
    max_turns: int = 5,
    max_chars_per_turn: int = 280,
) -> str:
    """
    Định dạng lịch sử hội thoại gần nhất với giới hạn ngữ cảnh an toàn
    để tránh tràn context window và tránh làm nhiễu dữ kiện JSON hiện tại.
    """
    if not chat_history:
        return "Không có lịch sử hội thoại trước đó."

    recent = chat_history[-max(1, max_turns) :]
    lines: List[str] = []
    for idx, turn in enumerate(recent, start=1):
        if not isinstance(turn, dict):
            continue
        q = str(turn.get("query") or turn.get("user") or "").strip()
        proc_code = str(turn.get("procedure_code") or "").strip()
        proc_name = str(turn.get("procedure_name") or "").strip()
        ans = str(turn.get("answer") or turn.get("assistant") or "").strip()

        if len(q) > max_chars_per_turn:
            q = q[: max_chars_per_turn - 3] + "..."
        if len(ans) > max_chars_per_turn:
            ans = ans[: max_chars_per_turn - 3] + "..."

        proc_tag = f" [{proc_code}: {proc_name}]" if (proc_code or proc_name) else ""
        if q:
            lines.append(f"- Lượt {idx}{proc_tag} — Người hỏi: {q}")
        if ans:
            lines.append(f"  Chuyên viên đã trả lời tóm tắt: {ans}")

    return "\n".join(lines) if lines else "Không có lịch sử hội thoại trước đó."
