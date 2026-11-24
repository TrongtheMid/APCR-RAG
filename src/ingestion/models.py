"""
Mô hình dữ liệu Pydantic cho quy trình thu thập, bóc tách và kiểm duyệt (Staging)
từ các nguồn dữ liệu pháp luật chính thống (vanban.chinhphu.vn, congbao.chinhphu.vn, PDF/URL).
"""

from __future__ import annotations

import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from src.schemas.procedure_schema import ProcedureRecord


class OfficialSourceDocument(BaseModel):
    """Tài liệu pháp lý / Quyết định công bố TTHC thu thập từ nguồn chính thống."""

    source_name: str = Field(..., description="Tên nguồn (VD: vanban.chinhphu.vn, congbao.chinhphu.vn, pdf_upload)")
    doc_id: str = Field(default="", description="Mã văn bản tại nguồn gốc")
    source_url: str = Field(default="", description="Đường dẫn gốc trên cổng chính thống")
    so_hieu: str = Field(..., description="Số ký hiệu văn bản (VD: 14/2026/TT-BTP)")
    trich_yeu: str = Field(default="", description="Trích yếu / tên đầy đủ văn bản")
    loai_van_ban: str = Field(default="", description="Loại văn bản (Luật, Nghị định, Thông tư, Quyết định...)")
    co_quan_ban_hanh: str = Field(default="", description="Cơ quan ban hành")
    ngay_ban_hanh: Optional[str] = Field(default=None, description="Ngày ban hành chuẩn YYYY-MM-DD")
    ngay_hieu_luc: Optional[str] = Field(default=None, description="Ngày có hiệu lực chuẩn YYYY-MM-DD")
    trang_thai_hieu_luc: Literal["ĐANG HIỆU LỰC", "HẾT HIỆU LỰC"] = Field(
        default="ĐANG HIỆU LỰC",
        description="Trạng thái hiệu lực xác định từ metadata/trích yếu",
    )
    pdf_urls: List[str] = Field(default_factory=list, description="Danh sách URL tệp PDF ký số đính kèm")
    raw_text: str = Field(default="", description="Văn bản toàn văn trích xuất từ PDF hoặc HTML")
    fetched_at: str = Field(
        default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds"),
        description="Thời điểm thu thập (ISO-8601)",
    )


class ProcedureExtractionBatch(BaseModel):
    """Schema đầu ra có cấu trúc cho LLM bóc tách từ 1 văn bản sang danh sách ProcedureRecord."""

    procedures: List[ProcedureRecord] = Field(
        default_factory=list,
        description="Danh sách thủ tục chuẩn 6 trường trích xuất được từ văn bản nguồn",
    )
    replaced_law_codes: List[str] = Field(
        default_factory=list,
        description="Số hiệu văn bản cũ bị văn bản mới thay thế hoặc bãi bỏ (nếu có trong văn bản)",
    )
    extraction_notes: str = Field(
        default="",
        description="Ghi chú độ tin cậy và các trường cần kiểm duyệt kỹ (nếu có)",
    )


class ExtractedProcedureCandidate(BaseModel):
    """Bản ghi nằm trong hàng đợi kiểm duyệt (Staging) trước khi gộp vào data_procedures.json."""

    candidate_id: str = Field(..., description="Mã duy nhất của mục chờ duyệt")
    action: Literal["ADD", "UPDATE_EXISTING", "UPDATE_EXPIRY"] = Field(
        ..., description="Hành động đề xuất: Thêm mới / Cập nhật thủ tục / Đánh dấu hết hiệu lực"
    )
    target_existing_code: Optional[str] = Field(
        default=None,
        description="Mã thủ tục hiện có trong kho nếu là UPDATE_EXISTING hoặc UPDATE_EXPIRY",
    )
    record: ProcedureRecord = Field(..., description="Dữ liệu 6 trường chuẩn hóa đã qua kiểm định Pydantic")
    source_name: str = Field(default="vanban.chinhphu.vn")
    source_url: str = Field(default="")
    source_so_hieu: str = Field(default="")
    extraction_mode: Literal["llm_structured", "deterministic_fallback", "expiry_detector"] = Field(
        default="llm_structured"
    )
    confidence_note: str = Field(default="")
    status: Literal["PENDING", "APPROVED", "REJECTED"] = Field(default="PENDING")
    created_at: str = Field(
        default_factory=lambda: datetime.datetime.now().isoformat(timespec="seconds")
    )


class StagingQueue(BaseModel):
    """Tệp lưu trạng thái hàng đợi kiểm duyệt & nhật ký đồng bộ."""

    last_sync_at: Optional[str] = Field(default=None)
    last_source_summary: str = Field(default="")
    candidates: List[ExtractedProcedureCandidate] = Field(default_factory=list)
