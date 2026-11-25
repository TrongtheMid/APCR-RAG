"""
Trích xuất toàn văn từ tệp PDF ký số chính thống (datafiles.chinhphu.vn / tệp tải lên)
bằng thư viện pypdf với cơ chế giới hạn dung lượng và dọn sạch khoảng trắng.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Optional

import httpx

from src.core.config import has_valid_api_key
from src.core.llm_factory import extract_response_text, get_candidate_models, get_chat_llm, get_llm_settings

DEFAULT_MAX_PDF_BYTES = 15 * 1024 * 1024  # 15 MB
DEFAULT_MAX_PAGES = 25
DEFAULT_MAX_OCR_BYTES = 8 * 1024 * 1024  # 8 MB cho Gemini Multimodal PDF


def _clean_extracted_text(text: str) -> str:
    """Chuẩn hóa khoảng trắng và dòng trống từ text trích xuất PDF."""
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf_bytes(
    pdf_bytes: bytes,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = 45000,
) -> str:
    """
    Trích xuất text tiếng Việt từ bytes của tệp PDF.
    Trả về chuỗi rỗng nếu bytes không hợp lệ, lỗi đọc hoặc là PDF dạng ảnh scan.
    """
    if not pdf_bytes or len(pdf_bytes) < 8:
        return ""

    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes), strict=False)
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:
                return ""

        pages_text = []
        total_chars = 0
        for idx, page in enumerate(reader.pages):
            if idx >= max_pages:
                break
            txt = page.extract_text() or ""
            txt = txt.strip()
            if txt:
                pages_text.append(txt)
                total_chars += len(txt)
                if total_chars >= max_chars:
                    break

        combined = "\n\n".join(pages_text)
        return _clean_extracted_text(combined)[:max_chars]
    except Exception:
        return ""


def extract_text_from_scanned_pdf_with_gemini(
    pdf_bytes: bytes,
    max_bytes: int = DEFAULT_MAX_OCR_BYTES,
    max_chars: int = 45000,
) -> str:
    """
    Nhận diện nội dung PDF dạng ảnh scan bằng Gemini Multimodal (`application/pdf`)
    khi `pypdf` không trích xuất được đủ text (< 200 ký tự). Trả về "" nếu lỗi hoặc offline.
    """
    if not pdf_bytes or len(pdf_bytes) < 8 or len(pdf_bytes) > max_bytes:
        return ""
    if b"%PDF" not in pdf_bytes[:1024]:
        return ""

    settings = get_llm_settings()
    if settings["provider"] in {"none", "offline", "ollama"}:
        return ""
    if not has_valid_api_key(settings["google_api_key"]):
        return ""

    try:
        from langchain_core.messages import HumanMessage
    except ImportError:
        return ""

    b64_pdf = base64.b64encode(pdf_bytes).decode("ascii")
    msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Hãy đọc và chép lại nguyên văn nội dung tiếng Việt trong tệp PDF văn bản pháp luật này "
                    "(giữ chính xác số hiệu, tên thủ tục, trình tự, thành phần hồ sơ, thời hạn và mức lệ phí). "
                    "Không bình luận thêm."
                ),
            },
            {
                "type": "media",
                "mime_type": "application/pdf",
                "data": b64_pdf,
            },
        ]
    )

    for candidate in get_candidate_models():
        llm = get_chat_llm(
            temperature=0.0,
            provider="gemini",
            model_name=candidate,
            api_key=settings["google_api_key"],
            max_retries=0,
            timeout=30,
        )
        if llm is None:
            continue
        try:
            resp = llm.invoke([msg])
            extracted = _clean_extracted_text(extract_response_text(resp))
            if extracted:
                return extracted[:max_chars]
        except Exception:
            continue

    return ""


def download_and_extract_pdf(
    url: str,
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
    max_pages: int = DEFAULT_MAX_PAGES,
    timeout: float = 20.0,
    client: Optional[httpx.Client] = None,
    ocr_fallback: bool = True,
) -> str:
    """Tải PDF từ URL chính thống và trích xuất toàn văn an toàn (tự động OCR Multimodal nếu là bản scan)."""
    if not url or not url.lower().startswith(("http://", "https://")):
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) APCR-RAG-Sync/1.0",
        "Accept": "application/pdf,application/octet-stream,*/*",
    }

    owns_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True, verify=False)
    try:
        resp = http.get(url, headers=headers)
        if resp.status_code != 200:
            return ""
        content = resp.content
        if not content or len(content) > max_bytes:
            return ""
        # Kiểm tra header %PDF để tránh cố parse trang HTML lỗi
        if b"%PDF" not in content[:1024]:
            return ""
        text = extract_text_from_pdf_bytes(content, max_pages=max_pages)
        if ocr_fallback and len(text.strip()) < 200:
            ocr_text = extract_text_from_scanned_pdf_with_gemini(content)
            if len(ocr_text) > len(text):
                text = ocr_text
        return text
    except Exception:
        return ""
    finally:
        if owns_client:
            http.close()
