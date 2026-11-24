"""
Trích xuất toàn văn từ tệp PDF ký số chính thống (datafiles.chinhphu.vn / tệp tải lên)
bằng thư viện pypdf với cơ chế giới hạn dung lượng và dọn sạch khoảng trắng.
"""

from __future__ import annotations

import io
import re
from typing import Optional

import httpx

DEFAULT_MAX_PDF_BYTES = 15 * 1024 * 1024  # 15 MB
DEFAULT_MAX_PAGES = 25


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


def download_and_extract_pdf(
    url: str,
    max_bytes: int = DEFAULT_MAX_PDF_BYTES,
    max_pages: int = DEFAULT_MAX_PAGES,
    timeout: float = 20.0,
    client: Optional[httpx.Client] = None,
) -> str:
    """Tải PDF từ URL chính thống và trích xuất toàn văn an toàn."""
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
        return extract_text_from_pdf_bytes(content, max_pages=max_pages)
    except Exception:
        return ""
    finally:
        if owns_client:
            http.close()
