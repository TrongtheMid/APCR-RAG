"""
Client thu thập văn bản quy phạm pháp luật & Quyết định công bố TTHC chính thống từ:
1. Cổng TTĐT Chính phủ: https://vanban.chinhphu.vn + https://datafiles.chinhphu.vn (HTTP 200 công khai)
2. Công báo Chính phủ: https://congbao.chinhphu.vn (HTTP 200 công khai)

Lưu ý kỹ thuật & tuân thủ:
- Cổng Dịch vụ công Quốc gia (dichvucong.gov.vn) hiện áp dụng bảo vệ chống bot JS (/TSPD/)
  và các API TTHC chuẩn NDXP/LGSP đòi hỏi khóa xác thực cơ quan (AccessKey/SecretKey).
  Hệ thống KHÔNG vượt rào/bỏ qua cơ chế bảo vệ mà ghi nhận trạng thái rõ ràng qua
  `get_official_sources_status()`, đồng thời ưu tiên thu thập hợp lệ từ Cổng TTĐT Chính phủ.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from src.ingestion.models import OfficialSourceDocument
from src.ingestion.pdf_extractor import download_and_extract_pdf

CHINHPHU_BASE_URL = "https://vanban.chinhphu.vn"
CHINHPHU_SEARCH_URL = f"{CHINHPHU_BASE_URL}/he-thong-van-ban"
CONGBAO_BASE_URL = "https://congbao.chinhphu.vn"

_DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) APCR-RAG-Sync/1.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}


def get_official_sources_status() -> List[Dict[str, str]]:
    """Trả về trạng thái sẵn sàng của các kênh dữ liệu pháp luật chính thống."""
    dvcqg_configured = bool(
        os.getenv("DVCQG_ACCESS_KEY", "").strip() and os.getenv("DVCQG_SECRET_KEY", "").strip()
    )
    return [
        {
            "id": "vanban_chinhphu",
            "name": "Cổng Văn bản Chính phủ (vanban.chinhphu.vn)",
            "status": "Sẵn sàng (Công khai HTTP 200 + PDF ký số)",
            "mode": "auto_public",
        },
        {
            "id": "congbao_chinhphu",
            "name": "Công báo Chính phủ (congbao.chinhphu.vn)",
            "status": "Sẵn sàng (Công khai HTTP 200)",
            "mode": "auto_public",
        },
        {
            "id": "dichvucong_ndxp",
            "name": "Cổng Dịch vụ công Quốc gia (dichvucong.gov.vn / NDXP)",
            "status": (
                "Đã cấu hình khóa NDXP"
                if dvcqg_configured
                else "Giới hạn truy cập (Tường /TSPD/ công khai · API TTHC cần AccessKey/SecretKey NDXP)"
            ),
            "mode": "authenticated_api" if dvcqg_configured else "gated_requires_key",
        },
    ]


def parse_vn_date(raw: Optional[str]) -> Optional[str]:
    """Chuẩn hóa ngày DD-MM-YYYY hoặc DD/MM/YYYY sang YYYY-MM-DD."""
    if not raw:
        return None
    text = str(raw).strip()
    m_ymd = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m_ymd:
        y, m, d = int(m_ymd.group(1)), int(m_ymd.group(2)), int(m_ymd.group(3))
        return f"{y:04d}-{m:02d}-{d:02d}"

    m_dmy = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", text)
    if m_dmy:
        d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        return f"{y:04d}-{m:02d}-{d:02d}"
    return None


def _extract_doc_id(href: str) -> str:
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    if "docid" in qs and qs["docid"]:
        return qs["docid"][0]
    m = re.search(r"docid=(\d+)", href, re.IGNORECASE)
    return m.group(1) if m else ""


def parse_search_results_html(
    html: str,
    base_url: str = CHINHPHU_BASE_URL,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Phân tích HTML trang danh sách/tìm kiếm văn bản của vanban.chinhphu.vn.
    Trả về danh sách dict gồm doc_id, url, so_hieu, ngay_ban_hanh, trich_yeu, pdf_urls.
    """
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    results: List[Dict[str, Any]] = []
    seen_urls = set()

    for tr in soup.find_all("tr"):
        doc_link = tr.find("a", href=lambda h: h and "docid=" in h.lower())
        if doc_link is None:
            continue

        href = str(doc_link.get("href", "")).strip()
        full_url = urljoin(base_url, href)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)

        code_el = tr.select_one("span.code")
        date_el = tr.select_one("span.issued-date") or tr.select_one("span.issue-v2")
        subj_el = tr.select_one("span.substract")

        so_hieu = code_el.get_text(" ", strip=True) if code_el else doc_link.get_text(" ", strip=True)
        ngay_raw = date_el.get_text(" ", strip=True) if date_el else ""
        trich_yeu = subj_el.get_text(" ", strip=True) if subj_el else ""

        pdf_urls = [
            urljoin(base_url, str(a.get("href", "")).strip())
            for a in tr.select(".bl-doc-file a[href], a[href$='.pdf']")
            if a.get("href")
        ]

        results.append(
            {
                "source_name": "vanban.chinhphu.vn",
                "doc_id": _extract_doc_id(full_url),
                "url": full_url,
                "so_hieu": so_hieu.strip(),
                "ngay_ban_hanh": parse_vn_date(ngay_raw) or ngay_raw,
                "trich_yeu": trich_yeu.strip(),
                "pdf_urls": list(dict.fromkeys(pdf_urls)),
            }
        )
        if len(results) >= limit:
            break

    return results


def parse_detail_page_html(
    html: str,
    source_url: str,
    base_url: str = CHINHPHU_BASE_URL,
) -> OfficialSourceDocument:
    """Phân tích HTML trang chi tiết `/?pageid=27160&docid=...` trên vanban.chinhphu.vn."""
    soup = BeautifulSoup(html, "lxml")

    fields: Dict[str, str] = {}
    for tr in soup.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
        if len(cells) == 2 and cells[0] and len(cells[0]) <= 45:
            fields[cells[0].strip().lower()] = cells[1].strip()

    so_hieu = fields.get("số ký hiệu") or ""
    if not so_hieu and soup.title and soup.title.string:
        m_code = re.search(r"số\s+([^\s:]+)", soup.title.string, re.IGNORECASE)
        if m_code:
            so_hieu = m_code.group(1)

    ngay_bh = parse_vn_date(fields.get("ngày ban hành"))
    ngay_hl = parse_vn_date(fields.get("ngày có hiệu lực")) or ngay_bh
    loai_vb = fields.get("loại văn bản", "")
    co_quan = fields.get("cơ quan ban hành", "")
    trich_yeu = fields.get("trích yếu", "")

    if not trich_yeu and soup.title and soup.title.string:
        trich_yeu = soup.title.string.strip()

    pdf_urls: List[str] = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        lower = href.lower()
        if ("datafiles.chinhphu.vn" in lower or lower.endswith(".pdf")) and ".pdf" in lower:
            full_pdf = urljoin(base_url, href)
            if full_pdf not in pdf_urls:
                pdf_urls.append(full_pdf)

    raw_text_parts: List[str] = []
    for sel in [".doc-content", ".toanvan", ".detail-content"]:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text("\n", strip=True)
            if len(txt) > 80:
                raw_text_parts.append(txt)

    if not raw_text_parts and trich_yeu:
        raw_text_parts.append(f"{loai_vb} {so_hieu} ({co_quan}): {trich_yeu}".strip())

    trang_thai = "ĐANG HIỆU LỰC"
    status_raw = (fields.get("tình trạng hiệu lực") or "").lower()
    if "hết hiệu lực" in status_raw:
        trang_thai = "HẾT HIỆU LỰC"

    return OfficialSourceDocument(
        source_name="vanban.chinhphu.vn",
        doc_id=_extract_doc_id(source_url),
        source_url=source_url,
        so_hieu=so_hieu or _extract_doc_id(source_url) or "CHUA-RO-SO-HIEU",
        trich_yeu=trich_yeu,
        loai_van_ban=loai_vb,
        co_quan_ban_hanh=co_quan,
        ngay_ban_hanh=ngay_bh,
        ngay_hieu_luc=ngay_hl,
        trang_thai_hieu_luc=trang_thai,
        pdf_urls=pdf_urls,
        raw_text="\n\n".join(raw_text_parts).strip(),
    )


def parse_congbao_html(
    html: str,
    base_url: str = CONGBAO_BASE_URL,
    limit: int = 15,
) -> List[Dict[str, Any]]:
    """Phân tích danh mục văn bản mới công bố trên trang chủ congbao.chinhphu.vn."""
    if not html:
        return []

    soup = BeautifulSoup(html, "lxml")
    results: List[Dict[str, Any]] = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = str(a["href"]).strip()
        if "/van-ban/" not in href or not href.endswith(".htm"):
            continue
        full_url = urljoin(base_url, href)
        title = a.get_text(" ", strip=True)
        if not title or full_url in seen:
            continue
        seen.add(full_url)

        m_code = re.search(r"số\s+([A-Za-z0-9./\-Đđ]+)", title, re.IGNORECASE)
        so_hieu = m_code.group(1) if m_code else title[:40]
        m_id = re.search(r"-(\d+)\.htm$", href)
        results.append(
            {
                "source_name": "congbao.chinhphu.vn",
                "doc_id": m_id.group(1) if m_id else "",
                "url": full_url,
                "so_hieu": so_hieu,
                "ngay_ban_hanh": "",
                "trich_yeu": title,
                "pdf_urls": [],
            }
        )
        if len(results) >= limit:
            break

    return results


def search_chinhphu_documents(
    keyword: str = "thủ tục hành chính",
    limit: int = 15,
    timeout: float = 15.0,
    client: Optional[httpx.Client] = None,
) -> List[Dict[str, Any]]:
    """
    Tìm kiếm văn bản mới hoặc theo từ khóa trên Cổng TTĐT Chính phủ (`vanban.chinhphu.vn`).
    Nếu keyword rỗng, lấy trực tiếp danh sách văn bản mới ban hành.
    """
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True, verify=False)
    try:
        r0 = http.get(CHINHPHU_SEARCH_URL, headers=_DEFAULT_HEADERS)
        if r0.status_code != 200:
            return []

        kw_clean = (keyword or "").strip()
        if not kw_clean:
            return parse_search_results_html(r0.text, limit=limit)

        soup0 = BeautifulSoup(r0.text, "lxml")
        payload: Dict[str, str] = {}
        for inp in soup0.find_all("input"):
            name = inp.get("name")
            if not name:
                continue
            if "btn" in str(name).lower() and name != "ctrl_191017_163$btnSearch":
                continue
            payload[str(name)] = str(inp.get("value") or "")

        payload["ctrl_191017_163$txtSearchKeyword"] = kw_clean
        payload["ctrl_191017_163$hidIsSearch"] = "1"
        payload["ctrl_191017_163$btnSearch"] = "Tìm kiếm"
        payload["__EVENTTARGET"] = ""
        payload["__EVENTARGUMENT"] = ""

        headers = {**_DEFAULT_HEADERS, "Referer": CHINHPHU_SEARCH_URL}
        r1 = http.post(CHINHPHU_SEARCH_URL, data=payload, headers=headers)
        if r1.status_code != 200:
            return []
        return parse_search_results_html(r1.text, limit=limit)
    finally:
        if owns_client:
            http.close()


def fetch_chinhphu_detail(
    doc_url_or_id: str,
    download_pdf_text: bool = True,
    timeout: float = 20.0,
    client: Optional[httpx.Client] = None,
) -> OfficialSourceDocument:
    """
    Tải chi tiết văn bản từ URL vanban.chinhphu.vn (hoặc docid) và trích xuất text PDF đính kèm.
    Cũng hỗ trợ URL từ congbao.chinhphu.vn.
    """
    target = str(doc_url_or_id).strip()
    if target.isdigit():
        target = f"{CHINHPHU_BASE_URL}/?pageid=27160&docid={target}"
    elif target.startswith("/"):
        target = urljoin(CHINHPHU_BASE_URL, target)

    owns_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True, verify=False)
    try:
        resp = http.get(target, headers=_DEFAULT_HEADERS)
        resp.raise_for_status()

        if "congbao.chinhphu.vn" in target:
            soup = BeautifulSoup(resp.text, "lxml")
            title = soup.title.string.strip() if soup.title and soup.title.string else target
            m_code = re.search(r"số\s+([A-Za-z0-9./\-Đđ]+)", title, re.IGNORECASE)
            pdfs = [
                urljoin(CONGBAO_BASE_URL, str(a["href"]))
                for a in soup.find_all("a", href=True)
                if ".pdf" in str(a["href"]).lower()
            ]
            doc = OfficialSourceDocument(
                source_name="congbao.chinhphu.vn",
                doc_id="",
                source_url=target,
                so_hieu=m_code.group(1) if m_code else "CONG-BAO",
                trich_yeu=title,
                pdf_urls=list(dict.fromkeys(pdfs)),
                raw_text=title,
            )
        else:
            doc = parse_detail_page_html(resp.text, source_url=target)

        if download_pdf_text and doc.pdf_urls:
            extracted_chunks: List[str] = []
            for pdf_url in doc.pdf_urls[:2]:
                pdf_txt = download_and_extract_pdf(pdf_url, client=http)
                if pdf_txt:
                    extracted_chunks.append(pdf_txt)
            if extracted_chunks:
                header = f"Số hiệu: {doc.so_hieu}\nCơ quan ban hành: {doc.co_quan_ban_hanh}\nTrích yếu: {doc.trich_yeu}"
                doc.raw_text = f"{header}\n\n" + "\n\n---\n\n".join(extracted_chunks)

        return doc
    finally:
        if owns_client:
            http.close()


def fetch_congbao_latest(
    limit: int = 15,
    timeout: float = 15.0,
    client: Optional[httpx.Client] = None,
) -> List[Dict[str, Any]]:
    """Lấy danh sách văn bản mới đăng Công báo Chính phủ (congbao.chinhphu.vn)."""
    owns_client = client is None
    http = client or httpx.Client(timeout=timeout, follow_redirects=True, verify=False)
    try:
        resp = http.get(CONGBAO_BASE_URL, headers=_DEFAULT_HEADERS)
        if resp.status_code != 200:
            return []
        return parse_congbao_html(resp.text, limit=limit)
    finally:
        if owns_client:
            http.close()
