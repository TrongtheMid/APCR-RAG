"""
Chuẩn hóa văn bản pháp luật / Quyết định công bố TTHC chính thống thành
Lược đồ 6 trường `ProcedureRecord` và phát hiện văn bản mới thay thế / bãi bỏ quy định cũ.
"""

from __future__ import annotations

import copy
import datetime
import json
import re
import unicodedata
import uuid
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from src.consultant.irac_synthesizer import _extract_json_object
from src.core.config import has_valid_api_key
from src.core.llm_factory import (
    extract_response_text,
    get_candidate_models,
    get_chat_llm,
    get_llm_settings,
)
from src.ingestion.models import (
    ExtractedProcedureCandidate,
    OfficialSourceDocument,
    ProcedureExtractionBatch,
)
from src.schemas.procedure_schema import ProcedureRecord

_EXTRACTION_SYSTEM_PROMPT = """Bạn là Chuyên gia Chuẩn hóa Dữ liệu Pháp luật Việt Nam cho hệ thống APCR-RAG.
Nhiệm vụ: từ văn bản pháp luật chính thống được cung cấp, hãy bóc tách các thủ tục hành chính hoặc thủ tục tố tụng hình sự cụ thể sang cấu trúc JSON chuẩn 6 trường (`ProcedureExtractionBatch`).

QUY TẮC BẮT BUỘC (KHÔNG ĐƯỢC VI PHẠM):
1. CHỈ trích xuất các con số lệ phí (`muc_phi_vnd` nguyên >= 0), thời hạn (`so_ngay_toi_da` nguyên >= 0), tên giấy tờ, cơ quan có thẩm quyền CÓ THẬT trong văn bản. Tuyệt đối KHÔNG tự bịa số liệu hoặc tên biểu mẫu không có trong tài liệu.
2. Nếu văn bản mang tính hướng dẫn chung mà không quy định mức phí/thời hạn cụ thể cho 1 thủ tục rõ ràng: đặt `muc_phi_vnd = 0` / `so_ngay_toi_da = 0` và giải thích rõ trong `mo_ta` / `chi_tiet`, đồng thời ghi chú trong `extraction_notes`.
3. `phan_nhom` CHỈ nhận đúng 1 trong 2 giá trị: "Thủ tục hành chính" hoặc "Tố tụng hình sự".
4. `can_cu_phap_ly[*].trang_thai_hieu_luc` CHỈ nhận "ĐANG HIỆU LỰC" hoặc "HẾT HIỆU LỰC"; `ngay_hieu_luc` định dạng `YYYY-MM-DD`.
5. Mỗi thủ tục cần `ma_thu_tuc` ngắn gọn, in hoa, không dấu cách (VD: TTHC-DT-2026-01 hoặc TTHS-AUTO-01).
6. Nếu văn bản có điều khoản thay thế, sửa đổi hoặc bãi bỏ văn bản cũ, liệt kê số hiệu văn bản cũ vào `replaced_law_codes`.
7. Tối đa {max_procedures} thủ tục tiêu biểu, rõ ràng nhất từ văn bản."""

_LAW_CODE_PATTERN = re.compile(
    r"\b\d{1,4}/\d{4}/[A-ZĐ0-9\-]+\b|\b\d{1,4}/QĐ-[A-ZĐ0-9]+\b",
    flags=re.IGNORECASE,
)


def _slugify_ascii(text: str, max_len: int = 24) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text or ""))
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c)).replace("đ", "d").replace("Đ", "D")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", no_marks).strip("-").upper()
    return cleaned[:max_len] or "AUTO"


def _infer_domain_and_field(text: str) -> tuple[str, str]:
    low = text.lower()
    criminal_kw = ("tố tụng hình sự", "bộ luật tố tụng hình sự", "bị can", "bị cáo", "khởi tố", "xóa án tích", "tạm giam")
    if any(k in low for k in criminal_kw):
        return "Tố tụng hình sự", "Tố tụng hình sự"
    if "hộ tịch" in low or "khai sinh" in low or "kết hôn" in low:
        return "Thủ tục hành chính", "Hộ tịch"
    if "cư trú" in low or "căn cước" in low:
        return "Thủ tục hành chính", "Cư trú & Căn cước"
    if "giấy phép lái xe" in low or "giao thông" in low:
        return "Thủ tục hành chính", "Giao thông vận tải"
    if "đất đai" in low or "quyền sử dụng đất" in low:
        return "Thủ tục hành chính", "Đất đai & Tài nguyên"
    return "Thủ tục hành chính", "Hành chính công"


def _extract_deadline_days(text: str) -> tuple[int, str, str]:
    m = re.search(
        r"(?:thời hạn(?:\s*giải quyết)?|trong\s*thời\s*hạn|tối\s*đa)\D{0,25}(\d{1,3})\s*(ngày\s*làm\s*việc|ngày)",
        text,
        re.IGNORECASE,
    )
    if m:
        days = int(m.group(1))
        unit = m.group(2).strip().lower()
        return days, unit, f"Theo quy định trong văn bản nguồn: {days} {unit}."
    return 0, "ngày làm việc", "Văn bản nguồn không nêu số ngày cố định; cần rà soát phụ lục hoặc điều khoản dẫn chiếu."


def _extract_fee_vnd(text: str) -> tuple[int, str]:
    if re.search(r"miễn\s*(?:phí|lệ phí)|không\s*thu\s*(?:phí|lệ phí)", text, re.IGNORECASE):
        return 0, "Miễn phí / không thu lệ phí theo nội dung văn bản nguồn."
    m = re.search(
        r"(?:lệ\s*phí|mức\s*thu|phí\s*thẩm\s*định)[^.\n]{0,40}?([\d]{1,3}(?:[.,]\d{3})+|\d{4,8})\s*(?:đồng|vnđ)",
        text,
        re.IGNORECASE,
    )
    if m:
        raw_num = re.sub(r"[.,]", "", m.group(1))
        try:
            fee = int(raw_num)
            return fee, f"Mức thu trích xuất từ văn bản nguồn: {fee:,} VNĐ.".replace(",", ".")
        except ValueError:
            pass
    return 0, "Chưa thấy mức phí bằng số cố định trong đoạn văn bản trích xuất; vui lòng kiểm duyệt kỹ."


def _extract_steps_from_text(doc: OfficialSourceDocument) -> List[str]:
    text = doc.raw_text or doc.trich_yeu
    steps: List[str] = []
    for m in re.finditer(r"(?:Bước\s*\d+|Điều\s*\d+)[^\n]{5,240}", text, re.IGNORECASE):
        line = re.sub(r"\s+", " ", m.group(0)).strip()
        if line and line not in steps:
            steps.append(line)
        if len(steps) >= 5:
            break
    if not steps:
        summary = (doc.trich_yeu or doc.so_hieu).strip()
        steps = [
            f"Bước 1: Chuẩn bị hồ sơ và biểu mẫu điện tử/giấy theo quy định tại {doc.so_hieu} ({summary}).",
            f"Bước 2: Nộp hồ sơ tại cơ quan có thẩm quyền ({doc.co_quan_ban_hanh or 'cơ quan tiếp nhận theo quy định'}) hoặc qua Cổng Dịch vụ công.",
            "Bước 3: Cơ quan có thẩm quyền kiểm tra, thụ lý và trả kết quả theo đúng thời hạn luật định.",
        ]
    return steps


def _extract_forms_from_text(doc: OfficialSourceDocument) -> List[Dict[str, str]]:
    text = doc.raw_text or ""
    forms: List[Dict[str, str]] = []
    for idx, m in enumerate(
        re.finditer(
            r"(?:Tờ khai|Đơn đề nghị|Văn bản đề nghị|Giấy chứng nhận|Bản sao|Hồ sơ)[^\n.;]{4,120}",
            text,
            re.IGNORECASE,
        ),
        start=1,
    ):
        item = re.sub(r"\s+", " ", m.group(0)).strip(" .;-:")
        if len(item) >= 8 and not any(item == f["ten_giay_to"] for f in forms):
            forms.append({"form_id": f"BM-{_slugify_ascii(doc.so_hieu, 10)}-{idx:02d}", "ten_giay_to": item})
        if len(forms) >= 4:
            break

    if not forms:
        forms.append(
            {
                "form_id": f"HS-{_slugify_ascii(doc.so_hieu, 12)}",
                "ten_giay_to": f"Hồ sơ đề nghị thực hiện thủ tục theo quy định tại {doc.so_hieu}",
            }
        )
    return forms


def detect_expiry_impacts(
    doc: OfficialSourceDocument,
    existing_procedures: List[Dict[str, Any]],
    llm_replaced_codes: Optional[List[str]] = None,
) -> List[ExtractedProcedureCandidate]:
    """
    Phát hiện các thủ tục hiện có trong data_procedures.json có căn cứ pháp lý bị văn bản mới
    thay thế, bãi bỏ hoặc tuyên bố hết hiệu lực.
    """
    if not existing_procedures:
        return []

    text = f"{doc.trich_yeu}\n{doc.raw_text or ''}"
    found_old_codes = {c.strip().lower() for c in (llm_replaced_codes or []) if c and c.strip()}

    for m in re.finditer(r"(?:thay thế|bãi bỏ|hết hiệu lực)[^.\n;]{0,180}", text, flags=re.IGNORECASE):
        snippet = m.group(0)
        for code_match in _LAW_CODE_PATTERN.findall(snippet):
            found_old_codes.add(code_match.strip().lower())

    doc_code_low = (doc.so_hieu or "").strip().lower()
    if doc_code_low in found_old_codes:
        found_old_codes.discard(doc_code_low)

    if not found_old_codes:
        return []

    candidates: List[ExtractedProcedureCandidate] = []
    for proc in existing_procedures:
        ma_tt = proc["ma_va_ten_thu_tuc"]["ma_thu_tuc"]
        laws = proc.get("can_cu_phap_ly") or []
        matched_indices: List[int] = []
        matched_law_names: List[str] = []

        for idx, law in enumerate(laws):
            if law.get("trang_thai_hieu_luc") == "HẾT HIỆU LỰC":
                continue
            so_hieu_law = str(law.get("so_hieu_van_ban") or "").lower()
            for old_code in found_old_codes:
                if old_code in so_hieu_law:
                    matched_indices.append(idx)
                    matched_law_names.append(law["so_hieu_van_ban"])
                    break

        if not matched_indices:
            continue

        updated_proc = copy.deepcopy(proc)
        for idx in matched_indices:
            updated_proc["can_cu_phap_ly"][idx]["trang_thai_hieu_luc"] = "HẾT HIỆU LỰC"
            updated_proc["can_cu_phap_ly"][idx]["van_ban_thay_the"] = doc.so_hieu

        validated = ProcedureRecord.model_validate(updated_proc)
        candidates.append(
            ExtractedProcedureCandidate(
                candidate_id=f"exp_{ma_tt}_{uuid.uuid4().hex[:6]}",
                action="UPDATE_EXPIRY",
                target_existing_code=ma_tt,
                record=validated,
                source_name=doc.source_name,
                source_url=doc.source_url,
                source_so_hieu=doc.so_hieu,
                extraction_mode="expiry_detector",
                confidence_note=(
                    f"Phát hiện văn bản mới {doc.so_hieu} đề cập thay thế/bãi bỏ văn bản cũ liên quan tới "
                    f"{', '.join(dict.fromkeys(matched_law_names))}."
                ),
            )
        )

    return candidates


def _ground_record_with_official_metadata(
    record: ProcedureRecord,
    doc: OfficialSourceDocument,
    existing_codes: set[str],
) -> tuple[ProcedureRecord, str, Optional[str]]:
    """Đảm bảo ngày hiệu lực, số hiệu văn bản và mã thủ tục khớp metadata chính thống."""
    raw = record.model_dump()
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    fallback_date = doc.ngay_hieu_luc or doc.ngay_ban_hanh or today_str

    for law in raw.get("can_cu_phap_ly") or []:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", str(law.get("ngay_hieu_luc") or "")):
            law["ngay_hieu_luc"] = fallback_date
        if doc.so_hieu and doc.so_hieu.lower() not in str(law.get("so_hieu_van_ban") or "").lower():
            if not str(law.get("so_hieu_van_ban") or "").strip():
                law["so_hieu_van_ban"] = f"{doc.loai_van_ban} số {doc.so_hieu}".strip()

    ma_tt = str(raw["ma_va_ten_thu_tuc"].get("ma_thu_tuc") or "").strip().upper()
    if not ma_tt:
        ma_tt = f"TTHC-AUTO-{_slugify_ascii(doc.so_hieu, 12)}"
    raw["ma_va_ten_thu_tuc"]["ma_thu_tuc"] = ma_tt

    action = "UPDATE_EXISTING" if ma_tt in existing_codes else "ADD"
    target_code = ma_tt if action == "UPDATE_EXISTING" else None
    validated = ProcedureRecord.model_validate(raw)
    return validated, action, target_code


def extract_procedures_deterministic(
    doc: OfficialSourceDocument,
    existing_codes: Optional[set[str]] = None,
) -> ExtractedProcedureCandidate:
    """Bộ dự phòng tất định trích xuất 6 trường trực tiếp bằng regex + metadata chính thống."""
    existing = existing_codes or set()
    full_text = f"{doc.trich_yeu}\n{doc.raw_text or ''}".strip()
    phan_nhom, linh_vuc = _infer_domain_and_field(full_text)
    days, unit, chi_tiet = _extract_deadline_days(full_text)
    fee, fee_desc = _extract_fee_vnd(full_text)
    steps = _extract_steps_from_text(doc)
    forms = _extract_forms_from_text(doc)

    ngay_hl = doc.ngay_hieu_luc or doc.ngay_ban_hanh or datetime.date.today().strftime("%Y-%m-%d")
    prefix = "TTHS" if phan_nhom == "Tố tụng hình sự" else "TTHC"
    ma_tt = f"{prefix}-SYNC-{_slugify_ascii(doc.so_hieu, 14)}"
    ten_tt = (doc.trich_yeu or f"Thủ tục theo văn bản {doc.so_hieu}").strip()
    if len(ten_tt) > 180:
        ten_tt = ten_tt[:177].rstrip() + "..."

    record_dict = {
        "ma_va_ten_thu_tuc": {
            "ma_thu_tuc": ma_tt,
            "ten_thu_tuc": ten_tt,
            "phan_nhom": phan_nhom,
            "linh_vuc": linh_vuc,
            "cap_tham_quyen": [doc.co_quan_ban_hanh] if doc.co_quan_ban_hanh else ["Cơ quan có thẩm quyền theo quy định"],
            "giai_doan_to_tung": ["Khởi tố - Giải quyết nguồn tin về tội phạm"] if prefix == "TTHS" else [],
        },
        "trinh_tu_thuc_hien": steps,
        "thanh_phan_ho_so": {
            "giay_to_bat_buoc": forms,
            "dieu_kien_bat_buoc": (
                f"Thực hiện theo quy định tại {doc.loai_van_ban or 'văn bản'} số {doc.so_hieu}: {doc.trich_yeu}."
            ),
        },
        "thoi_han_giai_quyet": {
            "so_ngay_toi_da": days,
            "don_vi_tinh": unit,
            "chi_tiet": chi_tiet,
        },
        "le_phi": {
            "muc_phi_vnd": fee,
            "mo_ta": fee_desc,
        },
        "can_cu_phap_ly": [
            {
                "law_id": f"{_slugify_ascii(doc.so_hieu, 16).lower()}",
                "so_hieu_van_ban": f"{doc.loai_van_ban} số {doc.so_hieu}".strip() if doc.loai_van_ban else doc.so_hieu,
                "dieu_khoan_diem": f"Toàn văn / hiệu lực từ {ngay_hl}",
                "ngay_hieu_luc": ngay_hl,
                "trang_thai_hieu_luc": doc.trang_thai_hieu_luc,
                "van_ban_thay_the": None,
            }
        ],
    }
    validated = ProcedureRecord.model_validate(record_dict)
    action = "UPDATE_EXISTING" if ma_tt in existing else "ADD"
    return ExtractedProcedureCandidate(
        candidate_id=f"cand_{uuid.uuid4().hex[:8]}",
        action=action,
        target_existing_code=ma_tt if action == "UPDATE_EXISTING" else None,
        record=validated,
        source_name=doc.source_name,
        source_url=doc.source_url,
        source_so_hieu=doc.so_hieu,
        extraction_mode="deterministic_fallback",
        confidence_note=(
            "Bóc tách tự động bằng Regex + metadata chính thống (chế độ ngoại tuyến/dự phòng). "
            "Nên đối soát lại thời hạn, lệ phí và thành phần hồ sơ trước khi duyệt."
        ),
    )


_JSON_FALLBACK_SCHEMA_HINT = """
Hãy trả về DUY NHẤT 1 đối tượng JSON (không giải thích thêm) đúng cấu trúc:
{
  "procedures": [
    {
      "ma_va_ten_thu_tuc": {
        "ma_thu_tuc": "TTHC-...",
        "ten_thu_tuc": "...",
        "phan_nhom": "Thủ tục hành chính",
        "linh_vuc": "...",
        "cap_tham_quyen": ["..."],
        "giai_doan_to_tung": []
      },
      "trinh_tu_thuc_hien": ["Bước 1: ...", "Bước 2: ..."],
      "thanh_phan_ho_so": {
        "giay_to_bat_buoc": [{"form_id": "BM-01", "ten_giay_to": "..."}],
        "dieu_kien_bat_buoc": "..."
      },
      "thoi_han_giai_quyet": {"so_ngay_toi_da": 0, "don_vi_tinh": "ngày làm việc", "chi_tiet": "..."},
      "le_phi": {"muc_phi_vnd": 0, "mo_ta": "..."},
      "can_cu_phap_ly": [
        {
          "law_id": "vb-01",
          "so_hieu_van_ban": "...",
          "dieu_khoan_diem": "...",
          "ngay_hieu_luc": "YYYY-MM-DD",
          "trang_thai_hieu_luc": "ĐANG HIỆU LỰC",
          "van_ban_thay_the": null
        }
      ]
    }
  ],
  "replaced_law_codes": [],
  "extraction_notes": ""
}
"""


def _sanitize_batch_dict(parsed_obj: Dict[str, Any], doc: OfficialSourceDocument) -> Dict[str, Any]:
    """Chuẩn hóa nhẹ các trường tối thiểu nếu LLM JSON fallback bỏ sót list mặc định."""
    fallback_date = doc.ngay_hieu_luc or doc.ngay_ban_hanh or datetime.date.today().strftime("%Y-%m-%d")
    procs = parsed_obj.get("procedures")
    if isinstance(procs, list):
        for p in procs:
            if not isinstance(p, dict):
                continue
            if not p.get("trinh_tu_thuc_hien"):
                p["trinh_tu_thuc_hien"] = [f"Bước 1: Thực hiện theo quy định tại {doc.so_hieu}."]
            if not p.get("can_cu_phap_ly"):
                p["can_cu_phap_ly"] = [
                    {
                        "law_id": _slugify_ascii(doc.so_hieu, 16).lower(),
                        "so_hieu_van_ban": doc.so_hieu,
                        "dieu_khoan_diem": "Quy định liên quan",
                        "ngay_hieu_luc": fallback_date,
                        "trang_thai_hieu_luc": doc.trang_thai_hieu_luc,
                        "van_ban_thay_the": None,
                    }
                ]
    return parsed_obj


def _try_llm_single_instance(llm: Any, prompt: str, doc: OfficialSourceDocument) -> Optional[ProcedureExtractionBatch]:
    if hasattr(llm, "with_structured_output"):
        try:
            structured = llm.with_structured_output(ProcedureExtractionBatch)
            res = structured.invoke(prompt)
            if isinstance(res, ProcedureExtractionBatch):
                return res
            if isinstance(res, dict):
                return ProcedureExtractionBatch.model_validate(_sanitize_batch_dict(res, doc))
        except Exception:
            pass

    try:
        raw_resp = llm.invoke(prompt + _JSON_FALLBACK_SCHEMA_HINT)
        parsed_obj = _extract_json_object(extract_response_text(raw_resp))
        if parsed_obj:
            return ProcedureExtractionBatch.model_validate(_sanitize_batch_dict(parsed_obj, doc))
    except (ValidationError, Exception):
        return None
    return None


def _invoke_llm_extraction(
    doc: OfficialSourceDocument,
    max_procedures: int = 4,
) -> Optional[ProcedureExtractionBatch]:
    settings = get_llm_settings()
    if settings["provider"] in {"none", "offline"}:
        return None

    snippet = (doc.raw_text or doc.trich_yeu or "")[:18000]
    prompt = (
        f"{_EXTRACTION_SYSTEM_PROMPT.format(max_procedures=max_procedures)}\n\n"
        f"=== METADATA CHÍNH THỐNG ===\n"
        f"- Nguồn: {doc.source_name}\n"
        f"- Số ký hiệu: {doc.so_hieu}\n"
        f"- Loại văn bản: {doc.loai_van_ban}\n"
        f"- Cơ quan ban hành: {doc.co_quan_ban_hanh}\n"
        f"- Ngày ban hành: {doc.ngay_ban_hanh or 'Không rõ'}\n"
        f"- Ngày hiệu lực: {doc.ngay_hieu_luc or 'Không rõ'}\n"
        f"- Trích yếu: {doc.trich_yeu}\n\n"
        f"=== NỘI DUNG VĂN BẢN ===\n{snippet}\n"
    )

    if settings["provider"] == "ollama":
        llm = get_chat_llm(temperature=0.0, provider="ollama", timeout=25)
        return _try_llm_single_instance(llm, prompt, doc) if llm is not None else None

    if has_valid_api_key(settings["google_api_key"]):
        for candidate_model in get_candidate_models():
            llm = get_chat_llm(
                temperature=0.0,
                provider="gemini",
                model_name=candidate_model,
                api_key=settings["google_api_key"],
                max_retries=0,
                timeout=25,
            )
            if llm is None:
                continue
            batch = _try_llm_single_instance(llm, prompt, doc)
            if batch is not None:
                return batch

    if settings["ollama_enabled"]:
        llm = get_chat_llm(temperature=0.0, provider="ollama", timeout=25)
        if llm is not None:
            return _try_llm_single_instance(llm, prompt, doc)

    return None


def extract_procedures_from_document(
    doc: OfficialSourceDocument,
    existing_procedures: Optional[List[Dict[str, Any]]] = None,
    use_llm: bool = True,
    max_procedures: int = 4,
) -> List[ExtractedProcedureCandidate]:
    """
    Chuyển đổi 1 OfficialSourceDocument thành danh sách ExtractedProcedureCandidate
    (bao gồm cả thủ tục mới/cập nhật VÀ cảnh báo hết hiệu lực cho kho hiện tại).
    """
    existing_list = existing_procedures or []
    existing_codes = {
        str((p.get("ma_va_ten_thu_tuc") or {}).get("ma_thu_tuc") or "").upper()
        for p in existing_list
    }

    candidates: List[ExtractedProcedureCandidate] = []
    llm_replaced_codes: List[str] = []

    if use_llm:
        batch = _invoke_llm_extraction(doc, max_procedures=max_procedures)
        if batch is not None:
            llm_replaced_codes = list(batch.replaced_law_codes or [])
            for rec in (batch.procedures or [])[:max_procedures]:
                grounded, action, target_code = _ground_record_with_official_metadata(
                    rec, doc=doc, existing_codes=existing_codes
                )
                candidates.append(
                    ExtractedProcedureCandidate(
                        candidate_id=f"cand_{uuid.uuid4().hex[:8]}",
                        action=action,
                        target_existing_code=target_code,
                        record=grounded,
                        source_name=doc.source_name,
                        source_url=doc.source_url,
                        source_so_hieu=doc.so_hieu,
                        extraction_mode="llm_structured",
                        confidence_note=batch.extraction_notes
                        or f"Trích xuất bằng LLM Structured Output từ nguồn chính thống {doc.so_hieu}.",
                    )
                )

    if not candidates:
        candidates.append(extract_procedures_deterministic(doc, existing_codes=existing_codes))

    expiry_candidates = detect_expiry_impacts(
        doc=doc,
        existing_procedures=existing_list,
        llm_replaced_codes=llm_replaced_codes,
    )
    candidates.extend(expiry_candidates)
    return candidates
