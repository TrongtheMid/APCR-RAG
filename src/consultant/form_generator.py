"""
NHÁNH VŨ GIA ANH ĐỨC — Module Form Generator (Sinh biểu mẫu DOCX điền sẵn):
Sinh tài liệu `.docx` từ dữ liệu hồ sơ thủ tục chuẩn hóa 6 trường (`ProcedureRecord` / dict)
và thông tin người dân truyền vào (`user_data`).

Nguyên tắc an toàn:
- Chỉ điền dữ kiện thực sự có trong `procedure` và `user_data`.
- Tuyệt đối KHÔNG tự tạo tên biểu mẫu, căn cứ pháp luật, nội dung tuyên bố hay thông tin cá nhân.
- Trường nào người dùng chưa cung cấp sẽ để chỗ trống rõ ràng (`........................`) để điền tay.
- Cung cấp API trả về cả `bytes` (cho Streamlit download button / web API) và ghi file ra đĩa.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.schemas import ProcedureRecord

_BLANK_PLACEHOLDER = "............................................................"

STANDARD_USER_FIELDS: List[Tuple[str, str]] = [
    ("ho_ten", "Họ và tên người đề nghị"),
    ("ngay_sinh", "Ngày, tháng, năm sinh"),
    ("cccd", "Số Thẻ Căn cước / CCCD / Hộ chiếu"),
    ("dia_chi", "Địa chỉ thường trú / Nơi ở hiện tại"),
    ("so_dien_thoai", "Số điện thoại liên hệ"),
    ("email", "Email liên hệ (nếu có)"),
]


def _normalize_procedure_dict(procedure: ProcedureRecord | Dict[str, Any]) -> Dict[str, Any]:
    """Chuẩn hóa và kiểm tra cấu trúc đầu vào của hồ sơ thủ tục."""
    if isinstance(procedure, ProcedureRecord):
        return procedure.model_dump()
    if not isinstance(procedure, dict):
        raise TypeError("Tham số 'procedure' phải là dict hoặc ProcedureRecord hợp lệ.")

    required_top_keys = (
        "ma_va_ten_thu_tuc",
        "thanh_phan_ho_so",
        "thoi_han_giai_quyet",
        "le_phi",
        "can_cu_phap_ly",
    )
    missing = [k for k in required_top_keys if k not in procedure or not procedure[k]]
    if missing:
        raise ValueError(
            f"Dữ liệu thủ tục thiếu các trường bắt buộc của lược đồ 6 trường: {', '.join(missing)}."
        )

    info = procedure["ma_va_ten_thu_tuc"]
    if not isinstance(info, dict) or not info.get("ma_thu_tuc") or not info.get("ten_thu_tuc"):
        raise ValueError("Trường 'ma_va_ten_thu_tuc' phải chứa đầy đủ 'ma_thu_tuc' và 'ten_thu_tuc'.")

    return procedure


def _validate_user_data(user_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Kiểm tra tính hợp lệ cơ bản của dữ liệu người dùng truyền vào."""
    if user_data is None:
        return {}
    if not isinstance(user_data, dict):
        raise TypeError("Tham số 'user_data' phải là một dictionary (hoặc None).")

    cleaned: Dict[str, Any] = {}
    for key, val in user_data.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Tên trường trong 'user_data' phải là chuỗi khác rỗng.")
        if val is None:
            continue
        if key == "thong_tin_bo_sung":
            if not isinstance(val, dict):
                raise ValueError("Trường 'thong_tin_bo_sung' phải là một dictionary.")
            cleaned[key] = {str(k): str(v).strip() for k, v in val.items() if v is not None and str(v).strip()}
        elif key == "tai_lieu_kem_theo":
            if not isinstance(val, list):
                raise ValueError("Trường 'tai_lieu_kem_theo' phải là danh sách mã form_id hoặc tên giấy tờ.")
            cleaned[key] = [str(item).strip() for item in val if str(item).strip()]
        elif isinstance(val, (str, int, float, bool)):
            cleaned[key] = str(val).strip()
        else:
            raise ValueError(
                f"Giá trị của trường '{key}' không hợp lệ (chỉ chấp nhận chuỗi, số hoặc kiểu đơn giản)."
            )
    return cleaned


def validate_form_inputs(
    procedure: ProcedureRecord | Dict[str, Any],
    user_data: Optional[Dict[str, Any]] = None,
    form_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, str]]]:
    """
    Kiểm tra toàn bộ đầu vào trước khi sinh biểu mẫu DOCX.
    Trả về `(proc_dict, cleaned_user_data, selected_form)`.
    """
    proc_dict = _normalize_procedure_dict(procedure)
    cleaned_user = _validate_user_data(user_data)

    selected_form: Optional[Dict[str, str]] = None
    forms = proc_dict["thanh_phan_ho_so"].get("giay_to_bat_buoc", [])
    if form_id is not None:
        target = form_id.strip()
        if not target:
            raise ValueError("Tham số 'form_id' không được là chuỗi rỗng.")
        for item in forms:
            if str(item.get("form_id", "")).strip().upper() == target.upper():
                selected_form = item
                break
        if selected_form is None:
            available = [f.get("form_id", "") for f in forms]
            raise ValueError(
                f"Mã biểu mẫu '{form_id}' không tồn tại trong thủ tục "
                f"'{proc_dict['ma_va_ten_thu_tuc']['ma_thu_tuc']}'. "
                f"Các mã hợp lệ: {available}."
            )

    return proc_dict, cleaned_user, selected_form


def get_supported_form_fields(
    procedure: ProcedureRecord | Dict[str, Any],
    form_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trả về danh sách trường biểu mẫu chuẩn và danh mục giấy tờ của thủ tục
    để giao diện Streamlit/Web có thể dựng form nhập liệu động.
    """
    proc_dict, _, selected_form = validate_form_inputs(procedure, None, form_id=form_id)
    info = proc_dict["ma_va_ten_thu_tuc"]
    forms = proc_dict["thanh_phan_ho_so"].get("giay_to_bat_buoc", [])
    return {
        "ma_thu_tuc": info["ma_thu_tuc"],
        "ten_thu_tuc": info["ten_thu_tuc"],
        "selected_form": selected_form,
        "available_forms": forms,
        "fields": [
            {"key": k, "label": lbl, "required": k in {"ho_ten", "cccd"}}
            for k, lbl in STANDARD_USER_FIELDS
        ]
        + [
            {"key": "co_quan_tiep_nhan", "label": "Cơ quan tiếp nhận (Kính gửi)", "required": False},
            {"key": "noi_dung_de_nghi", "label": "Nội dung đề nghị cụ thể", "required": False},
            {"key": "ngay_lap", "label": "Ngày lập đơn/tờ khai", "required": False},
        ],
    }


def _import_docx():
    try:
        from docx import Document  # type: ignore
        from docx.enum.text import WD_ALIGN_PARAGRAPH  # type: ignore
        from docx.shared import Pt  # type: ignore

        return Document, WD_ALIGN_PARAGRAPH, Pt
    except ImportError as exc:
        raise ImportError(
            "Thiếu thư viện 'python-docx' để sinh biểu mẫu DOCX. "
            "Vui lòng cài đặt: pip install python-docx"
        ) from exc


def generate_procedure_docx_bytes(
    procedure: ProcedureRecord | Dict[str, Any],
    user_data: Optional[Dict[str, Any]] = None,
    form_id: Optional[str] = None,
) -> bytes:
    """
    Sinh tài liệu DOCX điền sẵn dưới dạng `bytes` từ dữ liệu thủ tục JSON 6 trường
    và thông tin người dùng cung cấp (chỉ điền dữ kiện có thật, còn lại để chỗ trống).
    """
    proc_dict, cleaned_user, selected_form = validate_form_inputs(
        procedure=procedure,
        user_data=user_data,
        form_id=form_id,
    )
    Document, WD_ALIGN_PARAGRAPH, Pt = _import_docx()

    info = proc_dict["ma_va_ten_thu_tuc"]
    ho_so = proc_dict["thanh_phan_ho_so"]
    thoi_han = proc_dict["thoi_han_giai_quyet"]
    le_phi = proc_dict["le_phi"]
    laws = proc_dict["can_cu_phap_ly"]

    doc = Document()
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Times New Roman"
    normal_style.font.size = Pt(12)

    # Quốc hiệu & Tiêu ngữ
    p_nat = doc.add_paragraph()
    p_nat.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_nat = p_nat.add_run("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n")
    r_nat.bold = True
    r_sub = p_nat.add_run("Độc lập - Tự do - Hạnh phúc\n")
    r_sub.bold = True
    p_nat.add_run("--------------------")

    # Tiêu đề biểu mẫu (chỉ lấy từ dữ liệu gốc của thủ tục)
    title_text = (
        selected_form["ten_giay_to"]
        if selected_form is not None
        else f"PHIẾU KÊ KHAI THÔNG TIN THỰC HIỆN THỦ TỤC\n{info['ten_thu_tuc']}"
    )
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run(title_text.upper())
    r_title.bold = True
    r_title.font.size = Pt(13)

    code_line = f"Mã thủ tục: {info['ma_thu_tuc']}"
    if selected_form is not None:
        code_line += f"  |  Mã biểu mẫu: {selected_form['form_id']}"
    p_code = doc.add_paragraph()
    p_code.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_code.add_run(code_line).italic = True

    # Kính gửi (chỉ điền nếu người dùng truyền, nếu không gợi ý cơ quan thẩm quyền từ JSON và chỗ trống)
    authorities = info.get("cap_tham_quyen") or []
    kinh_gui = cleaned_user.get("co_quan_tiep_nhan")
    p_to = doc.add_paragraph()
    p_to.add_run("Kính gửi: ").bold = True
    if kinh_gui:
        p_to.add_run(kinh_gui)
    else:
        auth_hint = f" (Thẩm quyền theo quy định: {', '.join(authorities)})" if authorities else ""
        p_to.add_run(f"{_BLANK_PLACEHOLDER}{auth_hint}")

    # Phần I: Thông tin người đề nghị
    p_sec1 = doc.add_paragraph()
    p_sec1.add_run("I. THÔNG TIN NGƯỜI ĐỀ NGHỊ / KÊ KHAI").bold = True

    table = doc.add_table(rows=len(STANDARD_USER_FIELDS), cols=2)
    table.style = "Table Grid"
    for row_idx, (field_key, field_label) in enumerate(STANDARD_USER_FIELDS):
        row_cells = table.rows[row_idx].cells
        row_cells[0].text = field_label
        val = cleaned_user.get(field_key)
        row_cells[1].text = val if val else _BLANK_PLACEHOLDER

    # Phần II: Nội dung đề nghị & Thông tin bổ sung
    p_sec2 = doc.add_paragraph()
    p_sec2.paragraph_format.space_before = Pt(10)
    p_sec2.add_run("II. NỘI DUNG ĐỀ NGHỊ").bold = True

    p_proc_name = doc.add_paragraph()
    p_proc_name.add_run("Thủ tục đề nghị giải quyết: ").bold = True
    p_proc_name.add_run(f"{info['ten_thu_tuc']} ({info['ma_thu_tuc']})")

    p_req = doc.add_paragraph()
    p_req.add_run("Nội dung cụ thể: ").bold = True
    p_req.add_run(cleaned_user.get("noi_dung_de_nghi") or _BLANK_PLACEHOLDER)

    extra_info = cleaned_user.get("thong_tin_bo_sung")
    if isinstance(extra_info, dict) and extra_info:
        p_ext = doc.add_paragraph()
        p_ext.add_run("Thông tin bổ sung do người đề nghị cung cấp:").bold = True
        for k, v in extra_info.items():
            doc.add_paragraph(f"- {k}: {v}")

    # Phần III: Thành phần hồ sơ & Căn cứ pháp lý từ JSON gốc
    p_sec3 = doc.add_paragraph()
    p_sec3.paragraph_format.space_before = Pt(10)
    p_sec3.add_run("III. THÀNH PHẦN HỒ SƠ & THÔNG TIN PHÁP LÝ THAM CHIẾU").bold = True

    provided_docs = {d.upper() for d in cleaned_user.get("tai_lieu_kem_theo", [])}
    doc.add_paragraph("Danh mục giấy tờ theo hồ sơ chuẩn:")
    for item in ho_so.get("giay_to_bat_buoc", []):
        fid = str(item.get("form_id", ""))
        fname = str(item.get("ten_giay_to", ""))
        checked = "[x]" if (fid.upper() in provided_docs or fname.upper() in provided_docs) else "[ ]"
        doc.add_paragraph(f"  {checked} [{fid}] {fname}")

    p_law = doc.add_paragraph()
    p_law.add_run("Căn cứ pháp lý (theo dữ liệu hệ thống): ").bold = True
    law_texts = [
        f"{c['so_hieu_van_ban']} ({c['dieu_khoan_diem']} — {c['trang_thai_hieu_luc']})"
        for c in laws
    ]
    p_law.add_run("; ".join(law_texts))

    fee_val = int(le_phi["muc_phi_vnd"])
    fee_str = "Miễn phí (0 VNĐ)" if fee_val == 0 else f"{fee_val:,} VNĐ".replace(",", ".")
    p_meta = doc.add_paragraph()
    p_meta.add_run("Thời hạn giải quyết theo quy định: ").bold = True
    p_meta.add_run(f"Tối đa {thoi_han['so_ngay_toi_da']} {thoi_han['don_vi_tinh']}. ")
    p_meta.add_run("Lệ phí tham chiếu: ").bold = True
    p_meta.add_run(f"{fee_str}.")

    p_cond = doc.add_paragraph()
    p_cond.add_run("Điều kiện bắt buộc cần lưu ý: ").bold = True
    p_cond.add_run(str(ho_so.get("dieu_kien_bat_buoc", "")))

    # Phần chữ ký
    date_line = cleaned_user.get("ngay_lap") or "......., ngày ...... tháng ...... năm 20...."
    p_sign = doc.add_paragraph()
    p_sign.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_sign.paragraph_format.space_before = Pt(14)
    p_sign.add_run(f"{date_line}\n").italic = True
    p_sign.add_run("NGƯỜI ĐỀ NGHỊ\n").bold = True
    p_sign.add_run("(Ký và ghi rõ họ tên)\n\n\n")
    if cleaned_user.get("ho_ten"):
        p_sign.add_run(cleaned_user["ho_ten"]).bold = True

    p_note = doc.add_paragraph()
    p_note.paragraph_format.space_before = Pt(10)
    r_note = p_note.add_run(
        "Ghi chú: Mẫu biểu được tạo tự động từ dữ liệu chuẩn hóa của APCR-RAG; "
        "chỉ điền các thông tin do người dùng cung cấp và giữ nguyên các chỗ trống chưa khai báo."
    )
    r_note.italic = True
    r_note.font.size = Pt(10)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def generate_procedure_docx_file(
    procedure: ProcedureRecord | Dict[str, Any],
    output_path: str | Path,
    user_data: Optional[Dict[str, Any]] = None,
    form_id: Optional[str] = None,
) -> Path:
    """Sinh biểu mẫu DOCX điền sẵn và lưu ra đường dẫn `output_path`."""
    if not output_path or not str(output_path).strip():
        raise ValueError("Đường dẫn 'output_path' không được để trống.")

    target = Path(output_path)
    if target.exists() and target.is_dir():
        raise ValueError(f"Đường dẫn '{target}' là thư mục, cần truyền đường dẫn tệp .docx.")
    if target.suffix.lower() != ".docx":
        target = target.with_suffix(".docx")

    docx_bytes = generate_procedure_docx_bytes(
        procedure=procedure,
        user_data=user_data,
        form_id=form_id,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(docx_bytes)
    return target


class ProcedureFormGenerator:
    """Facade hướng đối tượng cho chức năng sinh biểu mẫu DOCX điền sẵn."""

    def get_supported_fields(
        self,
        procedure: ProcedureRecord | Dict[str, Any],
        form_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return get_supported_form_fields(procedure=procedure, form_id=form_id)

    def generate_bytes(
        self,
        procedure: ProcedureRecord | Dict[str, Any],
        user_data: Optional[Dict[str, Any]] = None,
        form_id: Optional[str] = None,
    ) -> bytes:
        return generate_procedure_docx_bytes(
            procedure=procedure,
            user_data=user_data,
            form_id=form_id,
        )

    def generate_file(
        self,
        procedure: ProcedureRecord | Dict[str, Any],
        output_path: str | Path,
        user_data: Optional[Dict[str, Any]] = None,
        form_id: Optional[str] = None,
    ) -> Path:
        return generate_procedure_docx_file(
            procedure=procedure,
            output_path=output_path,
            user_data=user_data,
            form_id=form_id,
        )
