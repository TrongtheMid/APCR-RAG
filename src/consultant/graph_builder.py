"""
NHÁNH VŨ GIA ANH ĐỨC — Module Graph Builder (Hạng mục 2.5):
Chuyển đổi kết quả JSON 6 trường của thủ tục thành danh sách Nodes và Edges
(Thủ tục -> Cơ quan, Hồ sơ, Thời hạn, Lệ phí, Điều luật) phục vụ trực quan hóa đồ thị.
"""

from __future__ import annotations

from typing import Any, Dict, List


def extract_procedure_graph_data(proc_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trích xuất Dữ liệu Đồ thị Quan hệ 6 trường (Nodes & Edges)
    để hiển thị trực quan bên cột phải màn hình Hỏi — Đáp.
    """
    info = proc_dict["ma_va_ten_thu_tuc"]
    ho_so = proc_dict["thanh_phan_ho_so"]
    thoi_han = proc_dict["thoi_han_giai_quyet"]
    le_phi = proc_dict["le_phi"]
    laws = proc_dict["can_cu_phap_ly"]

    root_id = info["ma_thu_tuc"]
    nodes: List[Dict[str, Any]] = [
        {
            "id": root_id,
            "label": f"{root_id}\n{info['ten_thu_tuc'][:42]}...",
            "full_title": info["ten_thu_tuc"],
            "group": "procedure",
            "color": "#2563eb",
            "size": 30,
            "details": f"Phân nhóm: {info['phan_nhom']} | Lĩnh vực: {info['linh_vuc']}",
        }
    ]
    edges: List[Dict[str, str]] = []

    for idx, auth in enumerate(info.get("cap_tham_quyen", [])):
        node_id = f"{root_id}_AUTH_{idx + 1}"
        nodes.append(
            {
                "id": node_id,
                "label": f"Cơ quan:\n{auth}",
                "full_title": f"Cơ quan thẩm quyền: {auth}",
                "group": "authority",
                "color": "#059669",
                "size": 22,
                "details": f"Giai đoạn: {', '.join(info.get('giai_doan_to_tung', [])) or 'Hành chính công'}",
            }
        )
        edges.append({"source": root_id, "target": node_id, "label": "THẨM_QUYỀN"})

    for form in ho_so.get("giay_to_bat_buoc", []):
        node_id = form["form_id"]
        nodes.append(
            {
                "id": node_id,
                "label": f"Biểu mẫu:\n{form['form_id']}",
                "full_title": form["ten_giay_to"],
                "group": "form",
                "color": "#d97706",
                "size": 20,
                "details": form["ten_giay_to"],
            }
        )
        edges.append({"source": root_id, "target": node_id, "label": "YÊU_CẦU_HỒ_SƠ"})

    time_id = f"{root_id}_TIME"
    nodes.append(
        {
            "id": time_id,
            "label": f"Thời hạn:\nTối đa {thoi_han['so_ngay_toi_da']} ngày",
            "full_title": f"Tối đa {thoi_han['so_ngay_toi_da']} {thoi_han['don_vi_tinh']}",
            "group": "deadline",
            "color": "#7c3aed",
            "size": 21,
            "details": thoi_han["chi_tiet"],
        }
    )
    edges.append({"source": root_id, "target": time_id, "label": "THỜI_HẠN"})

    fee_id = f"{root_id}_FEE"
    fee_val = le_phi["muc_phi_vnd"]
    fee_label = "Miễn phí (0đ)" if fee_val == 0 else f"{fee_val:,} VNĐ"
    nodes.append(
        {
            "id": fee_id,
            "label": f"Lệ phí:\n{fee_label}",
            "full_title": f"Mức phí: {fee_label}",
            "group": "fee",
            "color": "#db2777",
            "size": 21,
            "details": le_phi["mo_ta"],
        }
    )
    edges.append({"source": root_id, "target": fee_id, "label": "LỆ_PHÍ"})

    for law in laws:
        node_id = law["law_id"]
        is_active = law["trang_thai_hieu_luc"] == "ĐANG HIỆU LỰC"
        nodes.append(
            {
                "id": node_id,
                "label": f"Điều luật:\n{law['law_id']}\n({law['trang_thai_hieu_luc']})",
                "full_title": f"{law['so_hieu_van_ban']} — {law['dieu_khoan_diem']}",
                "group": "law",
                "color": "#0284c7" if is_active else "#dc2626",
                "size": 22,
                "details": f"Hiệu lực: {law['ngay_hieu_luc']} | Trạng thái: {law['trang_thai_hieu_luc']}",
            }
        )
        edges.append({"source": root_id, "target": node_id, "label": "CĂN_CỨ_PHÁP_LÝ"})

    return {"nodes": nodes, "edges": edges, "procedure": proc_dict}
