"""
NHÁNH VŨ GIA ANH ĐỨC — Module Graph Builder:
1. Duy trì API đơn-hồ-sơ `extract_procedure_graph_data(proc_dict)` tương thích ngược
   (Nodes & Edges 6 trường: Thủ tục -> Cơ quan, Hồ sơ, Thời hạn, Lệ phí, Điều luật).
2. Mở rộng Đồ thị Tri thức Liên thủ tục bằng `networkx` (`build_inter_procedure_nx_graph`,
   `extract_inter_procedure_graph_data`, `get_related_procedures`), chỉ tạo quan hệ dựa trên
   dữ liệu thực tế trong hồ sơ JSON hoặc quy tắc liên kết được khai báo tường minh,
   bảo đảm node/edge không trùng lặp với ID ổn định.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# Quy tắc liên kết liên thủ tục được khai báo tường minh theo dữ liệu chuẩn hóa trong kho 23 thủ tục.
# Chỉ tạo cạnh khi cả thủ tục nguồn và thủ tục đích đều có mặt trong tập hồ sơ đầu vào.
INTER_PROCEDURE_RELATIONS: List[Tuple[str, str, str, str]] = [
    (
        "TTHS-BL-121",
        "TTHS-DT-122",
        "BIỆN_PHÁP_THAY_THẾ_TƯƠNG_ĐƯƠNG",
        "Hai biện pháp ngăn chặn thay thế tạm giam (Bảo lĩnh Điều 121 và Đặt tiền bảo đảm Điều 122 BLTTHS).",
    ),
    (
        "TTHS-BL-121",
        "TTHC-CT-04",
        "YÊU_CẦU_LIÊN_QUAN",
        "Bị can, bị cáo đang tại ngoại đi khỏi cấp xã từ 01 ngày trở lên phải khai báo tạm vắng (TTHC-CT-04).",
    ),
    (
        "TTHS-DT-122",
        "TTHC-CT-04",
        "YÊU_CẦU_LIÊN_QUAN",
        "Bị can, bị cáo được đặt tiền bảo đảm tại ngoại khi đi khỏi nơi cư trú phải khai báo tạm vắng.",
    ),
    (
        "TTHS-TG-145",
        "TTHS-GD-205",
        "THỦ_TỤC_TIẾP_THEO",
        "Sau khi tiếp nhận tố giác/tin báo tội phạm, đương sự có thể yêu cầu trưng cầu giám định thương tích hoặc định giá tài sản.",
    ),
    (
        "TTHS-TG-145",
        "TTHS-KN-475",
        "THỦ_TỤC_KHIẾU_NẠI",
        "Khiếu nại quyết định không khởi tố hoặc hành vi tố tụng phát sinh từ quá trình giải quyết tố giác.",
    ),
    (
        "TTHS-XAT-070",
        "TTHS-LLTP-045",
        "LIÊN_THÔNG_HỒ_SƠ",
        "Xác nhận đương nhiên xóa án tích được cập nhật và thể hiện qua thủ tục cấp Phiếu lý lịch tư pháp tại Sở Tư pháp.",
    ),
    (
        "TTHS-XAT-070",
        "TTHS-XAT-071",
        "PHÂN_NHÁNH_THẨM_QUYỀN",
        "Phân nhánh xóa án tích: đương nhiên (Sở Tư pháp) vs theo quyết định Tòa án đối với tội Chương XIII, XXVI BLHS.",
    ),
    (
        "TTHC-HT-01",
        "TTHC-HT-06",
        "BIẾN_THỂ_LIÊN_THÔNG",
        "Đăng ký khai sinh thông thường và quy trình liên thông điện tử 3 trong 1 (Khai sinh - Thường trú - BHYT).",
    ),
    (
        "TTHC-HT-06",
        "TTHC-CT-01",
        "BAO_GỒM_LIÊN_THÔNG",
        "Thủ tục liên thông TTHC-HT-06 tự động chuyển hồ sơ đăng ký thường trú cho trẻ dưới 6 tuổi (TTHC-CT-01).",
    ),
    (
        "TTHC-HT-06",
        "TTHC-BH-02",
        "BAO_GỒM_LIÊN_THÔNG",
        "Thủ tục liên thông TTHC-HT-06 tự động chuyển hồ sơ cấp thẻ BHYT cho trẻ dưới 6 tuổi (TTHC-BH-02).",
    ),
    (
        "TTHC-HT-03",
        "TTHC-HT-02",
        "GIẤY_TỜ_TIỀN_ĐỀ",
        "Giấy xác nhận tình trạng hôn nhân là giấy tờ đầu vào cho đăng ký kết hôn trong nước (khi chưa có trên CSDL).",
    ),
    (
        "TTHC-HT-03",
        "TTHC-HT-05",
        "GIẤY_TỜ_TIỀN_ĐỀ",
        "Giấy xác nhận tình trạng hôn nhân phục vụ hồ sơ đăng ký kết hôn có yếu tố nước ngoài.",
    ),
    (
        "TTHC-HT-02",
        "TTHC-HT-05",
        "PHÂN_NHÁNH_THẨM_QUYỀN",
        "Phân nhánh thẩm quyền đăng ký kết hôn: trong nước (UBND cấp xã) vs có yếu tố nước ngoài (UBND cấp huyện).",
    ),
    (
        "TTHC-CT-OLD-01",
        "TTHC-CT-01",
        "THAY_THẾ_HIỆU_LỰC",
        "Quy định cấp Sổ hộ khẩu giấy đã HẾT HIỆU LỰC và được thay thế bởi đăng ký thường trú điện tử (TTHC-CT-01).",
    ),
    (
        "TTHC-CC-OLD-02",
        "TTHC-CC-01",
        "THAY_THẾ_HIỆU_LỰC",
        "Quy định cấp/dùng CMND 9 số, 12 số đã HẾT HIỆU LỰC từ 01/01/2025, thay thế bằng thẻ Căn cước (TTHC-CC-01).",
    ),
    (
        "TTHC-CC-01",
        "TTHC-CT-03",
        "ĐIỀU_KIỆN_LIÊN_QUAN",
        "Cấp tài khoản định danh điện tử VNeID mức 2 yêu cầu có thẻ Căn cước gắn chíp hoặc thực hiện đồng thời khi cấp thẻ.",
    ),
    (
        "TTHC-LD-01",
        "TTHC-BH-01",
        "CÙNG_SỬ_DỤNG_SỔ_BHXH",
        "Cùng yêu cầu Sổ BHXH đã chốt quá trình đóng bảo hiểm khi người lao động chấm dứt hợp đồng lao động.",
    ),
    (
        "TTHC-LD-02",
        "TTHS-LLTP-045",
        "GIẤY_TỜ_TIỀN_ĐỀ",
        "Hồ sơ xin Giấy phép lao động cho người nước ngoài yêu cầu Phiếu lý lịch tư pháp (cấp không quá 06 tháng).",
    ),
    (
        "TTHC-DD-02",
        "TTHC-DD-01",
        "ĐIỀU_KIỆN_TIỀN_ĐỀ",
        "Cần được cấp Giấy chứng nhận (Sổ đỏ) lần đầu trước khi thực hiện đăng ký biến động/sang tên đất đai.",
    ),
    (
        "TTHC-DD-02",
        "TTHC-DD-03",
        "ĐIỀU_KIỆN_TIỀN_ĐỀ",
        "Hồ sơ xin Giấy phép xây dựng nhà ở riêng lẻ bắt buộc có Giấy chứng nhận quyền sử dụng đất hợp pháp.",
    ),
    (
        "TTHC-KD-01",
        "TTHC-KD-02",
        "PHÂN_NHÁNH_LOẠI_HÌNH",
        "Phân nhánh đăng ký kinh doanh theo Nghị định 01/2021/NĐ-CP: Hộ kinh doanh (cấp huyện) vs Doanh nghiệp (cấp tỉnh).",
    ),
]


def _stable_slug(text: str) -> str:
    """Tạo định danh ổn định, không dấu, viết hoa từ chuỗi tiếng Việt."""
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_str = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    ascii_str = ascii_str.replace("đ", "d").replace("Đ", "D")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", ascii_str).strip("_").upper()
    return cleaned or "UNKNOWN"


def _add_unique_node(
    node_map: Dict[str, Dict[str, Any]],
    node_data: Dict[str, Any],
) -> None:
    node_id = node_data["id"]
    if node_id not in node_map:
        node_map[node_id] = node_data


def _add_unique_edge(
    edge_map: Dict[Tuple[str, str, str], Dict[str, Any]],
    source: str,
    target: str,
    label: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    key = (source, target, label)
    if key not in edge_map:
        edge: Dict[str, Any] = {"source": source, "target": target, "label": label}
        if extra:
            edge.update(extra)
        edge_map[key] = edge


def extract_procedure_graph_data(proc_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Trích xuất Dữ liệu Đồ thị Quan hệ 6 trường (Nodes & Edges) cho một thủ tục đơn lẻ
    để hiển thị trực quan trên giao diện Hỏi — Đáp (giữ nguyên cấu trúc tương thích ngược).
    """
    info = proc_dict["ma_va_ten_thu_tuc"]
    ho_so = proc_dict["thanh_phan_ho_so"]
    thoi_han = proc_dict["thoi_han_giai_quyet"]
    le_phi = proc_dict["le_phi"]
    laws = proc_dict["can_cu_phap_ly"]

    root_id = info["ma_thu_tuc"]
    node_map: Dict[str, Dict[str, Any]] = {}
    edge_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    _add_unique_node(
        node_map,
        {
            "id": root_id,
            "label": f"{root_id}\n{info['ten_thu_tuc'][:42]}...",
            "full_title": info["ten_thu_tuc"],
            "group": "procedure",
            "color": "#2563eb",
            "size": 30,
            "details": f"Phân nhóm: {info['phan_nhom']} | Lĩnh vực: {info['linh_vuc']}",
        },
    )

    for idx, auth in enumerate(info.get("cap_tham_quyen", [])):
        node_id = f"{root_id}_AUTH_{idx + 1}"
        _add_unique_node(
            node_map,
            {
                "id": node_id,
                "label": f"Cơ quan:\n{auth}",
                "full_title": f"Cơ quan thẩm quyền: {auth}",
                "group": "authority",
                "color": "#059669",
                "size": 22,
                "details": f"Giai đoạn: {', '.join(info.get('giai_doan_to_tung', [])) or 'Hành chính công'}",
            },
        )
        _add_unique_edge(edge_map, root_id, node_id, "THẨM_QUYỀN")

    for form in ho_so.get("giay_to_bat_buoc", []):
        node_id = form["form_id"]
        _add_unique_node(
            node_map,
            {
                "id": node_id,
                "label": f"Biểu mẫu:\n{form['form_id']}",
                "full_title": form["ten_giay_to"],
                "group": "form",
                "color": "#d97706",
                "size": 20,
                "details": form["ten_giay_to"],
            },
        )
        _add_unique_edge(edge_map, root_id, node_id, "YÊU_CẦU_HỒ_SƠ")

    time_id = f"{root_id}_TIME"
    _add_unique_node(
        node_map,
        {
            "id": time_id,
            "label": f"Thời hạn:\nTối đa {thoi_han['so_ngay_toi_da']} ngày",
            "full_title": f"Tối đa {thoi_han['so_ngay_toi_da']} {thoi_han['don_vi_tinh']}",
            "group": "deadline",
            "color": "#7c3aed",
            "size": 21,
            "details": thoi_han["chi_tiet"],
        },
    )
    _add_unique_edge(edge_map, root_id, time_id, "THỜI_HẠN")

    fee_id = f"{root_id}_FEE"
    fee_val = le_phi["muc_phi_vnd"]
    fee_label = "Miễn phí (0đ)" if fee_val == 0 else f"{fee_val:,} VNĐ"
    _add_unique_node(
        node_map,
        {
            "id": fee_id,
            "label": f"Lệ phí:\n{fee_label}",
            "full_title": f"Mức phí: {fee_label}",
            "group": "fee",
            "color": "#db2777",
            "size": 21,
            "details": le_phi["mo_ta"],
        },
    )
    _add_unique_edge(edge_map, root_id, fee_id, "LỆ_PHÍ")

    for law in laws:
        node_id = law["law_id"]
        is_active = law["trang_thai_hieu_luc"] == "ĐANG HIỆU LỰC"
        _add_unique_node(
            node_map,
            {
                "id": node_id,
                "label": f"Điều luật:\n{law['law_id']}\n({law['trang_thai_hieu_luc']})",
                "full_title": f"{law['so_hieu_van_ban']} — {law['dieu_khoan_diem']}",
                "group": "law",
                "color": "#0284c7" if is_active else "#dc2626",
                "size": 22,
                "details": f"Hiệu lực: {law['ngay_hieu_luc']} | Trạng thái: {law['trang_thai_hieu_luc']}",
            },
        )
        _add_unique_edge(edge_map, root_id, node_id, "CĂN_CỨ_PHÁP_LÝ")

    return {
        "nodes": list(node_map.values()),
        "edges": list(edge_map.values()),
        "procedure": proc_dict,
    }


def _import_networkx():
    try:
        import networkx as nx  # type: ignore

        return nx
    except ImportError as exc:
        raise ImportError(
            "Thiếu thư viện 'networkx' để xây dựng đồ thị tri thức liên thủ tục. "
            "Vui lòng cài đặt: pip install networkx"
        ) from exc


def build_inter_procedure_nx_graph(
    procedures: List[Dict[str, Any]],
    include_attribute_nodes: bool = True,
    include_declared_relations: bool = True,
) -> Any:
    """
    Xây dựng đồ thị tri thức liên thủ tục bằng NetworkX (`nx.DiGraph`).
    - Dùng ID ổn định cho mọi nút: `ma_thu_tuc`, `AUTH::<slug>`, `law_id`, `form_id`.
    - Các thủ tục dùng chung cơ quan thẩm quyền hoặc căn cứ điều luật sẽ tự động
      kết nối qua nút dùng chung (khi `include_attribute_nodes=True`).
    - Các quan hệ liên thông trực tiếp giữa thủ tục chỉ được thêm từ `INTER_PROCEDURE_RELATIONS`
      khi cả hai đầu đều có trong `procedures`.
    """
    nx = _import_networkx()
    graph = nx.DiGraph()
    proc_by_id: Dict[str, Dict[str, Any]] = {}

    for proc in procedures:
        info = proc.get("ma_va_ten_thu_tuc") or {}
        ma_tt = str(info.get("ma_thu_tuc") or "").strip()
        if not ma_tt:
            continue
        proc_by_id[ma_tt] = proc
        laws = proc.get("can_cu_phap_ly") or []
        status = laws[0].get("trang_thai_hieu_luc", "ĐANG HIỆU LỰC") if laws else "ĐANG HIỆU LỰC"
        ten_tt = str(info.get("ten_thu_tuc") or ma_tt)
        short_title = f"{ten_tt[:42]}..." if len(ten_tt) > 42 else ten_tt

        if not graph.has_node(ma_tt):
            graph.add_node(
                ma_tt,
                label=f"{ma_tt}\n{short_title}",
                full_title=ten_tt,
                group="procedure",
                color="#2563eb" if status == "ĐANG HIỆU LỰC" else "#9ca3af",
                size=30,
                details=f"Phân nhóm: {info.get('phan_nhom')} | Lĩnh vực: {info.get('linh_vuc')} | {status}",
                phan_nhom=info.get("phan_nhom", ""),
                linh_vuc=info.get("linh_vuc", ""),
                trang_thai_hieu_luc=status,
            )

        if include_attribute_nodes:
            for auth in info.get("cap_tham_quyen", []):
                auth_id = f"AUTH::{_stable_slug(str(auth))}"
                if not graph.has_node(auth_id):
                    graph.add_node(
                        auth_id,
                        label=f"Cơ quan:\n{auth}",
                        full_title=f"Cơ quan thẩm quyền: {auth}",
                        group="authority",
                        color="#059669",
                        size=22,
                        details=f"Cơ quan có thẩm quyền: {auth}",
                    )
                if not graph.has_edge(ma_tt, auth_id):
                    graph.add_edge(
                        ma_tt,
                        auth_id,
                        label="THẨM_QUYỀN",
                        relation_type="THẨM_QUYỀN",
                        description=f"{ma_tt} thuộc thẩm quyền của {auth}",
                    )

            for law in laws:
                law_id = str(law.get("law_id") or "").strip()
                if not law_id:
                    continue
                is_active = law.get("trang_thai_hieu_luc") == "ĐANG HIỆU LỰC"
                if not graph.has_node(law_id):
                    graph.add_node(
                        law_id,
                        label=f"Điều luật:\n{law_id}\n({law.get('trang_thai_hieu_luc', '')})",
                        full_title=f"{law.get('so_hieu_van_ban', '')} — {law.get('dieu_khoan_diem', '')}",
                        group="law",
                        color="#0284c7" if is_active else "#dc2626",
                        size=22,
                        details=f"Hiệu lực: {law.get('ngay_hieu_luc', '')} | Trạng thái: {law.get('trang_thai_hieu_luc', '')}",
                    )
                if not graph.has_edge(ma_tt, law_id):
                    graph.add_edge(
                        ma_tt,
                        law_id,
                        label="CĂN_CỨ_PHÁP_LÝ",
                        relation_type="CĂN_CỨ_PHÁP_LÝ",
                        description=f"{ma_tt} căn cứ {law_id}",
                    )

            ho_so = proc.get("thanh_phan_ho_so") or {}
            for form in ho_so.get("giay_to_bat_buoc", []):
                form_id = str(form.get("form_id") or "").strip()
                if not form_id:
                    continue
                if not graph.has_node(form_id):
                    graph.add_node(
                        form_id,
                        label=f"Biểu mẫu:\n{form_id}",
                        full_title=form.get("ten_giay_to", form_id),
                        group="form",
                        color="#d97706",
                        size=20,
                        details=form.get("ten_giay_to", ""),
                    )
                if not graph.has_edge(ma_tt, form_id):
                    graph.add_edge(
                        ma_tt,
                        form_id,
                        label="YÊU_CẦU_HỒ_SƠ",
                        relation_type="YÊU_CẦU_HỒ_SƠ",
                        description=f"{ma_tt} yêu cầu biểu mẫu {form_id}",
                    )

    if include_declared_relations:
        for src_id, dst_id, rel_label, desc in INTER_PROCEDURE_RELATIONS:
            if src_id in proc_by_id and dst_id in proc_by_id and not graph.has_edge(src_id, dst_id):
                graph.add_edge(
                    src_id,
                    dst_id,
                    label=rel_label,
                    relation_type=rel_label,
                    description=desc,
                    grounded_by="declared_rule",
                )

    # Tạo cạnh trực tiếp dựa trên dữ liệu dùng chung cơ quan / điều luật khi không bật nút thuộc tính
    if not include_attribute_nodes:
        proc_ids = sorted(proc_by_id.keys())
        for idx, pid_a in enumerate(proc_ids):
            pa = proc_by_id[pid_a]
            laws_a = {c["law_id"] for c in pa.get("can_cu_phap_ly", [])}
            auth_a = set((pa.get("ma_va_ten_thu_tuc") or {}).get("cap_tham_quyen", []))
            for pid_b in proc_ids[idx + 1 :]:
                pb = proc_by_id[pid_b]
                laws_b = {c["law_id"] for c in pb.get("can_cu_phap_ly", [])}
                auth_b = set((pb.get("ma_va_ten_thu_tuc") or {}).get("cap_tham_quyen", []))

                shared_laws = sorted(laws_a.intersection(laws_b))
                shared_auth = sorted(auth_a.intersection(auth_b))
                if shared_laws and not graph.has_edge(pid_a, pid_b):
                    graph.add_edge(
                        pid_a,
                        pid_b,
                        label="CÙNG_CĂN_CỨ_PHÁP_LÝ",
                        relation_type="CÙNG_CĂN_CỨ_PHÁP_LÝ",
                        description=f"Cùng viện dẫn điều luật: {', '.join(shared_laws)}",
                        grounded_by="shared_law_id",
                    )
                elif shared_auth and not graph.has_edge(pid_a, pid_b):
                    graph.add_edge(
                        pid_a,
                        pid_b,
                        label="CÙNG_THẨM_QUYỀN",
                        relation_type="CÙNG_THẨM_QUYỀN",
                        description=f"Cùng cơ quan có thẩm quyền: {', '.join(shared_auth)}",
                        grounded_by="shared_authority",
                    )

    return graph


def get_related_procedures(
    ma_thu_tuc: str,
    procedures: List[Dict[str, Any]],
    max_hops: int = 1,
) -> List[Dict[str, Any]]:
    """
    Truy vấn các thủ tục liên quan trực tiếp hoặc gián tiếp (qua cơ quan/điều luật dùng chung)
    từ đồ thị NetworkX liên thủ tục.
    """
    graph = build_inter_procedure_nx_graph(
        procedures=procedures,
        include_attribute_nodes=True,
        include_declared_relations=True,
    )
    if ma_thu_tuc not in graph:
        return []

    proc_by_id = {
        p["ma_va_ten_thu_tuc"]["ma_thu_tuc"]: p
        for p in procedures
        if (p.get("ma_va_ten_thu_tuc") or {}).get("ma_thu_tuc")
    }
    related: Dict[str, Dict[str, Any]] = {}

    # 1. Liên kết trực tiếp giữa hai thủ tục (vào hoặc ra)
    for _, target, data in graph.out_edges(ma_thu_tuc, data=True):
        if target in proc_by_id and target != ma_thu_tuc:
            related[target] = {
                "ma_thu_tuc": target,
                "ten_thu_tuc": proc_by_id[target]["ma_va_ten_thu_tuc"]["ten_thu_tuc"],
                "relation": data.get("relation_type", data.get("label", "")),
                "direction": "outgoing",
                "description": data.get("description", ""),
                "hop": 1,
            }

    for source, _, data in graph.in_edges(ma_thu_tuc, data=True):
        if source in proc_by_id and source != ma_thu_tuc and source not in related:
            related[source] = {
                "ma_thu_tuc": source,
                "ten_thu_tuc": proc_by_id[source]["ma_va_ten_thu_tuc"]["ten_thu_tuc"],
                "relation": data.get("relation_type", data.get("label", "")),
                "direction": "incoming",
                "description": data.get("description", ""),
                "hop": 1,
            }

    # 2. Nếu max_hops >= 2, tìm thêm thủ tục liên kết qua nút thuộc tính dùng chung (cùng cơ quan / cùng điều luật)
    if max_hops >= 2:
        undirected = graph.to_undirected()
        for nbr in undirected.neighbors(ma_thu_tuc):
            if nbr in proc_by_id:
                continue
            nbr_group = graph.nodes[nbr].get("group", "")
            nbr_title = graph.nodes[nbr].get("full_title", nbr)
            for hop2 in undirected.neighbors(nbr):
                if hop2 in proc_by_id and hop2 != ma_thu_tuc and hop2 not in related:
                    rel_lbl = "CÙNG_THẨM_QUYỀN" if nbr_group == "authority" else "DÙNG_CHUNG_THUỘC_TÍNH"
                    if nbr_group == "law":
                        rel_lbl = "CÙNG_CĂN_CỨ_PHÁP_LÝ"
                    related[hop2] = {
                        "ma_thu_tuc": hop2,
                        "ten_thu_tuc": proc_by_id[hop2]["ma_va_ten_thu_tuc"]["ten_thu_tuc"],
                        "relation": rel_lbl,
                        "direction": "shared_node",
                        "description": f"Liên kết qua nút dùng chung: {nbr_title}",
                        "hop": 2,
                    }

    return sorted(related.values(), key=lambda item: (item["hop"], item["ma_thu_tuc"]))


def extract_inter_procedure_graph_data(
    procedures: List[Dict[str, Any]],
    focus_ma_thu_tuc: Optional[str] = None,
    max_hops: int = 1,
    include_attribute_nodes: bool = True,
) -> Dict[str, Any]:
    """
    Xuất dữ liệu đồ thị liên thủ tục (Nodes, Edges, Related Procedures, Stats)
    từ NetworkX `DiGraph` để phục vụ trực quan hóa hoặc kiểm tra quan hệ liên thông.
    """
    graph = build_inter_procedure_nx_graph(
        procedures=procedures,
        include_attribute_nodes=include_attribute_nodes,
        include_declared_relations=True,
    )

    nx = _import_networkx()
    if focus_ma_thu_tuc and focus_ma_thu_tuc in graph:
        undirected = graph.to_undirected()
        sub_nodes = set(nx.single_source_shortest_path_length(undirected, focus_ma_thu_tuc, cutoff=max(1, max_hops)).keys())
        view_graph = graph.subgraph(sub_nodes)
    else:
        view_graph = graph

    nodes: List[Dict[str, Any]] = []
    for node_id, attrs in view_graph.nodes(data=True):
        item = {"id": node_id}
        item.update(attrs)
        nodes.append(item)

    edges: List[Dict[str, Any]] = []
    for src, dst, attrs in view_graph.edges(data=True):
        edge_item = {
            "source": src,
            "target": dst,
            "label": attrs.get("label", ""),
            "description": attrs.get("description", ""),
        }
        edges.append(edge_item)

    related = (
        get_related_procedures(focus_ma_thu_tuc, procedures, max_hops=max(1, max_hops))
        if focus_ma_thu_tuc
        else []
    )

    procedure_count = sum(1 for n in nodes if n.get("group") == "procedure")
    return {
        "nodes": nodes,
        "edges": edges,
        "focus_ma_thu_tuc": focus_ma_thu_tuc,
        "related_procedures": related,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "procedure_nodes": procedure_count,
        },
    }
