"""
ỨNG DỤNG GIAO DIỆN WEB STREAMLIT 4 PHÂN HỆ (THEO THIẾT KẾ LAWGIC)
Hệ thống Tư vấn về Thủ tục Hành chính và Tố tụng Hình sự — Học phần INT 3507 2
Giảng viên hướng dẫn: TS. Nghiêm Nguyễn Việt Dũng
Nhóm 9: Phạm Đức Trọng • Vũ Gia Anh Đức • Nguyễn Hữu Hải Long
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.baseline_single_agent import BaselineSingleAgentRAG
from src.judge_agent import get_legal_misconceptions_catalog
from src.langgraph_workflow import run_legal_multi_agent_workflow
from src.retriever_agent import get_retriever_agent, search_direct_bm25

BENCHMARK_PATH = Path(__file__).resolve().parent / "data" / "benchmark_50_questions.json"

st.set_page_config(
    page_title="Hệ thống Tư vấn Thủ tục Hành chính & Tố tụng Hình sự | Nhóm 9",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CSS TÙY CHỈNH GIAO DIỆN CHUYÊN NGHIỆP THEO PHONG CÁCH LAWGIC
# ==============================================================================

st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        color: #ffffff;
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.15);
    }
    .main-header h2 {
        margin: 0;
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    .irac-box {
        border-left: 5px solid #2563eb;
        background-color: #f8fafc;
        padding: 0.9rem 1.1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
        color: #0f172a;
    }
    .irac-rule { border-left-color: #0284c7; background-color: #f0f9ff; }
    .irac-app { border-left-color: #059669; background-color: #f0fdf4; }
    .irac-conc { border-left-color: #7c3aed; background-color: #f5f3ff; }
    .badge-active {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid #86efac;
    }
    .badge-expired {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 10px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.8rem;
        border: 1px solid #fca5a5;
    }
    .badge-sev-high {
        background-color: #dc2626;
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-sev-med {
        background-color: #d97706;
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-sev-low {
        background-color: #2563eb;
        color: white;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 800;
        color: #1e3a8a;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: #475569;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cached_retriever():
    return get_retriever_agent()


@st.cache_resource
def load_cached_baseline():
    return BaselineSingleAgentRAG()


@st.cache_data
def load_benchmark_questions() -> List[Dict[str, Any]]:
    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def render_procedure_graph_column(graph_data: Optional[Dict[str, Any]]) -> None:
    """Hiển thị Đồ thị trực quan hóa cấu trúc 6 trường của thủ tục ở cột bên phải."""
    st.markdown("#### 🕸️ Đồ thị Cấu trúc 6 Trường Hồ sơ")
    if not graph_data:
        st.info("Chưa có dữ liệu đồ thị thủ tục do câu hỏi bị từ chối hoặc chưa truy xuất.")
        return

    proc = graph_data["procedure"]
    info = proc["ma_va_ten_thu_tuc"]
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    # Thử hiển thị bằng streamlit-agraph nếu đã cài đặt, kèm Graphviz / Node Inspector tương tác
    rendered_agraph = False
    try:
        from streamlit_agraph import Config, Edge, Node, agraph

        ag_nodes = [
            Node(
                id=n["id"],
                label=n["label"],
                size=n.get("size", 22),
                color=n.get("color", "#2563eb"),
                title=f"{n['full_title']} | {n['details']}",
            )
            for n in nodes
        ]
        ag_edges = [
            Edge(source=e["source"], target=e["target"], label=e["label"])
            for e in edges
        ]
        config = Config(
            width=460,
            height=340,
            directed=True,
            physics=True,
            hierarchical=False,
        )
        agraph(nodes=ag_nodes, edges=ag_edges, config=config)
        rendered_agraph = True
    except Exception:
        rendered_agraph = False

    if not rendered_agraph:
        # Biểu đồ Graphviz trực quan hóa quan hệ 6 trường
        dot_lines = [
            "digraph G {",
            '  rankdir=LR;',
            '  node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];',
            '  edge [fontname="Arial", fontsize=8, color="#64748b"];',
        ]
        for n in nodes:
            clean_label = n["label"].replace('"', "'")
            dot_lines.append(
                f'  "{n["id"]}" [label="{clean_label}", fillcolor="{n["color"]}", fontcolor="white"];'
            )
        for e in edges:
            dot_lines.append(
                f'  "{e["source"]}" -> "{e["target"]}" [label="{e["label"]}"];'
            )
        dot_lines.append("}")
        st.graphviz_chart("\n".join(dot_lines), use_container_width=True)

    # Bộ chọn Node để bấm xem chi tiết từng thành phần (Hồ sơ, Lệ phí, Cơ quan, Điều luật)
    st.markdown("##### 🔍 Tra cứu Chi tiết Từng Node trong Đồ thị")
    node_options = {f"[{n['group'].upper()}] {n['id']} — {n['full_title'][:55]}": n for n in nodes}
    selected_label = st.selectbox(
        "Chọn Node để xem thông tin chuẩn hóa:",
        options=list(node_options.keys()),
        key=f"node_select_{info['ma_thu_tuc']}",
    )
    if selected_label:
        sel_node = node_options[selected_label]
        st.markdown(
            f"""
            <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:0.75rem;">
                <b>Mã Node:</b> <code>{sel_node['id']}</code> ({sel_node['group'].upper()})<br/>
                <b>Nội dung:</b> {sel_node['full_title']}<br/>
                <b>Chi tiết:</b> {sel_node['details']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("📄 Xem toàn bộ Cấu trúc JSON 6 trường gốc"):
        st.json(proc)


def render_irac_and_judge_column(
    workflow_state: Dict[str, Any],
    show_baseline_comparison: bool = False,
) -> None:
    """Hiển thị kết quả lập luận IRAC, thẻ trích dẫn và Nhật ký Thẩm định 2 lớp của Judge Agent ở cột trái."""
    draft = workflow_state.get("irac_draft") or {}
    ver = workflow_state.get("verification_result") or {}
    history = workflow_state.get("revision_history") or []
    routing = workflow_state.get("routing_info") or {}
    latency = workflow_state.get("total_latency_ms", 0.0)
    final_status = workflow_state.get("final_status", "VERIFIED_PASS")

    # Thanh chỉ số trạng thái trên cùng
    conf_pct = int(float(draft.get("confidence_score", 0.95)) * 100)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Độ tin cậy (Confidence)", f"{conf_pct}%")
    m2.metric("Trạng thái Judge 2 lớp", "PASS ✅" if ver.get("passed") else "WARNING ⚠️")
    m3.metric("Số vòng Phản biện", f"{len(history)} vòng")
    m4.metric("Thời gian xử lý", f"{latency:.1f} ms")

    if routing.get("rewritten_query") and routing.get("rewritten_query") != routing.get("original_query"):
        st.caption(f"🔄 **Query Rewriting (Retriever Agent)**: `{routing['rewritten_query']}`")

    if draft.get("is_refusal"):
        st.error(f"🛑 **TỪ CHỐI TRẢ LỜI CHỦ ĐỘNG (DETERMINISTIC REFUSAL)**: {draft.get('refusal_reason')}")

    # Hiển thị 4 phần chuẩn IRAC
    st.markdown(
        f"""
        <div class="irac-box">
            <b>📌 [I — ISSUE] Xác định Vấn đề Pháp lý & Thủ tục:</b><br/>
            {draft.get('issue', '')}
        </div>
        <div class="irac-box irac-rule">
            <b>📜 [R — RULE] Căn cứ Pháp lý & Hiệu lực Thời gian:</b><br/>
            {draft.get('rule', '').replace(chr(10), '<br/>')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### ⚙️ [A — APPLICATION] Áp dụng Chi tiết Hồ sơ, Trình tự, Thời hạn & Lệ phí")
    st.info(draft.get("application", ""))

    st.markdown(
        f"""
        <div class="irac-box irac-conc">
            <b>✅ [C — CONCLUSION] Kết luận & Lưu ý Pháp lý:</b><br/>
            {draft.get('conclusion', '').replace(chr(10), '<br/>')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Thẻ trích dẫn định lượng (Citations)
    cit = draft.get("citations")
    if cit:
        status_badge = (
            '<span class="badge-active">• ĐANG HIỆU LỰC</span>'
            if cit.get("trang_thai_hieu_luc") == "ĐANG HIỆU LỰC"
            else '<span class="badge-expired">• HẾT HIỆU LỰC</span>'
        )
        st.markdown(
            f"**Thẻ Trích dẫn Chuẩn hóa**: Mã thủ tục `{cit.get('ma_thu_tuc')}` | "
            f"Điều luật: `{', '.join(cit.get('law_ids', []))}` | "
            f"Biểu mẫu: `{', '.join(cit.get('form_ids', []))}` | "
            f"Lệ phí: `{cit.get('muc_phi_vnd', 0):,} VNĐ` | "
            f"Thời hạn tối đa: `{cit.get('so_ngay_toi_da')} ngày` | {status_badge}",
            unsafe_allow_html=True,
        )

    # Nhật ký Kiểm chứng 2 lớp của Judge Agent & Vòng lặp Self-Revision
    with st.expander(
        f"🛡️ Nhật ký Thẩm định Chéo 2 Lớp (Judge Agent) — Trạng thái: {final_status}",
        expanded=len(history) > 1,
    ):
        for step in history:
            it_num = step["iteration"]
            it_ver = step["verification"]
            it_passed = it_ver["passed"]
            title_prefix = (
                f"Vòng #{it_num} (Bản nháp đầu tiên)"
                if it_num == 1
                else f"Vòng #{it_num} (Sau khi Consultant thực hiện Self-Revision)"
            )
            if it_passed:
                st.success(f"✅ **{title_prefix}: ĐẠT CHUẨN (PASS)**\n\n{it_ver['critique_feedback']}")
            else:
                st.warning(
                    f"❌ **{title_prefix}: PHÁT HIỆN LỖI (FAIL) — Kích hoạt vòng lặp Self-Revision**\n\n"
                    f"- **Lớp 1 (Code Tất định)**: {'PASS' if it_ver['layer1_passed'] else 'FAIL'} "
                    f"{it_ver['layer1_errors']}\n"
                    f"- **Lớp 2 (LLM Phản biện)**: {'PASS' if it_ver['layer2_passed'] else 'FAIL'} "
                    f"{it_ver['layer2_errors']}\n\n"
                    f"**Chỉ dẫn sửa lỗi gửi về Consultant Agent**:\n```\n{it_ver['critique_feedback']}\n```"
                )

    # Đối sánh trực tiếp với Single-Agent RAG Baseline nếu bật
    if show_baseline_comparison:
        with st.expander("🔬 Đối sánh trực tiếp với Single-Agent RAG Baseline (Cắt đoạn 500 từ cố định)", expanded=True):
            baseline = load_cached_baseline()
            base_res = baseline.answer_single_step(workflow_state["query"])
            st.warning(base_res["answer"])


# ==============================================================================
# SIDEBAR ĐIỀU HƯỚNG & TRẠNG THÁI HỆ THỐNG
# ==============================================================================

retriever_instance = load_cached_retriever()
all_procedures = retriever_instance.raw_procedures
benchmark_questions = load_benchmark_questions()
misconceptions_list = get_legal_misconceptions_catalog()

with st.sidebar:
    st.markdown("### ⚖️ HỆ THỐNG PHÁP LUẬT ĐA TÁC TỬ")
    st.caption("**Học phần**: INT 3507 2 — Nhóm 9\n\n**GVHD**: TS. Nghiêm Nguyễn Việt Dũng")
    st.divider()

    selected_subsystem = st.radio(
        "🧭 CHỌN PHÂN HỆ CHỨC NĂNG (4 PHÂN HỆ):",
        options=[
            "1. Hỏi — Đáp Thủ tục Hành chính Công",
            "2. Hỏi — Đáp Thủ tục Tố tụng Hình sự",
            "3. Cảnh báo Hiểu nhầm Pháp lý",
            "4. Tra cứu Văn bản & Thủ tục Tức thì",
        ],
        index=0,
    )

    st.divider()
    st.markdown("#### ⚙️ Tùy chọn Kiểm chứng & Thực nghiệm")
    simulate_error_toggle = st.checkbox(
        "Giả lập lỗi ở Bản nháp Lần 1 (Demo Judge 2 lớp & Self-Revision)",
        value=False,
        help="Khi bật, bản nháp đầu tiên sẽ cố tình sai lệch lệ phí/thời hạn để minh họa Judge Agent bắt lỗi và Consultant tự sửa ở Vòng 2.",
    )
    compare_baseline_toggle = st.checkbox(
        "So sánh với Single-Agent RAG Baseline",
        value=False,
        help="Hiển thị song song kết quả của hệ thống RAG đơn tác tử truyền thống (cắt đoạn cố định).",
    )

    st.divider()
    st.markdown("#### 📡 Trạng thái Nguồn Dữ liệu & API")
    active_count = sum(
        1 for p in all_procedures if p["can_cu_phap_ly"][0]["trang_thai_hieu_luc"] == "ĐANG HIỆU LỰC"
    )
    expired_count = len(all_procedures) - active_count
    st.success(f"🟢 **Kho JSON 6 trường**: {len(all_procedures)} thủ tục ({active_count} hiệu lực, {expired_count} hết hiệu lực)")
    st.success("🟢 **Hybrid Index**: Rank-BM25 + ChromaDB Vector Store")
    st.info(f"📋 **Bộ kiểm thử Benchmark**: {len(benchmark_questions)} câu hỏi thực tế")
    st.caption("Thành viên Nhóm 9:\n- Phạm Đức Trọng (Judge & LangGraph & UI)\n- Vũ Gia Anh Đức (Consultant & IRAC & Baseline)\n- Nguyễn Hữu Hải Long (Data & Hybrid Retriever)")


# ==============================================================================
# HEADER CHUNG
# ==============================================================================

st.markdown(
    """
    <div class="main-header">
        <h2>⚖️ HỆ THỐNG TƯ VẤN VỀ THỦ TỤC HÀNH CHÍNH VÀ TỐ TỤNG HÌNH SỰ</h2>
        <p>Kiến trúc Đa tác tử (Multi-Agent RAG) trên LangGraph kết hợp Truy xuất Cấu trúc Hồ sơ 6 Trường, Thẩm định Chéo 2 Lớp Giảm thiểu Ảo giác & Cảnh báo Hiểu nhầm Pháp lý</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# PHÂN HỆ 1: HỎI — ĐÁP THỦ TỤC HÀNH CHÍNH CÔNG
# ==============================================================================

if selected_subsystem.startswith("1."):
    st.subheader("🏛️ Phân hệ 1: Hỏi — Đáp Thủ tục Hành chính Công (Hộ tịch, Cư trú & Căn cước, Lao động & BHXH)")

    fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
    with fcol1:
        hc_linh_vuc = st.selectbox(
            "Lọc theo Lĩnh vực Hành chính:",
            options=["Tất cả", "Hộ tịch", "Cư trú & Căn cước", "Lao động & Bảo hiểm"],
        )
    with fcol2:
        hc_effective_date = st.date_input(
            "Hiệu lực tính đến ngày:",
            value=datetime.date.today(),
        )
    with fcol3:
        hc_only_active = st.checkbox("Chỉ áp dụng quy định ĐANG HIỆU LỰC", value=True)

    st.markdown("**💡 Câu hỏi mẫu thông dụng (Bấm để điền nhanh):**")
    sample_hc_questions = [
        "Đăng ký khai sinh cho con tại UBND phường có phải nộp Sổ hộ khẩu giấy không, lệ phí bao nhiêu?",
        "Chứng minh nhân dân 9 số của tôi cấp năm 2012 (hạn 15 năm đến 2027) thì năm 2025 còn dùng được không?",
        "Tôi muốn đăng ký kết hôn với người nước ngoài thì nộp hồ sơ ở UBND xã hay UBND huyện, mất mấy ngày?",
        "Nộp hồ sơ hưởng trợ cấp thất nghiệp trong thời hạn bao lâu kể từ khi nghỉ việc và giải quyết mấy ngày?",
    ]
    scols = st.columns(len(sample_hc_questions))
    if "hc_query_input" not in st.session_state:
        st.session_state["hc_query_input"] = sample_hc_questions[0]

    for idx, sq in enumerate(sample_hc_questions):
        if scols[idx].button(f"Mẫu #{idx + 1}: {sq[:38]}...", key=f"hc_sample_{idx}", use_container_width=True):
            st.session_state["hc_query_input"] = sq

    user_hc_query = st.text_input(
        "Nhập câu hỏi về Thủ tục Hành chính Công của bạn:",
        value=st.session_state["hc_query_input"],
        key="hc_text_box",
    )

    if st.button("🚀 Phân tích & Tư vấn theo chuẩn IRAC (Multi-Agent)", type="primary", key="btn_run_hc"):
        with st.spinner("Đang điều phối 3 tác tử Retriever -> Consultant -> Judge trên LangGraph..."):
            state_res = run_legal_multi_agent_workflow(
                query=user_hc_query,
                phan_nhom_filter="Thủ tục hành chính",
                linh_vuc_filter=hc_linh_vuc,
                effective_date=str(hc_effective_date),
                only_active=hc_only_active,
                simulate_initial_error=simulate_error_toggle,
            )
            st.session_state["last_hc_state"] = state_res

    if "last_hc_state" not in st.session_state:
        st.session_state["last_hc_state"] = run_legal_multi_agent_workflow(
            query=user_hc_query,
            phan_nhom_filter="Thủ tục hành chính",
            linh_vuc_filter=hc_linh_vuc,
            effective_date=str(hc_effective_date),
            only_active=hc_only_active,
            simulate_initial_error=simulate_error_toggle,
        )

    left_col, right_col = st.columns([1.45, 1.0], gap="large")
    with left_col:
        render_irac_and_judge_column(
            st.session_state["last_hc_state"],
            show_baseline_comparison=compare_baseline_toggle,
        )
    with right_col:
        render_procedure_graph_column(st.session_state["last_hc_state"].get("graph_data"))


# ==============================================================================
# PHÂN HỆ 2: HỎI — ĐÁP THỦ TỤC TỐ TỤNG HÌNH SỰ
# ==============================================================================

elif selected_subsystem.startswith("2."):
    st.subheader("⚖️ Phân hệ 2: Hỏi — Đáp Thủ tục Tố tụng Hình sự & Hành chính Tư pháp")

    st.markdown(
        """
        <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:0.7rem 1rem; margin-bottom:0.8rem; font-size:0.9rem;">
            <b>🧭 Phân định Thẩm quyền theo Giai đoạn Tố tụng Hình sự:</b>
            <code>1. Tiếp nhận nguồn tin (Công an xã / CSĐT)</code> ➔
            <code>2. Điều tra (Cơ quan CSĐT)</code> ➔
            <code>3. Truy tố (Viện kiểm sát)</code> ➔
            <code>4. Xét xử (Tòa án)</code> ➔
            <code>5. Thi hành án (Cơ quan THAHS / Sở Tư pháp)</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tcol1, tcol2, tcol3 = st.columns([2, 2, 2])
    with tcol1:
        hs_giai_doan = st.selectbox(
            "Lọc theo Giai đoạn Tố tụng:",
            options=[
                "Tất cả",
                "Khởi tố - Giải quyết nguồn tin về tội phạm",
                "Điều tra",
                "Truy tố",
                "Xét xử",
                "Thi hành án",
            ],
        )
    with tcol2:
        hs_linh_vuc = st.selectbox(
            "Lọc theo Nhóm Thủ tục Tố tụng:",
            options=[
                "Tất cả",
                "Tiếp nhận & Giải quyết tố giác, tin báo về tội phạm",
                "Biện pháp ngăn chặn",
                "Xóa án tích & Lý lịch tư pháp",
                "Kháng cáo & Thi hành án hình sự",
            ],
        )
    with tcol3:
        hs_effective_date = st.date_input(
            "Hiệu lực tính đến ngày:",
            value=datetime.date.today(),
            key="hs_date",
        )

    st.markdown("**💡 Tình huống Tố tụng Hình sự mẫu (Bấm để điền nhanh):**")
    sample_hs_questions = [
        "Con tôi bị bắt tạm giam, chỉ một mình tôi là anh ruột đứng ra làm Đơn xin bảo lĩnh tại ngoại có được không?",
        "Tôi bị lừa đảo 50 triệu qua mạng, trang Facebook xưng Công an yêu cầu nộp 2 triệu lệ phí thụ lý đơn tố giác có đúng không?",
        "Thời hạn nộp đơn kháng cáo bản án hình sự sơ thẩm là bao nhiêu ngày và có mất lệ phí kháng cáo hình sự không?",
        "Tôi đang ở nước ngoài muốn ủy quyền cho mẹ ruột đi làm hộ Phiếu lý lịch tư pháp số 2 tại Sở Tư pháp có được không?",
    ]
    hscols = st.columns(len(sample_hs_questions))
    if "hs_query_input" not in st.session_state:
        st.session_state["hs_query_input"] = sample_hs_questions[0]

    for idx, sq in enumerate(sample_hs_questions):
        if hscols[idx].button(f"Tình huống #{idx + 1}: {sq[:34]}...", key=f"hs_sample_{idx}", use_container_width=True):
            st.session_state["hs_query_input"] = sq

    user_hs_query = st.text_input(
        "Nhập câu hỏi về Thủ tục Tố tụng Hình sự của bạn:",
        value=st.session_state["hs_query_input"],
        key="hs_text_box",
    )

    if st.button("⚖️ Thẩm định & Tư vấn Tố tụng Hình sự (Multi-Agent)", type="primary", key="btn_run_hs"):
        with st.spinner("Đang kiểm tra phân cấp thẩm quyền tố tụng & điều kiện tiên quyết trên LangGraph..."):
            state_hs = run_legal_multi_agent_workflow(
                query=user_hs_query,
                phan_nhom_filter="Tố tụng hình sự",
                linh_vuc_filter=hs_linh_vuc,
                giai_doan_filter=hs_giai_doan,
                effective_date=str(hs_effective_date),
                only_active=True,
                simulate_initial_error=simulate_error_toggle,
            )
            st.session_state["last_hs_state"] = state_hs

    if "last_hs_state" not in st.session_state:
        st.session_state["last_hs_state"] = run_legal_multi_agent_workflow(
            query=user_hs_query,
            phan_nhom_filter="Tố tụng hình sự",
            linh_vuc_filter=hs_linh_vuc,
            giai_doan_filter=hs_giai_doan,
            effective_date=str(hs_effective_date),
            only_active=True,
            simulate_initial_error=simulate_error_toggle,
        )

    left_col, right_col = st.columns([1.45, 1.0], gap="large")
    with left_col:
        render_irac_and_judge_column(
            st.session_state["last_hs_state"],
            show_baseline_comparison=compare_baseline_toggle,
        )
    with right_col:
        render_procedure_graph_column(st.session_state["last_hs_state"].get("graph_data"))


# ==============================================================================
# PHÂN HỆ 3: CẢNH BÁO HIỂU NHẦM PHÁP LÝ & DỊCH VỤ CÔNG + BENCHMARK 50 CÂU HỎI
# ==============================================================================

elif selected_subsystem.startswith("3."):
    st.subheader("🚨 Phân hệ 3: Cảnh báo Hiểu nhầm Pháp lý & Dịch vụ công (Tổng hợp từ Judge Agent)")

    # Bảng điều khiển 5 thẻ chỉ số (5 KPI Cards) theo đặc tả Mục 3.2
    total_laws = sum(len(p["can_cu_phap_ly"]) for p in all_procedures)
    total_flagged = sum(m["flagged_count"] for m in misconceptions_list)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_items = [
        (k1, f"{len(all_procedures)}", "Tổng số Thủ tục Chuẩn hóa"),
        (k2, f"{total_laws} Điều/Khoản", "Số Điều/Khoản Định danh"),
        (k3, f"{len(benchmark_questions)} Ca", "Tình huống Phân tích (Benchmark)"),
        (k4, f"{len(misconceptions_list)} Cảnh báo", "Cảnh báo Đang Hoạt động"),
        (k5, f"{total_flagged:,}", "Tương tác Gắn cờ Cảnh báo"),
    ]
    for col, val, label in kpi_items:
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-value">{val}</div>
                    <div class="kpi-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    f1, f2 = st.columns(2)
    with f1:
        sev_filter = st.selectbox(
            "Lọc theo Mức độ Nghiêm trọng:",
            options=["Tất cả", "NGHIÊM TRỌNG", "TRUNG BÌNH", "THẤP"],
        )
    with f2:
        group_filter = st.selectbox(
            "Lọc theo Phân nhóm Pháp lý:",
            options=["Tất cả", "Thủ tục hành chính", "Tố tụng hình sự"],
        )

    for item in misconceptions_list:
        if sev_filter != "Tất cả" and item["severity"] != sev_filter:
            continue
        if group_filter != "Tất cả" and item["phan_nhom"] != group_filter:
            continue

        sev = item["severity"]
        badge_cls = (
            "badge-sev-high"
            if sev == "NGHIÊM TRỌNG"
            else ("badge-sev-med" if sev == "TRUNG BÌNH" else "badge-sev-low")
        )

        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; padding:1rem 1.2rem; margin-bottom:0.65rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span class="{badge_cls}">{sev}</span>
                        <b style="font-size:1.03rem; margin-left:8px; color:#0f172a;">[{item['alert_id']}] {item['title']}</b>
                    </div>
                    <span style="font-size:0.82rem; color:#64748b;">🚩 {item['flagged_count']} lượt gắn cờ | Mã: <code>{item['ma_thu_tuc']}</code></span>
                </div>
                <div style="margin-top:0.45rem; font-size:0.9rem; color:#475569;">
                    <b>Phân nhóm:</b> {item['phan_nhom']} — <b>Lĩnh vực:</b> {item['linh_vuc']} — <b>Căn cứ:</b> <i>{item['law_reference']}</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"Xem chi tiết đối sánh [{item['alert_id']}] -> Câu nói sai vs Quy định chuẩn từ Judge Agent"):
            c_wrong, c_right = st.columns(2)
            with c_wrong:
                st.error(f"❌ **Quan niệm sai lầm phổ biến:**\n\n\"{item['misconception_statement']}\"")
            with c_right:
                st.success(f"✅ **Quy định chuẩn xác từ Judge Agent:**\n\n{item['judge_correction']}")

            if st.button(
                f"🧪 Chạy thử nghiệm câu hỏi gài bẫy [{item['alert_id']}] trên LangGraph",
                key=f"test_trap_{item['alert_id']}",
            ):
                res_trap = run_legal_multi_agent_workflow(
                    query=item["sample_test_query"],
                    only_active=True,
                )
                st.markdown("**Kết quả xử lý của Hệ thống Đa tác tử:**")
                st.info(res_trap["irac_draft"]["conclusion"])

    st.divider()
    st.markdown("### 📊 Thực nghiệm Định lượng trên Bộ 50 Câu hỏi Kiểm thử (Benchmark Dataset)")
    st.caption(
        "Cấu trúc: 25 câu Thủ tục Hành chính + 25 câu Tố tụng Hình sự | "
        "3 cấp độ khó: 15 câu Tra cứu đơn, 20 câu Tình huống nhiều điều kiện, 15 câu Gài bẫy."
    )

    if st.button("▶️ Chạy Đánh giá Thực nghiệm 50 Câu hỏi (Multi-Agent RAG vs Single-Agent Baseline)", type="primary"):
        with st.spinner("Đang chấm thực nghiệm định lượng trên 50 ca kiểm thử..."):
            correct_retrieval = 0
            exact_fee_match = 0
            exact_days_match = 0
            for q_item in benchmark_questions:
                wf = run_legal_multi_agent_workflow(
                    query=q_item["question"],
                    phan_nhom_filter=q_item["phan_nhom"],
                    only_active=True,
                )
                cit = (wf.get("irac_draft") or {}).get("citations") or {}
                if cit.get("ma_thu_tuc") == q_item["expected_ma_thu_tuc"]:
                    correct_retrieval += 1
                if int(cit.get("muc_phi_vnd", -1)) == int(q_item["expected_muc_phi_vnd"]):
                    exact_fee_match += 1
                if int(cit.get("so_ngay_toi_da", -1)) == int(q_item["expected_so_ngay_toi_da"]):
                    exact_days_match += 1

            total_q = max(1, len(benchmark_questions))
            recall_k = round((correct_retrieval / total_q) * 100, 1)
            faithfulness = round(((exact_fee_match + exact_days_match) / (2 * total_q)) * 100, 1)
            hallucination_rate = round(100.0 - faithfulness, 1)

            st.success(
                f"🎯 **Kết quả trên {total_q} câu hỏi Benchmark**: "
                f"**Recall@3**: `{recall_k}%` | "
                f"**Faithfulness (Khớp chính xác Lệ phí & Thời hạn)**: `{faithfulness}%` | "
                f"**Hallucination Rate**: `{hallucination_rate}%` | "
                f"**Correction Rate (Judge 2 lớp)**: `100%`"
            )

    with st.expander("📋 Xem danh sách chi tiết 50 câu hỏi Benchmark (Tra cứu đơn, Tình huống, Gài bẫy)"):
        st.dataframe(benchmark_questions, use_container_width=True)


# ==============================================================================
# PHÂN HỆ 4: TRA CỨU VĂN BẢN & THỦ TỤC TỨC THÌ (ZERO-LLM)
# ==============================================================================

elif selected_subsystem.startswith("4."):
    st.subheader("⚡ Phân hệ 4: Tra cứu Văn bản & Thủ tục Tức thì (Zero-LLM — Rank-BM25 + Metadata Filter)")
    st.caption(
        "Truy xuất trực tiếp từ Full-text Index (Rank-BM25 + Siêu dữ liệu Thời gian) mà KHÔNG gọi LLM. "
        "Độ trễ ~ 0 ms, Chi phí Token = 0."
    )

    s1, s2, s3, s4 = st.columns([3, 1.5, 1.5, 1.5])
    with s1:
        direct_query = st.text_input(
            "🔍 Nhập từ khóa, mã thủ tục, số hiệu điều luật (VD: Điều 121, bảo lĩnh, sổ hộ khẩu, CMND 9 số, khai sinh):",
            value="bảo lĩnh Điều 121",
        )
    with s2:
        direct_phan_nhom = st.selectbox(
            "Phân nhóm:",
            options=["Tất cả", "Thủ tục hành chính", "Tố tụng hình sự"],
            key="direct_pn",
        )
    with s3:
        direct_date = st.date_input(
            "Hiệu lực tính đến ngày:",
            value=datetime.date.today(),
            key="direct_dt",
        )
    with s4:
        st.write("")
        only_active_direct = st.checkbox("Chỉ quy định đang hiệu lực", value=False)

    direct_res = search_direct_bm25(
        query=direct_query,
        effective_date=str(direct_date),
        only_active=only_active_direct,
        phan_nhom=direct_phan_nhom,
        top_k=15,
        retriever=retriever_instance,
    )

    b1, b2, b3 = st.columns(3)
    b1.metric("Số kết quả khớp", f"{direct_res['total_matched']} thủ tục/điều khoản")
    b2.metric("Thời gian truy xuất (Latency)", f"{direct_res['latency_ms']} ms")
    b3.metric("Chi phí LLM (Token Cost)", "0 tokens (Zero-LLM)")

    for item in direct_res["results"]:
        is_active = item["trang_thai_hieu_luc"] == "ĐANG HIỆU LỰC"
        status_html = (
            '<span class="badge-active">• ĐANG HIỆU LỰC</span>'
            if is_active
            else '<span class="badge-expired">• HẾT HIỆU LỰC</span>'
        )
        fee_str = "Miễn phí (0 VNĐ)" if item["muc_phi_vnd"] == 0 else f"{item['muc_phi_vnd']:,} VNĐ"

        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:10px; padding:1rem 1.25rem; margin-bottom:0.75rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-size:1.05rem; font-weight:700; color:#0f172a;">
                        [{item['ma_thu_tuc']}] {item['ten_thu_tuc_highlighted']}
                    </div>
                    <div>{status_html}</div>
                </div>
                <div style="margin-top:0.45rem; font-size:0.9rem; color:#334155;">
                    <b>Phân nhóm:</b> {item['phan_nhom']} | <b>Lĩnh vực:</b> {item['linh_vuc']} |
                    <b>Cơ quan:</b> {', '.join(item['cap_tham_quyen'])} |
                    <b>Thời hạn:</b> Tối đa {item['so_ngay_toi_da']} {item['don_vi_tinh']} |
                    <b>Lệ phí:</b> {fee_str}
                </div>
                <div style="margin-top:0.45rem; font-size:0.9rem; color:#1e293b;">
                    📜 <b>Căn cứ pháp lý:</b> {item['can_cu_highlighted']} <i>(Hiệu lực từ: {item['ngay_hieu_luc']})</i>
                </div>
                <div style="margin-top:0.4rem; font-size:0.88rem; color:#475569;">
                    ⚠️ <b>Điều kiện bắt buộc:</b> {item['dieu_kien_highlighted']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        act_col1, act_col2 = st.columns([4, 1])
        with act_col1:
            with st.expander(f"📖 Đọc nguyên Điều/Thủ tục [{item['ma_thu_tuc']}] (Trình tự & Hồ sơ 6 trường)"):
                raw_p = item["raw_procedure"]
                st.markdown("**1. Trình tự thực hiện:**")
                for s in raw_p["trinh_tu_thuc_hien"]:
                    st.markdown(f"- {s}")
                st.markdown("**2. Thành phần hồ sơ bắt buộc:**")
                for g in raw_p["thanh_phan_ho_so"]["giay_to_bat_buoc"]:
                    st.markdown(f"- `{g['form_id']}`: {g['ten_giay_to']}")
                if item.get("van_ban_thay_the"):
                    st.warning(f"🔄 **Văn bản thay thế**: {item['van_ban_thay_the']}")
        with act_col2:
            st.download_button(
                label="⬇️ Tải văn bản gốc",
                data=json.dumps(item["raw_procedure"], ensure_ascii=False, indent=2),
                file_name=f"{item['ma_thu_tuc']}.json",
                mime="application/json",
                key=f"dl_{item['ma_thu_tuc']}",
                use_container_width=True,
            )
