"""
APCR-RAG — ADMINISTRATIVE & PENAL CRIMINAL RAG
Cổng tư vấn Thủ tục Hành chính & Tố tụng Hình sự (Multi-Agent RAG) — Nhóm 9, INT 3507 2

Giao diện tông Đỏ — Trắng, gồm 4 chức năng chính:
  1. Hỏi — Đáp Thủ tục Hành chính công   (Retriever -> Consultant IRAC -> Judge 2 lớp)
  2. Hỏi — Đáp Thủ tục Tố tụng Hình sự   (phân định thẩm quyền theo giai đoạn tố tụng)
  3. Cảnh báo hiểu nhầm pháp lý          (danh mục từ Judge Agent + đánh giá Benchmark 50 câu)
  4. Tra cứu văn bản & thủ tục tức thì   (Zero-LLM: Rank-BM25 + lọc hiệu lực thời gian)
"""

from __future__ import annotations

import datetime
import html as html_lib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.consultant_agent import extract_procedure_graph_data
from src.judge_agent import get_legal_misconceptions_catalog
from src.langgraph_workflow import run_legal_multi_agent_workflow
from src.retriever_agent import get_retriever_agent, search_direct_bm25

BENCHMARK_PATH = Path(__file__).resolve().parent / "data" / "benchmark_50_questions.json"

st.set_page_config(
    page_title="APCR-RAG | Tư vấn Thủ tục Hành chính & Tố tụng Hình sự",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==============================================================================
# TIỆN ÍCH
# ==============================================================================

def h(markup: str) -> None:
    """Render HTML: bỏ thụt lề từng dòng để Markdown không hiểu nhầm thành khối code."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(text: Any) -> str:
    return html_lib.escape(str(text or ""))


def fmt_fee(vnd: int) -> str:
    return "Miễn phí (0 đ)" if int(vnd) == 0 else f"{int(vnd):,} đ".replace(",", ".")


@st.cache_resource
def load_retriever():
    return get_retriever_agent()


@st.cache_data
def load_benchmark() -> List[Dict[str, Any]]:
    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


retriever = load_retriever()
ALL_PROCS: List[Dict[str, Any]] = retriever.raw_procedures
BENCHMARK = load_benchmark()
MISCONCEPTIONS = get_legal_misconceptions_catalog()

PAGES = {
    "home": "Trang chủ",
    "hanh_chinh": "Thủ tục hành chính",
    "to_tung": "Tố tụng hình sự",
    "canh_bao": "Cảnh báo hiểu nhầm",
    "tra_cuu": "Tra cứu văn bản",
    "huong_dan": "Hướng dẫn sử dụng",
}
current_page = st.query_params.get("page", "home")
if current_page not in PAGES:
    current_page = "home"


# ==============================================================================
# CSS — TÔNG ĐỎ / TRẮNG
# ==============================================================================

h(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap');
:root {
  --red: #B91C1C; --red-dark: #7F1D1D; --red-soft: #FEE2E2; --red-bg: #FFF5F5;
  --gold: #D97706; --ink: #1F2937; --muted: #6B7280; --line: #F1D5D5;
}
html, body, [class*="css"], .stMarkdown, button, input, textarea {
  font-family: 'Be Vietnam Pro', 'Segoe UI', Roboto, sans-serif !important;
}
header[data-testid="stHeader"], section[data-testid="stSidebar"],
div[data-testid="collapsedControl"], #MainMenu, footer { display: none !important; }
.stApp { background: #ffffff; }
.block-container { padding: 0 2.4rem 0 2.4rem !important; max-width: 1280px !important; }

/* ---------- HEADER ---------- */
.apcr-header { display:flex; justify-content:space-between; align-items:center;
  padding: 16px 4px 14px 4px; border-bottom: 3px solid var(--red); }
.apcr-brand { display:flex; align-items:center; gap:14px; }
.apcr-logo { width:52px; height:52px; border-radius:14px;
  background: linear-gradient(145deg, #DC2626 0%, #7F1D1D 100%);
  display:flex; align-items:center; justify-content:center; font-size:26px;
  box-shadow: 0 4px 12px rgba(185,28,28,.30); color:#fff; }
.apcr-name { font-size:22px; font-weight:800; color:var(--red); letter-spacing:.02em; line-height:1.1; }
.apcr-tag { font-size:12.5px; font-weight:600; color:var(--red-dark); text-transform:uppercase; letter-spacing:.03em; }
.apcr-team { border:1.5px solid var(--red); color:var(--red); border-radius:999px;
  padding:6px 16px; font-size:12.5px; font-weight:600; background:#fff; }

/* ---------- NAV ---------- */
.apcr-nav { display:flex; gap:6px; padding:10px 0; border-bottom:1px solid var(--line); flex-wrap:wrap; }
.apcr-nav a { text-decoration:none !important; color:var(--red-dark) !important; font-weight:700;
  font-size:12.5px; padding:7px 14px; border-radius:8px; text-transform:uppercase; letter-spacing:.02em; }
.apcr-nav a:hover { background:var(--red-bg); }
.apcr-nav a.active { background:var(--red); color:#fff !important; box-shadow:0 2px 8px rgba(185,28,28,.25); }

/* ---------- HERO ---------- */
.apcr-hero { position:relative; overflow:hidden; margin:0 -2.4rem; padding:46px 3.2rem;
  background: linear-gradient(100deg, #FFF7F5 0%, #FFFFFF 45%, #FEE2E2 100%);
  display:flex; justify-content:space-between; align-items:center; min-height:250px; }
.apcr-hero h1 { font-size:34px; font-weight:800; color:#111827; margin:0 0 10px 0; letter-spacing:.01em; line-height:1.2; }
.apcr-hero h1 span { color:var(--red); }
.apcr-hero p { font-size:15px; color:#4B5563; max-width:560px; line-height:1.6; margin:0; }
.hero-chips { margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; }
.hero-chip { background:#fff; border:1px solid var(--line); color:var(--red-dark);
  border-radius:999px; padding:5px 12px; font-size:12px; font-weight:600; }

/* ---------- HOME FUNCTION CARDS ---------- */
.section-title { font-size:15px; font-weight:800; color:var(--red-dark); text-transform:uppercase;
  letter-spacing:.04em; margin:28px 0 14px 0; display:flex; align-items:center; gap:10px; }
.section-title::before { content:""; width:5px; height:20px; background:var(--red); border-radius:3px; }
.fn-grid { display:grid; grid-template-columns: 1fr 1fr; gap:18px; }
a.fn-card { display:flex; align-items:center; gap:18px; text-decoration:none !important;
  background:#fff; border:1px solid #F3E3E3; border-radius:14px; padding:24px 22px;
  box-shadow:0 2px 10px rgba(127,29,29,.05); transition:all .18s ease; }
a.fn-card:hover { border-color:var(--red); box-shadow:0 8px 22px rgba(185,28,28,.14); transform:translateY(-2px); }
.fn-icon { min-width:54px; height:54px; border-radius:12px; background:var(--red-soft);
  display:flex; align-items:center; justify-content:center; font-size:26px; }
.fn-body { flex:1; }
.fn-title { font-size:16.5px; font-weight:700; color:#111827; margin-bottom:5px; }
.fn-desc { font-size:13.5px; color:var(--muted); line-height:1.55; }
.fn-arrow { color:var(--red); font-size:22px; font-weight:700; }

.stat-strip { display:grid; grid-template-columns: repeat(4, 1fr); gap:14px; }
.stat-box { background:var(--red-bg); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
.stat-num { font-size:26px; font-weight:800; color:var(--red); line-height:1.1; }
.stat-lbl { font-size:12.5px; color:#6B7280; margin-top:4px; font-weight:500; }

.flow { display:flex; align-items:stretch; gap:10px; }
.flow-step { flex:1; background:#fff; border:1px dashed #F0B4B4; border-radius:12px; padding:14px 16px; }
.flow-step b { color:var(--red); font-size:13px; text-transform:uppercase; letter-spacing:.03em; }
.flow-step div { font-size:13px; color:#4B5563; margin-top:4px; line-height:1.5; }
.flow-arrow { display:flex; align-items:center; color:var(--red); font-size:22px; font-weight:800; }

/* ---------- PAGE BANNER ---------- */
.page-banner { margin:0 -2.4rem 20px -2.4rem; padding:22px 3.2rem;
  background: linear-gradient(90deg, #991B1B 0%, #B91C1C 55%, #DC2626 100%); color:#fff; }
.crumb { font-size:12px; opacity:.85; margin-bottom:6px; }
.crumb a { color:#fff !important; text-decoration:none !important; }
.page-banner h2 { margin:0; font-size:24px; font-weight:800; color:#fff; }
.page-banner p { margin:6px 0 0 0; font-size:13.5px; opacity:.92; max-width:900px; }

/* ---------- COMMON ---------- */
.field-lbl { font-size:11.5px; font-weight:700; color:var(--red-dark); text-transform:uppercase; letter-spacing:.04em; margin-bottom:4px; }
.mini-stats { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin:6px 0 14px 0; }
.mini { border:1px solid #F3E3E3; border-left:4px solid var(--red); border-radius:10px; padding:10px 14px; background:#fff; }
.mini .v { font-size:18px; font-weight:800; color:#111827; }
.mini .l { font-size:11.5px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.03em; }
.irac-lbl { display:inline-block; font-size:11.5px; font-weight:800; letter-spacing:.05em; text-transform:uppercase;
  padding:3px 10px; border-radius:6px; margin:6px 0 4px 0; }
.irac-i { background:#FEE2E2; color:#991B1B; }
.irac-r { background:#FEF3C7; color:#92400E; }
.irac-a { background:#FFE4E6; color:#9F1239; }
.irac-c { background:#DCFCE7; color:#166534; }
.badge { display:inline-block; font-size:10.5px; font-weight:700; padding:3px 10px; border-radius:999px; letter-spacing:.02em; }
.b-active { background:#DCFCE7; color:#166534; border:1px solid #86EFAC; }
.b-expired { background:#FEE2E2; color:#991B1B; border:1px solid #FCA5A5; }
.b-grey { background:#F3F4F6; color:#4B5563; border:1px solid #E5E7EB; }
.proc-card { border:1px solid #F3E3E3; border-top:4px solid var(--red); border-radius:12px; padding:16px 18px; background:#fff; }
.proc-code { font-size:12px; font-weight:700; color:var(--red); letter-spacing:.03em; }
.proc-name { font-size:15px; font-weight:700; color:#111827; margin:4px 0 10px 0; line-height:1.4; }
.kv { display:grid; grid-template-columns: 120px 1fr; gap:6px 10px; font-size:13px; }
.kv .k { color:var(--muted); font-weight:600; }
.kv .v { color:#111827; }

.timeline { display:flex; gap:0; margin:4px 0 16px 0; }
.tl-step { flex:1; text-align:center; position:relative; padding-top:28px; font-size:12px; color:#9CA3AF; font-weight:600; }
.tl-step::before { content:""; position:absolute; top:10px; left:0; right:0; height:3px; background:#F3D3D3; }
.tl-dot { position:absolute; top:3px; left:50%; transform:translateX(-50%); width:17px; height:17px;
  border-radius:50%; background:#fff; border:3px solid #F3D3D3; z-index:1; }
.tl-step.on { color:var(--red-dark); }
.tl-step.on::before { background:var(--red); }
.tl-step.on .tl-dot { border-color:var(--red); background:var(--red); }

.alert-card { border:1px solid #F3E3E3; border-left:6px solid var(--red); border-radius:12px;
  padding:16px 18px; background:#fff; margin-bottom:6px; }
.alert-card.med { border-left-color:#D97706; }
.alert-card.low { border-left-color:#2563EB; }
.sev { font-size:10.5px; font-weight:800; color:#fff; padding:3px 9px; border-radius:6px; letter-spacing:.03em; }
.sev.high { background:#B91C1C; } .sev.med { background:#D97706; } .sev.low { background:#2563EB; }
.alert-title { font-size:15px; font-weight:700; color:#111827; margin:8px 0 6px 0; }
.alert-meta { font-size:12.5px; color:var(--muted); }

.res-card { border:1px solid #F3E3E3; border-radius:12px; padding:16px 18px; background:#fff; margin-bottom:4px; }
.res-top { display:flex; justify-content:space-between; align-items:center; gap:10px; }
.res-title { font-size:15px; font-weight:700; color:#111827; }
.res-body { font-size:13.5px; color:#374151; line-height:1.6; margin:8px 0; }
.res-foot { font-size:12px; color:#9CA3AF; }
mark { background:#FDE68A; color:#111827; padding:0 3px; border-radius:3px; }

.apcr-footer { margin:40px -2.4rem 0 -2.4rem; padding:22px 3.2rem; background:#7F1D1D; color:#FEE2E2;
  display:flex; justify-content:space-between; flex-wrap:wrap; gap:14px; font-size:12.5px; }
.apcr-footer b { color:#fff; }

/* Nút Streamlit */
div.stButton > button, div.stDownloadButton > button { border-radius:9px; font-weight:600; }
div.stButton > button[kind="secondary"] { border-color:#F0C4C4; color:var(--red-dark); }
div.stButton > button[kind="secondary"]:hover { border-color:var(--red); color:var(--red); background:var(--red-bg); }
div[data-testid="stExpander"] details { border-color:#F3E3E3 !important; border-radius:10px !important; }
</style>
"""
)


# ==============================================================================
# HEADER + NAV
# ==============================================================================

nav_items = [
    ("home", "🏠 Trang chủ"),
    ("hanh_chinh", "Thủ tục hành chính"),
    ("to_tung", "Tố tụng hình sự"),
    ("canh_bao", "Cảnh báo hiểu nhầm"),
    ("tra_cuu", "Tra cứu văn bản"),
    ("huong_dan", "Hướng dẫn sử dụng"),
]
nav_html = "".join(
    f'<a href="?page={key}" target="_self" class="{"active" if key == current_page else ""}">{label}</a>'
    for key, label in nav_items
)
h(
    f"""
<div class="apcr-header">
  <div class="apcr-brand">
    <div class="apcr-logo">⚖</div>
    <div>
      <div class="apcr-name">APCR-RAG</div>
      <div class="apcr-tag">Hệ thống tư vấn Thủ tục hành chính &amp; Tố tụng hình sự</div>
    </div>
  </div>
  <div class="apcr-team">👥 Nhóm 9 · INT 3507 2</div>
</div>
<div class="apcr-nav">{nav_html}</div>
"""
)


def page_banner(title: str, desc: str) -> None:
    h(
        f"""
<div class="page-banner">
  <div class="crumb"><a href="?page=home" target="_self">Trang chủ</a> &nbsp;›&nbsp; {esc(title)}</div>
  <h2>{esc(title)}</h2>
  <p>{esc(desc)}</p>
</div>
"""
    )


def footer() -> None:
    h(
        """
<div class="apcr-footer">
  <div><b>APCR-RAG</b> — Administrative &amp; Penal Criminal RAG<br/>Multi-Agent RAG trên LangGraph · Lược đồ hồ sơ 6 trường · Thẩm định chéo 2 lớp</div>
  <div><b>Nhóm 9 — INT 3507 2</b><br/>Phạm Đức Trọng · Vũ Gia Anh Đức · Nguyễn Hữu Hải Long</div>
  <div><b>GVHD</b><br/>TS. Nghiêm Nguyễn Việt Dũng</div>
</div>
<div style="font-size:11.5px; color:#9CA3AF; text-align:center; padding:10px 0 18px 0;">
Thông tin chỉ mang tính tham khảo, không thay thế văn bản quy phạm pháp luật chính thức hoặc ý kiến của cơ quan có thẩm quyền.
</div>
"""
    )


# ==============================================================================
# THÀNH PHẦN DÙNG CHUNG CHO 2 PHÂN HỆ HỎI — ĐÁP
# ==============================================================================

def build_graph_dot(proc: Dict[str, Any]) -> str:
    """Đồ thị 6 trường của thủ tục theo bảng màu đỏ — trắng (render bằng st.graphviz_chart)."""
    data = extract_procedure_graph_data(proc)
    palette = {
        "procedure": ("#B91C1C", "#FFFFFF"),
        "authority": ("#FEE2E2", "#7F1D1D"),
        "form": ("#FFFFFF", "#991B1B"),
        "deadline": ("#FEF3C7", "#92400E"),
        "fee": ("#FEF3C7", "#92400E"),
        "law": ("#7F1D1D", "#FFFFFF"),
    }
    lines = [
        "digraph G {",
        "rankdir=LR; bgcolor=\"transparent\"; nodesep=0.25; ranksep=0.5;",
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, color="#E8B4B4", penwidth=1.2];',
        'edge [fontname="Arial", fontsize=8, color="#D18C8C", fontcolor="#9CA3AF", arrowsize=0.6];',
    ]
    for n in data["nodes"]:
        fill, font = palette.get(n["group"], ("#FFFFFF", "#111827"))
        if n["group"] == "law" and "HẾT HIỆU LỰC" in n["label"]:
            fill, font = "#9CA3AF", "#FFFFFF"
        label = n["label"].replace("\\", "").replace('"', "'").replace("\n", "\\n")
        lines.append(f'"{n["id"]}" [label="{label}", fillcolor="{fill}", fontcolor="{font}"];')
    for e in data["edges"]:
        lines.append(f'"{e["source"]}" -> "{e["target"]}" [label="{e["label"].replace("_", " ").lower()}"];')
    lines.append("}")
    return "\n".join(lines)


def render_procedure_panel(proc: Optional[Dict[str, Any]]) -> None:
    if not proc:
        st.info("Không có hồ sơ thủ tục phù hợp để hiển thị.")
        return
    info = proc["ma_va_ten_thu_tuc"]
    laws = proc["can_cu_phap_ly"]
    status = laws[0]["trang_thai_hieu_luc"]
    badge = '<span class="badge b-active">● ĐANG HIỆU LỰC</span>' if status == "ĐANG HIỆU LỰC" else '<span class="badge b-expired">● HẾT HIỆU LỰC</span>'
    h(
        f"""
<div class="proc-card">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <span class="proc-code">{esc(info["ma_thu_tuc"])} · {esc(info["linh_vuc"])}</span>{badge}
  </div>
  <div class="proc-name">{esc(info["ten_thu_tuc"])}</div>
  <div class="kv">
    <div class="k">Cơ quan</div><div class="v">{esc(", ".join(info["cap_tham_quyen"]))}</div>
    <div class="k">Thời hạn</div><div class="v">Tối đa <b>{proc["thoi_han_giai_quyet"]["so_ngay_toi_da"]}</b> {esc(proc["thoi_han_giai_quyet"]["don_vi_tinh"])}</div>
    <div class="k">Lệ phí</div><div class="v"><b>{fmt_fee(proc["le_phi"]["muc_phi_vnd"])}</b></div>
    <div class="k">Căn cứ</div><div class="v">{esc("; ".join(f'{l["so_hieu_van_ban"]} — {l["dieu_khoan_diem"]}' for l in laws))}</div>
  </div>
</div>
"""
    )
    st.markdown('<div class="field-lbl" style="margin-top:14px;">Sơ đồ hồ sơ 6 trường</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.graphviz_chart(build_graph_dot(proc))
    with st.expander("📎 Thành phần hồ sơ & điều kiện bắt buộc"):
        for g in proc["thanh_phan_ho_so"]["giay_to_bat_buoc"]:
            st.markdown(f"- `{g['form_id']}` — {g['ten_giay_to']}")
        st.warning(f"**Điều kiện bắt buộc:** {proc['thanh_phan_ho_so']['dieu_kien_bat_buoc']}")
    with st.expander("📄 Dữ liệu JSON gốc"):
        st.json(proc)


def render_answer(state: Dict[str, Any]) -> None:
    draft = state.get("irac_draft") or {}
    ver = state.get("verification_result") or {}
    history = state.get("revision_history") or []
    status_txt = "Từ chối an toàn" if draft.get("is_refusal") else ("Đạt (PASS)" if ver.get("passed") else "Cảnh báo")
    h(
        f"""
<div class="mini-stats">
  <div class="mini"><div class="l">Kết quả thẩm định</div><div class="v">{status_txt}</div></div>
  <div class="mini"><div class="l">Số vòng kiểm chứng</div><div class="v">{len(history)}</div></div>
  <div class="mini"><div class="l">Độ tin cậy</div><div class="v">{int(float(draft.get("confidence_score", 0)) * 100)}%</div></div>
  <div class="mini"><div class="l">Thời gian xử lý</div><div class="v">{state.get("total_latency_ms", 0):.0f} ms</div></div>
</div>
"""
    )

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown('<div class="field-lbl">Ý kiến tư vấn (cấu trúc IRAC)</div>', unsafe_allow_html=True)
        with st.container(border=True):
            if draft.get("is_refusal"):
                st.error(f"⛔ **Hệ thống từ chối trả lời:** {draft.get('refusal_reason')}")
                st.markdown(draft.get("application", ""))
            else:
                st.markdown('<span class="irac-lbl irac-i">I · Vấn đề pháp lý</span>', unsafe_allow_html=True)
                st.markdown(draft.get("issue", ""))
                st.markdown('<span class="irac-lbl irac-r">R · Căn cứ pháp luật</span>', unsafe_allow_html=True)
                st.markdown(draft.get("rule", ""))
                st.markdown('<span class="irac-lbl irac-a">A · Áp dụng vào tình huống</span>', unsafe_allow_html=True)
                st.markdown(draft.get("application", ""))
                st.markdown('<span class="irac-lbl irac-c">C · Kết luận & lưu ý</span>', unsafe_allow_html=True)
                st.markdown(draft.get("conclusion", ""))

        with st.expander(f"🛡️ Nhật ký thẩm định 2 lớp của Judge Agent — {state.get('final_status', '')}", expanded=len(history) > 1):
            for step in history:
                v = step["verification"]
                tag = "Bản nháp đầu" if step["iteration"] == 1 else "Sau tự sửa lỗi"
                if v["passed"]:
                    st.success(f"**Vòng {step['iteration']} ({tag}) — ĐẠT.** {v['critique_feedback']}")
                else:
                    errs = "\n".join(f"- {e}" for e in v["layer1_errors"] + v["layer2_errors"])
                    st.error(f"**Vòng {step['iteration']} ({tag}) — PHÁT HIỆN LỖI, yêu cầu Consultant sửa:**\n{errs}")
        routing = state.get("routing_info") or {}
        if routing.get("matched_expansions"):
            st.caption(f"🔎 Truy vấn đã được chuẩn hóa thuật ngữ · Miền: **{routing.get('detected_domain')}**")
    with right:
        st.markdown('<div class="field-lbl">Hồ sơ thủ tục được truy xuất</div>', unsafe_allow_html=True)
        render_procedure_panel(state.get("primary_procedure"))


def set_query(key: str, value: str) -> None:
    st.session_state[key] = value


def render_qa_page(kind: str) -> None:
    is_tthc = kind == "hanh_chinh"
    q_key, res_key = f"{kind}_question", f"{kind}_result"

    if is_tthc:
        page_banner(
            "Hỏi — Đáp Thủ tục hành chính công",
            "Hộ tịch · Cư trú & Căn cước · Lao động & Bảo hiểm. Mỗi câu trả lời đều dẫn chiếu mã thủ tục, "
            "điều luật, biểu mẫu và được Judge Agent đối soát lệ phí, thời hạn với dữ liệu gốc.",
        )
        samples = [
            "Đăng ký khai sinh cho con có phải nộp Sổ hộ khẩu giấy không, lệ phí bao nhiêu?",
            "CMND 9 số cấp năm 2012 thì năm 2025 còn dùng được không?",
            "Kết hôn với người nước ngoài nộp hồ sơ ở UBND xã hay huyện?",
            "Nghỉ việc bao lâu thì phải nộp hồ sơ trợ cấp thất nghiệp?",
        ]
        linh_vuc_opts = ["Tất cả", "Hộ tịch", "Cư trú & Căn cước", "Lao động & Bảo hiểm"]
        phan_nhom = "Thủ tục hành chính"
    else:
        page_banner(
            "Hỏi — Đáp Thủ tục Tố tụng hình sự",
            "Tố giác tội phạm · Biện pháp ngăn chặn · Xóa án tích & Lý lịch tư pháp · Kháng cáo & Thi hành án. "
            "Hệ thống xác định đúng cơ quan thụ lý theo từng giai đoạn tố tụng.",
        )
        samples = [
            "Chỉ một mình tôi đứng ra làm đơn bảo lĩnh cho em trai có được không?",
            "Bị lừa đảo qua mạng, có người đòi 2 triệu lệ phí thụ lý đơn tố giác có đúng không?",
            "Thời hạn kháng cáo bản án hình sự sơ thẩm là bao nhiêu ngày?",
            "Có được ủy quyền cho mẹ làm hộ Phiếu lý lịch tư pháp số 2 không?",
        ]
        linh_vuc_opts = [
            "Tất cả",
            "Tiếp nhận & Giải quyết tố giác, tin báo về tội phạm",
            "Biện pháp ngăn chặn",
            "Xóa án tích & Lý lịch tư pháp",
            "Kháng cáo & Thi hành án hình sự",
        ]
        phan_nhom = "Tố tụng hình sự"

    if q_key not in st.session_state:
        st.session_state[q_key] = samples[0]

    with st.container(border=True):
        st.markdown('<div class="field-lbl">Nội dung câu hỏi / tình huống</div>', unsafe_allow_html=True)
        st.text_area("Câu hỏi", key=q_key, height=90, label_visibility="collapsed")

        c1, c2, c3, c4 = st.columns([2.2, 1.3, 1.9, 1.0])
        with c1:
            linh_vuc = st.selectbox("Lĩnh vực", linh_vuc_opts, key=f"{kind}_lv")
        with c2:
            eff_date = st.date_input("Hiệu lực tính đến ngày", value=datetime.date.today(), format="DD/MM/YYYY", key=f"{kind}_date")
        with c3:
            giai_doan = None
            if not is_tthc:
                giai_doan = st.selectbox(
                    "Giai đoạn tố tụng",
                    ["Tất cả", "Khởi tố - Giải quyết nguồn tin về tội phạm", "Điều tra", "Truy tố", "Xét xử", "Thi hành án"],
                    key=f"{kind}_gd",
                )
            else:
                st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            simulate = st.checkbox("Demo: cố ý sai ở bản nháp đầu", key=f"{kind}_sim",
                                   help="Bản nháp đầu bị làm sai lệ phí/thời hạn để minh họa Judge Agent bắt lỗi và vòng tự sửa.")
        with c4:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            ask = st.button("Gửi câu hỏi", type="primary", key=f"{kind}_ask", width="stretch")

        st.markdown('<div class="field-lbl" style="margin-top:6px;">Câu hỏi gợi ý</div>', unsafe_allow_html=True)
        s_cols = st.columns(len(samples))
        for i, s in enumerate(samples):
            s_cols[i].button(s, key=f"{kind}_s{i}", on_click=set_query, args=(q_key, s), width="stretch")

    if ask or res_key not in st.session_state:
        with st.spinner("Đang truy xuất hồ sơ, lập luận IRAC và thẩm định 2 lớp..."):
            st.session_state[res_key] = run_legal_multi_agent_workflow(
                query=st.session_state[q_key],
                phan_nhom_filter=phan_nhom,
                linh_vuc_filter=linh_vuc,
                giai_doan_filter=giai_doan,
                effective_date=str(eff_date) if eff_date else None,
                only_active=True,
                simulate_initial_error=simulate,
            )

    state = st.session_state[res_key]

    if not is_tthc:
        stages = ["Khởi tố - Giải quyết nguồn tin về tội phạm", "Điều tra", "Truy tố", "Xét xử", "Thi hành án"]
        short = ["Tiếp nhận nguồn tin", "Điều tra", "Truy tố", "Xét xử", "Thi hành án"]
        proc = state.get("primary_procedure") or {}
        active = set((proc.get("ma_va_ten_thu_tuc") or {}).get("giai_doan_to_tung", []))
        steps = "".join(
            f'<div class="tl-step {"on" if st_full in active else ""}"><span class="tl-dot"></span>{lbl}</div>'
            for st_full, lbl in zip(stages, short)
        )
        st.markdown('<div class="field-lbl" style="margin-top:8px;">Giai đoạn tố tụng áp dụng</div>', unsafe_allow_html=True)
        h(f'<div class="timeline">{steps}</div>')

    render_answer(state)


# ==============================================================================
# TRANG CHỦ
# ==============================================================================

def render_home() -> None:
    h(
        """
<div class="apcr-hero">
  <div>
    <h1>TRA CỨU ĐÚNG — <span>TƯ VẤN RÕ</span> — KIỂM CHỨNG CHẶT</h1>
    <p>Trợ lý pháp lý đa tác tử giúp người dân nắm chắc hồ sơ, thời hạn, lệ phí và cơ quan có thẩm quyền
    khi thực hiện thủ tục hành chính công và các thủ tục trong tố tụng hình sự.</p>
    <div class="hero-chips">
      <span class="hero-chip">📑 Hồ sơ chuẩn hóa 6 trường</span>
      <span class="hero-chip">🧭 Lập luận IRAC</span>
      <span class="hero-chip">🛡️ Thẩm định chéo 2 lớp</span>
      <span class="hero-chip">⏱️ Lọc hiệu lực theo thời gian</span>
    </div>
  </div>
  <svg width="300" height="210" viewBox="0 0 300 210" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="gr" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#DC2626"/><stop offset="1" stop-color="#7F1D1D"/></linearGradient>
      <linearGradient id="gd" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FBBF24"/><stop offset="1" stop-color="#D97706"/></linearGradient>
    </defs>
    <circle cx="170" cy="105" r="98" fill="url(#gr)" opacity="0.10"/>
    <rect x="200" y="40" width="78" height="100" rx="8" fill="#fff" stroke="#F3C4C4" stroke-width="2"/>
    <rect x="212" y="56" width="54" height="6" rx="3" fill="#FCA5A5"/>
    <rect x="212" y="70" width="44" height="5" rx="2.5" fill="#FDE2E2"/>
    <rect x="212" y="82" width="50" height="5" rx="2.5" fill="#FDE2E2"/>
    <rect x="212" y="94" width="38" height="5" rx="2.5" fill="#FDE2E2"/>
    <circle cx="258" cy="122" r="11" fill="#16A34A"/><path d="M252 122 l4 4 l8 -8" stroke="#fff" stroke-width="2.6" fill="none"/>
    <rect x="146" y="44" width="8" height="122" rx="4" fill="url(#gr)"/>
    <rect x="96" y="58" width="108" height="7" rx="3.5" fill="url(#gr)"/>
    <circle cx="150" cy="44" r="10" fill="url(#gd)"/>
    <line x1="104" y1="65" x2="88" y2="118" stroke="#B91C1C" stroke-width="2"/>
    <line x1="104" y1="65" x2="120" y2="118" stroke="#B91C1C" stroke-width="2"/>
    <path d="M78 118 h52 a26 12 0 0 1 -52 0 z" fill="url(#gd)"/>
    <line x1="196" y1="65" x2="180" y2="118" stroke="#B91C1C" stroke-width="2"/>
    <line x1="196" y1="65" x2="212" y2="118" stroke="#B91C1C" stroke-width="2"/>
    <path d="M170 118 h52 a26 12 0 0 1 -52 0 z" fill="url(#gd)" opacity="0.9"/>
    <rect x="118" y="166" width="64" height="12" rx="6" fill="url(#gr)"/>
    <rect x="104" y="180" width="92" height="12" rx="6" fill="#7F1D1D"/>
  </svg>
</div>
"""
    )

    h('<div class="section-title">Chức năng chính</div>')
    cards = [
        ("hanh_chinh", "🏛️", "Hỏi — Đáp Thủ tục hành chính công",
         "Khai sinh, kết hôn, khai tử, thường trú, tạm trú, căn cước, trợ cấp thất nghiệp, BHXH một lần, giấy phép lao động."),
        ("to_tung", "⚖️", "Hỏi — Đáp Thủ tục tố tụng hình sự",
         "Tố giác tội phạm, bảo lĩnh, đặt tiền bảo đảm, thăm gặp, kháng cáo, hoãn chấp hành án, xóa án tích, lý lịch tư pháp."),
        ("canh_bao", "🚨", "Cảnh báo hiểu nhầm pháp lý",
         "Những quan niệm sai phổ biến (Sổ hộ khẩu giấy, CMND 9 số, phí tố giác, bảo lĩnh 1 người…) kèm đính chính có căn cứ."),
        ("tra_cuu", "🔍", "Tra cứu văn bản & thủ tục tức thì",
         "Tìm Điều — Khoản theo từ khóa, lọc trạng thái hiệu lực theo ngày, không gọi mô hình ngôn ngữ, phản hồi tức thì."),
    ]
    h(
        '<div class="fn-grid">'
        + "".join(
            f'<a class="fn-card" href="?page={k}" target="_self"><div class="fn-icon">{ic}</div>'
            f'<div class="fn-body"><div class="fn-title">{t}</div><div class="fn-desc">{d}</div></div>'
            f'<div class="fn-arrow">›</div></a>'
            for k, ic, t, d in cards
        )
        + "</div>"
    )

    n_tthc = sum(1 for p in ALL_PROCS if p["ma_va_ten_thu_tuc"]["phan_nhom"] == "Thủ tục hành chính")
    n_tths = len(ALL_PROCS) - n_tthc
    n_laws = len({l["law_id"] for p in ALL_PROCS for l in p["can_cu_phap_ly"]})
    h('<div class="section-title">Dữ liệu hệ thống</div>')
    h(
        f"""
<div class="stat-strip">
  <div class="stat-box"><div class="stat-num">{n_tthc}</div><div class="stat-lbl">Thủ tục hành chính chuẩn hóa</div></div>
  <div class="stat-box"><div class="stat-num">{n_tths}</div><div class="stat-lbl">Thủ tục tố tụng hình sự</div></div>
  <div class="stat-box"><div class="stat-num">{n_laws}</div><div class="stat-lbl">Căn cứ pháp lý định danh</div></div>
  <div class="stat-box"><div class="stat-num">{len(BENCHMARK)}</div><div class="stat-lbl">Câu hỏi kiểm thử Benchmark</div></div>
</div>
"""
    )

    h('<div class="section-title">Quy trình xử lý một câu hỏi</div>')
    h(
        """
<div class="flow">
  <div class="flow-step"><b>① Retriever Agent</b><div>Chuẩn hóa thuật ngữ, định tuyến miền, lọc hiệu lực và tìm kiếm lai BM25 + Vector.</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-step"><b>② Consultant Agent</b><div>Lập luận IRAC từ hồ sơ 6 trường; từ chối trả lời khi thiếu căn cứ.</div></div>
  <div class="flow-arrow">→</div>
  <div class="flow-step"><b>③ Judge Agent</b><div>Lớp 1 đối soát mã, lệ phí, thời hạn bằng code; Lớp 2 kiểm điều kiện & thẩm quyền.</div></div>
  <div class="flow-arrow">↺</div>
  <div class="flow-step"><b>④ Tự sửa lỗi</b><div>Nếu chưa đạt, trả phản hồi cho Consultant sửa tối đa 2 vòng trước khi trả kết quả.</div></div>
</div>
"""
    )


# ==============================================================================
# PHÂN HỆ 3 — CẢNH BÁO HIỂU NHẦM
# ==============================================================================

def render_canh_bao() -> None:
    page_banner(
        "Cảnh báo hiểu nhầm pháp lý",
        "Danh mục quan niệm sai thường gặp khi người dân làm thủ tục, được Judge Agent đối chiếu với căn cứ pháp lý "
        "và gắn mức độ rủi ro. Có thể chạy thử từng tình huống bẫy trên hệ thống đa tác tử.",
    )
    n_high = sum(1 for m in MISCONCEPTIONS if m["severity"] == "NGHIÊM TRỌNG")
    total_flag = sum(m["flagged_count"] for m in MISCONCEPTIONS)
    n_trap = sum(1 for b in BENCHMARK if b["difficulty"] == "Gài bẫy")
    h(
        f"""
<div class="stat-strip">
  <div class="stat-box"><div class="stat-num">{len(MISCONCEPTIONS)}</div><div class="stat-lbl">Cảnh báo đang theo dõi</div></div>
  <div class="stat-box"><div class="stat-num">{n_high}</div><div class="stat-lbl">Mức độ nghiêm trọng</div></div>
  <div class="stat-box"><div class="stat-num">{n_trap}</div><div class="stat-lbl">Câu hỏi bẫy trong Benchmark</div></div>
  <div class="stat-box"><div class="stat-num">{total_flag:,}</div><div class="stat-lbl">Lượt gắn cờ ghi nhận</div></div>
</div>
""".replace(",", ".")
    )

    st.write("")
    f1, f2, _ = st.columns([1.3, 1.3, 2])
    sev_f = f1.selectbox("Mức độ", ["Tất cả", "NGHIÊM TRỌNG", "TRUNG BÌNH", "THẤP"])
    grp_f = f2.selectbox("Phân nhóm", ["Tất cả", "Thủ tục hành chính", "Tố tụng hình sự"])

    sev_cls = {"NGHIÊM TRỌNG": "high", "TRUNG BÌNH": "med", "THẤP": "low"}
    for m in MISCONCEPTIONS:
        if sev_f != "Tất cả" and m["severity"] != sev_f:
            continue
        if grp_f != "Tất cả" and m["phan_nhom"] != grp_f:
            continue
        cls = sev_cls[m["severity"]]
        h(
            f"""
<div class="alert-card {cls}">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
    <div><span class="sev {cls}">{m["severity"]}</span>
    <span class="badge b-grey" style="margin-left:6px;">{esc(m["phan_nhom"])} · {esc(m["linh_vuc"])}</span></div>
    <span class="alert-meta">🚩 {m["flagged_count"]} lượt · Mã <b>{esc(m["ma_thu_tuc"])}</b></span>
  </div>
  <div class="alert-title">{esc(m["title"])}</div>
  <div class="alert-meta">Căn cứ: <i>{esc(m["law_reference"])}</i></div>
</div>
"""
        )
        with st.expander(f"Xem đính chính — {m['alert_id']}"):
            a, b = st.columns(2)
            a.error(f"**❌ Hiểu nhầm phổ biến**\n\n{m['misconception_statement']}")
            b.success(f"**✅ Quy định đúng**\n\n{m['judge_correction']}")
            st.caption(f"Tình huống kiểm thử: “{m['sample_test_query']}”")
            if st.button("Chạy thử tình huống này trên hệ thống", key=f"trap_{m['alert_id']}"):
                with st.spinner("Đang xử lý..."):
                    res = run_legal_multi_agent_workflow(query=m["sample_test_query"], phan_nhom_filter=m["phan_nhom"])
                st.markdown((res.get("irac_draft") or {}).get("conclusion", ""))
        st.write("")

    h('<div class="section-title">Đánh giá trên bộ 50 câu hỏi kiểm thử</div>')
    st.caption("25 câu hành chính + 25 câu tố tụng hình sự · 15 tra cứu đơn · 20 tình huống nhiều điều kiện · 15 câu gài bẫy.")
    if st.button("▶ Chạy đánh giá Benchmark", type="primary"):
        ok_id = ok_fee = ok_day = 0
        prog = st.progress(0.0)
        for i, q in enumerate(BENCHMARK):
            res = run_legal_multi_agent_workflow(query=q["question"], phan_nhom_filter=q["phan_nhom"])
            cit = (res.get("irac_draft") or {}).get("citations") or {}
            ok_id += cit.get("ma_thu_tuc") == q["expected_ma_thu_tuc"]
            ok_fee += int(cit.get("muc_phi_vnd", -1)) == q["expected_muc_phi_vnd"]
            ok_day += int(cit.get("so_ngay_toi_da", -1)) == q["expected_so_ngay_toi_da"]
            prog.progress((i + 1) / len(BENCHMARK))
        n = max(1, len(BENCHMARK))
        r1, r2, r3 = st.columns(3)
        r1.metric("Truy xuất đúng thủ tục", f"{ok_id / n * 100:.1f}%")
        r2.metric("Khớp lệ phí", f"{ok_fee / n * 100:.1f}%")
        r3.metric("Khớp thời hạn", f"{ok_day / n * 100:.1f}%")
    with st.expander("Danh sách 50 câu hỏi kiểm thử"):
        st.dataframe(
            [{"Mã": b["question_id"], "Câu hỏi": b["question"], "Nhóm": b["phan_nhom"], "Độ khó": b["difficulty"],
              "Thủ tục chuẩn": b["expected_ma_thu_tuc"]} for b in BENCHMARK],
            width="stretch", hide_index=True,
        )


# ==============================================================================
# PHÂN HỆ 4 — TRA CỨU VĂN BẢN
# ==============================================================================

def render_tra_cuu() -> None:
    page_banner(
        "Tra cứu văn bản & thủ tục tức thì",
        "Tìm theo từ khóa, mã thủ tục hoặc số hiệu điều luật. Kết quả lấy trực tiếp từ chỉ mục toàn văn (BM25), "
        "không gọi mô hình ngôn ngữ, có nhãn trạng thái hiệu lực.",
    )
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 1.4, 1.3, 1.2])
        kw = c1.text_input("Từ khóa", value=st.session_state.get("tc_kw", "bảo lĩnh"),
                           placeholder="VD: khai sinh, Điều 121, sổ hộ khẩu, kháng cáo...", key="tc_kw")
        pn = c2.selectbox("Phân nhóm", ["Tất cả", "Thủ tục hành chính", "Tố tụng hình sự"], key="tc_pn")
        dt = c3.date_input("Hiệu lực đến ngày", value=datetime.date.today(), format="DD/MM/YYYY", key="tc_dt")
        with c4:
            st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
            only_active = st.checkbox("Chỉ còn hiệu lực", key="tc_active")
        st.caption("Gợi ý: `khai sinh` · `Điều 121` · `sổ hộ khẩu` · `CMND` · `kháng cáo` · `lý lịch tư pháp` · `thất nghiệp`")

    res = search_direct_bm25(query=kw, effective_date=str(dt), only_active=only_active,
                             phan_nhom=pn, top_k=20, retriever=retriever)
    h(
        f"""
<div style="display:flex; gap:18px; align-items:center; margin:14px 0 10px 0; font-size:13px; color:#6B7280;">
  <span><b style="color:#B91C1C; font-size:16px;">{res["total_matched"]}</b> kết quả cho “{esc(kw)}”</span>
  <span>⏱ {res["latency_ms"]} ms</span><span>🪙 0 token</span>
</div>
"""
    )
    if not res["results"]:
        st.info("Không tìm thấy kết quả. Hãy thử từ khóa khác hoặc bỏ bộ lọc hiệu lực.")
    for it in res["results"]:
        badge = '<span class="badge b-active">● ĐANG HIỆU LỰC</span>' if it["trang_thai_hieu_luc"] == "ĐANG HIỆU LỰC" else '<span class="badge b-expired">● HẾT HIỆU LỰC</span>'
        h(
            f"""
<div class="res-card">
  <div class="res-top">
    <div class="res-title">[{esc(it["ma_thu_tuc"])}] {it["ten_thu_tuc_highlighted"]}</div>
    <div style="white-space:nowrap;">{badge} <span class="badge b-grey">{esc(it["phan_nhom"]).upper()}</span></div>
  </div>
  <div class="res-body">📜 {it["can_cu_highlighted"]}<br/>⚠️ {it["dieu_kien_highlighted"]}</div>
  <div class="res-foot">Cơ quan: {esc(", ".join(it["cap_tham_quyen"]))} · Thời hạn tối đa {it["so_ngay_toi_da"]} ngày ·
  Lệ phí: {fmt_fee(it["muc_phi_vnd"])} · Hiệu lực từ {esc(it["ngay_hieu_luc"])}</div>
</div>
"""
        )
        a, b = st.columns([4, 1])
        with a.expander("Xem toàn văn thủ tục"):
            p = it["raw_procedure"]
            st.markdown("**Trình tự thực hiện**")
            for s in p["trinh_tu_thuc_hien"]:
                st.markdown(f"- {s}")
            st.markdown("**Thành phần hồ sơ**")
            for g in p["thanh_phan_ho_so"]["giay_to_bat_buoc"]:
                st.markdown(f"- `{g['form_id']}` — {g['ten_giay_to']}")
            if it.get("van_ban_thay_the"):
                st.warning(f"Văn bản thay thế: {it['van_ban_thay_the']}")
        b.download_button("⬇ Tải JSON", data=json.dumps(it["raw_procedure"], ensure_ascii=False, indent=2),
                          file_name=f"{it['ma_thu_tuc']}.json", mime="application/json",
                          key=f"dl_{it['ma_thu_tuc']}", width="stretch")


# ==============================================================================
# HƯỚNG DẪN SỬ DỤNG
# ==============================================================================

def render_huong_dan() -> None:
    page_banner("Hướng dẫn sử dụng", "Các bước khai thác hệ thống APCR-RAG hiệu quả.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("#### 1. Đặt câu hỏi")
            st.markdown(
                "- Chọn đúng phân hệ: **Thủ tục hành chính** hoặc **Tố tụng hình sự**.\n"
                "- Mô tả tình huống cụ thể (ai, việc gì, đang ở giai đoạn nào).\n"
                "- Có thể chọn lĩnh vực, giai đoạn tố tụng và ngày hiệu lực để thu hẹp kết quả."
            )
        with st.container(border=True):
            st.markdown("#### 2. Đọc kết quả IRAC")
            st.markdown(
                "- **I** — vấn đề pháp lý và thủ tục cần làm.\n"
                "- **R** — căn cứ pháp luật kèm trạng thái hiệu lực.\n"
                "- **A** — cơ quan, trình tự, hồ sơ, thời hạn, lệ phí.\n"
                "- **C** — kết luận và cảnh báo tránh hiểu nhầm."
            )
    with c2:
        with st.container(border=True):
            st.markdown("#### 3. Kiểm tra độ tin cậy")
            st.markdown(
                "- Mở **Nhật ký thẩm định 2 lớp** để xem Judge Agent đã đối soát những gì.\n"
                "- Bật *Demo: cố ý sai ở bản nháp đầu* để quan sát vòng tự sửa lỗi.\n"
                "- Câu hỏi ngoài phạm vi sẽ bị **từ chối an toàn** thay vì trả lời phỏng đoán."
            )
        with st.container(border=True):
            st.markdown("#### 4. Tra cứu nhanh")
            st.markdown(
                "- Dùng **Tra cứu văn bản** khi chỉ cần tìm điều luật/thủ tục theo từ khóa.\n"
                "- Nhãn đỏ **HẾT HIỆU LỰC** cho biết quy định cũ đã bị thay thế.\n"
                "- Tải dữ liệu JSON gốc để đối chiếu."
            )


# ==============================================================================
# ĐIỀU HƯỚNG
# ==============================================================================

if current_page == "home":
    render_home()
elif current_page in ("hanh_chinh", "to_tung"):
    render_qa_page(current_page)
elif current_page == "canh_bao":
    render_canh_bao()
elif current_page == "tra_cuu":
    render_tra_cuu()
elif current_page == "huong_dan":
    render_huong_dan()

footer()
