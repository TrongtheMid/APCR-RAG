"""
APCR-RAG — ADMINISTRATIVE & PENAL CRIMINAL RAG
Giao diện Trợ lý Pháp lý phong cách Google Gemini Chatbot:
  - Thanh bên trái (Sidebar): Tạo cuộc trò chuyện mới, Lịch sử các phiên trò chuyện (lưu bền vững),
    chuyển nhanh Công cụ (Trò chuyện Pháp lý, Tra cứu BM25 Zero-LLM, Cảnh báo & Benchmark) và Cài đặt lọc.
  - Không gian chính (Main Canvas): Màn hình chào kiểu Gemini, luồng hội thoại tập trung thoáng đãng,
    tích hợp gọn gàng Phân tích IRAC, Nhật ký Judge 2 lớp và Hồ sơ/Sơ đồ 6 trường ngay dưới câu trả lời.
"""

from __future__ import annotations

import base64
import datetime
import html as html_lib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from src.consultant_agent import extract_procedure_graph_data
from src.evaluation import run_full_benchmark_evaluation
from src.ingestion import (
    OfficialSourceDocument,
    add_candidates_to_staging,
    approve_and_merge_candidate,
    clear_processed_candidates,
    extract_procedures_from_document,
    extract_text_from_pdf_bytes,
    fetch_chinhphu_detail,
    fetch_congbao_latest,
    get_official_sources_status,
    load_staging_queue,
    reject_candidate,
    search_chinhphu_documents,
)
from src.judge_agent import get_legal_misconceptions_catalog
from src.langgraph_workflow import (
    run_legal_multi_agent_workflow,
    stream_legal_multi_agent_workflow,
)
from src.retriever_agent import get_retriever_agent, search_direct_bm25

PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark_50_questions.json"
CHAT_HISTORY_FILE = PROJECT_ROOT / "data" / ".apcr_chat_sessions.json"

st.set_page_config(
    page_title="APCR-RAG · Trợ lý Pháp lý AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================================================================
# TIỆN ÍCH & LƯU TRỮ LỊCH SỬ TRÒ CHUYỆN
# ==============================================================================

def h(markup: str) -> None:
    """Render HTML an toàn (loại bỏ thụt lề để Markdown không hiểu nhầm thành code block)."""
    cleaned = "\n".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def esc(text: Any) -> str:
    return html_lib.escape(str(text or ""))


def fmt_fee(vnd: int) -> str:
    return "Miễn phí (0 đ)" if int(vnd) == 0 else f"{int(vnd):,} đ".replace(",", ".")


def _md_basic_to_html(text: Any) -> str:
    """Chuyển đổi an toàn Markdown cơ bản (**đậm**, *nghiêng*, `code`, xuống dòng) sang HTML cho ngăn kéo mượt."""
    safe = esc(text or "")
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe)
    safe = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", safe)
    return safe.replace("\n", "<br/>")


@st.cache_resource
def load_retriever():
    return get_retriever_agent()


@st.cache_data
def load_benchmark() -> List[Dict[str, Any]]:
    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _load_saved_conversations() -> List[Dict[str, Any]]:
    if CHAT_HISTORY_FILE.exists():
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _save_conversations(convs: List[Dict[str, Any]]) -> None:
    try:
        CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(convs[:30], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _new_conversation() -> Dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:10],
        "title": "Cuộc trò chuyện mới",
        "created_at": datetime.datetime.now().strftime("%H:%M · %d/%m"),
        "turns": [],
    }


def _init_session_state() -> None:
    if "conversations" not in st.session_state:
        saved = _load_saved_conversations()
        st.session_state["conversations"] = saved if saved else [_new_conversation()]
    if not st.session_state["conversations"]:
        st.session_state["conversations"] = [_new_conversation()]
    if "active_conv_id" not in st.session_state:
        st.session_state["active_conv_id"] = st.session_state["conversations"][0]["id"]
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "chat"
    if "pending_prompt" not in st.session_state:
        st.session_state["pending_prompt"] = None


def _get_active_conversation() -> Dict[str, Any]:
    convs: List[Dict[str, Any]] = st.session_state["conversations"]
    active_id = st.session_state.get("active_conv_id")
    for c in convs:
        if c["id"] == active_id:
            return c
    st.session_state["active_conv_id"] = convs[0]["id"]
    return convs[0]


def _create_and_select_new_chat() -> None:
    convs: List[Dict[str, Any]] = st.session_state["conversations"]
    if convs and not convs[0]["turns"]:
        st.session_state["active_conv_id"] = convs[0]["id"]
    else:
        new_c = _new_conversation()
        convs.insert(0, new_c)
        st.session_state["active_conv_id"] = new_c["id"]
        _save_conversations(convs)
    st.session_state["active_view"] = "chat"


def _delete_conversation(conv_id: str) -> None:
    convs: List[Dict[str, Any]] = [
        c for c in st.session_state["conversations"] if c["id"] != conv_id
    ]
    if not convs:
        convs = [_new_conversation()]
    st.session_state["conversations"] = convs
    if st.session_state.get("active_conv_id") == conv_id:
        st.session_state["active_conv_id"] = convs[0]["id"]
    _save_conversations(convs)


retriever = load_retriever()
ALL_PROCS: List[Dict[str, Any]] = retriever.raw_procedures
BENCHMARK = load_benchmark()
MISCONCEPTIONS = get_legal_misconceptions_catalog()
_init_session_state()


# ==============================================================================
# CSS — PHONG CÁCH GOOGLE GEMINI CHATBOT
# ==============================================================================

h(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap');

:root {
  --bg: #FFFFFF;
  --sidebar-bg: #FFF5F5;
  --surface: #FFF5F5;
  --surface-hover: #FEE2E2;
  --active-pill: #FEE2E2;
  --active-ink: #991B1B;
  --ink: #1F2937;
  --muted: #6B7280;
  --border: #F3D5D5;
  --red: #B91C1C;
  --red-dark: #7F1D1D;
  --green-bg: #DCFCE7;
  --green-ink: #166534;
  --red-bg: #FEE2E2;
  --red-ink: #991B1B;
  --amber-bg: #FEF3C7;
  --amber-ink: #92400E;
}

section[data-testid="stMain"] {
  scroll-behavior: auto !important;
  -webkit-overflow-scrolling: touch;
  overscroll-behavior-y: contain;
  overflow-anchor: auto;
}
html, body, [class*="css"], .stMarkdown, button, input, textarea {
  font-family: 'Be Vietnam Pro', 'Segoe UI', Roboto, sans-serif !important;
  color: var(--ink);
}

#MainMenu, footer { display: none !important; }
header[data-testid="stHeader"] {
  background: #FFFFFF !important;
}
.stApp { background: var(--bg); }
.block-container {
  padding-top: 3.6rem !important;
  padding-bottom: 5.5rem !important;
  max-width: 920px !important;
}

/* Thanh cuộn tinh gọn tông đỏ nhạt */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #E5B4B4; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #DC2626; }

/* ---------- SIDEBAR TÔNG ĐỎ NHẠT - TRẮNG ---------- */
section[data-testid="stSidebar"] {
  background: var(--sidebar-bg) !important;
  border-right: 1px solid #F3D5D5 !important;
}
section[data-testid="stSidebar"] .block-container {
  padding: 1rem 0.9rem 1.5rem 0.9rem !important;
}
.sb-brand {
  display: flex; align-items: center; gap: 10px;
  padding: 4px 6px 14px 6px;
}
.sb-spark {
  width: 34px; height: 34px; border-radius: 10px;
  background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
  color: #FFFFFF; font-size: 18px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 2px 8px rgba(185, 28, 28, 0.25);
}
.sb-title { font-size: 17px; font-weight: 700; color: var(--red-dark); line-height: 1.15; }
.sb-sub { font-size: 11px; color: var(--muted); font-weight: 500; }
.sb-section {
  font-size: 11px; font-weight: 700; color: var(--red-dark);
  text-transform: uppercase; letter-spacing: 0.05em;
  margin: 14px 4px 6px 6px;
}

/* Nút sidebar dạng pill đỏ nhạt */
section[data-testid="stSidebar"] div.stButton > button {
  border-radius: 999px !important;
  border: none !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 0.45rem 0.85rem !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  box-shadow: none !important;
  transition: background 0.15s ease !important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
  background: transparent !important;
  color: #4B5563 !important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
  background: var(--surface-hover) !important;
  color: var(--red-dark) !important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
  background: var(--active-pill) !important;
  color: var(--active-ink) !important;
  font-weight: 600 !important;
}
section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
  background: #FECACA !important;
}

/* ---------- TOP BAR CỐ ĐỊNH NHẸ ---------- */
.gem-topbar {
  position: sticky; top: 2.9rem; z-index: 20;
  display: flex; justify-content: space-between; align-items: center;
  background: #FFFFFF;
  border-bottom: 1px solid #F8E4E4;
  padding: 8px 4px 10px 4px;
  margin-bottom: 10px;
}
.gem-model-pill {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 18px; font-weight: 700; color: var(--red-dark);
}
.gem-model-sub {
  font-size: 12px; font-weight: 600; color: var(--red-ink);
  background: var(--surface); border: 1px solid #FECACA;
  padding: 4px 12px; border-radius: 999px;
}
.gem-status-chip {
  font-size: 11.5px; font-weight: 600; color: #166534;
  background: #DCFCE7; border-radius: 999px; padding: 4px 12px;
}

/* ---------- MÀN HÌNH CHÀO TÔNG ĐỎ ẤM ---------- */
.gem-welcome {
  max-width: 780px; margin: 4.5vh auto 26px auto; padding: 0 8px;
}
.gem-hello {
  font-size: 40px; font-weight: 700; line-height: 1.2; letter-spacing: -0.02em;
  background: linear-gradient(90deg, #991B1B 0%, #DC2626 55%, #EA580C 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 6px;
}
.gem-subhello {
  font-size: 24px; font-weight: 500; color: #9CA3AF; line-height: 1.3;
  margin-bottom: 14px;
}
.gem-desc {
  font-size: 14px; color: var(--muted); line-height: 1.6; max-width: 660px;
}

/* ---------- CHAT BUBBLES ĐỎ NHẠT - TRẮNG ---------- */
div[data-testid="stChatMessage"] {
  background: transparent !important;
  border: none !important;
  padding: 8px 0 !important;
}
.user-bubble-wrap {
  display: flex; justify-content: flex-end; margin: 6px 0 6px 0;
  scroll-margin-top: 76px;
}
.user-bubble {
  background: #FFF1F2;
  border: 1px solid #FECDD3;
  color: var(--ink);
  padding: 12px 18px;
  border-radius: 20px 20px 4px 20px;
  max-width: 80%;
  font-size: 14.5px;
  line-height: 1.55;
  box-shadow: 0 1px 2px rgba(185, 28, 28, 0.05);
}
.ai-meta-row {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
  margin: 10px 0 8px 0;
}
.chip {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 11.5px; font-weight: 600;
  padding: 3px 10px; border-radius: 999px;
}
.chip-pass { background: var(--green-bg); color: var(--green-ink); }
.chip-warn { background: var(--red-bg); color: var(--red-ink); }
.chip-info { background: #FFF5F5; color: #4B5563; border: 1px solid #F3D5D5; }
.chip-red { background: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }

/* ---------- NGĂN KÉO CHI TIẾT PHÁP LÝ MƯỢT (0ms, KHÔNG GIẬT, KHÔNG NHẢY CUỘN) ---------- */
details.legal-drawer {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: #FFFFFF;
  margin: 6px 0;
  overflow: hidden;
  overflow-anchor: none;
}
details.legal-drawer > summary {
  cursor: pointer;
  list-style: none;
  padding: 9px 14px;
  font-size: 13px;
  font-weight: 600;
  color: var(--red-dark);
  background: #FFF9F9;
  display: flex;
  align-items: center;
  justify-content: space-between;
  user-select: none;
  transition: background 0.15s ease;
}
details.legal-drawer > summary::-webkit-details-marker { display: none; }
details.legal-drawer > summary::after {
  content: "▾";
  font-size: 13px;
  color: var(--red);
  transition: transform 0.18s ease;
}
details.legal-drawer[open] > summary::after {
  transform: rotate(180deg);
}
details.legal-drawer > summary:hover {
  background: #FFF1F2;
}
@keyframes drawerReveal {
  from { opacity: 0; transform: translateY(-4px); }
  to { opacity: 1; transform: translateY(0); }
}
details.legal-drawer[open] > .drawer-body {
  animation: drawerReveal 0.16s cubic-bezier(0.2, 0, 0, 1);
}
.drawer-body {
  padding: 12px 16px;
  border-top: 1px solid #F8E4E4;
  font-size: 13.5px;
  line-height: 1.6;
}
/* Sơ đồ 6 trường HTML gọn nhẹ & khung JSON */
.schema6-hub {
  background: linear-gradient(135deg, #B91C1C 0%, #991B1B 100%);
  color: #FFFFFF;
  border-radius: 10px;
  padding: 10px 14px;
  margin-bottom: 10px;
}
.schema6-hub-code {
  font-size: 11px;
  font-weight: 700;
  opacity: 0.9;
  letter-spacing: 0.04em;
}
.schema6-hub-title {
  font-size: 14px;
  font-weight: 700;
  margin-top: 2px;
}
.schema6-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 8px;
  margin-bottom: 10px;
}
.schema6-node {
  border: 1px solid var(--border);
  border-left: 3px solid var(--red);
  border-radius: 9px;
  padding: 8px 11px;
  background: #FFFDFD;
  font-size: 12.5px;
  line-height: 1.45;
}
.schema6-node.auth { border-left-color: #991B1B; background: #FFF5F5; }
.schema6-node.deadline { border-left-color: #D97706; background: #FFFBEB; }
.schema6-node.fee { border-left-color: #16A34A; background: #F0FDF4; }
.schema6-node.law { border-left-color: #7F1D1D; }
.schema6-node-rel {
  font-size: 10.5px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--red-dark);
  margin-bottom: 3px;
}
.json-box {
  background: #FFF9F9;
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  font-size: 12px;
  line-height: 1.45;
  color: #374151;
  max-height: 220px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
  margin: 6px 0 10px 0;
}
.dl-json-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 13px;
  border-radius: 999px;
  background: #FFF1F2;
  border: 1px solid #FCA5A5;
  color: #991B1B !important;
  text-decoration: none !important;
  font-size: 12.5px;
  font-weight: 600;
  transition: background 0.15s ease;
}
.dl-json-btn:hover {
  background: #FEE2E2;
}
/* Ẩn khoảng trống container của iframe 0px dùng để cuộn 1 lần (không dùng display:none để script vẫn chạy) */
div.stElementContainer:has(iframe[height="0"]),
div.element-container:has(iframe[height="0"]) {
  position: fixed !important;
  width: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
  pointer-events: none !important;
}
.drawer-body code {
  background: #FFF1F2;
  color: #991B1B;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 12.5px;
}
.judge-step {
  border-radius: 8px;
  padding: 9px 12px;
  margin-bottom: 8px;
  font-size: 13px;
  line-height: 1.5;
}
.judge-step.pass { background: #F0FDF4; border: 1px solid #BBF7D0; color: #166534; }
.judge-step.fail { background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; }
.judge-score { font-size: 11.5px; color: var(--muted); margin-top: 4px; }

/* ---------- THẺ PHỤ TRỢ (IRAC / HỒ SƠ / CẢNH BÁO / TRA CỨU) ---------- */
.irac-lbl {
  display: inline-block; font-size: 11px; font-weight: 700;
  letter-spacing: 0.04em; text-transform: uppercase;
  padding: 3px 9px; border-radius: 6px; margin: 8px 0 4px 0;
}
.irac-i { background: #FEE2E2; color: #991B1B; }
.irac-r { background: #FEF3C7; color: #92400E; }
.irac-a { background: #FFE4E6; color: #9F1239; }
.irac-c { background: #DCFCE7; color: #166534; }

.proc-card {
  border: 1px solid var(--border); border-top: 3px solid var(--red); border-radius: 12px;
  padding: 12px 15px; background: #FFFDFD; margin: 8px 0;
}
.proc-code { font-size: 11.5px; font-weight: 700; color: var(--red); }
.proc-name { font-size: 14.5px; font-weight: 700; color: var(--ink); margin: 3px 0 8px 0; }
.kv { display: grid; grid-template-columns: 110px 1fr; gap: 5px 10px; font-size: 13px; }
.kv .k { color: var(--muted); font-weight: 600; }
.kv .v { color: var(--ink); }

.alert-card {
  border: 1px solid var(--border); border-left: 5px solid #B91C1C;
  border-radius: 14px; padding: 14px 16px; background: #FFFFFF; margin-bottom: 8px;
}
.alert-card.med { border-left-color: #D97706; }
.alert-card.low { border-left-color: #2563EB; }
.sev {
  font-size: 10.5px; font-weight: 700; color: #FFFFFF;
  padding: 2px 8px; border-radius: 999px;
}
.sev.high { background: #B91C1C; }
.sev.med { background: #D97706; }
.sev.low { background: #2563EB; }

.res-card {
  border: 1px solid var(--border); border-radius: 14px;
  padding: 14px 16px; background: #FFFFFF; margin-bottom: 6px;
}
.res-title { font-size: 15px; font-weight: 700; color: var(--ink); }
.res-body { font-size: 13.5px; color: #374151; line-height: 1.6; margin: 6px 0; }
.res-foot { font-size: 12px; color: var(--muted); }
mark { background: #FDE68A; color: #1F2937; padding: 0 3px; border-radius: 3px; }

/* Nút gợi ý chính dạng thẻ bo tròn đỏ nhạt */
div.stButton > button[kind="secondary"] {
  border-radius: 14px !important;
  border: 1px solid var(--border) !important;
  background: #FFFDFD !important;
  color: var(--ink) !important;
  font-weight: 500 !important;
  transition: all 0.15s ease !important;
}
div.stButton > button[kind="secondary"]:hover {
  background: #FFF1F2 !important;
  border-color: #FCA5A5 !important;
  color: var(--red-ink) !important;
}
div[data-testid="stExpander"] details {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  background: #FFFFFF;
}
div[data-testid="stChatInput"] {
  max-width: 880px;
  margin: 0 auto;
}
div[data-testid="stChatInput"] > div {
  border-radius: 28px !important;
  border: 1px solid #F1C4C4 !important;
  background: #FFF5F5 !important;
  padding: 4px 10px !important;
}
div[data-testid="stChatInput"] > div:focus-within {
  border-color: var(--red) !important;
  background: #FFFFFF !important;
  box-shadow: 0 2px 12px rgba(185, 28, 28, 0.12) !important;
}
</style>
"""
)


# ==============================================================================
# SIDEBAR — LỊCH SỬ TRÒ CHUYỆN & CÔNG CỤ (PHONG CÁCH GEMINI)
# ==============================================================================

with st.sidebar:
    h(
        """
<div class="sb-brand">
  <div class="sb-spark">✦</div>
  <div>
    <div class="sb-title">APCR-RAG</div>
    <div class="sb-sub">Trợ lý Pháp lý AI · Nhóm 9</div>
  </div>
</div>
"""
    )

    if st.button("＋  Cuộc trò chuyện mới", key="sb_new_chat", type="primary", width="stretch"):
        _create_and_select_new_chat()
        st.rerun()

    h('<div class="sb-section">Gần đây</div>')
    convs: List[Dict[str, Any]] = st.session_state["conversations"]
    active_conv_id = st.session_state["active_conv_id"]
    active_view = st.session_state["active_view"]

    for conv in convs:
        is_selected = active_view == "chat" and conv["id"] == active_conv_id
        title_short = conv["title"]
        if len(title_short) > 26:
            title_short = title_short[:25].rstrip() + "…"
        c_title, c_del = st.columns([5.2, 1.0], gap="small")
        with c_title:
            if st.button(
                f"💬 {title_short}",
                key=f"sb_conv_{conv['id']}",
                type="primary" if is_selected else "secondary",
                width="stretch",
                help=f"{conv['title']} ({conv['created_at']})",
            ):
                st.session_state["active_conv_id"] = conv["id"]
                st.session_state["active_view"] = "chat"
                st.rerun()
        with c_del:
            if len(convs) > 1 or conv["turns"]:
                if st.button("✕", key=f"sb_del_{conv['id']}", help="Xóa cuộc trò chuyện này", width="stretch"):
                    _delete_conversation(conv["id"])
                    st.rerun()

    h('<div class="sb-section">Công cụ pháp lý</div>')
    staging_q_sb = load_staging_queue()
    pending_cnt_sb = sum(1 for c in staging_q_sb.candidates if c.status == "PENDING")
    dong_bo_lbl = f"📥  Đồng bộ Nguồn ({pending_cnt_sb})" if pending_cnt_sb else "📥  Đồng bộ Nguồn Chính thống"
    nav_tools = [
        ("chat", "✦  Trò chuyện Pháp lý"),
        ("tra_cuu", "🔍  Tra cứu Văn bản (BM25)"),
        ("canh_bao", "🚨  Cảnh báo & Benchmark"),
        ("dong_bo", dong_bo_lbl),
    ]
    for view_key, label in nav_tools:
        if st.button(
            label,
            key=f"sb_view_{view_key}",
            type="primary" if active_view == view_key else "secondary",
            width="stretch",
        ):
            st.session_state["active_view"] = view_key
            st.rerun()

    h('<div class="sb-section">Tùy chọn truy vấn</div>')
    with st.expander("⚙️ Bộ lọc & Chế độ Demo", expanded=False):
        domain_choice = st.selectbox(
            "Phạm vi tư vấn",
            ["Tự động nhận diện", "Thủ tục hành chính", "Tố tụng hình sự"],
            key="sb_domain",
        )
        eff_date = st.date_input(
            "Hiệu lực đến ngày",
            value=datetime.date.today(),
            format="DD/MM/YYYY",
            key="sb_eff_date",
        )
        giai_doan_choice = "Tất cả"
        if domain_choice == "Tố tụng hình sự":
            giai_doan_choice = st.selectbox(
                "Giai đoạn tố tụng",
                ["Tất cả", "Khởi tố - Giải quyết nguồn tin về tội phạm", "Điều tra", "Truy tố", "Xét xử", "Thi hành án"],
                key="sb_stage",
            )
        simulate_err = st.checkbox(
            "Demo: cố ý sai bản nháp đầu",
            key="sb_sim_err",
            help="Giả lập bản nháp đầu sai lệ phí/thời hạn để xem Judge Agent 2 lớp bắt lỗi và tự sửa.",
        )
        if st.button("🗑️ Xóa toàn bộ lịch sử", key="sb_clear_all", width="stretch"):
            fresh = [_new_conversation()]
            st.session_state["conversations"] = fresh
            st.session_state["active_conv_id"] = fresh[0]["id"]
            _save_conversations(fresh)
            st.rerun()

    st.caption(f"Kho dữ liệu: **{len(ALL_PROCS)} thủ tục** · **{len(BENCHMARK)} câu Benchmark**")


# ==============================================================================
# THÀNH PHẦN HIỂN THỊ ĐỒ THỊ & TIN NHẮN TRỢ LÝ
# ==============================================================================

def build_schema_6_html(proc: Dict[str, Any]) -> str:
    """Hiển thị Sơ đồ cấu trúc 6 trường dạng lưới HTML nhẹ (0 WASM) + khung JSON & nút tải."""
    data = extract_procedure_graph_data(proc)
    info = proc.get("ma_va_ten_thu_tuc") or {}
    ma_tt = info.get("ma_thu_tuc", "thu_tuc")

    rel_by_target: Dict[str, str] = {}
    for edge in data.get("edges", []):
        rel_by_target[edge["target"]] = edge.get("label", "").replace("_", " ").upper()

    group_meta = {
        "authority": ("auth", "🏛️ CƠ QUAN THẨM QUYỀN"),
        "deadline": ("deadline", "⏳ THỜI HẠN GIẢI QUYẾT"),
        "fee": ("fee", "💰 LỆ PHÍ"),
        "form": ("form", "📎 THÀNH PHẦN HỒ SƠ"),
        "law": ("law", "📜 CĂN CỨ PHÁP LÝ"),
    }

    hub_html = ""
    node_cards: List[str] = []
    for n in data.get("nodes", []):
        grp = n.get("group", "")
        raw_lbl = str(n.get("label", "")).strip()
        lbl_html = esc(raw_lbl).replace("\n", "<br/>")
        if grp == "procedure":
            hub_html = (
                f'<div class="schema6-hub">'
                f'<div class="schema6-hub-code">THỦ TỤC TRUNG TÂM · {esc(ma_tt)}</div>'
                f'<div class="schema6-hub-title">{lbl_html}</div>'
                f'</div>'
            )
            continue

        css_cls, default_rel = group_meta.get(grp, ("", grp.upper()))
        rel_lbl = rel_by_target.get(n.get("id", ""), default_rel) or default_rel
        node_cards.append(
            f'<div class="schema6-node {css_cls}">'
            f'<div class="schema6-node-rel">{esc(rel_lbl)}</div>'
            f'<div>{lbl_html}</div>'
            f'</div>'
        )

    json_str = json.dumps(proc, ensure_ascii=False, indent=2)
    b64_json = base64.b64encode(json_str.encode("utf-8")).decode("ascii")
    return (
        f'{hub_html}'
        f'<div class="schema6-grid">{"".join(node_cards)}</div>'
        f'<div style="font-size:12px; font-weight:600; color:var(--red-dark); margin-top:4px;">📄 Hồ sơ JSON chuẩn hóa</div>'
        f'<pre class="json-box">{esc(json_str)}</pre>'
        f'<a class="dl-json-btn" download="{esc(ma_tt)}.json" '
        f'href="data:application/json;base64,{b64_json}">⬇ Tải hồ sơ JSON chuẩn hóa</a>'
    )


def _scroll_to_turn_once(turn_idx: int) -> None:
    """Cuộn mượt đúng 1 lần tới câu hỏi vừa gửi mà không dùng cơ chế ghim đáy liên tục."""
    components.html(
        f"""
<script>
(function() {{
  try {{
    var doc = window.parent.document;
    var el = doc.getElementById("turn-anchor-{int(turn_idx)}");
    if (el) {{
      setTimeout(function() {{
        el.scrollIntoView({{ behavior: "smooth", block: "start" }});
      }}, 40);
    }}
  }} catch (e) {{}}
}})();
</script>
""",
        height=0,
    )


def build_graph_dot(proc: Dict[str, Any]) -> str:
    """Đồ thị 6 trường của thủ tục theo tông đỏ nhạt — trắng (st.graphviz_chart)."""
    data = extract_procedure_graph_data(proc)
    palette = {
        "procedure": ("#B91C1C", "#FFFFFF"),
        "authority": ("#FEE2E2", "#7F1D1D"),
        "form": ("#FFFFFF", "#991B1B"),
        "deadline": ("#FEF3C7", "#92400E"),
        "fee": ("#DCFCE7", "#166534"),
        "law": ("#7F1D1D", "#FFFFFF"),
    }
    lines = [
        "digraph G {",
        'rankdir=LR; bgcolor="transparent"; nodesep=0.25; ranksep=0.5;',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, color="#F3C4C4", penwidth=1.1];',
        'edge [fontname="Arial", fontsize=8, color="#D18C8C", fontcolor="#6B7280", arrowsize=0.6];',
    ]
    for n in data["nodes"]:
        fill, font = palette.get(n["group"], ("#FFFFFF", "#1F2937"))
        if n["group"] == "law" and "HẾT HIỆU LỰC" in n["label"]:
            fill, font = "#9CA3AF", "#FFFFFF"
        label = n["label"].replace("\\", "").replace('"', "'").replace("\n", "\\n")
        lines.append(f'"{n["id"]}" [label="{label}", fillcolor="{fill}", fontcolor="{font}"];')
    for e in data["edges"]:
        lines.append(f'"{e["source"]}" -> "{e["target"]}" [label="{e["label"].replace("_", " ").lower()}"];')
    lines.append("}")
    return "\n".join(lines)


def render_assistant_turn(state: Dict[str, Any], turn_idx: int) -> None:
    """Hiển thị 1 lượt trả lời của APCR-RAG với ngăn kéo chi tiết pháp lý mượt 0ms."""
    draft = state.get("irac_draft") or {}
    ver = state.get("verification_result") or {}
    history = state.get("revision_history") or []
    proc = state.get("primary_procedure")
    lat_ms = float(state.get("total_latency_ms", 0.0))

    # 1. Hội thoại tự nhiên (Chào hỏi / Liệt kê dịch vụ / Tóm tắt nối tiếp)
    if state.get("final_status") == "CONVERSATIONAL_REPLY" or state.get("is_conversational"):
        st.markdown(draft.get("direct_advice") or draft.get("conclusion", ""))
        if proc:
            info = proc["ma_va_ten_thu_tuc"]
            h(
                f'<div class="ai-meta-row">'
                f'<span class="chip chip-red">📎 Ngữ cảnh: {esc(info["ma_thu_tuc"])}</span>'
                f'<span class="chip chip-info">⏱ {lat_ms:.0f} ms</span>'
                f'</div>'
            )
        return

    # 2. Từ chối an toàn
    if draft.get("is_refusal"):
        st.markdown(draft.get("direct_advice") or draft.get("application", ""))
        h(
            f'<div class="ai-meta-row">'
            f'<span class="chip chip-warn">⛔ Từ chối an toàn</span>'
            f'<span class="chip chip-info">⏱ {lat_ms:.0f} ms</span>'
            f'</div>'
        )
        if draft.get("refusal_reason"):
            st.caption(f"Lý do: {draft['refusal_reason']}")
        return

    # 3. Tư vấn pháp lý đầy đủ (RAG + IRAC + Judge 2 lớp)
    direct_txt = (draft.get("direct_advice") or "").strip() or (draft.get("conclusion") or "")
    st.markdown(direct_txt)

    info = (proc or {}).get("ma_va_ten_thu_tuc") or {}
    ma_tt = info.get("ma_thu_tuc", "")
    conf_pct = int(float(draft.get("confidence_score", 0)) * 100)
    passed = bool(ver.get("passed", False))
    ver_chip = (
        '<span class="chip chip-pass">✓ Đã kiểm chứng 2 lớp</span>'
        if passed
        else '<span class="chip chip-warn">⚠ Cần đối chiếu thêm</span>'
    )
    fee_chip = ""
    days_chip = ""
    if proc:
        fee_chip = f'<span class="chip chip-info">💰 {esc(fmt_fee(proc["le_phi"]["muc_phi_vnd"]))}</span>'
        days_chip = f'<span class="chip chip-info">⏳ Tối đa {proc["thoi_han_giai_quyet"]["so_ngay_toi_da"]} ngày</span>'

    code_chip = f'<span class="chip chip-red">{esc(ma_tt)}</span>' if ma_tt else ""
    h(
        f'<div class="ai-meta-row">'
        f'{ver_chip}'
        f'{code_chip}'
        f'{fee_chip}{days_chip}'
        f'<span class="chip chip-info">Độ tin cậy {conf_pct}% · {lat_ms:.0f} ms</span>'
        f'</div>'
    )

    # Ngăn kéo 1: Phân tích IRAC & Hồ sơ bắt buộc (HTML <details> mở tức thì, mượt tuyệt đối)
    proc_block = ""
    if proc:
        laws = proc["can_cu_phap_ly"]
        status = laws[0]["trang_thai_hieu_luc"]
        st_chip = (
            '<span class="chip chip-pass">● ĐANG HIỆU LỰC</span>'
            if status == "ĐANG HIỆU LỰC"
            else '<span class="chip chip-warn">● HẾT HIỆU LỰC</span>'
        )
        forms_html = "".join(
            f"<li><code>{esc(g['form_id'])}</code> — {esc(g['ten_giay_to'])}</li>"
            for g in proc["thanh_phan_ho_so"]["giay_to_bat_buoc"]
        )
        laws_str = "; ".join(f"{l['so_hieu_van_ban']} — {l['dieu_khoan_diem']}" for l in laws)
        proc_block = (
            f'<div class="proc-card">'
            f'<div style="display:flex; justify-content:space-between; align-items:center;">'
            f'<span class="proc-code">{esc(info["ma_thu_tuc"])} · {esc(info["linh_vuc"])}</span>{st_chip}</div>'
            f'<div class="proc-name">{esc(info["ten_thu_tuc"])}</div>'
            f'<div class="kv">'
            f'<div class="k">Cơ quan</div><div class="v">{esc(", ".join(info["cap_tham_quyen"]))}</div>'
            f'<div class="k">Thời hạn</div><div class="v">Tối đa <b>{proc["thoi_han_giai_quyet"]["so_ngay_toi_da"]}</b> {esc(proc["thoi_han_giai_quyet"]["don_vi_tinh"])}</div>'
            f'<div class="k">Lệ phí</div><div class="v"><b>{esc(fmt_fee(proc["le_phi"]["muc_phi_vnd"]))}</b></div>'
            f'<div class="k">Căn cứ</div><div class="v">{esc(laws_str)}</div>'
            f'</div>'
            f'<div style="margin-top:8px; font-size:12.5px;"><b>Thành phần hồ sơ:</b><ul style="margin:4px 0 6px 18px; padding:0;">{forms_html}</ul></div>'
            f'<div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:8px; padding:7px 10px; font-size:12.5px; color:#92400E;">'
            f'<b>Điều kiện bắt buộc:</b> {esc(proc["thanh_phan_ho_so"]["dieu_kien_bat_buoc"])}</div>'
            f'</div>'
        )

    h(
        f"""
<details class="legal-drawer">
  <summary>📑 Phân tích pháp lý IRAC &amp; Hồ sơ bắt buộc</summary>
  <div class="drawer-body">
    <span class="irac-lbl irac-i">I · Vấn đề pháp lý</span>
    <div style="margin-bottom:6px;">{_md_basic_to_html(draft.get("issue", ""))}</div>
    <span class="irac-lbl irac-r">R · Căn cứ pháp luật</span>
    <div style="margin-bottom:6px;">{_md_basic_to_html(draft.get("rule", ""))}</div>
    <span class="irac-lbl irac-a">A · Hướng dẫn áp dụng</span>
    <div style="margin-bottom:6px;">{_md_basic_to_html(draft.get("application", ""))}</div>
    <span class="irac-lbl irac-c">C · Kết luận &amp; lưu ý</span>
    <div style="margin-bottom:10px;">{_md_basic_to_html(draft.get("conclusion", ""))}</div>
    {proc_block}
  </div>
</details>
"""
    )

    # Ngăn kéo 2: Nhật ký thẩm định Judge 2 lớp (HTML <details>)
    judge_steps_html: List[str] = []
    for step in history:
        v = step["verification"]
        tag = "Bản nháp đầu" if step["iteration"] == 1 else f"Tự hiệu chỉnh lần #{step['iteration'] - 1}"
        l1_sc = int(float(v.get("layer1_score", 1.0)) * 100)
        l2_sc = int(float(v.get("claim_coverage_score", 1.0)) * 100)
        v_claims = v.get("verified_claims_count", 0)
        t_claims = v.get("total_claims_count", 0)
        if v["passed"]:
            msg_html = f"<b>Vòng {step['iteration']} ({esc(tag)}) — ĐẠT.</b> {esc(v['critique_feedback'])}"
            cls_step = "pass"
        else:
            errs_li = "".join(f"<li>{esc(e)}</li>" for e in v["layer1_errors"] + v["layer2_errors"])
            msg_html = f"<b>Vòng {step['iteration']} ({esc(tag)}) — PHÁT HIỆN LỖI:</b><ul style='margin:4px 0 0 18px;'>{errs_li}</ul>"
            cls_step = "fail"
        judge_steps_html.append(
            f'<div class="judge-step {cls_step}">{msg_html}'
            f'<div class="judge-score">Lớp 1 (Regex &amp; Exact Match): <b>{l1_sc}%</b> · '
            f'Lớp 2 (Claim Coverage): <b>{l2_sc}%</b> ({v_claims}/{t_claims} mệnh đề)</div></div>'
        )

    open_attr = "open" if len(history) > 1 else ""
    h(
        f"""
<details class="legal-drawer" {open_attr}>
  <summary>🛡️ Nhật ký thẩm định Judge 2 lớp ({len(history)} vòng · {esc(state.get("final_status", ""))})</summary>
  <div class="drawer-body">{"".join(judge_steps_html)}</div>
</details>
"""
    )

    if proc:
        h(
            f"""
<details class="legal-drawer">
  <summary>📊 Xem Sơ đồ cấu trúc 6 trường &amp; JSON gốc</summary>
  <div class="drawer-body">{build_schema_6_html(proc)}</div>
</details>
"""
        )


# ==============================================================================
# VIEW 1 — MÀN HÌNH CHAT CHÍNH (PHONG CÁCH GOOGLE GEMINI)
# ==============================================================================

def render_gemini_chat() -> None:
    conv = _get_active_conversation()
    phan_nhom = None if domain_choice == "Tự động nhận diện" else domain_choice
    giai_doan = None if giai_doan_choice == "Tất cả" else giai_doan_choice

    # Top bar tối giản
    h(
        f"""
<div class="gem-topbar">
  <div class="gem-model-pill">
    <span>✦ APCR-RAG</span>
    <span class="gem-model-sub">{esc(domain_choice)}</span>
  </div>
  <span class="gem-status-chip">✓ Multi-Agent · Judge 2 lớp</span>
</div>
"""
    )

    # Đặt chat_input trong st.bottom để ghim đáy màn hình nhưng TẮT vòng lặp stAppScrollToBottomContainer 17ms
    with st.bottom:
        typed_prompt = st.chat_input(
            "Hỏi APCR-RAG bất kỳ điều gì về thủ tục hành chính, tố tụng hình sự...",
            key="gemini_chat_input",
        )
    incoming = typed_prompt or st.session_state.pop("pending_prompt", None)

    if incoming and incoming.strip():
        q_clean = incoming.strip()
        history_ctx: List[Dict[str, str]] = []
        for t in conv["turns"][-5:]:
            p_info = ((t["state"].get("primary_procedure") or {}).get("ma_va_ten_thu_tuc")) or {}
            d_prev = t["state"].get("irac_draft") or {}
            history_ctx.append(
                {
                    "query": t["query"],
                    "procedure_code": p_info.get("ma_thu_tuc", ""),
                    "procedure_name": p_info.get("ten_thu_tuc", ""),
                    "direct_advice": d_prev.get("direct_advice", ""),
                }
            )

        for idx, turn in enumerate(conv["turns"], start=1):
            h(f'<div id="turn-anchor-{idx}" class="user-bubble-wrap"><div class="user-bubble">{esc(turn["query"])}</div></div>')
            with st.chat_message("assistant", avatar="✨"):
                render_assistant_turn(turn["state"], turn_idx=idx)

        new_idx = len(conv["turns"]) + 1
        h(f'<div id="turn-anchor-{new_idx}" class="user-bubble-wrap"><div class="user-bubble">{esc(q_clean)}</div></div>')
        _scroll_to_turn_once(new_idx)
        with st.chat_message("assistant", avatar="✨"):
            final_state: Dict[str, Any] = {}
            with st.status("Đang suy luận và đối soát căn cứ pháp lý...", expanded=False) as status_box:
                for event in stream_legal_multi_agent_workflow(
                    query=q_clean,
                    phan_nhom_filter=phan_nhom,
                    giai_doan_filter=giai_doan,
                    effective_date=str(eff_date) if eff_date else None,
                    only_active=True,
                    simulate_initial_error=simulate_err,
                    chat_history=history_ctx,
                ):
                    if event["node"] != "__end__":
                        st.write(f"• **{event['label']}** (`{event['latency_ms']} ms`): {event['summary']}")
                    final_state = event["state"]
                status_box.update(
                    label=f"Hoàn tất ({final_state.get('total_latency_ms', 0):.0f} ms)",
                    state="complete",
                    expanded=False,
                )

            if not conv["turns"]:
                conv["title"] = q_clean[:44] + ("…" if len(q_clean) > 44 else "")
            conv["turns"].append({"query": q_clean, "state": final_state})
            _save_conversations(st.session_state["conversations"])
            render_assistant_turn(final_state, turn_idx=len(conv["turns"]))
        st.session_state["scroll_to_turn"] = len(conv["turns"])
        st.rerun()

    # Nếu cuộc trò chuyện đang rỗng -> hiển thị màn hình chào kiểu Gemini
    if not conv["turns"]:
        h(
            """
<div class="gem-welcome">
  <div class="gem-hello">Xin chào!</div>
  <div class="gem-subhello">Hôm nay tôi có thể giúp gì cho bạn?</div>
  <div class="gem-desc">
    Trò chuyện tự nhiên về <b>40 thủ tục Hành chính công &amp; Tố tụng Hình sự</b>.
    Mọi câu trả lời pháp lý đều có trích dẫn điều luật và được <b>Judge Agent 2 lớp</b> đối soát lệ phí, thời hạn.
  </div>
</div>
"""
        )
        starters = [
            ("📋", "Đâu là các dịch vụ mà bạn hỗ trợ?"),
            ("🚗", "Tôi muốn xin cấp lại giấy phép lái xe"),
            ("⚖️", "Chỉ một mình tôi đứng ra làm đơn bảo lĩnh cho em trai có được không?"),
            ("🪪", "CMND 9 số cấp năm 2012 thì năm 2025 còn dùng được không?"),
        ]
        c1, c2 = st.columns(2, gap="small")
        for idx, (icon, prompt_txt) in enumerate(starters):
            col = c1 if idx % 2 == 0 else c2
            with col:
                if st.button(f"{icon}  {prompt_txt}", key=f"starter_{idx}", width="stretch"):
                    st.session_state["pending_prompt"] = prompt_txt
                    st.rerun()
        return

    # Hiển thị toàn bộ luồng hội thoại trực tiếp trên trang (cuộn chuột mượt toàn trang)
    for idx, turn in enumerate(conv["turns"], start=1):
        h(f'<div id="turn-anchor-{idx}" class="user-bubble-wrap"><div class="user-bubble">{esc(turn["query"])}</div></div>')
        with st.chat_message("assistant", avatar="✨"):
            render_assistant_turn(turn["state"], turn_idx=idx)

    # Chip gợi ý hỏi tiếp nhanh nếu lượt cuối đang tư vấn 1 thủ tục cụ thể
    latest_state = conv["turns"][-1]["state"]
    if latest_state.get("primary_procedure"):
        h('<div style="overflow-anchor: none; height: 8px;"></div>')
        followups = [
            "📎 Cần giấy tờ gì?",
            "📋 Trình tự các bước?",
            "💰 Lệ phí bao nhiêu?",
            "✨ Tóm tắt ngắn gọn",
        ]
        f_cols = st.columns(4, gap="small")
        for j, fq in enumerate(followups):
            if f_cols[j].button(fq, key=f"fq_{conv['id']}_{j}", width="stretch"):
                st.session_state["pending_prompt"] = fq.split(" ", 1)[1]
                st.rerun()

    target_scroll = st.session_state.pop("scroll_to_turn", None)
    if target_scroll is not None:
        _scroll_to_turn_once(int(target_scroll))


# ==============================================================================
# VIEW 2 — TRA CỨU VĂN BẢN TỨC THÌ (ZERO-LLM BM25)
# ==============================================================================

def render_tra_cuu() -> None:
    h(
        """
<div class="gem-topbar">
  <div class="gem-model-pill">
    <span>🔍 Tra cứu Văn bản &amp; Thủ tục</span>
    <span class="gem-model-sub">Zero-LLM · Rank-BM25</span>
  </div>
  <span class="gem-status-chip">🪙 0 token · Phản hồi tức thì</span>
</div>
"""
    )
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([3, 1.4, 1.3, 1.2])
        kw = c1.text_input(
            "Từ khóa tìm kiếm",
            value=st.session_state.get("tc_kw", "bảo lĩnh"),
            placeholder="VD: khai sinh, Điều 121, sổ hộ khẩu, GPLX, hộ chiếu...",
            key="tc_kw",
        )
        pn = c2.selectbox("Phân nhóm", ["Tất cả", "Thủ tục hành chính", "Tố tụng hình sự"], key="tc_pn")
        dt = c3.date_input("Hiệu lực đến ngày", value=datetime.date.today(), format="DD/MM/YYYY", key="tc_dt")
        with c4:
            st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
            only_active = st.checkbox("Chỉ còn hiệu lực", key="tc_active")

    res = search_direct_bm25(
        query=kw,
        effective_date=str(dt),
        only_active=only_active,
        phan_nhom=pn,
        top_k=20,
        retriever=retriever,
    )
    st.caption(f"Tìm thấy **{res['total_matched']}** kết quả cho “{kw}” · ⏱ {res['latency_ms']} ms · 🪙 0 token")
    if not res["results"]:
        st.info("Không tìm thấy kết quả phù hợp. Hãy thử từ khóa khác hoặc bỏ lọc hiệu lực.")
    for it in res["results"]:
        badge = (
            '<span class="chip chip-pass">● ĐANG HIỆU LỰC</span>'
            if it["trang_thai_hieu_luc"] == "ĐANG HIỆU LỰC"
            else '<span class="chip chip-warn">● HẾT HIỆU LỰC</span>'
        )
        h(
            f"""
<div class="res-card">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
    <div class="res-title">[{esc(it["ma_thu_tuc"])}] {it["ten_thu_tuc_highlighted"]}</div>
    <div style="white-space:nowrap;">{badge} <span class="chip chip-info">{esc(it["phan_nhom"])}</span></div>
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
        b.download_button(
            "⬇ Tải JSON",
            data=json.dumps(it["raw_procedure"], ensure_ascii=False, indent=2),
            file_name=f"{it['ma_thu_tuc']}.json",
            mime="application/json",
            key=f"dl_{it['ma_thu_tuc']}",
            width="stretch",
        )


# ==============================================================================
# VIEW 3 — CẢNH BÁO HIỂU NHẦM PHÁP LÝ & BENCHMARK 50 CÂU
# ==============================================================================

def render_canh_bao() -> None:
    h(
        """
<div class="gem-topbar">
  <div class="gem-model-pill">
    <span>🚨 Cảnh báo Hiểu nhầm &amp; Benchmark</span>
    <span class="gem-model-sub">Judge 2 lớp · 50 câu kiểm thử</span>
  </div>
</div>
"""
    )
    tab_alerts, tab_bench = st.tabs(["🚨 10 Cảnh báo Hiểu nhầm Pháp lý", "📊 Đánh giá Benchmark & Ablation Study"])

    with tab_alerts:
        f1, f2 = st.columns(2)
        sev_f = f1.selectbox("Mức độ rủi ro", ["Tất cả", "NGHIÊM TRỌNG", "TRUNG BÌNH", "THẤP"])
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
  <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
    <div><span class="sev {cls}">{m["severity"]}</span>
    <span class="chip chip-info" style="margin-left:6px;">{esc(m["phan_nhom"])} · {esc(m["linh_vuc"])}</span></div>
    <span style="font-size:12px; color:#5F6368;">🚩 {m["flagged_count"]} lượt · <b>{esc(m["ma_thu_tuc"])}</b></span>
  </div>
  <div style="font-size:15px; font-weight:700; margin:8px 0 4px 0;">{esc(m["title"])}</div>
  <div style="font-size:12.5px; color:#5F6368;">Căn cứ: <i>{esc(m["law_reference"])}</i></div>
</div>
"""
            )
            with st.expander(f"Xem đính chính & hỏi thử — {m['alert_id']}"):
                a, b = st.columns(2)
                a.error(f"**❌ Hiểu nhầm phổ biến**\n\n{m['misconception_statement']}")
                b.success(f"**✅ Quy định đúng**\n\n{m['judge_correction']}")
                if st.button("💬 Hỏi thử tình huống này trong Chatbot", key=f"try_trap_{m['alert_id']}"):
                    _create_and_select_new_chat()
                    st.session_state["pending_prompt"] = m["sample_test_query"]
                    st.rerun()

    with tab_bench:
        st.caption("25 câu Hành chính + 25 câu Tố tụng hình sự · Đối chứng 4 cấu hình Ablation Study & Confusion Matrix.")
        if st.button("▶ Chạy đánh giá Toàn diện (Benchmark 50 câu + Ablation Study)", type="primary"):
            with st.spinner("Đang chạy thực nghiệm đối chứng trên bộ 50 câu hỏi..."):
                st.session_state["benchmark_full_report"] = run_full_benchmark_evaluation(include_ablation=True)

        report = st.session_state.get("benchmark_full_report")
        if report:
            ma = report["multi_agent_apcr_rag"]
            stress = report["self_revision_stress_test"]
            cm_data = report["routing_confusion_matrix"]
            r1, r2, r3, r4, r5 = st.columns(5)
            r1.metric("Precision@1", f"{ma['precision_at_1_pct']:.1f}%")
            r2.metric("Faithfulness", f"{ma['faithfulness_pct']:.1f}%")
            r3.metric("Hallucination", f"{ma['hallucination_rate_pct']:.1f}%")
            r4.metric("Self-Revision", f"{stress['self_revision_correction_rate_pct']:.1f}%")
            r5.metric("Routing Acc", f"{cm_data['accuracy_pct']:.1f}%")

            st.markdown("##### Bảng thí nghiệm cắt bỏ (Ablation Study — 4 cấu hình)")
            st.dataframe(
                [
                    {
                        "Cấu hình": row["config"],
                        "P@1 (%)": row["precision_at_1_pct"],
                        "R@3 (%)": row["recall_at_3_pct"],
                        "Faithfulness (%)": row["faithfulness_pct"],
                        "Hallucination (%)": row["hallucination_rate_pct"],
                        "Claim Cov (%)": row["claim_coverage_pct"],
                        "Latency (ms)": row["avg_latency_ms"],
                    }
                    for row in (report.get("ablation_study") or [])
                ],
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "⬇ Tải báo cáo Benchmark JSON",
                data=json.dumps(report, ensure_ascii=False, indent=2),
                file_name="apcr_rag_benchmark_ablation_report.json",
                mime="application/json",
            )


# ==============================================================================
# VIEW 4 — ĐỒNG BỘ NGUỒN DỮ LIỆU CHÍNH THỐNG (AUTO INGESTION & STAGING)
# ==============================================================================

def render_dong_bo() -> None:
    staging_queue = load_staging_queue()
    pending_items = [c for c in staging_queue.candidates if c.status == "PENDING"]
    active_cnt = sum(
        1 for p in ALL_PROCS if (p.get("can_cu_phap_ly") or [{}])[0].get("trang_thai_hieu_luc") == "ĐANG HIỆU LỰC"
    )

    h(
        f"""
<div class="gem-topbar">
  <div class="gem-model-pill">
    <span>📥 Đồng bộ Nguồn Chính thống</span>
    <span class="gem-model-sub">vanban.chinhphu.vn · pypdf · Staging</span>
  </div>
  <span class="gem-status-chip">{len(ALL_PROCS)} thủ tục ({active_cnt} hiệu lực) · {len(pending_items)} chờ duyệt</span>
</div>
"""
    )

    # Trạng thái các kênh dữ liệu chính thống
    sources = get_official_sources_status()
    src_cols = st.columns(len(sources), gap="small")
    for idx, src in enumerate(sources):
        is_ready = src["mode"] == "auto_public"
        badge_cls = "chip-pass" if is_ready else "chip-warn"
        badge_txt = "● MỞ CÔNG KHAI" if is_ready else "● CẦN KHÓA / TƯỜNG CHỐNG BOT"
        with src_cols[idx]:
            h(
                f"""
<div class="res-card" style="padding:10px 12px; margin-bottom:8px;">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:4px;">
    <span class="chip {badge_cls}">{badge_txt}</span>
  </div>
  <div style="font-size:13px; font-weight:700; color:var(--red-dark);">{esc(src["name"])}</div>
  <div style="font-size:12px; color:var(--muted); margin-top:2px;">{esc(src["status"])}</div>
</div>
"""
            )

    tab_portal, tab_upload, tab_staging = st.tabs(
        [
            "🌐 Thu thập từ Cổng Chính phủ",
            "📄 Nạp từ URL / Tải tệp PDF",
            f"✅ Hàng đợi Kiểm duyệt ({len(pending_items)} chờ duyệt)",
        ]
    )

    with tab_portal:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2.6, 1.5, 0.9, 1.3])
            sync_kw = c1.text_input(
                "Từ khóa tìm văn bản / TTHC",
                value=st.session_state.get("sync_kw", "thủ tục hành chính"),
                placeholder="VD: hộ tịch, cư trú, giấy phép lái xe, tố tụng hình sự...",
                key="sync_kw",
            )
            src_channel = c2.selectbox(
                "Nguồn thu thập",
                ["vanban.chinhphu.vn", "congbao.chinhphu.vn"],
                key="sync_channel",
            )
            sync_limit = int(c3.number_input("Số văn bản", min_value=1, max_value=20, value=6, key="sync_limit"))
            use_llm_extract = c4.checkbox(
                "Dùng Gemini bóc tách",
                value=True,
                key="sync_use_llm",
                help="Dùng Gemini Structured Output bóc tách PDF sang chuẩn 6 trường (tắt để test Regex ngoại tuyến).",
            )

            b1, b2 = st.columns([1.4, 2.6])
            if b1.button("🔍 Quét văn bản chính thống", type="primary", width="stretch"):
                with st.spinner(f"Đang kết nối {src_channel}..."):
                    try:
                        if src_channel == "congbao.chinhphu.vn":
                            hits = fetch_congbao_latest(limit=sync_limit)
                        else:
                            hits = search_chinhphu_documents(keyword=sync_kw, limit=sync_limit)
                        st.session_state["sync_hits"] = hits
                    except Exception as exc:
                        st.error(f"Không thể kết nối nguồn {src_channel}: {exc}")

            hits: List[Dict[str, Any]] = st.session_state.get("sync_hits") or []
            if hits and b2.button("⚡ Bóc tách nhanh 2 văn bản đầu vào Staging", width="stretch"):
                added_total = 0
                with st.spinner("Đang tải PDF ký số & chuẩn hóa sang 6 trường..."):
                    for item in hits[:2]:
                        doc = fetch_chinhphu_detail(item["url"], download_pdf_text=True)
                        cands = extract_procedures_from_document(
                            doc,
                            existing_procedures=ALL_PROCS,
                            use_llm=use_llm_extract,
                        )
                        res = add_candidates_to_staging(
                            cands,
                            source_summary=f"{src_channel} ({doc.so_hieu})",
                        )
                        added_total += res["added"]
                st.success(f"Đã đưa {added_total} bản ghi vào Hàng đợi Kiểm duyệt! Chuyển sang tab Kiểm duyệt để xem.")
                st.rerun()

        if not hits:
            st.info("Bấm **🔍 Quét văn bản chính thống** để tải danh sách Luật, Nghị định, Thông tư, Quyết định công bố TTHC mới nhất.")
        else:
            st.caption(f"Tìm thấy **{len(hits)}** văn bản chính thống.")
            for i, it in enumerate(hits):
                pdf_badge = (
                    f'<span class="chip chip-pass">PDF gốc ({len(it.get("pdf_urls") or [])})</span>'
                    if it.get("pdf_urls")
                    else '<span class="chip chip-info">Trang chi tiết</span>'
                )
                h(
                    f"""
<div class="res-card">
  <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
    <div class="res-title">[{esc(it["so_hieu"])}] {esc(it["trich_yeu"] or it["url"])}</div>
    <div style="white-space:nowrap;">{pdf_badge} <span class="chip chip-info">{esc(it.get("ngay_ban_hanh") or "Mới")}</span></div>
  </div>
  <div class="res-foot">Nguồn: <a href="{esc(it["url"])}" target="_blank">{esc(it["url"])}</a></div>
</div>
"""
                )
                if st.button("⚙️ Tải toàn văn PDF & Đưa vào Staging", key=f"ingest_hit_{i}_{it['doc_id']}"):
                    with st.spinner(f"Đang tải và chuẩn hóa {it['so_hieu']}..."):
                        doc = fetch_chinhphu_detail(it["url"], download_pdf_text=True)
                        cands = extract_procedures_from_document(
                            doc,
                            existing_procedures=ALL_PROCS,
                            use_llm=use_llm_extract,
                        )
                        res = add_candidates_to_staging(
                            cands,
                            source_summary=f"Thu thập thủ công {doc.so_hieu}",
                        )
                    st.success(f"Đã thêm {res['added']} bản ghi chờ duyệt từ {doc.so_hieu} (bỏ qua trùng: {res['skipped']}).")
                    st.rerun()

    with tab_upload:
        with st.container(border=True):
            st.markdown("##### Cách 1: Dán trực tiếp URL văn bản Chính phủ / Công báo")
            url_in = st.text_input(
                "Đường dẫn chi tiết (`vanban.chinhphu.vn/?pageid=27160&docid=...` hoặc `congbao.chinhphu.vn/van-ban/...`)",
                placeholder="https://vanban.chinhphu.vn/?pageid=27160&docid=219796",
                key="sync_direct_url",
            )
            use_llm_direct = st.checkbox("Dùng Gemini chuẩn hóa 6 trường", value=True, key="sync_direct_llm")
            if st.button("🌐 Tải & Bóc tách từ URL", type="primary", key="btn_ingest_url"):
                if not url_in.strip():
                    st.warning("Vui lòng nhập đường dẫn văn bản chính thống.")
                else:
                    with st.spinner("Đang tải metadata + PDF ký số và chuẩn hóa 6 trường..."):
                        doc = fetch_chinhphu_detail(url_in.strip(), download_pdf_text=True)
                        cands = extract_procedures_from_document(
                            doc,
                            existing_procedures=ALL_PROCS,
                            use_llm=use_llm_direct,
                        )
                        res = add_candidates_to_staging(cands, source_summary=f"URL {doc.so_hieu}")
                    st.success(f"Đã nạp {res['added']} bản ghi vào Staging từ {doc.so_hieu} ({len(doc.raw_text)} ký tự).")
                    st.rerun()

        with st.container(border=True):
            st.markdown("##### Cách 2: Tải lên tệp PDF / TXT Quyết định công bố TTHC (kể cả xuất từ DVCQG)")
            uploaded = st.file_uploader("Chọn tệp PDF hoặc TXT", type=["pdf", "txt"], key="sync_file_up")
            u1, u2 = st.columns(2)
            up_code = u1.text_input("Số ký hiệu văn bản", value="QD-CONGBO-TTHC", key="up_so_hieu")
            up_org = u2.text_input("Cơ quan ban hành", value="Cơ quan có thẩm quyền", key="up_org")
            if st.button("📤 Đọc tệp & Đưa vào Staging", key="btn_ingest_file"):
                if uploaded is None:
                    st.warning("Vui lòng chọn tệp PDF hoặc TXT.")
                else:
                    raw_bytes = uploaded.read()
                    if uploaded.name.lower().endswith(".pdf"):
                        txt = extract_text_from_pdf_bytes(raw_bytes)
                    else:
                        txt = raw_bytes.decode("utf-8", errors="ignore")
                    if not txt.strip():
                        st.error("Không đọc được text từ tệp (PDF có thể là bản scan ảnh chưa OCR).")
                    else:
                        doc = OfficialSourceDocument(
                            source_name="pdf_upload",
                            doc_id=uploaded.name,
                            so_hieu=up_code.strip() or uploaded.name,
                            trich_yeu=f"Tài liệu tải lên: {uploaded.name}",
                            loai_van_ban="Quyết định / Văn bản tải lên",
                            co_quan_ban_hanh=up_org.strip(),
                            ngay_hieu_luc=str(datetime.date.today()),
                            raw_text=txt,
                        )
                        cands = extract_procedures_from_document(
                            doc,
                            existing_procedures=ALL_PROCS,
                            use_llm=use_llm_direct,
                        )
                        res = add_candidates_to_staging(cands, source_summary=f"Upload {uploaded.name}")
                        st.success(f"Đã đọc {len(txt)} ký tự và thêm {res['added']} mục vào Staging.")
                        st.rerun()

    with tab_staging:
        if staging_queue.last_sync_at:
            st.caption(f"Đồng bộ gần nhất: **{staging_queue.last_sync_at}** · {staging_queue.last_source_summary}")

        if not pending_items:
            st.info("Hàng đợi kiểm duyệt đang trống. Hãy thu thập văn bản ở 2 tab bên cạnh để kiểm duyệt trước khi đưa vào kho chính.")
        else:
            for cand in pending_items:
                rec = cand.record.model_dump()
                info = rec["ma_va_ten_thu_tuc"]
                laws = rec["can_cu_phap_ly"]
                action_badge = {
                    "ADD": ('<span class="chip chip-pass">THÊM MỚI</span>'),
                    "UPDATE_EXISTING": ('<span class="chip chip-info">CẬP NHẬT</span>'),
                    "UPDATE_EXPIRY": ('<span class="chip chip-warn">CẬP NHẬT HẾT HIỆU LỰC</span>'),
                }.get(cand.action, '<span class="chip chip-info">STAGING</span>')

                h(
                    f"""
<div class="proc-card" style="margin-bottom:10px;">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
    <span class="proc-code">{esc(info["ma_thu_tuc"])} · {esc(cand.source_so_hieu)}</span>
    <div>{action_badge} <span class="chip chip-info">{esc(cand.extraction_mode)}</span></div>
  </div>
  <div class="proc-name">{esc(info["ten_thu_tuc"])}</div>
  <div style="font-size:12.5px; color:var(--muted);">{esc(cand.confidence_note)}</div>
</div>
"""
                )
                with st.container(border=True):
                    e1, e2, e3 = st.columns([1.4, 2.2, 1.4])
                    new_code = e1.text_input("Mã thủ tục", value=info["ma_thu_tuc"], key=f"st_code_{cand.candidate_id}")
                    new_name = e2.text_input("Tên thủ tục", value=info["ten_thu_tuc"], key=f"st_name_{cand.candidate_id}")
                    new_group = e3.selectbox(
                        "Phân nhóm",
                        ["Thủ tục hành chính", "Tố tụng hình sự"],
                        index=0 if info["phan_nhom"] == "Thủ tục hành chính" else 1,
                        key=f"st_grp_{cand.candidate_id}",
                    )
                    e4, e5, e6 = st.columns([1.2, 1.2, 1.6])
                    new_fee = int(
                        e4.number_input(
                            "Lệ phí (VNĐ)",
                            min_value=0,
                            step=1000,
                            value=int(rec["le_phi"]["muc_phi_vnd"]),
                            key=f"st_fee_{cand.candidate_id}",
                        )
                    )
                    new_days = int(
                        e5.number_input(
                            "Thời hạn tối đa (ngày)",
                            min_value=0,
                            step=1,
                            value=int(rec["thoi_han_giai_quyet"]["so_ngay_toi_da"]),
                            key=f"st_days_{cand.candidate_id}",
                        )
                    )
                    cur_status = laws[0]["trang_thai_hieu_luc"]
                    new_status = e6.selectbox(
                        "Trạng thái hiệu lực",
                        ["ĐANG HIỆU LỰC", "HẾT HIỆU LỰC"],
                        index=0 if cur_status == "ĐANG HIỆU LỰC" else 1,
                        key=f"st_status_{cand.candidate_id}",
                    )

                    json_preview = esc(json.dumps(rec, ensure_ascii=False, indent=2))
                    h(
                        f"""
<details class="legal-drawer">
  <summary>🔍 Xem trước JSON 6 trường đầy đủ ({esc(info["ma_thu_tuc"])})</summary>
  <div class="drawer-body"><pre class="json-box">{json_preview}</pre></div>
</details>
"""
                    )

                    act_l, act_r, _ = st.columns([1.6, 1.2, 2.0])
                    if act_l.button("✅ Duyệt & Nạp vào kho chính", type="primary", key=f"app_{cand.candidate_id}", width="stretch"):
                        rec["ma_va_ten_thu_tuc"]["ma_thu_tuc"] = new_code.strip() or info["ma_thu_tuc"]
                        rec["ma_va_ten_thu_tuc"]["ten_thu_tuc"] = new_name.strip() or info["ten_thu_tuc"]
                        rec["ma_va_ten_thu_tuc"]["phan_nhom"] = new_group
                        rec["le_phi"]["muc_phi_vnd"] = new_fee
                        rec["thoi_han_giai_quyet"]["so_ngay_toi_da"] = new_days
                        rec["can_cu_phap_ly"][0]["trang_thai_hieu_luc"] = new_status
                        try:
                            res = approve_and_merge_candidate(
                                cand.candidate_id,
                                edited_record_dict=rec,
                                reload_retriever=True,
                            )
                            load_retriever.clear()
                            st.success(
                                f"Đã gộp {res['merged_code']} vào data_procedures.json (Tổng: {res['total_procedures']} thủ tục) và làm mới BM25/ChromaDB!"
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Lỗi kiểm định dữ liệu: {exc}")

                    if act_r.button("🗑️ Loại bỏ", key=f"rej_{cand.candidate_id}", width="stretch"):
                        reject_candidate(cand.candidate_id)
                        st.rerun()

        processed_cnt = sum(1 for c in staging_queue.candidates if c.status != "PENDING")
        if processed_cnt > 0:
            if st.button(f"🧹 Dọn {processed_cnt} mục đã duyệt/loại bỏ khỏi lịch sử Staging"):
                clear_processed_candidates()
                st.rerun()


# ==============================================================================
# ĐIỀU HƯỚNG VIEW CHÍNH
# ==============================================================================

view = st.session_state.get("active_view", "chat")
if view == "chat":
    render_gemini_chat()
elif view == "tra_cuu":
    render_tra_cuu()
elif view == "canh_bao":
    render_canh_bao()
elif view == "dong_bo":
    render_dong_bo()
