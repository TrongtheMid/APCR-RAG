"""
NHÁNH PHẠM ĐỨC TRỌNG — Bộ Điều phối Đồ thị Trạng thái LangGraph (LegalMultiAgentOrchestrator):
- Kết nối 3 tác tử (Retriever -> Consultant -> Judge) với vòng lặp tự sửa lỗi (tối đa 2 vòng).
- Hỗ trợ Streaming tiến trình từng Agent theo thời gian thực (`stream()`).
- Hỗ trợ Hội thoại đa lượt (`chat_history`) cho các câu hỏi nối tiếp ngắn gọn.
- Hỗ trợ nhánh Clarification (khi Consultant cần hỏi ngược làm rõ).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Literal, Optional

from src.consultant import ConsultantAgent, extract_procedure_graph_data
from src.core.config import MAX_REVISION_LOOPS
from src.judge import JudgeAgent
from src.retriever import RetrieverAgent, get_retriever_agent
from src.schemas import IRACOutput, JudgeVerificationResult
from src.workflow.edges import route_after_consultant, route_after_judge
from src.workflow.intent_router import build_conversational_reply, classify_chat_intent
from src.workflow.state import LegalWorkflowState


_FOLLOWUP_TRIGGERS = (
    "lệ phí", "bao nhiêu tiền", "mất phí", "tốn bao nhiêu", "phí bao nhiêu",
    "bao lâu", "mấy ngày", "thời hạn", "nộp ở đâu", "cơ quan nào", "làm ở đâu",
    "cần giấy tờ gì", "giấy tờ gì", "hồ sơ gồm", "biểu mẫu nào", "cần những gì", "chuẩn bị gì",
    "các bước", "trình tự", "quy trình", "làm thế nào", "điều kiện", "lưu ý gì",
    "online", "trực tuyến", "thế còn", "vậy thì", "còn điều kiện", "trình tự thế nào",
)


def contextualize_followup_query(
    query: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Nếu người dùng đặt câu hỏi nối tiếp trong Chatbot (VD: 'Nộp ở đâu?', 'Lệ phí bao nhiêu?',
    'Cần chuẩn bị giấy tờ gì?'), tự động kiểm tra xem câu hỏi mới đã chuyển sang thủ tục khác chưa;
    nếu chưa chuyển chủ đề thì ghép ngữ cảnh thủ tục của lượt liền trước để Retriever giữ đúng mạch hội thoại.
    """
    if not chat_history:
        return query
    q_clean = query.strip()
    q_lower = q_clean.lower()

    # Nếu câu hỏi mới đã nhắc rõ một thủ tục mới trong từ điển pháp lý -> người dùng chuyển chủ đề
    from src.retriever.query_rewriter import rewrite_query_and_route

    new_routing = rewrite_query_and_route(q_clean)
    if new_routing.get("matched_expansions"):
        return query

    is_short_followup = len(q_clean.split()) <= 16 and any(t in q_lower for t in _FOLLOWUP_TRIGGERS)
    if not is_short_followup:
        return query

    last_turn = chat_history[-1]
    prev_code = last_turn.get("procedure_code") or ""
    prev_proc = last_turn.get("procedure_name") or ""
    prev_q = last_turn.get("query") or ""
    anchor = f"{prev_code} {prev_proc}".strip() if (prev_code or prev_proc) else prev_q
    if not anchor or anchor.lower() in q_lower:
        return query
    return f"{q_clean} (liên quan đến thủ tục trước đó: {anchor})"


class LegalMultiAgentOrchestrator:
    """Bộ điều phối 3 tác tử (Retriever -> Consultant -> Judge) trên đồ thị trạng thái LangGraph."""

    def __init__(
        self,
        retriever: Optional[RetrieverAgent] = None,
        consultant: Optional[ConsultantAgent] = None,
        judge: Optional[JudgeAgent] = None,
        max_revisions: int = MAX_REVISION_LOOPS,
    ) -> None:
        self.retriever = retriever or get_retriever_agent()
        self.consultant = consultant or ConsultantAgent()
        self.judge = judge or JudgeAgent()
        self.max_revisions = max_revisions
        self.compiled_graph = self._build_langgraph()
        self.execution_engine = (
            "LangGraph StateGraph" if self.compiled_graph is not None else "Sequential Graph Engine"
        )

    def retrieve_node(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """Node 1: Thực thi Retriever Agent (Query Rewriting + Metadata Filter + Hybrid Search)."""
        t0 = time.perf_counter()
        eff_query = contextualize_followup_query(
            state["query"], state.get("chat_history")
        )
        res = self.retriever.retrieve_hybrid(
            query=eff_query,
            top_k=3,
            phan_nhom=state.get("phan_nhom_filter"),
            linh_vuc=state.get("linh_vuc_filter"),
            giai_doan_to_tung=state.get("giai_doan_filter"),
            only_active=state.get("only_active", True),
            effective_date=state.get("effective_date"),
        )
        items = res["results"]
        primary_proc = items[0]["procedure"] if items else None
        graph_data = extract_procedure_graph_data(primary_proc) if primary_proc else None
        node_ms = round((time.perf_counter() - t0) * 1000, 2)

        top_code = items[0]["ma_thu_tuc"] if items else "NONE"
        top_score = items[0]["hybrid_score"] if items else 0.0
        trace = list(state.get("workflow_trace", []))
        trace.append(
            {
                "node": "retriever",
                "label": "Retriever Agent (Hybrid Search)",
                "latency_ms": node_ms,
                "summary": f"Truy xuất {len(items)} hồ sơ · Top-1: {top_code} (score={top_score:.2f})",
            }
        )

        return {
            "effective_query": eff_query,
            "routing_info": res["routing_info"],
            "retrieved_items": items,
            "expired_alerts": res["expired_alerts"],
            "primary_procedure": primary_proc,
            "graph_data": graph_data,
            "workflow_trace": trace,
        }

    def consultant_node(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """Node 2: Thực thi Consultant Agent (Sinh bản nháp IRAC hoặc Tự hiệu chỉnh Self-Revision)."""
        t0 = time.perf_counter()
        rev_count = state.get("revision_count", 0)
        eff_query = state.get("effective_query") or state["query"]
        draft_obj: IRACOutput = self.consultant.generate_irac_consultation(
            query=eff_query,
            retrieved_items=state.get("retrieved_items", []),
            expired_alerts=state.get("expired_alerts", []),
            critique_feedback=state.get("critique_feedback"),
            revision_count=rev_count,
            simulate_initial_error=state.get("simulate_initial_error", False),
            chat_history=state.get("chat_history"),
        )
        node_ms = round((time.perf_counter() - t0) * 1000, 2)
        trace = list(state.get("workflow_trace", []))
        action_lbl = "Bản nháp IRAC đầu tiên" if rev_count == 0 else f"Tự hiệu chỉnh lần #{rev_count}"
        summary_msg = (
            "Kích hoạt Từ chối An toàn (Deterministic Refusal)"
            if draft_obj.is_refusal
            else f"{action_lbl} · Độ tin cậy {int(draft_obj.confidence_score * 100)}%"
        )
        trace.append(
            {
                "node": "consultant",
                "label": f"Consultant Agent ({action_lbl})",
                "latency_ms": node_ms,
                "summary": summary_msg,
            }
        )
        return {
            "irac_draft": draft_obj.model_dump(),
            "needs_clarification": False,
            "workflow_trace": trace,
        }

    def judge_node(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """Node 3: Thực thi Judge Agent 2 lớp (Lớp 1 Code Tất định + Regex, Lớp 2 Claim-Level Critic)."""
        t0 = time.perf_counter()
        draft_dict = state["irac_draft"]
        draft_obj = IRACOutput.model_validate(draft_dict)
        primary_proc = state.get("primary_procedure")
        rev_count = state.get("revision_count", 0)
        eff_query = state.get("effective_query") or state["query"]

        ver_obj: JudgeVerificationResult = self.judge.evaluate_draft(
            query=eff_query,
            draft=draft_obj,
            ground_truth_proc=primary_proc,
        )

        history = list(state.get("revision_history", []))
        history.append(
            {
                "iteration": rev_count + 1,
                "is_revision": rev_count > 0,
                "draft": draft_obj.model_dump(),
                "verification": ver_obj.model_dump(),
            }
        )

        if draft_obj.is_refusal:
            status = "REFUSED_SAFELY"
        elif ver_obj.passed:
            status = "VERIFIED_PASS" if rev_count == 0 else f"REVISED_PASS_AFTER_{rev_count}_LOOP"
        else:
            status = "VERIFICATION_FAILED"

        next_rev_count = rev_count if ver_obj.passed else (rev_count + 1)
        node_ms = round((time.perf_counter() - t0) * 1000, 2)
        trace = list(state.get("workflow_trace", []))
        trace.append(
            {
                "node": "judge",
                "label": f"Judge Agent 2 Lớp (Vòng #{rev_count + 1})",
                "latency_ms": node_ms,
                "summary": (
                    f"PASS · Lớp 1: {ver_obj.layer1_score * 100:.0f}% · "
                    f"Lớp 2 Claim Coverage: {ver_obj.claim_coverage_score * 100:.0f}%"
                    if ver_obj.passed
                    else f"FAIL ({len(ver_obj.layer1_errors)} lỗi L1, {len(ver_obj.layer2_errors)} lỗi L2) -> Trả về Consultant sửa"
                ),
            }
        )

        return {
            "verification_result": ver_obj.model_dump(),
            "critique_feedback": ver_obj.critique_feedback,
            "revision_count": next_rev_count,
            "revision_history": history,
            "final_status": status,
            "workflow_trace": trace,
        }

    def fallback_warning_node(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """Node 4: Kích hoạt cảnh báo an toàn nếu sau tối đa 2 vòng sửa lỗi bản nháp vẫn không vượt qua Judge."""
        t0 = time.perf_counter()
        draft_dict = dict(state.get("irac_draft") or {})
        draft_dict["conclusion"] = (
            str(draft_dict.get("conclusion", ""))
            + "\n\n⚠️ **[CẢNH BÁO AN TOÀN TỪ HỆ THỐNG ĐA TÁC TỬ]**: Sau 02 vòng tự hiệu chỉnh, "
            "một số chi tiết trong bản nháp vẫn chưa đạt độ khớp tuyệt đối theo tiêu chuẩn của Judge Agent. "
            "Người dân vui lòng đối chiếu trực tiếp văn bản gốc ở cột bên phải."
        )
        trace = list(state.get("workflow_trace", []))
        trace.append(
            {
                "node": "fallback_warning",
                "label": "Fallback Safety Guard",
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "summary": "Vượt quá số vòng tự sửa tối đa -> Chèn cảnh báo đối chiếu văn bản gốc.",
            }
        )
        return {
            "irac_draft": draft_dict,
            "final_status": "MAX_REVISIONS_WARNING",
            "workflow_trace": trace,
        }

    def conversational_node(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """Node Hội thoại Tự nhiên: Xử lý chào hỏi, liệt kê dịch vụ hỗ trợ hoặc tóm tắt nối tiếp."""
        t0 = time.perf_counter()
        intent = state.get("intent", "greeting")
        chat_history = state.get("chat_history") or []
        reply = build_conversational_reply(
            intent=intent,  # type: ignore[arg-type]
            query=state["query"],
            procedures=self.retriever.raw_procedures,
            chat_history=chat_history,
            phan_nhom_filter=state.get("phan_nhom_filter"),
            use_llm=self.consultant.use_llm,
        )
        draft_obj = IRACOutput(
            is_refusal=False,
            refusal_reason=None,
            direct_advice=reply,
            issue="Hội thoại tự nhiên & Định hướng dịch vụ hỗ trợ APCR-RAG.",
            rule="APCR-RAG Intent Router (Conversational Mode).",
            application=reply,
            conclusion=reply,
            citations=None,
            confidence_score=0.96,
        )
        primary_proc = None
        if intent == "followup_freeform" and chat_history:
            prev_code = chat_history[-1].get("procedure_code") or ""
            primary_proc = self.retriever.procedures_by_id.get(prev_code)
        graph_data = extract_procedure_graph_data(primary_proc) if primary_proc else None

        node_ms = round((time.perf_counter() - t0) * 1000, 2)
        trace = list(state.get("workflow_trace", []))
        trace.append(
            {
                "node": "conversational",
                "label": "Trợ lý Hội thoại (Intent Router)",
                "latency_ms": node_ms,
                "summary": f"Phân loại ý định `{intent}` · Phản hồi tự nhiên trực tiếp.",
            }
        )
        return {
            "irac_draft": draft_obj.model_dump(),
            "primary_procedure": primary_proc,
            "graph_data": graph_data,
            "verification_result": {
                "passed": True,
                "layer1_passed": True,
                "layer1_errors": [],
                "layer2_passed": True,
                "layer2_errors": [],
                "critique_feedback": "Lượt hội thoại tự nhiên / tra cứu phạm vi dịch vụ (không áp dụng khung IRAC đơn thủ tục).",
            },
            "final_status": "CONVERSATIONAL_REPLY",
            "workflow_trace": trace,
        }

    def route_after_consultant(
        self, state: LegalWorkflowState
    ) -> Literal["clarify", "judge"]:
        return route_after_consultant(state)

    def route_after_judge(
        self, state: LegalWorkflowState
    ) -> Literal["accept", "revise", "fallback"]:
        return route_after_judge(state, default_max_revisions=self.max_revisions)

    def _build_langgraph(self) -> Any:
        """Khởi tạo và biên dịch đồ thị trạng thái StateGraph của thư viện LangGraph."""
        try:
            from langgraph.graph import END, START, StateGraph

            workflow = StateGraph(LegalWorkflowState)
            workflow.add_node("retriever", self.retrieve_node)
            workflow.add_node("consultant", self.consultant_node)
            workflow.add_node("judge", self.judge_node)
            workflow.add_node("fallback_warning", self.fallback_warning_node)

            workflow.add_edge(START, "retriever")
            workflow.add_edge("retriever", "consultant")
            workflow.add_conditional_edges(
                "consultant",
                self.route_after_consultant,
                {
                    "clarify": END,
                    "judge": "judge",
                },
            )
            workflow.add_conditional_edges(
                "judge",
                self.route_after_judge,
                {
                    "accept": END,
                    "revise": "consultant",
                    "fallback": "fallback_warning",
                },
            )
            workflow.add_edge("fallback_warning", END)
            return workflow.compile()
        except Exception:
            return None

    def _build_initial_state(
        self,
        query: str,
        phan_nhom_filter: Optional[str] = None,
        linh_vuc_filter: Optional[str] = None,
        giai_doan_filter: Optional[str] = None,
        effective_date: Optional[str] = None,
        only_active: bool = True,
        simulate_initial_error: bool = False,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> LegalWorkflowState:
        return {
            "query": query,
            "effective_query": query,
            "chat_history": list(chat_history or []),
            "phan_nhom_filter": phan_nhom_filter,
            "linh_vuc_filter": linh_vuc_filter,
            "giai_doan_filter": giai_doan_filter,
            "effective_date": effective_date,
            "only_active": only_active,
            "simulate_initial_error": simulate_initial_error,
            "needs_clarification": False,
            "clarification_question": None,
            "revision_count": 0,
            "max_revisions": self.max_revisions,
            "revision_history": [],
            "workflow_trace": [],
            "execution_engine": self.execution_engine,
        }

    def stream(
        self,
        query: str,
        phan_nhom_filter: Optional[str] = None,
        linh_vuc_filter: Optional[str] = None,
        giai_doan_filter: Optional[str] = None,
        effective_date: Optional[str] = None,
        only_active: bool = True,
        simulate_initial_error: bool = False,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Chạy luồng Multi-Agent ở chế độ Streaming, phát ra từng sự kiện node
        (retriever -> consultant -> judge -> ...) để giao diện hiển thị tiến trình thời gian thực.
        Sự kiện cuối cùng có `node == "__end__"` chứa toàn bộ `state` hoàn chỉnh.
        """
        start_time = time.perf_counter()
        state: Dict[str, Any] = dict(
            self._build_initial_state(
                query=query,
                phan_nhom_filter=phan_nhom_filter,
                linh_vuc_filter=linh_vuc_filter,
                giai_doan_filter=giai_doan_filter,
                effective_date=effective_date,
                only_active=only_active,
                simulate_initial_error=simulate_initial_error,
                chat_history=chat_history,
            )
        )

        routing_preview = self.retriever.rewrite_query_and_route(
            query, preferred_domain=phan_nhom_filter
        )
        intent = classify_chat_intent(
            query=query,
            chat_history=chat_history,
            matched_expansions=routing_preview.get("matched_expansions"),
        )
        state["intent"] = intent
        state["is_conversational"] = intent != "legal_qa"

        if intent != "legal_qa":
            state.update(self.conversational_node(state))  # type: ignore[arg-type]
            yield self._make_stream_event("conversational", state)
        elif self.compiled_graph is not None:
            for chunk in self.compiled_graph.stream(state):
                for node_name, update in chunk.items():
                    if isinstance(update, dict):
                        state.update(update)
                        last_step = (state.get("workflow_trace") or [{}])[-1]
                        yield {
                            "node": node_name,
                            "label": last_step.get("label", node_name),
                            "summary": last_step.get("summary", ""),
                            "latency_ms": last_step.get("latency_ms", 0.0),
                            "state": dict(state),
                        }
        else:
            state.update(self.retrieve_node(state))  # type: ignore[arg-type]
            yield self._make_stream_event("retriever", state)

            while True:
                state.update(self.consultant_node(state))  # type: ignore[arg-type]
                yield self._make_stream_event("consultant", state)

                if self.route_after_consultant(state) == "clarify":  # type: ignore[arg-type]
                    state["final_status"] = "NEEDS_CLARIFICATION"
                    break

                state.update(self.judge_node(state))  # type: ignore[arg-type]
                yield self._make_stream_event("judge", state)

                decision = self.route_after_judge(state)  # type: ignore[arg-type]
                if decision == "accept":
                    break
                if decision == "fallback":
                    state.update(self.fallback_warning_node(state))  # type: ignore[arg-type]
                    yield self._make_stream_event("fallback_warning", state)
                    break

        state["total_latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
        yield {
            "node": "__end__",
            "label": "Hoàn tất quy trình Multi-Agent",
            "summary": f"Trạng thái: {state.get('final_status', '')} ({state['total_latency_ms']} ms)",
            "latency_ms": state["total_latency_ms"],
            "state": state,
        }

    @staticmethod
    def _make_stream_event(node_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        last_step = (state.get("workflow_trace") or [{}])[-1]
        return {
            "node": node_name,
            "label": last_step.get("label", node_name),
            "summary": last_step.get("summary", ""),
            "latency_ms": last_step.get("latency_ms", 0.0),
            "state": dict(state),
        }

    def invoke(
        self,
        query: str,
        phan_nhom_filter: Optional[str] = None,
        linh_vuc_filter: Optional[str] = None,
        giai_doan_filter: Optional[str] = None,
        effective_date: Optional[str] = None,
        only_active: bool = True,
        simulate_initial_error: bool = False,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> LegalWorkflowState:
        """Chạy toàn bộ luồng Multi-Agent RAG trên đồ thị LangGraph và trả về trạng thái cuối cùng."""
        final_state: Dict[str, Any] = {}
        for event in self.stream(
            query=query,
            phan_nhom_filter=phan_nhom_filter,
            linh_vuc_filter=linh_vuc_filter,
            giai_doan_filter=giai_doan_filter,
            effective_date=effective_date,
            only_active=only_active,
            simulate_initial_error=simulate_initial_error,
            chat_history=chat_history,
        ):
            final_state = event["state"]
        return final_state  # type: ignore[return-value]


_DEFAULT_ORCHESTRATOR: Optional[LegalMultiAgentOrchestrator] = None


def get_orchestrator() -> LegalMultiAgentOrchestrator:
    """Lấy thực thể LegalMultiAgentOrchestrator dùng chung (Singleton)."""
    global _DEFAULT_ORCHESTRATOR
    if _DEFAULT_ORCHESTRATOR is None:
        _DEFAULT_ORCHESTRATOR = LegalMultiAgentOrchestrator()
    return _DEFAULT_ORCHESTRATOR


def run_legal_multi_agent_workflow(
    query: str,
    phan_nhom_filter: Optional[str] = None,
    linh_vuc_filter: Optional[str] = None,
    giai_doan_filter: Optional[str] = None,
    effective_date: Optional[str] = None,
    only_active: bool = True,
    simulate_initial_error: bool = False,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> LegalWorkflowState:
    """Hàm tiện ích gọi nhanh luồng Multi-Agent LangGraph từ giao diện Streamlit hoặc script đánh giá."""
    return get_orchestrator().invoke(
        query=query,
        phan_nhom_filter=phan_nhom_filter,
        linh_vuc_filter=linh_vuc_filter,
        giai_doan_filter=giai_doan_filter,
        effective_date=effective_date,
        only_active=only_active,
        simulate_initial_error=simulate_initial_error,
        chat_history=chat_history,
    )


def stream_legal_multi_agent_workflow(
    query: str,
    phan_nhom_filter: Optional[str] = None,
    linh_vuc_filter: Optional[str] = None,
    giai_doan_filter: Optional[str] = None,
    effective_date: Optional[str] = None,
    only_active: bool = True,
    simulate_initial_error: bool = False,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Iterator[Dict[str, Any]]:
    """Hàm tiện ích phát tiến trình từng Agent (Streaming) cho giao diện Streamlit."""
    return get_orchestrator().stream(
        query=query,
        phan_nhom_filter=phan_nhom_filter,
        linh_vuc_filter=linh_vuc_filter,
        giai_doan_filter=giai_doan_filter,
        effective_date=effective_date,
        only_active=only_active,
        simulate_initial_error=simulate_initial_error,
        chat_history=chat_history,
    )
