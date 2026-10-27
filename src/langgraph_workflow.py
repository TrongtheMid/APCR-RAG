"""
NHÁNH CỦA PHẠM ĐỨC TRỌNG (HẠNG MỤC 3.2):
Điều phối Đồ thị Trạng thái LangGraph (Multi-Agent RAG Workflow):
- Kết nối 3 tác tử: Retriever Agent -> Consultant Agent -> Judge Agent.
- Cạnh điều kiện (Conditional Edges):
  + Nếu Judge chấm PASS (hoặc Consultant thực hiện Deterministic Refusal): Kết thúc và xuất kết quả chuẩn IRAC + Đồ thị 6 trường.
  + Nếu Judge chấm FAIL và số vòng sửa lỗi < max_revisions (tối đa 2 vòng): Quay lại Consultant Agent kèm critique_feedback (Self-Revision Loop).
  + Nếu sau 2 vòng sửa lỗi vẫn FAIL: Kích hoạt nút Fallback An toàn (Safe Warning / Refusal) để không trả thông tin sai cho người dân.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict

from src.consultant_agent import ConsultantAgent, extract_procedure_graph_data
from src.judge_agent import JudgeAgent
from src.retriever_agent import RetrieverAgent, get_retriever_agent
from src.schema import IRACOutput, JudgeVerificationResult


class LegalWorkflowState(TypedDict, total=False):
    """Trạng thái xuyên suốt đồ thị LangGraph của hệ thống Đa tác tử."""
    query: str
    phan_nhom_filter: Optional[str]
    linh_vuc_filter: Optional[str]
    giai_doan_filter: Optional[str]
    effective_date: Optional[str]
    only_active: bool
    simulate_initial_error: bool

    # Kết quả từ AGENT 01 — RETRIEVER
    routing_info: Dict[str, Any]
    retrieved_items: List[Dict[str, Any]]
    expired_alerts: List[Dict[str, Any]]
    primary_procedure: Optional[Dict[str, Any]]

    # Kết quả từ AGENT 02 — CONSULTANT
    irac_draft: Optional[Dict[str, Any]]

    # Kết quả từ AGENT 03 — JUDGE
    verification_result: Optional[Dict[str, Any]]
    critique_feedback: Optional[str]

    # Quản lý vòng lặp phản biện (tối đa 2 vòng)
    revision_count: int
    max_revisions: int
    revision_history: List[Dict[str, Any]]

    # Đầu ra cuối cùng cho giao diện Streamlit
    graph_data: Optional[Dict[str, Any]]
    final_status: str
    total_latency_ms: float


class LegalMultiAgentOrchestrator:
    """Bộ điều phối 3 tác tử (Retriever -> Consultant -> Judge) trên đồ thị trạng thái LangGraph."""

    def __init__(
        self,
        retriever: Optional[RetrieverAgent] = None,
        consultant: Optional[ConsultantAgent] = None,
        judge: Optional[JudgeAgent] = None,
        max_revisions: int = 2,
    ) -> None:
        self.retriever = retriever or get_retriever_agent()
        self.consultant = consultant or ConsultantAgent()
        self.judge = judge or JudgeAgent()
        self.max_revisions = int(os.getenv("MAX_REVISION_LOOPS", max_revisions))
        self.compiled_graph = self._build_langgraph()

    # --------------------------------------------------------------------------
    # CÁC NÚT XỬ LÝ TRÊN ĐỒ THỊ (GRAPH NODES)
    # --------------------------------------------------------------------------

    def retrieve_node(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """Node 1: Thực thi Retriever Agent (Query Rewriting + Metadata Filter + Hybrid Search)."""
        res = self.retriever.retrieve_hybrid(
            query=state["query"],
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

        return {
            "routing_info": res["routing_info"],
            "retrieved_items": items,
            "expired_alerts": res["expired_alerts"],
            "primary_procedure": primary_proc,
            "graph_data": graph_data,
        }

    def consultant_node(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """Node 2: Thực thi Consultant Agent (Sinh bản nháp IRAC hoặc Tự hiệu chỉnh Self-Revision)."""
        rev_count = state.get("revision_count", 0)
        draft_obj: IRACOutput = self.consultant.generate_irac_consultation(
            query=state["query"],
            retrieved_items=state.get("retrieved_items", []),
            expired_alerts=state.get("expired_alerts", []),
            critique_feedback=state.get("critique_feedback"),
            revision_count=rev_count,
            simulate_initial_error=state.get("simulate_initial_error", False),
        )
        return {
            "irac_draft": draft_obj.model_dump(),
        }

    def judge_node(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """Node 3: Thực thi Judge Agent 2 lớp (Lớp 1 Code Tất định + Lớp 2 Phản biện Ngữ nghĩa)."""
        draft_dict = state["irac_draft"]
        draft_obj = IRACOutput.model_validate(draft_dict)
        primary_proc = state.get("primary_procedure")
        rev_count = state.get("revision_count", 0)

        ver_obj: JudgeVerificationResult = self.judge.evaluate_draft(
            query=state["query"],
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

        return {
            "verification_result": ver_obj.model_dump(),
            "critique_feedback": ver_obj.critique_feedback,
            "revision_count": next_rev_count,
            "revision_history": history,
            "final_status": status,
        }

    def fallback_warning_node(self, state: LegalWorkflowState) -> Dict[str, Any]:
        """Node 4: Kích hoạt cảnh báo an toàn nếu sau tối đa 2 vòng sửa lỗi bản nháp vẫn không vượt qua Judge."""
        draft_dict = dict(state.get("irac_draft") or {})
        draft_dict["conclusion"] = (
            str(draft_dict.get("conclusion", ""))
            + "\n\n⚠️ **[CẢNH BÁO AN TOÀN TỪ HỆ THỐNG ĐA TÁC TỬ]**: Sau 02 vòng tự hiệu chỉnh, "
            "một số chi tiết trong bản nháp vẫn chưa đạt độ khớp tuyệt đối theo tiêu chuẩn của Judge Agent. "
            "Người dân vui lòng đối chiếu trực tiếp văn bản gốc ở cột bên phải."
        )
        return {
            "irac_draft": draft_dict,
            "final_status": "MAX_REVISIONS_WARNING",
        }

    # --------------------------------------------------------------------------
    # CẠNH ĐIỀU KIỆN (CONDITIONAL EDGE ROUTER)
    # --------------------------------------------------------------------------

    def route_after_judge(
        self, state: LegalWorkflowState
    ) -> Literal["accept", "revise", "fallback"]:
        """
        Quyết định luồng đi sau khi Judge Agent chấm điểm:
        - 'accept': Bản nháp đạt chuẩn (PASS) hoặc đã từ chối an toàn -> Kết thúc (END).
        - 'revise': Phát hiện lỗi (FAIL) và còn lượt sửa (revision_count <= max_revisions) -> Quay lại Consultant.
        - 'fallback': Vượt quá số vòng tối đa (2 vòng) -> Chuyển sang nút cảnh báo an toàn rồi kết thúc.
        """
        ver = state.get("verification_result") or {}
        if ver.get("passed", False):
            return "accept"

        if state.get("revision_count", 0) <= state.get("max_revisions", self.max_revisions):
            return "revise"

        return "fallback"

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
            workflow.add_edge("consultant", "judge")
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

    def invoke(
        self,
        query: str,
        phan_nhom_filter: Optional[str] = None,
        linh_vuc_filter: Optional[str] = None,
        giai_doan_filter: Optional[str] = None,
        effective_date: Optional[str] = None,
        only_active: bool = True,
        simulate_initial_error: bool = False,
    ) -> LegalWorkflowState:
        """Chạy toàn bộ luồng Multi-Agent RAG trên đồ thị LangGraph và trả về trạng thái cuối cùng."""
        start_time = time.perf_counter()
        initial_state: LegalWorkflowState = {
            "query": query,
            "phan_nhom_filter": phan_nhom_filter,
            "linh_vuc_filter": linh_vuc_filter,
            "giai_doan_filter": giai_doan_filter,
            "effective_date": effective_date,
            "only_active": only_active,
            "simulate_initial_error": simulate_initial_error,
            "revision_count": 0,
            "max_revisions": self.max_revisions,
            "revision_history": [],
        }

        if self.compiled_graph is not None:
            final_state: LegalWorkflowState = self.compiled_graph.invoke(initial_state)
        else:
            # Chế độ thực thi tuần tự tương đương nếu môi trường chưa cài gói langgraph
            state = dict(initial_state)
            state.update(self.retrieve_node(state))  # type: ignore[arg-type]
            while True:
                state.update(self.consultant_node(state))  # type: ignore[arg-type]
                state.update(self.judge_node(state))  # type: ignore[arg-type]
                decision = self.route_after_judge(state)  # type: ignore[arg-type]
                if decision == "accept":
                    break
                if decision == "fallback":
                    state.update(self.fallback_warning_node(state))  # type: ignore[arg-type]
                    break
            final_state = state  # type: ignore[assignment]

        final_state["total_latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
        return final_state


_DEFAULT_ORCHESTRATOR: Optional[LegalMultiAgentOrchestrator] = None


def run_legal_multi_agent_workflow(
    query: str,
    phan_nhom_filter: Optional[str] = None,
    linh_vuc_filter: Optional[str] = None,
    giai_doan_filter: Optional[str] = None,
    effective_date: Optional[str] = None,
    only_active: bool = True,
    simulate_initial_error: bool = False,
) -> LegalWorkflowState:
    """Hàm tiện ích gọi nhanh luồng Multi-Agent LangGraph từ giao diện Streamlit hoặc script đánh giá."""
    global _DEFAULT_ORCHESTRATOR
    if _DEFAULT_ORCHESTRATOR is None:
        _DEFAULT_ORCHESTRATOR = LegalMultiAgentOrchestrator()
    return _DEFAULT_ORCHESTRATOR.invoke(
        query=query,
        phan_nhom_filter=phan_nhom_filter,
        linh_vuc_filter=linh_vuc_filter,
        giai_doan_filter=giai_doan_filter,
        effective_date=effective_date,
        only_active=only_active,
        simulate_initial_error=simulate_initial_error,
    )
