"""
NHÁNH PHẠM ĐỨC TRỌNG — Bộ Điều phối Đồ thị Trạng thái LangGraph (LegalMultiAgentOrchestrator):
Kết nối 3 tác tử (Retriever -> Consultant -> Judge) với vòng lặp tự sửa lỗi (tối đa 2 vòng).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Literal, Optional

from src.consultant import ConsultantAgent, extract_procedure_graph_data
from src.core.config import MAX_REVISION_LOOPS
from src.judge import JudgeAgent
from src.retriever import RetrieverAgent, get_retriever_agent
from src.schemas import IRACOutput, JudgeVerificationResult
from src.workflow.edges import route_after_judge
from src.workflow.state import LegalWorkflowState


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
        return {"irac_draft": draft_obj.model_dump()}

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
