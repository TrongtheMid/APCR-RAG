"""
NHÁNH VŨ GIA ANH ĐỨC — Module Clarification (Hỏi ngược làm rõ dữ kiện pháp lý):
Nhận diện các tình huống thiếu dữ kiện có thể làm thay đổi thủ tục áp dụng hoặc cơ quan có thẩm quyền,
và sinh câu hỏi làm rõ ngắn gọn, cụ thể, chỉ hỏi đúng thông tin tối thiểu cần thiết.

Thiết kế theo kiến trúc mở rộng:
1. Quy tắc suy luận theo cấu trúc hồ sơ/trường dữ liệu (`ClarificationRule` registry) — không đóng kín
   theo mã thủ tục cố định; có thể đăng ký thêm quy tắc mới qua `register_clarification_rule()`.
2. Bộ phân tích động ứng viên truy xuất (`_evaluate_dynamic_candidate_ambiguity`): tự động phát hiện
   hai thủ tục ứng viên sát điểm nhau nhưng khác cơ quan thẩm quyền khi câu hỏi chưa nêu đặc trưng phân nhánh.
3. Không tự chọn một nhánh pháp lý khi dữ kiện còn thiếu; giữ tương thích với `IRACOutput` và workflow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from src.schemas import IRACOutput


class ClarificationResult(BaseModel):
    """Kết quả đánh giá nhu cầu hỏi ngược làm rõ của Consultant Agent."""

    needs_clarification: bool = Field(
        default=False,
        description="True nếu thiếu dữ kiện phân định thủ tục hoặc thẩm quyền giải quyết",
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Câu hỏi ngắn, cụ thể yêu cầu người dùng bổ sung dữ kiện tối thiểu",
    )
    rule_id: Optional[str] = Field(
        default=None,
        description="Định danh quy tắc hoặc bộ phân tích đã kích hoạt hỏi ngược",
    )
    missing_facts: List[str] = Field(
        default_factory=list,
        description="Danh sách các dữ kiện còn thiếu cần người dùng xác nhận",
    )
    candidate_procedures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Danh sách các nhánh thủ tục/thẩm quyền ứng viên đang bị phân vân",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Giải thích ngắn gọn lý do cần hỏi ngược làm rõ",
    )


@dataclass
class ClarificationRule:
    """Định nghĩa một quy tắc nhận diện thiếu dữ kiện có thể mở rộng."""

    rule_id: str
    description: str
    evaluator: Callable[
        [str, List[Dict[str, Any]], Optional[List[Dict[str, str]]]],
        Optional[ClarificationResult],
    ]


_CLARIFICATION_RULES: List[ClarificationRule] = []

_AUTHORITY_QUERY_MARKERS = (
    "nộp ở đâu",
    "nộp tại đâu",
    "cơ quan nào",
    "nộp cho cơ quan nào",
    "nộp cho ai",
    "gửi đến đâu",
    "gửi cho cơ quan nào",
    "gửi ở đâu",
    "nơi nào tiếp nhận",
    "thẩm quyền",
    "ubnd xã hay",
    "xã hay huyện",
    "sở tư pháp hay tòa án",
    "công an hay viện kiểm sát",
)

_DELEGATION_OR_CHOICE_MARKERS = (
    "ủy quyền",
    "làm hộ",
    "đi hộ",
    "nộp hộ",
    "làm thay",
    "nộp thay",
    "nhờ người",
    "nhờ mẹ",
    "nhờ cha",
    "nhờ bố",
    "nhờ vợ",
    "nhờ chồng",
    "nhờ con",
    "loại nào",
    "số nào",
    "số 1 hay số 2",
    "khác nhau",
)

_TITLE_STOPWORDS = {
    "thủ", "tục", "đăng", "ký", "cấp", "xin", "yêu", "cầu", "giải", "quyết",
    "cho", "của", "và", "hoặc", "tại", "về", "trong", "theo", "mới", "lại",
    "các", "người", "công", "dân", "hiện", "nay", "quy", "định",
}


def _combine_query_and_history(
    query: str,
    chat_history: Optional[List[Dict[str, str]]],
) -> str:
    """Ghép câu hỏi hiện tại với các lượt hội thoại gần nhất để kiểm tra dữ kiện đã cung cấp."""
    parts: List[str] = [query or ""]
    if chat_history:
        for turn in chat_history[-5:]:
            if not isinstance(turn, dict):
                continue
            q = str(turn.get("query") or turn.get("user") or "").strip()
            if q:
                parts.append(q)
    return " ".join(parts).lower()


def _proc_summary(proc: Dict[str, Any]) -> Dict[str, Any]:
    info = proc.get("ma_va_ten_thu_tuc") or {}
    return {
        "ma_thu_tuc": info.get("ma_thu_tuc", ""),
        "ten_thu_tuc": info.get("ten_thu_tuc", ""),
        "cap_tham_quyen": list(info.get("cap_tham_quyen") or []),
    }


def _is_active_proc(proc: Dict[str, Any]) -> bool:
    laws = proc.get("can_cu_phap_ly") or []
    return bool(laws) and laws[0].get("trang_thai_hieu_luc") == "ĐANG HIỆU LỰC"


def _evaluate_marriage_jurisdiction_rule(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]],
) -> Optional[ClarificationResult]:
    """
    Phân định kết hôn trong nước (UBND cấp xã, 0 VNĐ, 1 ngày) vs có yếu tố nước ngoài
    (UBND cấp huyện, 1.000.000 VNĐ, 15 ngày) dựa trên dữ liệu lĩnh vực/tên thủ tục.
    """
    q_lower = query.lower()
    if "kết hôn" not in q_lower:
        return None
    # Không hỏi ngược nếu người dùng đang hỏi thủ tục Giấy xác nhận tình trạng hôn nhân hoặc hỏi tình huống ký nhận một mình
    if any(
        k in q_lower
        for k in (
            "tình trạng hôn nhân",
            "độc thân",
            "ly hôn",
            "một mình",
            "vắng mặt",
            "ký nhận",
            "bao nhiêu tuổi",
            "đủ tuổi",
        )
    ):
        return None

    combined = _combine_query_and_history(query, chat_history)
    domestic_markers = (
        "trong nước",
        "công dân việt nam",
        "người việt nam",
        "cùng quốc tịch",
        "ubnd xã",
        "ubnd phường",
        "ubnd thị trấn",
        "cấp xã",
    )
    foreign_markers = (
        "nước ngoài",
        "quốc tịch",
        "ngoại quốc",
        "việt kiều",
        "định cư ở nước ngoài",
        "hợp pháp hóa lãnh sự",
        "ubnd huyện",
        "ubnd quận",
        "cấp huyện",
        "hàn quốc",
        "nhật bản",
        "trung quốc",
        "đài loan",
        "singapore",
        "thái lan",
        "pháp",
        "mỹ",
        "anh",
        "đức",
        "úc",
        "canada",
        "nga",
    )
    if any(m in combined for m in domestic_markers) or any(m in combined for m in foreign_markers):
        return ClarificationResult(
            needs_clarification=False,
            rule_id="marriage_jurisdiction_branch",
            reason="Đã có đủ dữ kiện phân định kết hôn trong nước hay có yếu tố nước ngoài.",
        )

    marriage_candidates = [
        it["procedure"]
        for it in retrieved_items
        if _is_active_proc(it["procedure"])
        and "kết hôn" in (it["procedure"].get("ma_va_ten_thu_tuc", {}).get("ten_thu_tuc", "").lower())
    ]
    if not marriage_candidates:
        return None

    return ClarificationResult(
        needs_clarification=True,
        clarification_question=(
            "Để xác định đúng cơ quan có thẩm quyền (UBND cấp xã hay UBND cấp huyện), "
            "thời hạn và lệ phí, bạn vui lòng cho biết hai bên dự định đăng ký kết hôn đều là "
            "công dân Việt Nam cư trú trong nước hay có yếu tố nước ngoài?"
        ),
        rule_id="marriage_jurisdiction_branch",
        missing_facts=["yếu tố quốc tịch / cư trú của hai bên kết hôn (trong nước hay có yếu tố nước ngoài)"],
        candidate_procedures=[_proc_summary(p) for p in marriage_candidates[:3]],
        reason="Đăng ký kết hôn trong nước và có yếu tố nước ngoài khác nhau về thẩm quyền, lệ phí và hồ sơ.",
    )


def _evaluate_lltp_type_rule(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]],
) -> Optional[ClarificationResult]:
    """
    Suy luận từ trường `dieu_kien_bat_buoc` của hồ sơ Lý lịch tư pháp (chứa cả 'số 1' và 'số 2'):
    khi người dùng hỏi về ủy quyền / nhờ người làm hộ / lựa chọn loại phiếu mà chưa nêu số 1 hay số 2.
    """
    if not retrieved_items:
        return None
    top_proc = retrieved_items[0]["procedure"]
    info = top_proc.get("ma_va_ten_thu_tuc") or {}
    ho_so = top_proc.get("thanh_phan_ho_so") or {}
    title_lower = str(info.get("ten_thu_tuc") or "").lower()
    cond_lower = str(ho_so.get("dieu_kien_bat_buoc") or "").lower()

    if "lý lịch tư pháp" not in title_lower and "lý lịch tư pháp" not in query.lower():
        return None
    if not ("số 1" in cond_lower and "số 2" in cond_lower):
        return None

    q_lower = query.lower()
    if not any(m in q_lower for m in _DELEGATION_OR_CHOICE_MARKERS):
        return None

    combined = _combine_query_and_history(query, chat_history)
    type_specified = any(
        m in combined
        for m in ("số 1", "số 2", "phiếu 1", "phiếu 2", "loại 1", "loại 2", "mẫu số 1", "mẫu số 2")
    )
    if type_specified:
        return None

    return ClarificationResult(
        needs_clarification=True,
        clarification_question=(
            "Bạn đang muốn xin cấp **Phiếu lý lịch tư pháp số 1** hay **Phiếu lý lịch tư pháp số 2**? "
            "(Vì Phiếu số 1 có thể ủy quyền cho người khác làm thủ tục, còn Phiếu số 2 tuyệt đối không được ủy quyền.)"
        ),
        rule_id="lltp_type_branch",
        missing_facts=["loại Phiếu lý lịch tư pháp (số 1 hay số 2)"],
        candidate_procedures=[_proc_summary(top_proc)],
        reason="Quy định về quyền ủy quyền khác nhau hoàn toàn giữa Phiếu lý lịch tư pháp số 1 và số 2.",
    )


def _evaluate_criminal_stage_authority_rule(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]],
) -> Optional[ClarificationResult]:
    """
    Suy luận từ các trường `phan_nhom`, `giai_doan_to_tung` (>= 2 giai đoạn) và `cap_tham_quyen` (>= 2 cơ quan):
    khi người dùng hỏi nơi nộp / cơ quan có thẩm quyền trong tố tụng hình sự mà chưa nêu giai đoạn hiện tại.
    """
    if not retrieved_items:
        return None
    top_proc = retrieved_items[0]["procedure"]
    if not _is_active_proc(top_proc):
        return None

    info = top_proc.get("ma_va_ten_thu_tuc") or {}
    if info.get("phan_nhom") != "Tố tụng hình sự":
        return None

    stages = list(info.get("giai_doan_to_tung") or [])
    authorities = list(info.get("cap_tham_quyen") or [])
    if len(stages) < 2 or len(authorities) < 2:
        return None

    q_lower = query.lower()
    if not any(m in q_lower for m in _AUTHORITY_QUERY_MARKERS):
        return None

    combined = _combine_query_and_history(query, chat_history)
    stage_or_auth_markers = (
        "điều tra",
        "truy tố",
        "xét xử",
        "sơ thẩm",
        "phúc thẩm",
        "khởi tố",
        "thi hành án",
        "cơ quan điều tra",
        "cảnh sát điều tra",
        "csđt",
        "công an",
        "viện kiểm sát",
        "vks",
        "tòa án",
        "trại tạm giam",
        "nhà tạm giữ",
    )
    if any(m in combined for m in stage_or_auth_markers):
        return None

    return ClarificationResult(
        needs_clarification=True,
        clarification_question=(
            f"Thủ tục **{info.get('ten_thu_tuc')}** thuộc thẩm quyền của cơ quan đang trực tiếp thụ lý hồ sơ "
            f"theo từng giai đoạn ({', '.join(stages)}). Hiện vụ án đang ở giai đoạn nào "
            f"(Cơ quan điều tra, Viện kiểm sát hay Tòa án đang thụ lý)?"
        ),
        rule_id="criminal_stage_authority_branch",
        missing_facts=["giai đoạn tố tụng hiện tại của vụ án"],
        candidate_procedures=[_proc_summary(top_proc)],
        reason="Cơ quan có thẩm quyền tiếp nhận thay đổi theo giai đoạn tố tụng.",
    )


def _evaluate_expungement_offense_rule(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]],
) -> Optional[ClarificationResult]:
    """
    Phân định Xóa án tích đương nhiên (Sở Tư pháp) vs Theo quyết định của Tòa án (Tòa án sơ thẩm)
    khi câu hỏi chưa nêu tội danh hoặc cơ chế xóa án tích.
    """
    q_lower = query.lower()
    if "xóa án tích" not in q_lower:
        return None

    combined = _combine_query_and_history(query, chat_history)
    known_markers = (
        "đương nhiên",
        "quyết định của tòa",
        "an ninh quốc gia",
        "phá hoại hòa bình",
        "chống loài người",
        "tội phạm chiến tranh",
        "chương xiii",
        "chương xxvi",
        "điều 70",
        "điều 71",
        "trộm cắp",
        "đánh bạc",
        "cố ý gây thương tích",
        "lừa đảo",
        "ma túy",
        "giao thông",
        "cướp",
        "tham ô",
    )
    if any(m in combined for m in known_markers):
        return ClarificationResult(
            needs_clarification=False,
            rule_id="expungement_offense_branch",
            reason="Đã có đủ dữ kiện phân định nhóm tội / hình thức xóa án tích.",
        )

    candidates = [
        it["procedure"]
        for it in retrieved_items
        if _is_active_proc(it["procedure"])
        and "xóa án tích" in (it["procedure"].get("ma_va_ten_thu_tuc", {}).get("ten_thu_tuc", "").lower())
    ]
    if len(candidates) < 2:
        return None

    return ClarificationResult(
        needs_clarification=True,
        clarification_question=(
            "Để xác định đúng nơi nộp hồ sơ (Sở Tư pháp hay Tòa án đã xét xử sơ thẩm), "
            "bạn vui lòng cho biết bản án trước đây thuộc nhóm tội xâm phạm an ninh quốc gia / phá hoại hòa bình "
            "hay là tội danh thông thường khác (như trộm cắp, đánh bạc, gây thương tích...)?"
        ),
        rule_id="expungement_offense_branch",
        missing_facts=["nhóm tội danh trong bản án trước đây"],
        candidate_procedures=[_proc_summary(p) for p in candidates[:2]],
        reason="Xóa án tích đương nhiên nộp tại Sở Tư pháp, còn tội Chương XIII/XXVI BLHS phải xin Quyết định của Tòa án.",
    )


def _evaluate_dynamic_candidate_ambiguity(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]],
) -> Optional[ClarificationResult]:
    """
    Bộ phát hiện tổng quát không phụ thuộc mã thủ tục cố định:
    Khi 2 thủ tục đang hiệu lực đứng đầu có điểm truy xuất rất sát nhau, cùng lĩnh vực nhưng
    khác cơ quan thẩm quyền và câu hỏi chưa chứa từ khóa phân biệt giữa 2 phương án.
    """
    active_items = [
        it for it in retrieved_items if _is_active_proc(it.get("procedure", {}))
    ]
    if len(active_items) < 2:
        return None

    top1, top2 = active_items[0], active_items[1]
    s1 = float(top1.get("hybrid_score", 0.0))
    s2 = float(top2.get("hybrid_score", 0.0))
    if s1 < 0.2 or (s1 - s2) > 0.08:
        return None

    p1 = top1["procedure"]
    p2 = top2["procedure"]
    i1 = p1.get("ma_va_ten_thu_tuc") or {}
    i2 = p2.get("ma_va_ten_thu_tuc") or {}
    if i1.get("ma_thu_tuc") == i2.get("ma_thu_tuc"):
        return None
    if i1.get("linh_vuc") != i2.get("linh_vuc"):
        return None

    auth1 = set(i1.get("cap_tham_quyen") or [])
    auth2 = set(i2.get("cap_tham_quyen") or [])
    if auth1 == auth2:
        return None

    t1_words = {
        w for w in re.findall(r"[^\W\d_]{2,}", str(i1.get("ten_thu_tuc", "")).lower(), flags=re.UNICODE)
        if w not in _TITLE_STOPWORDS
    }
    t2_words = {
        w for w in re.findall(r"[^\W\d_]{2,}", str(i2.get("ten_thu_tuc", "")).lower(), flags=re.UNICODE)
        if w not in _TITLE_STOPWORDS
    }
    shared = t1_words.intersection(t2_words)
    if len(shared) < 2:
        return None

    combined = _combine_query_and_history(query, chat_history)
    diff1 = t1_words - t2_words
    diff2 = t2_words - t1_words
    has_diff1 = any(w in combined for w in diff1)
    has_diff2 = any(w in combined for w in diff2)
    if has_diff1 or has_diff2:
        return None

    name1 = i1.get("ten_thu_tuc", "")
    name2 = i2.get("ten_thu_tuc", "")
    return ClarificationResult(
        needs_clarification=True,
        clarification_question=(
            f"Trường hợp của bạn hiện khớp với hai thủ tục có cơ quan thẩm quyền khác nhau:\n"
            f"1. **{name1}** (Thẩm quyền: {', '.join(i1.get('cap_tham_quyen') or [])})\n"
            f"2. **{name2}** (Thẩm quyền: {', '.join(i2.get('cap_tham_quyen') or [])})\n"
            f"Bạn vui lòng cho biết bạn đang thuộc trường hợp (1) hay (2)?"
        ),
        rule_id="dynamic_candidate_ambiguity",
        missing_facts=["xác nhận đúng trường hợp thủ tục cụ thể"],
        candidate_procedures=[_proc_summary(p1), _proc_summary(p2)],
        reason="Hai thủ tục ứng viên có điểm tương đồng sát nhau nhưng khác cơ quan có thẩm quyền.",
    )


def register_clarification_rule(rule: ClarificationRule, prepend: bool = False) -> None:
    """Đăng ký thêm quy tắc hỏi ngược làm rõ mới mà không phải sửa lõi clarifier."""
    if prepend:
        _CLARIFICATION_RULES.insert(0, rule)
    else:
        _CLARIFICATION_RULES.append(rule)


def _init_default_rules() -> None:
    if _CLARIFICATION_RULES:
        return
    defaults = [
        ClarificationRule(
            rule_id="marriage_jurisdiction_branch",
            description="Phân định đăng ký kết hôn trong nước (UBND xã) vs có yếu tố nước ngoài (UBND huyện).",
            evaluator=_evaluate_marriage_jurisdiction_rule,
        ),
        ClarificationRule(
            rule_id="lltp_type_branch",
            description="Phân định Phiếu lý lịch tư pháp số 1 (được ủy quyền) vs số 2 (cấm ủy quyền).",
            evaluator=_evaluate_lltp_type_rule,
        ),
        ClarificationRule(
            rule_id="criminal_stage_authority_branch",
            description="Phân định cơ quan thụ lý theo giai đoạn tố tụng hình sự khi hỏi nơi nộp.",
            evaluator=_evaluate_criminal_stage_authority_rule,
        ),
        ClarificationRule(
            rule_id="expungement_offense_branch",
            description="Phân định xóa án tích đương nhiên (Sở Tư pháp) vs theo quyết định Tòa án.",
            evaluator=_evaluate_expungement_offense_rule,
        ),
        ClarificationRule(
            rule_id="dynamic_candidate_ambiguity",
            description="Bộ phát hiện tổng quát dựa trên chênh lệch ứng viên truy xuất và thẩm quyền.",
            evaluator=_evaluate_dynamic_candidate_ambiguity,
        ),
    ]
    _CLARIFICATION_RULES.extend(defaults)


_init_default_rules()


def analyze_clarification_need(
    query: str,
    retrieved_items: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> ClarificationResult:
    """
    Đánh giá xem câu hỏi hiện tại (kèm lịch sử hội thoại) có thiếu dữ kiện
    làm thay đổi thủ tục hoặc thẩm quyền giải quyết hay không.
    """
    if not query or not query.strip() or not retrieved_items:
        return ClarificationResult(needs_clarification=False)

    for rule in _CLARIFICATION_RULES:
        res = rule.evaluator(query, retrieved_items, chat_history)
        if res is not None:
            if res.needs_clarification:
                return res
            if res.rule_id:
                return ClarificationResult(needs_clarification=False)

    return ClarificationResult(needs_clarification=False)


def build_clarification_irac(
    clarification: ClarificationResult,
    query: str,
) -> IRACOutput:
    """
    Đóng gói câu hỏi làm rõ thành `IRACOutput` tương thích với caller/UI hiện tại
    khi không tự chọn nhánh pháp lý còn thiếu.
    """
    question = clarification.clarification_question or (
        "Bạn vui lòng cung cấp thêm thông tin cụ thể để hệ thống xác định đúng thủ tục và cơ quan có thẩm quyền."
    )
    candidates_desc = "; ".join(
        f"{c.get('ten_thu_tuc')} ({c.get('ma_thu_tuc')})"
        for c in clarification.candidate_procedures
        if c.get("ten_thu_tuc")
    )
    missing_str = ", ".join(clarification.missing_facts) or "thông tin phân định tình huống"

    return IRACOutput(
        is_refusal=False,
        refusal_reason=None,
        direct_advice=question,
        issue=f"Câu hỏi *\"{query}\"* đang thiếu dữ kiện để phân định chính xác nhánh thủ tục / thẩm quyền ({missing_str}).",
        rule=(
            f"Nguyên tắc an toàn pháp lý: Không tự chọn nhánh khi thiếu dữ kiện phân định "
            f"({candidates_desc or 'cần xác nhận thông tin bổ sung'})."
        ),
        application=(
            f"Hệ thống cần bạn làm rõ thông tin tối thiểu sau trước khi hướng dẫn chi tiết hồ sơ, "
            f"thời hạn và lệ phí:\n\n{question}"
        ),
        conclusion=question,
        citations=None,
        confidence_score=0.5,
    )
