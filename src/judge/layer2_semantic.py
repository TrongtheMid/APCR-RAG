"""
NHÁNH PHẠM ĐỨC TRỌNG — Module Judge Lớp 2 (General Claim-Level Semantic Critic):
Thay thế các quy tắc viết cứng theo mã thủ tục bằng cơ chế kiểm chứng tổng quát cấp mệnh đề
(Claim-Level Atomic Verification) áp dụng cho MỌI thủ tục trong kho dữ liệu:
1. Tách trường `dieu_kien_bat_buoc` thành các mệnh đề nguyên tử (atomic claims).
2. Trích xuất các neo pháp lý trọng yếu từ từng mệnh đề (mệnh đề cấm/phủ định, ngưỡng định lượng,
   thuật ngữ cốt lõi) và đối soát độ phủ + tính nhất quán cực tính (polarity) trên bản nháp IRAC.
3. Kiểm định phân cấp thẩm quyền (`cap_tham_quyen`) và giai đoạn tố tụng (`giai_doan_to_tung`).
4. Gọi bổ sung LLM Critic nếu có `GOOGLE_API_KEY` để phát hiện mâu thuẫn ngữ nghĩa tinh vi.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.llm_factory import extract_response_text, get_chat_llm
from src.schemas import IRACOutput


# Tập từ dừng tiếng Việt phổ thông để trích xuất từ khóa thực chất trong mệnh đề pháp lý
_VI_STOPWORDS = {
    "các", "của", "và", "là", "có", "được", "trong", "cho", "với", "khi", "theo",
    "tại", "từ", "đến", "về", "để", "nếu", "thì", "mà", "hoặc", "những", "này",
    "đó", "người", "việc", "trường", "hợp", "quy", "định", "phải", "cần", "làm",
    "một", "hai", "các", "như", "sau", "trước", "bởi", "vào", "ra", "cũng", "đã",
    "đang", "sẽ", "vẫn", "nhưng", "tuy", "nhiên", "hiện", "nay", "lưu", "ý", "quan",
    "trọng", "cảnh", "báo", "pháp", "lý", "thủ", "tục", "hồ", "sơ",
}

# Các mẫu mệnh đề cấm / phủ định / miễn phí bắt buộc không được đảo ngược nghĩa
_PROHIBITION_MARKERS = [
    ("tuyệt đối không", "không"),
    ("không được", "không"),
    ("không thuộc", "không"),
    ("không cấp", "không"),
    ("không yêu cầu", "không"),
    ("hết hiệu lực", "hết hiệu lực"),
    ("hết giá trị", "hết giá trị"),
    ("miễn phí", "miễn"),
    ("miễn lệ phí", "miễn"),
]

# Neo định lượng (VD: "02 người", "15 ngày", "60 ngày", "03 tháng", "36 tháng", "08 m2", "18 tuổi", "20 tuổi")
_QUANT_ANCHOR_RE = re.compile(
    r"\b0?(\d+(?:[.,]\d{3})*)\s*(người|ngày|tháng|năm|tuổi|giờ|lần|m2|đồng|vnđ)\b",
    flags=re.IGNORECASE | re.UNICODE,
)


def extract_mandatory_claims(dieu_kien_bat_buoc: str) -> List[Dict[str, Any]]:
    """
    Tách chuỗi `dieu_kien_bat_buoc` thành danh sách mệnh đề nguyên tử (atomic claims),
    mỗi mệnh đề kèm neo định lượng, cờ cấm/phủ định và tập từ khóa nội dung.
    """
    if not dieu_kien_bat_buoc or not dieu_kien_bat_buoc.strip():
        return []

    raw_parts = re.split(r"(?<=[.;])\s+|\(\d+\)\s*|\([a-d]\)\s*", dieu_kien_bat_buoc.strip())
    claims: List[Dict[str, Any]] = []

    for part in raw_parts:
        clause = part.strip(" .;:-")
        if len(clause) < 20:
            continue
        c_lower = clause.lower()

        # Trích neo định lượng chuẩn hóa (bỏ số 0 dẫn đầu: "02 người" -> ("2", "người"))
        quant_anchors: List[Tuple[str, str]] = []
        for m in _QUANT_ANCHOR_RE.finditer(c_lower):
            num_norm = m.group(1).lstrip("0") or "0"
            unit = m.group(2).lower()
            quant_anchors.append((num_norm, unit))

        # Phát hiện dấu hiệu mệnh đề cấm / miễn phí / hết hiệu lực
        prohibition_required: List[str] = []
        for trigger, required_token in _PROHIBITION_MARKERS:
            if trigger in c_lower and required_token not in prohibition_required:
                prohibition_required.append(required_token)

        # Trích từ khóa nội dung có ý nghĩa
        words = re.findall(r"[^\W\d_]{2,}", c_lower, flags=re.UNICODE)
        content_tokens = sorted({w for w in words if w not in _VI_STOPWORDS})

        claims.append(
            {
                "text": clause,
                "quant_anchors": quant_anchors,
                "prohibition_required": prohibition_required,
                "content_tokens": content_tokens,
            }
        )

    return claims


def evaluate_claims_coverage(
    draft_text: str,
    dieu_kien_bat_buoc: str,
) -> Dict[str, Any]:
    """
    Đánh giá tổng quát mức độ bao phủ mệnh đề (Claim-Level Verification) giữa bản nháp IRAC
    và trường `dieu_kien_bat_buoc` của JSON gốc — không hardcode theo mã thủ tục.
    """
    claims = extract_mandatory_claims(dieu_kien_bat_buoc)
    if not claims:
        return {
            "coverage_score": 1.0,
            "verified_count": 0,
            "total_count": 0,
            "missing_claims": [],
            "polarity_errors": [],
        }

    d_lower = draft_text.lower()
    draft_quants = {
        ((m.group(1).lstrip("0") or "0"), m.group(2).lower())
        for m in _QUANT_ANCHOR_RE.finditer(d_lower)
    }

    verified_count = 0
    missing_claims: List[str] = []
    polarity_errors: List[str] = []

    for claim in claims:
        tokens = claim["content_tokens"]
        token_overlap = (
            sum(1 for t in tokens if t in d_lower) / max(1, len(tokens))
            if tokens
            else 1.0
        )

        # Kiểm tra các neo định lượng quan trọng trong mệnh đề
        quants = claim["quant_anchors"]
        quant_ok = True
        if quants:
            matched_quants = sum(1 for q in quants if q in draft_quants)
            quant_ok = (matched_quants / len(quants)) >= 0.5

        # Kiểm tra mệnh đề phủ định / cấm / miễn phí có bị bỏ sót hoặc lật nghĩa không
        prohibitions = claim["prohibition_required"]
        prohibition_ok = all(req in d_lower for req in prohibitions)

        if not prohibition_ok:
            polarity_errors.append(
                f"[Lớp 2 - Thiếu Mệnh Đề Cấm/Miễn Phí/Hiệu Lực] Bản nháp chưa thể hiện rõ điều kiện bắt buộc: \"{claim['text']}\""
            )

        if token_overlap >= 0.45 and quant_ok and prohibition_ok:
            verified_count += 1
        else:
            missing_claims.append(claim["text"])

    coverage_score = round(verified_count / len(claims), 4)
    return {
        "coverage_score": coverage_score,
        "verified_count": verified_count,
        "total_count": len(claims),
        "missing_claims": missing_claims,
        "polarity_errors": polarity_errors,
    }


def try_llm_critic(
    query: str,
    draft: IRACOutput,
    ground_truth_proc: Dict[str, Any],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[str]:
    """Gọi LLM Judge kiểm tra bổ sung mâu thuẫn ngữ nghĩa nếu có GOOGLE_API_KEY."""
    llm = get_chat_llm(temperature=0.0, model_name=model_name, api_key=api_key)
    if llm is None:
        return []
    try:
        prompt = (
            "Bạn là Tác tử Thẩm định Pháp lý (JUDGE AGENT - Lớp 2: Claim-level Semantic Critic).\n"
            "Hãy kiểm tra xem BẢN NHÁP TƯ VẤN có bị mâu thuẫn về thẩm quyền, giai đoạn tố tụng "
            "hoặc bỏ sót/lật ngược điều kiện bắt buộc so với HỒ SƠ JSON GỐC hay không.\n"
            "Nếu hoàn toàn chính xác, chỉ trả lời đúng một từ: PASS.\n"
            "Nếu có lỗi sai nghiêm trọng, hãy mô tả ngắn gọn lỗi đó trong 1 câu.\n\n"
            f"CÂU HỎI: {query}\n"
            f"ĐIỀU KIỆN BẮT BUỘC GỐC: {ground_truth_proc['thanh_phan_ho_so']['dieu_kien_bat_buoc']}\n"
            f"CƠ QUAN THẨM QUYỀN GỐC: {ground_truth_proc['ma_va_ten_thu_tuc']['cap_tham_quyen']}\n"
            f"GIAI ĐOẠN TỐ TỤNG GỐC: {ground_truth_proc['ma_va_ten_thu_tuc'].get('giai_doan_to_tung', [])}\n"
            f"BẢN NHÁP APPLICATION: {draft.application}\n"
            f"BẢN NHÁP CONCLUSION: {draft.conclusion}"
        )
        resp = llm.invoke(prompt)
        content = extract_response_text(resp)
        if content.upper().startswith("PASS"):
            return []
        return [f"[Lớp 2 - LLM Semantic Critic] {content}"]
    except Exception:
        return []


def verify_layer2_semantic_critic_detailed(
    query: str,
    draft: IRACOutput,
    ground_truth_proc: Dict[str, Any],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    use_llm: bool = True,
) -> Dict[str, Any]:
    """
    LỚP 2 — THẨM ĐỊNH NGỮ NGHĨA TỔNG QUÁT CẤP MỆNH ĐỀ (CLAIM-LEVEL SEMANTIC CRITIC):
    1. Kiểm tra phân cấp cơ quan có thẩm quyền (`cap_tham_quyen`).
    2. Kiểm tra giai đoạn tố tụng (`giai_doan_to_tung`) đối với nhóm Tố tụng hình sự.
    3. Kiểm tra độ phủ & cực tính từng mệnh đề điều kiện bắt buộc (`dieu_kien_bat_buoc`).
    4. Kiểm tra độ đầy đủ nội dung lập luận IRAC (`application`).
    5. Gọi bổ sung LLM Critic nếu có API key.
    """
    if draft.is_refusal:
        return {
            "passed": True,
            "errors": [],
            "score": 1.0,
            "claim_coverage_score": 1.0,
            "verified_claims_count": 0,
            "total_claims_count": 0,
        }

    errors: List[str] = []
    checks_total = 0
    checks_passed = 0

    info = ground_truth_proc["ma_va_ten_thu_tuc"]
    ho_so = ground_truth_proc["thanh_phan_ho_so"]
    gt_authorities = info.get("cap_tham_quyen", [])
    gt_stages = info.get("giai_doan_to_tung", [])
    gt_condition = ho_so.get("dieu_kien_bat_buoc", "")

    combined_draft_text = f"{draft.issue}\n{draft.rule}\n{draft.application}\n{draft.conclusion}"
    combined_lower = combined_draft_text.lower()

    # 1. Thẩm định Cơ quan có thẩm quyền
    checks_total += 1
    if gt_authorities:
        has_valid_auth = any(auth.lower() in combined_lower for auth in gt_authorities)
        if not has_valid_auth:
            errors.append(
                f"[Lớp 2 - Nhầm Lẫn Thẩm Quyền] Bản nháp chưa nêu đúng cơ quan có thẩm quyền chuẩn: "
                f"{', '.join(gt_authorities)}."
            )
        else:
            checks_passed += 1
    else:
        checks_passed += 1

    # 2. Thẩm định Giai đoạn tố tụng (đối với Tố tụng hình sự)
    if info.get("phan_nhom") == "Tố tụng hình sự" and gt_stages:
        checks_total += 1
        has_stage = any(st.lower() in combined_lower for st in gt_stages)
        if not has_stage:
            errors.append(
                f"[Lớp 2 - Thiếu Giai Đoạn Tố Tụng] Thủ tục hình sự '{info['ma_thu_tuc']}' cần nêu rõ "
                f"giai đoạn tố tụng áp dụng: {', '.join(gt_stages)}."
            )
        else:
            checks_passed += 1

    # 3. Thẩm định Tổng quát Cấp Mệnh đề (Claim-Level Verification) trên `dieu_kien_bat_buoc`
    checks_total += 1
    claim_eval = evaluate_claims_coverage(combined_draft_text, gt_condition)
    if claim_eval["polarity_errors"]:
        errors.extend(claim_eval["polarity_errors"])
    elif claim_eval["total_count"] > 0 and claim_eval["coverage_score"] < 0.6:
        sample_missing = claim_eval["missing_claims"][0]
        errors.append(
            f"[Lớp 2 - Thiếu Điều Kiện Bắt Buộc (Claim Coverage {claim_eval['coverage_score'] * 100:.0f}%)] "
            f"Bản nháp bỏ sót mệnh đề tiên quyết trong JSON gốc: \"{sample_missing}\""
        )
    else:
        checks_passed += 1

    # 4. Thẩm định Độ đầy đủ của lập luận Application trong IRAC
    checks_total += 1
    if len(draft.application.strip()) < 120:
        errors.append(
            "[Lớp 2 - Sơ Sài Nội Dung Application] Phần Application trong cấu trúc IRAC quá ngắn, "
            "chưa trình bày đầy đủ trình tự các bước và điều kiện bắt buộc."
        )
    else:
        checks_passed += 1

    # 5. Gọi bổ sung LLM Semantic Critic nếu bật use_llm và có GOOGLE_API_KEY
    if use_llm and not errors:
        llm_errors = try_llm_critic(
            query=query,
            draft=draft,
            ground_truth_proc=ground_truth_proc,
            model_name=model_name,
            api_key=api_key,
        )
        errors.extend(llm_errors)

    base_score = checks_passed / max(1, checks_total)
    final_score = round(
        (0.6 * base_score) + (0.4 * claim_eval["coverage_score"]),
        4,
    )
    if errors:
        final_score = min(final_score, 0.75)

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "score": final_score,
        "claim_coverage_score": claim_eval["coverage_score"],
        "verified_claims_count": claim_eval["verified_count"],
        "total_claims_count": claim_eval["total_count"],
    }


def verify_layer2_semantic_critic(
    query: str,
    draft: IRACOutput,
    ground_truth_proc: Dict[str, Any],
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """API tương thích ngược trả về (passed, errors) cho Lớp 2."""
    detail = verify_layer2_semantic_critic_detailed(
        query=query,
        draft=draft,
        ground_truth_proc=ground_truth_proc,
        model_name=model_name,
        api_key=api_key,
    )
    return bool(detail["passed"]), list(detail["errors"])
