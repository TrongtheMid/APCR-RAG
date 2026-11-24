"""
Factory khởi tạo mô hình ngôn ngữ lớn (Google Gemini) dùng chung cho Consultant Agent,
Judge Agent (Lớp 2), Intent Conversational Handler và Baseline Single-Agent.
"""

from __future__ import annotations

from typing import Any, List, Optional

from src.core.config import GEMINI_MODEL, GOOGLE_API_KEY, has_valid_api_key

FALLBACK_GEMINI_MODELS: List[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
]


def get_chat_llm(
    temperature: float = 0.1,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    max_retries: int = 1,
    timeout: int = 20,
) -> Optional[Any]:
    """
    Trả về đối tượng ChatGoogleGenerativeAI nếu có API key hợp lệ,
    ngược lại trả về None để hệ thống chuyển sang chế độ Deterministic/Rule Fallback.
    """
    key = api_key if api_key is not None else GOOGLE_API_KEY
    if not has_valid_api_key(key):
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name or GEMINI_MODEL,
            google_api_key=key,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )
    except Exception:
        return None


def get_candidate_models(preferred_model: Optional[str] = None) -> List[str]:
    """Trả về danh sách model ứng viên theo thứ tự ưu tiên, loại bỏ trùng lặp."""
    primary = preferred_model or GEMINI_MODEL
    candidates = [primary] + [m for m in FALLBACK_GEMINI_MODELS if m != primary]
    return candidates


def extract_response_text(resp: Any) -> str:
    """Trích xuất chuỗi văn bản thuần túy từ phản hồi LangChain/Gemini (hỗ trợ cả str lẫn list blocks)."""
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    parts.append(str(txt))
            else:
                txt = getattr(item, "text", None)
                if txt:
                    parts.append(str(txt))
        return "\n".join(parts).strip()
    return str(content).strip()


def invoke_text_with_fallback(
    prompt: str,
    temperature: float = 0.2,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Gọi LLM sinh văn bản tự nhiên, tự động chuyển sang model dự phòng nếu model chính
    gặp lỗi 404/503/504 nhất thời. Trả về None nếu không gọi được.
    """
    key = api_key if api_key is not None else GOOGLE_API_KEY
    if not has_valid_api_key(key):
        return None
    for candidate in get_candidate_models(model_name):
        llm = get_chat_llm(
            temperature=temperature,
            model_name=candidate,
            api_key=key,
            max_retries=0,
            timeout=15,
        )
        if llm is None:
            continue
        try:
            resp = llm.invoke(prompt)
            text = extract_response_text(resp)
            if text:
                return text
        except Exception:
            continue
    return None
