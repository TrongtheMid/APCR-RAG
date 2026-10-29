"""
Factory khởi tạo mô hình ngôn ngữ lớn (Google Gemini) dùng chung cho Consultant Agent,
Judge Agent (Lớp 2) và Baseline Single-Agent.
"""

from __future__ import annotations

from typing import Any, Optional

from src.core.config import GEMINI_MODEL, GOOGLE_API_KEY, has_valid_api_key


def get_chat_llm(
    temperature: float = 0.1,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
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
        )
    except Exception:
        return None
