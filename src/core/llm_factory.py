"""
Factory khởi tạo mô hình ngôn ngữ lớn (Google Gemini & Local Ollama) dùng chung cho
Consultant Agent, Judge Agent (Lớp 2), Intent Conversational Handler và Baseline Single-Agent.

Thiết kế an toàn:
- Lazy import thư viện provider (`langchain_google_genai`, `langchain_ollama`).
- Không yêu cầu kết nối mạng hoặc API key để import hay chạy fallback offline.
- Mọi lỗi thiếu package, Ollama chưa bật, model chưa tải hoặc lỗi API đều trả về `None`
  để caller chuyển về chế độ Deterministic/Template Fallback mà không làm sập ứng dụng.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from src.core.config import GEMINI_MODEL, GOOGLE_API_KEY, has_valid_api_key

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
SUPPORTED_PROVIDERS = ("auto", "gemini", "ollama", "none", "offline")
FALLBACK_GEMINI_MODELS: List[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
]


def get_llm_settings() -> Dict[str, Any]:
    """Đọc cấu hình LLM provider/model hiện tại từ biến môi trường."""
    raw_provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
    if raw_provider not in SUPPORTED_PROVIDERS:
        raw_provider = "auto"

    ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    ollama_base_url = (
        os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip()
        or DEFAULT_OLLAMA_BASE_URL
    ).rstrip("/")
    ollama_model = (
        os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
        or DEFAULT_OLLAMA_MODEL
    )
    gemini_model = os.getenv("GEMINI_MODEL", GEMINI_MODEL).strip() or GEMINI_MODEL
    google_api_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY).strip()

    return {
        "provider": raw_provider,
        "gemini_model": gemini_model,
        "google_api_key": google_api_key,
        "ollama_enabled": ollama_enabled,
        "ollama_base_url": ollama_base_url,
        "ollama_model": ollama_model,
    }


def is_ollama_available(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 1.5,
) -> bool:
    """
    Kiểm tra nhanh dịch vụ Ollama cục bộ có đang chạy và đã tải model hay chưa.
    Trả về False an toàn nếu Ollama chưa khởi động hoặc không kết nối được.
    """
    settings = get_llm_settings()
    resolved_url = (base_url or settings["ollama_base_url"]).rstrip("/")
    target_model = (model_name or settings["ollama_model"]).strip()

    try:
        req = urllib.request.Request(f"{resolved_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return False
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False

    models = payload.get("models")
    if not isinstance(models, list) or not models:
        return False

    if not target_model:
        return True

    target_lower = target_model.lower()
    target_base = target_lower.split(":", 1)[0]
    for item in models:
        name = str(item.get("name") or item.get("model") or "").strip().lower()
        if not name:
            continue
        if name == target_lower:
            return True
        if ":" not in target_lower and name.split(":", 1)[0] == target_base:
            return True
    return False


def resolve_llm_config(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Chuẩn hóa provider và model sẽ sử dụng để Consultant, Baseline và Judge
    chia sẻ cùng một cơ chế cấu hình đồng nhất.
    """
    settings = get_llm_settings()
    selected_provider = (provider or settings["provider"] or "auto").strip().lower()
    resolved_key = api_key if api_key is not None else settings["google_api_key"]
    resolved_base_url = (base_url or settings["ollama_base_url"]).rstrip("/")

    if selected_provider in {"none", "offline"}:
        return {
            "provider": "offline",
            "model_name": None,
            "api_key": None,
            "base_url": None,
        }

    if selected_provider == "gemini":
        return {
            "provider": "gemini",
            "model_name": model_name or settings["gemini_model"],
            "api_key": resolved_key,
            "base_url": None,
        }

    if selected_provider == "ollama":
        return {
            "provider": "ollama",
            "model_name": model_name or settings["ollama_model"],
            "api_key": None,
            "base_url": resolved_base_url,
        }

    if has_valid_api_key(resolved_key):
        return {
            "provider": "gemini",
            "model_name": model_name or settings["gemini_model"],
            "api_key": resolved_key,
            "base_url": None,
        }

    if settings["ollama_enabled"]:
        return {
            "provider": "ollama",
            "model_name": model_name or settings["ollama_model"],
            "api_key": None,
            "base_url": resolved_base_url,
        }

    return {
        "provider": "offline",
        "model_name": None,
        "api_key": None,
        "base_url": None,
    }


def _build_gemini_llm(
    temperature: float,
    model_name: str,
    api_key: str,
    max_retries: int = 1,
    timeout: int = 20,
) -> Optional[Any]:
    if not has_valid_api_key(api_key):
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
            timeout=timeout,
        )
    except Exception:
        return None


def _build_ollama_llm(
    temperature: float,
    model_name: str,
    base_url: str,
    check_availability: bool = True,
) -> Optional[Any]:
    if check_availability and not is_ollama_available(
        model_name=model_name,
        base_url=base_url,
    ):
        return None

    try:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama as CommunityChatOllama

            return CommunityChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
            )
        except Exception:
            return None
    except Exception:
        return None


def get_chat_llm(
    temperature: float = 0.1,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    check_ollama_availability: bool = True,
    max_retries: int = 1,
    timeout: int = 20,
) -> Optional[Any]:
    """
    Khởi tạo Chat LLM theo cấu hình provider/model thống nhất:
    - Hỗ trợ Google Gemini (`langchain_google_genai.ChatGoogleGenerativeAI`)
    - Hỗ trợ Local Ollama (`langchain_ollama.ChatOllama`, ưu tiên Qwen2.5 hoặc Vistral)
    - Giữ tương thích ngược chữ ký cũ `get_chat_llm(temperature=..., model_name=..., api_key=...)`.
    - Nếu không cấu hình được hoặc phát sinh lỗi, trả về `None` để kích hoạt fallback offline.
    """
    cfg = resolve_llm_config(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    resolved_provider = cfg["provider"]

    if resolved_provider == "gemini":
        llm = _build_gemini_llm(
            temperature=temperature,
            model_name=str(cfg["model_name"]),
            api_key=str(cfg["api_key"] or ""),
            max_retries=max_retries,
            timeout=timeout,
        )
        if llm is not None:
            return llm
        if (provider or get_llm_settings()["provider"]) == "auto" and get_llm_settings()["ollama_enabled"]:
            settings = get_llm_settings()
            return _build_ollama_llm(
                temperature=temperature,
                model_name=model_name or settings["ollama_model"],
                base_url=base_url or settings["ollama_base_url"],
                check_availability=check_ollama_availability,
            )
        return None

    if resolved_provider == "ollama":
        return _build_ollama_llm(
            temperature=temperature,
            model_name=str(cfg["model_name"]),
            base_url=str(cfg["base_url"]),
            check_availability=check_ollama_availability,
        )

    return None


def get_candidate_models(preferred_model: Optional[str] = None) -> List[str]:
    """Trả về danh sách model Gemini ứng viên theo thứ tự ưu tiên, loại bỏ trùng lặp."""
    primary = preferred_model or get_llm_settings()["gemini_model"] or GEMINI_MODEL
    return [primary] + [m for m in FALLBACK_GEMINI_MODELS if m != primary]


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
    settings = get_llm_settings()
    if settings["provider"] in {"none", "offline"}:
        return None

    if settings["provider"] == "ollama":
        llm = get_chat_llm(temperature=temperature, provider="ollama", model_name=model_name)
        if llm is None:
            return None
        try:
            text = extract_response_text(llm.invoke(prompt))
            return text or None
        except Exception:
            return None

    key = api_key if api_key is not None else settings["google_api_key"]
    if has_valid_api_key(key):
        for candidate in get_candidate_models(model_name):
            llm = get_chat_llm(
                temperature=temperature,
                model_name=candidate,
                api_key=key,
                provider="gemini",
                max_retries=0,
                timeout=15,
            )
            if llm is None:
                continue
            try:
                text = extract_response_text(llm.invoke(prompt))
                if text:
                    return text
            except Exception:
                continue

    if settings["ollama_enabled"]:
        llm = get_chat_llm(temperature=temperature, provider="ollama")
        if llm is not None:
            try:
                text = extract_response_text(llm.invoke(prompt))
                if text:
                    return text
            except Exception:
                pass
    return None
