from __future__ import annotations

import os

import httpx
from langchain_openai import ChatOpenAI
from loguru import logger

from evomodelkg.clients.deepseek_client import _llm_timeout_seconds


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def role_api_key(role: str) -> str | None:
    role = role.upper()
    return _first_env(
        f"SELF_EVOLVE_{role}_API_KEY",
        f"{role}_OPENAI_COMPAT_API_KEY",
        f"{role}_DEEPSEEK_API_KEY",
    )


def role_base_url(role: str) -> str | None:
    role = role.upper()
    return _first_env(
        f"SELF_EVOLVE_{role}_BASE_URL",
        f"{role}_OPENAI_COMPAT_BASE_URL",
        f"{role}_DEEPSEEK_BASE_URL",
    )


def mask_secret(value: str | None) -> str:
    if not value:
        return "(鏈缃?"
    return (value[:7] + "..." + value[-4:]) if len(value) > 12 else "(宸茶缃?"


def create_role_llm(
    *,
    role: str,
    model: str | None,
    temperature: float,
    api_key: str | None = None,
    base_url: str | None = None,
    max_tokens: int | None = None,
):
    resolved_key = api_key or role_api_key(role)
    resolved_url = base_url or role_base_url(role)
    model_name = (
        model
        or os.getenv("OPENAI_COMPAT_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "gemma-4-31b-it"
    )
    timeout_s = _llm_timeout_seconds()
    trust_env = _env_flag("SELF_EVOLVE_LLM_TRUST_ENV", default=False)
    logger.info(
        "Create LLM client: role={}, model={}, base_url={}, max_tokens={}, timeout_s={}, trust_env={}",
        role,
        model_name,
        resolved_url or "http://192.168.15.121:8000/v1",
        max_tokens,
        timeout_s,
        trust_env,
    )
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=resolved_key or "EMPTY",
        base_url=resolved_url or "http://192.168.15.121:8000/v1",
        max_tokens=max_tokens,
        timeout=timeout_s,
        http_client=httpx.Client(trust_env=trust_env, timeout=timeout_s),
        http_async_client=httpx.AsyncClient(trust_env=trust_env, timeout=timeout_s),
    )
