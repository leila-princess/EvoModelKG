from __future__ import annotations

import asyncio
import json
import os
from typing import Any, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def is_timeout_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "timeout" in name or "timed out" in text or "request timeout" in text


def _llm_timeout_seconds() -> float:
    """Per-request timeout for LLM calls (seconds)."""
    raw = os.getenv("MULTI_AGENT_LLM_TIMEOUT_SEC", "240")
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 240.0
    return val if val > 0 else 240.0


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if "```" not in cleaned:
        return cleaned
    cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "")
    return cleaned.strip()


def _extract_json_candidates(text: str) -> list[str]:
    """Generate possible JSON substrings from noisy LLM output."""
    cleaned = _strip_code_fence(text)
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)

    first_obj = cleaned.find("{")
    last_obj = cleaned.rfind("}")
    if first_obj != -1 and last_obj != -1 and last_obj > first_obj:
        candidates.append(cleaned[first_obj:last_obj + 1].strip())

    first_arr = cleaned.find("[")
    last_arr = cleaned.rfind("]")
    if first_arr != -1 and last_arr != -1 and last_arr > first_arr:
        candidates.append(cleaned[first_arr:last_arr + 1].strip())

    # Deduplicate while preserving order.
    uniq: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def create_deepseek_llm(
    model: str | None = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    # OpenAI-compatible endpoints in LAN often do not enforce auth.
    # Keep compatibility with old DEEPSEEK_* envs while adding generic names.
    api_key = (
        os.getenv("OPENAI_COMPAT_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or "EMPTY"
    )
    base_url = (
        os.getenv("OPENAI_COMPAT_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or "https://api.deepseek.com/v1"
    )
    model_name = (
        model
        or os.getenv("OPENAI_COMPAT_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "gemma-4-31b-it"
    )

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        timeout=_llm_timeout_seconds(),
    )


def _coerce_model_payload(data: Any, model_cls: type[BaseModel]) -> Any:
    """将 LLM 常见畸形 JSON（如空数组）规范化为 model 可接受的 dict。"""
    if not isinstance(data, list):
        return data
    if model_cls.__name__ != "ReadmeExtractionResult":
        return data
    if not data:
        return {"relations": [], "attributes": [], "summary": ""}
    relations: list[dict] = []
    attributes: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("relation_type"):
            relations.append(item)
        elif item.get("attribute"):
            attributes.append(item)
    if relations or attributes:
        return {"relations": relations, "attributes": attributes, "summary": ""}
    return {"relations": [], "attributes": [], "summary": ""}


def parse_json_model(text: str, model_cls: type[T]) -> T:
    last_err: Exception | None = None
    decoder = json.JSONDecoder()

    def _validate(data: Any) -> T:
        return model_cls.model_validate(_coerce_model_payload(data, model_cls))

    for candidate in _extract_json_candidates(text):
        # 1) Strict JSON parse first.
        try:
            data = json.loads(candidate)
            return _validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e

        # 2) Try extracting first decodable JSON fragment.
        starts = [i for i, ch in enumerate(candidate) if ch in "{["]
        for start in starts:
            try:
                data, _end = decoder.raw_decode(candidate[start:])
                return _validate(data)
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e

    if last_err is not None:
        raise last_err
    raise ValueError("empty LLM response, cannot parse JSON")


def invoke_json(llm: ChatOpenAI, prompt: str, model_cls: type[T]) -> T:
    response = llm.invoke(prompt)
    content: Any = response.content
    if isinstance(content, list):
        content = "\n".join(str(x) for x in content)
    return parse_json_model(str(content), model_cls)


async def ainvoke_json(llm: ChatOpenAI, prompt: str, model_cls: type[T]) -> T:
    timeout_s = _llm_timeout_seconds()
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"LLM async request timeout after {timeout_s:.1f}s") from e
    content: Any = response.content
    if isinstance(content, list):
        content = "\n".join(str(x) for x in content)
    return parse_json_model(str(content), model_cls)
