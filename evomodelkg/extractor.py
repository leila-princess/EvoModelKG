from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from evomodelkg.clients.deepseek_client import (
    _llm_timeout_seconds,
    is_timeout_error,
    parse_json_model,
)
from evomodelkg.clients.schemas import CandidateAttribute, ReadmeExtractionResult
from evomodelkg.llm_config import create_role_llm
from evomodelkg.prompt_store import PromptStore
from evomodelkg.readme_text import strip_readme_front_matter
from evomodelkg.tool_store import ToolStore


ESTIMATED_CHARS_PER_TOKEN = 3
README_CHUNK_BUDGET_RATIO = 0.8
CONTEXT_RETRY_README_RATIO = 0.75


def _safe_confidence(attr: CandidateAttribute) -> float:
    try:
        return float(attr.confidence)
    except (TypeError, ValueError):
        return 0.0


def _merge_results(parts: list[ReadmeExtractionResult]) -> ReadmeExtractionResult:
    best_attrs: dict[tuple[str, str, str], CandidateAttribute] = {}
    for p in parts:
        for a in p.attributes:
            k = (a.entity_type.lower(), a.entity_id.strip(), a.attribute.lower())
            current = best_attrs.get(k)
            if current is None:
                best_attrs[k] = a
                continue
            current_score = (_safe_confidence(current), len(current.evidence_span or current.evidence or ""))
            candidate_score = (_safe_confidence(a), len(a.evidence_span or a.evidence or ""))
            if candidate_score > current_score:
                best_attrs[k] = a
    attrs = list(best_attrs.values())
    return ReadmeExtractionResult(relations=[], attributes=attrs, summary="merged")


def _to_audit_dict(result: ReadmeExtractionResult) -> dict[str, Any]:
    return {
        "relations": [],
        "attributes": [a.model_dump() for a in result.attributes],
        "summary": result.summary,
    }


def _empty_extraction_with_error(
    *,
    resource_id: str,
    error_type: str,
    error: str,
    sample_timeout_sec: float | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "resource_id": resource_id,
        "extraction_error_type": error_type,
        "extraction_error": error,
    }
    if sample_timeout_sec is not None:
        meta["sample_timeout_sec"] = sample_timeout_sec
    return {
        **_to_audit_dict(ReadmeExtractionResult()),
        "_meta": meta,
    }


def _extraction_item_count(extraction: dict[str, Any]) -> int:
    attrs = extraction.get("attributes") or []
    return len(attrs)


def _raw_llm_trace_dir() -> Path | None:
    raw = os.getenv("SELF_EVOLVE_TRACE_RAW_LLM_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def _safe_trace_name(resource_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in resource_id)
    return safe[:120] or "unknown"


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(x) for x in content)
    return str(content)


def _is_context_length_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "maximum context length" in text
        or "reduce the length of the input prompt" in text
        or "context length" in text
    )


def _shrink_readme_in_prompt(prompt: str, ratio: float = CONTEXT_RETRY_README_RATIO) -> str:
    readme_marker = "\n[README]\n"
    output_marker = "\n[OUTPUT_JSON_SHAPE]"
    start = prompt.find(readme_marker)
    end = prompt.find(output_marker)
    if start < 0 or end <= start:
        keep = max(1024, int(len(prompt) * ratio))
        return prompt[:keep]
    body_start = start + len(readme_marker)
    readme_body = prompt[body_start:end]
    keep = max(512, int(len(readme_body) * ratio))
    if keep >= len(readme_body):
        keep = max(512, len(readme_body) - 2048)
    shortened = readme_body[:keep].rstrip()
    notice = (
        "\n\n[README TRUNCATED]\n"
        "The README was shortened automatically because the original prompt exceeded "
        "the model context window. Extract only attributes supported by the visible text.\n"
    )
    return prompt[:body_start] + shortened + notice + prompt[end:]


def _json_candidates_from_text(text: str) -> list[Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)
    first_obj = cleaned.find("{")
    last_obj = cleaned.rfind("}")
    if first_obj != -1 and last_obj > first_obj:
        candidates.append(cleaned[first_obj : last_obj + 1].strip())
    first_arr = cleaned.find("[")
    last_arr = cleaned.rfind("]")
    if first_arr != -1 and last_arr > first_arr:
        candidates.append(cleaned[first_arr : last_arr + 1].strip())

    out: list[Any] = []
    decoder = json.JSONDecoder()
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            out.append(json.loads(candidate))
            continue
        except json.JSONDecodeError:
            pass
        for start, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                data, _end = decoder.raw_decode(candidate[start:])
                out.append(data)
                break
            except json.JSONDecodeError:
                continue
    return out


def _stringify_attr_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _normalize_extraction_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        items = [item for item in data if isinstance(item, dict)]
        return {
            "relations": [item for item in items if item.get("relation_type")],
            "attributes": [item for item in items if item.get("attribute")],
            "summary": "",
        }
    if not isinstance(data, dict):
        return {"relations": [], "attributes": [], "summary": ""}

    attrs = data.get("attributes") or []
    normalized_attrs: list[dict[str, Any]] = []
    if isinstance(attrs, list):
        for attr in attrs:
            if not isinstance(attr, dict) or not attr.get("attribute"):
                continue
            new_attr = dict(attr)
            new_attr["entity_type"] = str(new_attr.get("entity_type") or "")
            new_attr["entity_id"] = str(new_attr.get("entity_id") or "")
            new_attr["attribute"] = str(new_attr.get("attribute") or "")
            new_attr["value"] = _stringify_attr_value(new_attr.get("value"))
            new_attr["evidence"] = str(new_attr.get("evidence") or "")
            new_attr["source"] = str(new_attr.get("source") or "readme")
            new_attr["evidence_span"] = str(new_attr.get("evidence_span") or "")
            new_attr["normalization_note"] = str(new_attr.get("normalization_note") or "")
            normalized_attrs.append(new_attr)

    rels = data.get("relations") or []
    normalized_rels = rels if isinstance(rels, list) else []
    return {
        "relations": normalized_rels,
        "attributes": normalized_attrs,
        "summary": str(data.get("summary") or ""),
    }


def _parse_extraction_json(response_text: str) -> ReadmeExtractionResult:
    last_err: Exception | None = None
    best: ReadmeExtractionResult | None = None
    for data in _json_candidates_from_text(response_text):
        try:
            normalized = _normalize_extraction_payload(data)
            parsed = ReadmeExtractionResult.model_validate(normalized)
        except Exception as e:
            last_err = e
            continue
        if best is None or len(parsed.attributes) > len(best.attributes):
            best = parsed
    if best is not None:
        return best
    if last_err is not None:
        raise last_err
    return parse_json_model(response_text, ReadmeExtractionResult)


def _write_raw_llm_trace(
    *,
    resource_id: str,
    prompt: str,
    response_text: str,
    status: str,
    error: str = "",
) -> None:
    trace_root = _raw_llm_trace_dir()
    if trace_root is None:
        return
    try:
        trace_root.mkdir(parents=True, exist_ok=True)
        stem = f"{_safe_trace_name(resource_id)}__{uuid.uuid4().hex[:10]}"
        (trace_root / f"{stem}.prompt.txt").write_text(prompt, encoding="utf-8")
        (trace_root / f"{stem}.response.txt").write_text(response_text, encoding="utf-8")
        (trace_root / f"{stem}.meta.json").write_text(
            json.dumps(
                {
                    "resource_id": resource_id,
                    "status": status,
                    "error": error,
                    "prompt_chars": len(prompt),
                    "response_chars": len(response_text),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning(f"raw LLM trace write failed ({resource_id or 'unknown'}): {e}")


def _invoke_extraction_json_traced(llm: Any, prompt: str, *, resource_id: str = "") -> ReadmeExtractionResult:
    response = llm.invoke(prompt)
    response_text = _message_content_to_text(response.content)
    try:
        parsed = _parse_extraction_json(response_text)
        _write_raw_llm_trace(
            resource_id=resource_id,
            prompt=prompt,
            response_text=response_text,
            status=f"parsed_attrs={len(parsed.attributes)}",
        )
        return parsed
    except Exception as e:
        _write_raw_llm_trace(
            resource_id=resource_id,
            prompt=prompt,
            response_text=response_text,
            status="parse_failed",
            error=str(e),
        )
        raise


async def _ainvoke_extraction_json_traced(llm: Any, prompt: str, *, resource_id: str = "") -> ReadmeExtractionResult:
    timeout_s = _llm_timeout_seconds()
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        _write_raw_llm_trace(
            resource_id=resource_id,
            prompt=prompt,
            response_text="",
            status="timeout",
            error=f"LLM async request timeout after {timeout_s:.1f}s",
        )
        raise TimeoutError(f"LLM async request timeout after {timeout_s:.1f}s") from e
    response_text = _message_content_to_text(response.content)
    try:
        parsed = _parse_extraction_json(response_text)
        _write_raw_llm_trace(
            resource_id=resource_id,
            prompt=prompt,
            response_text=response_text,
            status=f"parsed_attrs={len(parsed.attributes)}",
        )
        return parsed
    except Exception as e:
        _write_raw_llm_trace(
            resource_id=resource_id,
            prompt=prompt,
            response_text=response_text,
            status="parse_failed",
            error=str(e),
        )
        raise


def _attr_confidence_value(attr: dict[str, Any]) -> float:
    try:
        return float(attr.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _dedupe_extraction_attributes_by_confidence(extraction: dict[str, Any]) -> dict[str, Any]:
    attrs = extraction.get("attributes") or []
    if not isinstance(attrs, list):
        return extraction
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    passthrough: list[Any] = []
    for attr in attrs:
        if not isinstance(attr, dict):
            passthrough.append(attr)
            continue
        key = (
            str(attr.get("entity_type") or "").lower(),
            str(attr.get("entity_id") or "").strip(),
            str(attr.get("attribute") or "").lower(),
        )
        current = best.get(key)
        if current is None:
            best[key] = attr
            continue
        current_score = (
            _attr_confidence_value(current),
            len(str(current.get("evidence_span") or current.get("evidence") or "")),
        )
        candidate_score = (
            _attr_confidence_value(attr),
            len(str(attr.get("evidence_span") or attr.get("evidence") or "")),
        )
        if candidate_score > current_score:
            best[key] = attr
    return {**extraction, "attributes": [*best.values(), *passthrough], "relations": []}


def _force_attribute_entity(
    extraction: dict[str, Any],
    *,
    resource_type: str,
    resource_id: str,
) -> dict[str, Any]:
    attrs = extraction.get("attributes") or []
    if not isinstance(attrs, list):
        return extraction
    fixed: list[Any] = []
    changed = False
    for attr in attrs:
        if not isinstance(attr, dict):
            fixed.append(attr)
            continue
        new_attr = dict(attr)
        if new_attr.get("entity_type") != resource_type:
            new_attr["entity_type"] = resource_type
            changed = True
        if new_attr.get("entity_id") != resource_id:
            new_attr["entity_id"] = resource_id
            changed = True
        fixed.append(new_attr)
    if not changed:
        return extraction
    return {**extraction, "attributes": fixed, "relations": []}


CANDIDATE_GROUNDED_FIELDS = {"architecture", "config_model_type"}


def _candidate_grounding_inputs(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attrs = extraction.get("attributes") or []
    if not isinstance(attrs, list):
        return rows
    for attr in attrs:
        if not isinstance(attr, dict):
            continue
        field = str(attr.get("attribute") or "").strip()
        if field not in CANDIDATE_GROUNDED_FIELDS:
            continue
        evidence = str(attr.get("evidence_span") or attr.get("evidence") or "").strip()
        rows.append(
            {
                "attribute": field,
                "value": attr.get("value"),
                "evidence_span": evidence[:1000],
            }
        )
    return rows


def _tool_use_decision_prompt(
    *,
    resource_type: str,
    resource_id: str,
    tool_entry: dict[str, Any],
    extraction: dict[str, Any],
) -> str:
    payload = {
        "resource_type": resource_type,
        "resource_id": resource_id,
        "tool": {
            "name": tool_entry.get("name"),
            "description": tool_entry.get("description"),
            "purpose": tool_entry.get("purpose"),
            "expected_behavior": tool_entry.get("expected_behavior"),
        },
        "candidate_grounding_inputs": _candidate_grounding_inputs(extraction),
    }
    return (
        "[IDENTITY]\n"
        "You are a README extraction tool-selection agent.\n\n"
        "[GOAL]\n"
        "Decide whether the managed tool should be called for this already-extracted result.\n\n"
        "[INPUT_JSON]\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "[DECISION_RULES]\n"
        "1. Return use_tool=true only when this sample has evidence-bearing fields that may benefit from the tool.\n"
        "2. For candidate_grounded_normalizer, use it only for architecture/config_model_type values that look noisy, ambiguous, abbreviated, or likely need canonical vocabulary grounding.\n"
        "3. Return use_tool=false when there are no relevant fields, no evidence span, or the extracted value already looks directly supported and does not need candidate grounding.\n"
        "4. Do not optimize for recall by always using the tool; skipping is valid when the normal prompt extraction is enough.\n\n"
        "[OUTPUT_JSON_SHAPE]\n"
        '{"use_tool": false, "reason": "short reason"}\n\n'
        "[END] Return JSON only."
    )


def _without_readme_relations(extraction: dict[str, Any]) -> dict[str, Any]:
    return {**extraction, "relations": []}


class ToolUseDecision(BaseModel):
    use_tool: bool = False
    reason: str = ""


def _top1_candidate_from_attr(attr: dict[str, Any]) -> str | None:
    grounding = attr.get("candidate_grounding") or {}
    rows = grounding.get("top_candidates") or []
    if not rows or not isinstance(rows[0], dict):
        return None
    top = rows[0]
    value = str(top.get("value") or "").strip()
    if not value:
        return None
    if len(rows) == 1:
        return value
    score = float(top.get("score") or 0.0)
    second = float((rows[1] or {}).get("score") or 0.0) if isinstance(rows[1], dict) else 0.0
    margin = score - second
    method = str(grounding.get("method") or top.get("method") or "")
    exact = score >= 0.999 and len(rows) == 1
    if exact:
        return value
    if method == "local_embedding_candidate_grounding":
        return value if score >= 0.70 and margin >= 0.04 else None
    return value if score >= 2.0 and margin >= 0.25 else None


def _candidate_values_from_attr(attr: dict[str, Any], limit: int = 10) -> list[str]:
    grounding = attr.get("candidate_grounding") or {}
    rows = grounding.get("top_candidates") or []
    values: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = str(row.get("value") or "").strip()
        if value and value not in values:
            values.append(value)
        if len(values) >= limit:
            break
    return values


def _replace_attr_value(
    attr: dict[str, Any],
    selected_value: str,
    *,
    candidates: list[str],
) -> dict[str, Any]:
    new_attr = dict(attr)
    old_value = new_attr.get("value")
    new_attr["value"] = selected_value
    note = str(new_attr.get("normalization_note") or "").strip()
    suffix = (
        f"Top-1 candidate grounding: raw_value={old_value!r}, selected={selected_value!r}, "
        f"candidate_count={len(candidates)}."
    )
    new_attr["normalization_note"] = f"{note} {suffix}".strip() if note else suffix
    grounding = dict(new_attr.get("candidate_grounding") or {})
    grounding["top1_selected"] = selected_value
    grounding["top1_selected_from_candidates"] = True
    new_attr["candidate_grounding"] = grounding
    return new_attr


def _invoke_extraction_json(llm: Any, prompt: str, *, resource_id: str = "") -> ReadmeExtractionResult:
    last_err: Exception | None = None
    current_prompt = prompt
    for attempt in range(2):
        try:
            if _raw_llm_trace_dir() is not None:
                return _invoke_extraction_json_traced(llm, current_prompt, resource_id=resource_id)
            response = llm.invoke(current_prompt)
            return _parse_extraction_json(_message_content_to_text(response.content))
        except Exception as e:
            last_err = e
            if attempt == 0:
                reason = "timeout" if is_timeout_error(e) else "JSON parse/request"
                logger.warning(
                    f"LLM extraction {reason} failed ({resource_id or 'unknown'}), retrying: {e}"
                )
                if _is_context_length_error(e):
                    current_prompt = _shrink_readme_in_prompt(current_prompt)
    logger.warning(f"LLM extraction failed ({resource_id or 'unknown'}), empty pass returned: {last_err}")
    return ReadmeExtractionResult()


async def _ainvoke_extraction_json(llm: Any, prompt: str, *, resource_id: str = "") -> ReadmeExtractionResult:
    last_err: Exception | None = None
    current_prompt = prompt
    for attempt in range(2):
        try:
            if _raw_llm_trace_dir() is not None:
                return await _ainvoke_extraction_json_traced(llm, current_prompt, resource_id=resource_id)
            timeout_s = _llm_timeout_seconds()
            response = await asyncio.wait_for(llm.ainvoke(current_prompt), timeout=timeout_s)
            return _parse_extraction_json(_message_content_to_text(response.content))
        except Exception as e:
            last_err = e
            if attempt == 0:
                reason = "timeout" if is_timeout_error(e) else "JSON parse/request"
                logger.warning(
                    f"LLM extraction {reason} failed ({resource_id or 'unknown'}), retrying: {e}"
                )
                if _is_context_length_error(e):
                    current_prompt = _shrink_readme_in_prompt(current_prompt)
    logger.warning(f"LLM extraction failed ({resource_id or 'unknown'}), empty pass returned: {last_err}")
    return ReadmeExtractionResult()


class ReadmeExtractor:
    """按 manifest.workflow 执行 README 抽取（实验用，独立于 LangGraph 管线）。"""

    def __init__(
        self,
        store: PromptStore,
        *,
        temperature: float = 0.0,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        context_window: int | None = None,
        output_token_budget: int = 3072,
        context_safety_tokens: int = 768,
        chunk_overlap_tokens: int = 256,
    ):
        self.store = store
        self.tool_store = ToolStore()
        self.context_window = int(context_window) if context_window else None
        self.output_token_budget = max(256, int(output_token_budget))
        self.context_safety_tokens = max(0, int(context_safety_tokens))
        self.chunk_overlap_tokens = max(0, int(chunk_overlap_tokens))
        self.llm = create_role_llm(
            role="extract",
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            max_tokens=self.output_token_budget,
        )

    @staticmethod
    def _readme_for_llm(readme_content: str) -> str:
        return strip_readme_front_matter(readme_content)

    @staticmethod
    def _estimated_tokens(text: str) -> int:
        # Conservative tokenizer-free estimate for README markdown sent to vLLM.
        return (
            max(1, int((len(text) + ESTIMATED_CHARS_PER_TOKEN - 1) / ESTIMATED_CHARS_PER_TOKEN))
            if text
            else 0
        )

    def _readme_token_budget_for_prompt(
        self,
        prompt_file: str,
        *,
        resource_type: str,
        resource_id: str,
        target_fields: str,
    ) -> int | None:
        if not self.context_window:
            return None
        prompt_without_readme = self.store.render_template(
            prompt_file,
            variables={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "readme_content": "",
                "target_fields": target_fields,
            },
        )
        reserved = (
            self._estimated_tokens(prompt_without_readme)
            + self.output_token_budget
            + self.context_safety_tokens
        )
        return max(128, self.context_window - reserved)

    def _split_readme_into_chunks(
        self,
        readme_content: str,
        *,
        token_budget: int | None,
    ) -> list[str]:
        if not token_budget:
            return [readme_content]
        effective_token_budget = max(128, int(token_budget * README_CHUNK_BUDGET_RATIO))
        if self._estimated_tokens(readme_content) <= effective_token_budget:
            return [readme_content]

        max_chars = max(512, effective_token_budget * ESTIMATED_CHARS_PER_TOKEN)
        overlap_chars = min(
            max_chars // 4,
            self.chunk_overlap_tokens * ESTIMATED_CHARS_PER_TOKEN,
        )
        chunks: list[str] = []
        start = 0
        text_len = len(readme_content)
        while start < text_len:
            hard_end = min(text_len, start + max_chars)
            end = hard_end
            if hard_end < text_len:
                split_at = readme_content.rfind("\n", start, hard_end)
                if split_at > start + max_chars // 2:
                    end = split_at
            chunk = readme_content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            start = max(end - overlap_chars, start + 1)
        return chunks or [readme_content[:max_chars]]

    def _chunked_readmes_for_prompt(
        self,
        prompt_file: str,
        *,
        resource_type: str,
        resource_id: str,
        readme_content: str,
        target_fields: str,
    ) -> tuple[list[str], dict[str, Any]]:
        token_budget = self._readme_token_budget_for_prompt(
            prompt_file,
            resource_type=resource_type,
            resource_id=resource_id,
            target_fields=target_fields,
        )
        chunks = self._split_readme_into_chunks(readme_content, token_budget=token_budget)
        meta = {
            "readme_chunking_enabled": bool(self.context_window),
            "readme_chunk_count": len(chunks),
            "readme_chunk_token_budget": token_budget,
            "readme_chunk_overlap_tokens": self.chunk_overlap_tokens if len(chunks) > 1 else 0,
        }
        return chunks, meta

    def _prepare_readme(
        self,
        *,
        resource_type: str,
        resource_id: str,
        readme_content: str,
    ) -> tuple[str, dict[str, Any]]:
        before = self._readme_for_llm(readme_content)
        after = self._apply_preprocess_tools(
            resource_type=resource_type,
            resource_id=resource_id,
            readme_content=before,
        )
        before_chars = len(before)
        after_chars = len(after)
        reduction = (
            max(0, before_chars - after_chars) / before_chars
            if before_chars > 0
            else 0.0
        )
        return after, {
            "readme_chars_before_preprocess": before_chars,
            "readme_chars_after_preprocess": after_chars,
            "readme_chars_reduction_rate": reduction,
            "estimated_readme_tokens_before_preprocess": self._estimated_tokens(before),
            "estimated_readme_tokens_after_preprocess": self._estimated_tokens(after),
        }

    @staticmethod
    def _attach_extraction_meta(
        extraction: dict[str, Any],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        current_meta = extraction.get("_meta")
        merged_meta = dict(current_meta) if isinstance(current_meta, dict) else {}
        merged_meta.update(meta)
        return {**extraction, "_meta": merged_meta}

    def _apply_preprocess_tools(
        self,
        *,
        resource_type: str,
        resource_id: str,
        readme_content: str,
    ) -> str:
        current = readme_content
        for entry in self.tool_store.active_tools("preprocess_readme"):
            name = str(entry.get("name"))
            try:
                result = self.tool_store.run_tool(
                    name,
                    {
                        "stage": "preprocess_readme",
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "readme_content": current,
                    },
                )
                if isinstance(result.get("readme_content"), str):
                    current = result["readme_content"]
                elif isinstance(result.get("clean_readme"), str):
                    current = result["clean_readme"]
            except Exception as e:
                logger.warning(f"managed preprocess tool failed {name} ({resource_id}): {e}")
        return current

    def _agent_should_use_tool(
        self,
        *,
        entry: dict[str, Any],
        resource_type: str,
        resource_id: str,
        extraction: dict[str, Any],
    ) -> bool:
        if not entry.get("agent_select"):
            return True
        name = str(entry.get("name") or "")
        if name == "candidate_grounded_normalizer" and not _candidate_grounding_inputs(extraction):
            logger.info(f"agent skipped managed tool {name} ({resource_id}): no candidate-grounded fields")
            return False
        try:
            decision = invoke_json(
                self.llm,
                _tool_use_decision_prompt(
                    resource_type=resource_type,
                    resource_id=resource_id,
                    tool_entry=entry,
                    extraction=extraction,
                ),
                ToolUseDecision,
            )
        except Exception as e:
            logger.warning(f"agent tool-selection failed {name} ({resource_id}); skipping tool: {e}")
            return False
        logger.info(
            f"agent {'selected' if decision.use_tool else 'skipped'} managed tool "
            f"{name} ({resource_id}): {decision.reason}"
        )
        return bool(decision.use_tool)

    async def _agent_should_use_tool_async(
        self,
        *,
        entry: dict[str, Any],
        resource_type: str,
        resource_id: str,
        extraction: dict[str, Any],
    ) -> bool:
        if not entry.get("agent_select"):
            return True
        name = str(entry.get("name") or "")
        if name == "candidate_grounded_normalizer" and not _candidate_grounding_inputs(extraction):
            logger.info(f"agent skipped managed tool {name} ({resource_id}): no candidate-grounded fields")
            return False
        try:
            decision = await ainvoke_json(
                self.llm,
                _tool_use_decision_prompt(
                    resource_type=resource_type,
                    resource_id=resource_id,
                    tool_entry=entry,
                    extraction=extraction,
                ),
                ToolUseDecision,
            )
        except Exception as e:
            logger.warning(f"agent tool-selection failed {name} ({resource_id}); skipping tool: {e}")
            return False
        logger.info(
            f"agent {'selected' if decision.use_tool else 'skipped'} managed tool "
            f"{name} ({resource_id}): {decision.reason}"
        )
        return bool(decision.use_tool)

    def _apply_postprocess_tools(
        self,
        *,
        resource_type: str,
        resource_id: str,
        readme_content: str,
        extraction: dict[str, Any],
    ) -> dict[str, Any]:
        current = extraction
        for entry in self.tool_store.active_tools("postprocess_extraction"):
            name = str(entry.get("name"))
            try:
                if not self._agent_should_use_tool(
                    entry=entry,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    extraction=current,
                ):
                    continue
                before_count = _extraction_item_count(current)
                result = self.tool_store.run_tool(
                    name,
                    {
                        "stage": "postprocess_extraction",
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "readme_content": readme_content,
                        "extraction": current,
                        "extraction_result": current,
                    },
                )
                candidate: dict[str, Any] | None = None
                if isinstance(result.get("extraction"), dict):
                    candidate = result["extraction"]
                elif isinstance(result.get("extraction_result"), dict):
                    candidate = result["extraction_result"]
                elif "attributes" in result or "relations" in result:
                    candidate = dict(current)
                    if "attributes" in result:
                        candidate["attributes"] = result.get("attributes") or []
                    if "relations" in result:
                        candidate["relations"] = result.get("relations") or []
                    for key, value in result.items():
                        if key not in {"attributes", "relations"}:
                            candidate[key] = value
                if candidate is None:
                    continue
                after_count = _extraction_item_count(candidate)
                if before_count > 0 and after_count == 0 and not entry.get("allow_empty_output"):
                    logger.warning(
                        "managed postprocess tool produced empty extraction; "
                        f"skipping {name} ({resource_id}). Set allow_empty_output=true "
                        "in tool_registry.json to permit this intentionally."
                    )
                    continue
                current = candidate
            except Exception as e:
                logger.warning(f"managed postprocess tool failed {name} ({resource_id}): {e}")
        return current

    async def _apply_postprocess_tools_async(
        self,
        *,
        resource_type: str,
        resource_id: str,
        readme_content: str,
        extraction: dict[str, Any],
    ) -> dict[str, Any]:
        current = extraction
        for entry in self.tool_store.active_tools("postprocess_extraction"):
            name = str(entry.get("name"))
            try:
                if not await self._agent_should_use_tool_async(
                    entry=entry,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    extraction=current,
                ):
                    continue
                before_count = _extraction_item_count(current)
                result = self.tool_store.run_tool(
                    name,
                    {
                        "stage": "postprocess_extraction",
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "readme_content": readme_content,
                        "extraction": current,
                        "extraction_result": current,
                    },
                )
                candidate: dict[str, Any] | None = None
                if isinstance(result.get("extraction"), dict):
                    candidate = result["extraction"]
                elif isinstance(result.get("extraction_result"), dict):
                    candidate = result["extraction_result"]
                elif "attributes" in result or "relations" in result:
                    candidate = dict(current)
                    if "attributes" in result:
                        candidate["attributes"] = result.get("attributes") or []
                    if "relations" in result:
                        candidate["relations"] = result.get("relations") or []
                    for key, value in result.items():
                        if key not in {"attributes", "relations"}:
                            candidate[key] = value
                if candidate is None:
                    continue
                after_count = _extraction_item_count(candidate)
                if before_count > 0 and after_count == 0 and not entry.get("allow_empty_output"):
                    logger.warning(
                        "managed postprocess tool produced empty extraction; "
                        f"skipping {name} ({resource_id}). Set allow_empty_output=true "
                        "in tool_registry.json to permit this intentionally."
                    )
                    continue
                current = candidate
            except Exception as e:
                logger.warning(f"managed postprocess tool failed {name} ({resource_id}): {e}")
        return current

    def _apply_candidate_choice_pass(
        self,
        *,
        resource_type: str,
        resource_id: str,
        extraction: dict[str, Any],
    ) -> dict[str, Any]:
        attributes = extraction.get("attributes") or []
        if not isinstance(attributes, list):
            return extraction
        changed = False
        new_attrs: list[dict[str, Any]] = []
        for attr in attributes:
            if not isinstance(attr, dict):
                new_attrs.append(attr)
                continue
            field = str(attr.get("attribute") or "").strip()
            if field not in CANDIDATE_GROUNDED_FIELDS:
                new_attrs.append(attr)
                continue
            candidates = _candidate_values_from_attr(attr)
            selected = _top1_candidate_from_attr(attr)
            if not candidates or not selected:
                new_attrs.append(attr)
                continue
            new_attrs.append(_replace_attr_value(attr, selected, candidates=candidates))
            changed = True
        if not changed:
            return extraction
        return {**extraction, "attributes": new_attrs, "relations": []}

    async def _apply_candidate_choice_pass_async(
        self,
        *,
        resource_type: str,
        resource_id: str,
        extraction: dict[str, Any],
    ) -> dict[str, Any]:
        attributes = extraction.get("attributes") or []
        if not isinstance(attributes, list):
            return extraction
        changed = False
        new_attrs: list[dict[str, Any]] = []
        for attr in attributes:
            if not isinstance(attr, dict):
                new_attrs.append(attr)
                continue
            field = str(attr.get("attribute") or "").strip()
            if field not in CANDIDATE_GROUNDED_FIELDS:
                new_attrs.append(attr)
                continue
            candidates = _candidate_values_from_attr(attr)
            selected = _top1_candidate_from_attr(attr)
            if not candidates or not selected:
                new_attrs.append(attr)
                continue
            new_attrs.append(_replace_attr_value(attr, selected, candidates=candidates))
            changed = True
        if not changed:
            return extraction
        return {**extraction, "attributes": new_attrs, "relations": []}

    def extract_one(
        self,
        *,
        resource_type: str,
        resource_id: str,
        readme_content: str,
    ) -> dict[str, Any]:
        readme_body, extraction_meta = self._prepare_readme(
            resource_type=resource_type,
            resource_id=resource_id,
            readme_content=readme_content,
        )
        manifest = self.store.load_manifest()
        workflow = str(manifest.get("workflow", "unified")).strip().lower()
        files = manifest.get("prompt_files") or {}
        groups = manifest.get("attribute_field_groups") or []
        if workflow in {"unified", "full_split", "split_relations"}:
            workflow = "split_attributes"
        if workflow not in {"split_attributes", "full_split"}:
            logger.warning(f"未知 workflow={workflow}，回退 split_attributes")
            workflow = "split_attributes"

        parts: list[ReadmeExtractionResult] = []

        if workflow in {"split_attributes", "full_split"}:
            attr_file = files.get("attribute_pass", "attribute_extract.txt")
            for group in groups:
                if not group:
                    continue
                target_fields = ", ".join(group)
                readme_chunks, chunk_meta = self._chunked_readmes_for_prompt(
                    attr_file,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    readme_content=readme_body,
                    target_fields=target_fields,
                )
                extraction_meta.update(chunk_meta)
                for chunk_index, readme_chunk in enumerate(readme_chunks, start=1):
                    chunk_body = readme_chunk
                    if len(readme_chunks) > 1:
                        chunk_body = (
                            f"[README CHUNK {chunk_index}/{len(readme_chunks)}]\n"
                            f"{readme_chunk}\n"
                            "[END README CHUNK]"
                        )
                    attr_prompt = self.store.render_template(
                        attr_file,
                        variables={
                            "resource_type": resource_type,
                            "resource_id": resource_id,
                            "readme_content": chunk_body,
                            "target_fields": target_fields,
                        },
                    )
                    parts.append(
                        _invoke_extraction_json(
                            self.llm, attr_prompt, resource_id=resource_id
                        )
                    )

        if not parts:
            logger.warning(f"未知 workflow={workflow}，回退 unified")
            extraction = _to_audit_dict(ReadmeExtractionResult())
            return self._attach_extraction_meta(
                _without_readme_relations(extraction), extraction_meta
            )

        extraction = _force_attribute_entity(
            _to_audit_dict(_merge_results(parts)),
            resource_type=resource_type,
            resource_id=resource_id,
        )
        extraction = self._apply_postprocess_tools(
            resource_type=resource_type,
            resource_id=resource_id,
            readme_content=readme_body,
            extraction=extraction,
        )
        extraction = _force_attribute_entity(
            extraction,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        extraction = self._apply_candidate_choice_pass(
            resource_type=resource_type,
            resource_id=resource_id,
            extraction=extraction,
        )
        extraction = _dedupe_extraction_attributes_by_confidence(extraction)
        return self._attach_extraction_meta(
            _without_readme_relations(extraction), extraction_meta
        )

    async def extract_one_async(
        self,
        *,
        resource_type: str,
        resource_id: str,
        readme_content: str,
    ) -> dict[str, Any]:
        readme_body, extraction_meta = self._prepare_readme(
            resource_type=resource_type,
            resource_id=resource_id,
            readme_content=readme_content,
        )
        manifest = self.store.load_manifest()
        workflow = str(manifest.get("workflow", "unified")).strip().lower()
        files = manifest.get("prompt_files") or {}
        groups = manifest.get("attribute_field_groups") or []
        if workflow in {"unified", "full_split", "split_relations"}:
            workflow = "split_attributes"
        if workflow not in {"split_attributes", "full_split"}:
            logger.warning(f"未知 workflow={workflow}，回退 split_attributes")
            workflow = "split_attributes"

        async def _call(prompt: str) -> ReadmeExtractionResult:
            return await _ainvoke_extraction_json(
                self.llm, prompt, resource_id=resource_id
            )

        parts: list[ReadmeExtractionResult] = []

        if workflow in {"split_attributes", "full_split"}:
            attr_file = files.get("attribute_pass", "attribute_extract.txt")
            for group in groups:
                if not group:
                    continue
                target_fields = ", ".join(group)
                readme_chunks, chunk_meta = self._chunked_readmes_for_prompt(
                    attr_file,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    readme_content=readme_body,
                    target_fields=target_fields,
                )
                extraction_meta.update(chunk_meta)
                for chunk_index, readme_chunk in enumerate(readme_chunks, start=1):
                    chunk_body = readme_chunk
                    if len(readme_chunks) > 1:
                        chunk_body = (
                            f"[README CHUNK {chunk_index}/{len(readme_chunks)}]\n"
                            f"{readme_chunk}\n"
                            "[END README CHUNK]"
                        )
                    attr_prompt = self.store.render_template(
                        attr_file,
                        variables={
                            "resource_type": resource_type,
                            "resource_id": resource_id,
                            "readme_content": chunk_body,
                            "target_fields": target_fields,
                        },
                    )
                    parts.append(await _call(attr_prompt))

        if not parts:
            logger.warning(f"未知 workflow={workflow}，回退 unified")
            extraction = _to_audit_dict(ReadmeExtractionResult())
            return self._attach_extraction_meta(
                _without_readme_relations(extraction), extraction_meta
            )

        extraction = _force_attribute_entity(
            _to_audit_dict(_merge_results(parts)),
            resource_type=resource_type,
            resource_id=resource_id,
        )
        extraction = await self._apply_postprocess_tools_async(
            resource_type=resource_type,
            resource_id=resource_id,
            readme_content=readme_body,
            extraction=extraction,
        )
        extraction = _force_attribute_entity(
            extraction,
            resource_type=resource_type,
            resource_id=resource_id,
        )
        extraction = await self._apply_candidate_choice_pass_async(
            resource_type=resource_type,
            resource_id=resource_id,
            extraction=extraction,
        )
        extraction = _dedupe_extraction_attributes_by_confidence(extraction)
        return self._attach_extraction_meta(
            _without_readme_relations(extraction), extraction_meta
        )

    async def extract_many_async(
        self,
        cases: list[tuple[str, str, str]],
        *,
        concurrency: int = 2,
        sample_timeout_sec: float | None = None,
    ) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(max(1, concurrency))
        timeout_s = float(sample_timeout_sec) if sample_timeout_sec else None

        async def _one(rt: str, rid: str, readme: str) -> dict[str, Any]:
            async with sem:
                try:
                    coro = self.extract_one_async(
                        resource_type=rt,
                        resource_id=rid,
                        readme_content=readme,
                    )
                    if timeout_s is not None and timeout_s > 0:
                        return await asyncio.wait_for(coro, timeout=timeout_s)
                    return await coro
                except asyncio.TimeoutError:
                    logger.warning(
                        f"sample extraction timeout {rid}: exceeded {timeout_s:.1f}s; returning empty extraction"
                    )
                    return _empty_extraction_with_error(
                        resource_id=rid,
                        error_type="sample_timeout",
                        error=f"sample extraction timeout after {timeout_s:.1f}s",
                        sample_timeout_sec=timeout_s,
                    )
                except Exception as e:
                    error_type = "timeout" if is_timeout_error(e) else type(e).__name__
                    logger.warning(f"sample extraction failed {rid}: {e}")
                    return _empty_extraction_with_error(
                        resource_id=rid,
                        error_type=error_type,
                        error=str(e),
                        sample_timeout_sec=timeout_s,
                    )

        tasks = [_one(rt, rid, readme) for rt, rid, readme in cases]
        return await asyncio.gather(*tasks)
