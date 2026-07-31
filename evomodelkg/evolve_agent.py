from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from evomodelkg.clients.deepseek_client import parse_json_model
from evomodelkg.llm_config import create_role_llm
from evomodelkg.prompt_store import PromptStore
from evomodelkg.schemas import EvolutionPlan
from evomodelkg.tools import EvolutionToolRunner, tools_prompt_block


MAX_EVOLVE_PLAN_PARSE_RETRIES = 2
MAX_RAW_RESPONSE_LOG_CHARS = 12000
MAX_EVOLVE_MISMATCH_SAMPLES = 250
MAX_EVOLVE_INTERESTING_CASES = 8
MAX_MISMATCH_VALUE_CHARS = 360
MAX_MISMATCH_EVIDENCE_CHARS = 280
MAX_MISMATCH_NOTE_CHARS = 180
MAX_INTRA_ITERATION_FACTS_CHARS = 1800


EVOLVE_ACTION_ENFORCEMENT = """

[ACTION_ENFORCEMENT]
You have write-capable tools. Use them when the strategy says to modify prompts or tools.

Write-capable tool names:
- write_prompt: overwrite a prompt file after read_prompt. For multi-line text, prefer arguments.content_lines.
- write_tool_file: create or overwrite a managed tool file after read_tool_contract and overlap checks.
  For multi-line Python code, prefer arguments.content_lines instead of one large escaped content string.
- update_tool_registry: register or update managed tools. There is no tool named write_tool_registry.
- run_managed_tool: validate a managed tool after writing or registering it.

Do not keep reading the same files across rounds. If previous rounds already read the prompt,
tool contract, tool registry, or candidate tool files, the next round must include at least one of:
write_prompt, write_tool_file, update_tool_registry, run_managed_tool, or done=true.

If your strategy says "create tool", "update tool", "register tool", or "update prompt", your
tool_calls must include the corresponding write-capable tool in the same round whenever you already
have enough context. Do not only describe the write in analysis or strategy.

When writing code, keep JSON simple: use content_lines, avoid triple-quoted strings inside the JSON,
and prefer single quotes in Python code when possible.

When changing prompts, first read the current prompt. Use write_prompt only when a complete
prompt rewrite is safer than tool or registry changes. Avoid repeatedly appending constraint sections.

Do not invent manifest workflows or manifest tool keys. Managed preprocess/postprocess tools
are enabled only through tool_registry.json. update_manifest may only use existing workflows:
unified, split_attributes, full_split, split_relations.

If no concrete write is safe, return done=true with tool_calls=[] and explain why in analysis.
Never return done=false with tool_calls=[].
"""


def _is_relation_feedback_key(key: Any) -> bool:
    text = str(key).lower()
    return "relation" in text or "relations" in text or "关系" in text


def _strip_relation_feedback(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_relation_feedback(child)
            for key, child in value.items()
            if not _is_relation_feedback_key(key)
        }
    if isinstance(value, list):
        out = []
        for child in value:
            cleaned = _strip_relation_feedback(child)
            if isinstance(cleaned, str) and (
                "relation" in cleaned.lower() or "关系" in cleaned
            ):
                continue
            out.append(cleaned)
        return out
    return value


def _slim_evolve_summary(summary: Any, meta: dict[str, Any]) -> dict[str, Any]:
    """Keep only high-signal summary metrics for the evolve prompt."""
    if not isinstance(summary, dict):
        return {}

    completeness = summary.get("completeness")
    if isinstance(completeness, dict):
        completeness = {
            key: completeness.get(key)
            for key in (
                "attributes_extracted",
                "avg_attributes_per_case",
                "avg_unique_attribute_names_per_case",
                "zero_attribute_cases",
            )
            if key in completeness
        }

    completeness_fit = summary.get("completeness_fit")
    if isinstance(completeness_fit, dict):
        completeness_fit = {
            key: completeness_fit.get(key)
            for key in ("score",)
            if key in completeness_fit
        }

    slim: dict[str, Any] = {
        "accuracy": summary.get("accuracy"),
        "dataset_accuracy": summary.get("dataset_accuracy"),
        "completeness": completeness,
        "completeness_fit": completeness_fit,
    }
    input_cost = meta.get("input_cost")
    if isinstance(input_cost, dict):
        slim["input_cost"] = input_cost

    return {key: value for key, value in slim.items() if value not in (None, {}, [])}


def _shorten_feedback_value(value: Any, limit: int) -> Any:
    if value is None:
        return value
    if isinstance(value, (list, dict)):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _compact_mismatch_for_prompt(item: Any) -> Any:
    if not isinstance(item, dict):
        return _shorten_feedback_value(item, MAX_MISMATCH_VALUE_CHARS)
    compact: dict[str, Any] = {}
    for key in (
        "kind",
        "resource_id",
        "attribute",
        "readme_value",
        "structured_value",
        "evidence_span",
        "normalization_note",
    ):
        if key not in item:
            continue
        value = item.get(key)
        if key == "evidence_span":
            value = _shorten_feedback_value(value, MAX_MISMATCH_EVIDENCE_CHARS)
        elif key == "normalization_note":
            value = _shorten_feedback_value(value, MAX_MISMATCH_NOTE_CHARS)
        elif key in {"readme_value", "structured_value"}:
            value = _shorten_feedback_value(value, MAX_MISMATCH_VALUE_CHARS)
        if value not in (None, "", [], {}):
            compact[key] = value
    return compact


def _unique_preserve_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _active_tool_summary(tool_runner: EvolutionToolRunner) -> list[dict[str, Any]]:
    try:
        active = tool_runner.tool_store.active_tools()
    except Exception as e:
        return [{"error": _shorten_feedback_value(repr(e), 300)}]
    return [
        {
            "name": row.get("name"),
            "filename": row.get("filename") or row.get("file"),
            "stage": row.get("stage"),
            "enabled": row.get("enabled", True),
        }
        for row in active
    ]


def _build_intra_iteration_memory(
    log_entries: list[dict[str, Any]],
    tool_runner: EvolutionToolRunner,
) -> dict[str, Any]:
    """Build a small factual state for this EvolveAgent.run call."""
    written_tool_files: dict[str, dict[str, Any]] = {}
    written_prompts: dict[str, dict[str, Any]] = {}
    registered_tool_names: list[str] = []
    validation_runs: list[dict[str, Any]] = []

    for entry in log_entries:
        round_no = entry.get("round")
        for row in entry.get("tool_results") or []:
            tool = str(row.get("tool") or "")
            args = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
            result = row.get("result") if isinstance(row.get("result"), dict) else {}
            ok = result.get("ok")

            if tool == "write_tool_file":
                filename = str(args.get("filename") or "").strip()
                if filename:
                    prev = written_tool_files.get(filename, {"filename": filename, "write_count": 0})
                    prev["write_count"] = int(prev.get("write_count") or 0) + 1
                    prev["last_round"] = round_no
                    prev["last_ok"] = ok
                    written_tool_files[filename] = prev
            elif tool == "write_prompt":
                filename = str(args.get("filename") or "").strip()
                if filename:
                    prev = written_prompts.get(filename, {"filename": filename, "write_count": 0})
                    prev["write_count"] = int(prev.get("write_count") or 0) + 1
                    prev["last_round"] = round_no
                    prev["last_ok"] = ok
                    written_prompts[filename] = prev
            elif tool == "update_tool_registry":
                patch = args.get("patch") if isinstance(args.get("patch"), dict) else {}
                tool_names: list[str] = []
                patch_tools = patch.get("tools")
                if isinstance(patch_tools, dict):
                    tool_names = [str(name) for name in patch_tools]
                elif isinstance(patch_tools, list):
                    tool_names = [
                        str(item.get("name") or item.get("filename") or item.get("file") or "")
                        for item in patch_tools
                        if isinstance(item, dict)
                    ]
                if ok is not False:
                    registered_tool_names.extend(_unique_preserve_order(tool_names))
            elif tool == "run_managed_tool":
                validation_runs.append(
                    {
                        "round": round_no,
                        "name": args.get("name"),
                        "ok": ok,
                    }
                )

    active_tools = _active_tool_summary(tool_runner)
    successfully_validated = {
        str(row.get("name"))
        for row in validation_runs
        if row.get("name") and row.get("ok") is True
    }
    registered_tool_names = _unique_preserve_order(registered_tool_names)
    active_tool_names = {
        str(row.get("name")) for row in active_tools if row.get("name") and not row.get("error")
    }
    touched_tool_names = set(registered_tool_names)
    for row in written_tool_files.values():
        filename = str(row.get("filename") or "")
        if filename.endswith(".py"):
            touched_tool_names.add(Path(filename).stem)
    pending_validation = sorted(
        name for name in touched_tool_names if name in active_tool_names and name not in successfully_validated
    )
    repeated_writes = [
        {"filename": filename, "write_count": row.get("write_count"), "last_round": row.get("last_round")}
        for filename, row in written_tool_files.items()
        if int(row.get("write_count") or 0) > 1
    ]

    state = {
        "rounds_completed": len(log_entries),
        "tool_files_written": list(written_tool_files.values())[-5:],
        "prompts_written": list(written_prompts.values())[-4:],
        "tools_registered_or_updated": registered_tool_names[-8:],
        "validation_runs": validation_runs[-5:],
        "pending_validation": pending_validation,
        "repeated_tool_file_writes": repeated_writes[-5:],
        "update_rule": (
            "Existing tools/prompts may be updated when source has been read or validation shows a concrete defect. "
            "Do not blindly overwrite the same file or create another tool with the same purpose."
        ),
    }
    shrink_order = (
        "tool_files_written",
        "prompts_written",
        "tools_registered_or_updated",
        "validation_runs",
        "pending_validation",
        "repeated_tool_file_writes",
    )
    while len(json.dumps(state, ensure_ascii=False, indent=2, default=str)) > MAX_INTRA_ITERATION_FACTS_CHARS:
        for key in shrink_order:
            if len(state.get(key) or []) > 2:
                state[key] = state[key][1:]
                break
        else:
            state["update_rule"] = "Update existing files only after reading source or seeing validation evidence."
            break
    return state


def _bounded_json_dumps(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


EVOLVE_SYSTEM = """你是 README 抽取管道的进化智能体。你的任务是根据本代 evolve_report.json 的指标、错配样例和样本观察，改进下一代的抽取效果。

一、你可以改什么
1. 修改 prompts 目录下的提示词。改提示词前必须先 read_prompt；只能用 write_prompt 完整改写。不要反复追加新的约束段落。
2. 创建或修改 managed tools。工具只有写入 tool_registry.json，且 stage 为 preprocess_readme 或 postprocess_extraction、enabled=true 时，才会影响下一代。
3. 不要修改 attribute_field_groups 等锁定的字段分组。

二、如何理解指标
1. 准确性 summary.accuracy：README 抽取值与结构化 baseline 的对齐情况。
2. 完整性 summary.completeness / completeness_fit：只衡量 README 侧抽取数量和合理性，不代表 baseline 覆盖率。
3. 输入成本 meta.input_cost：README 预处理前后的字符数、估算 token 和压缩率。只有准确性和完整性不明显下降时，压缩才有价值。
4. MISMATCH_SAMPLES：用于定位可改进问题。请结合 readme_value、structured_value、evidence_span、normalization_note 判断根因是提示词歧义、证据不足、字段口径不一致，还是可程序化归一化问题。

三、优先改进策略（优先修改提示词！！！）
1. 如果问题来自字段定义、证据选择、禁止臆造、schema 解释或抽取口径，优先修改 prompt。
2. 如果问题可复用、确定、程序化，优先创建或更新 managed tool，而不是只在 prompt 里反复强调。

四、工具治理规则
1. 创建或修改任何 managed tool 前，必须调用 read_tool_contract，并遵守 payload、返回格式、extraction schema 和 registry schema。
2. 如果本代可能需要工具，尽量在前 2 轮内读取契约，给后续写入、注册和 run_managed_tool 验证留出轮次。
3. 新建工具前，先查看工具列表和 tool_registry.json。只读取名称、stage 或 registry 描述疑似重复的候选工具源码。
4. 同类工具优先合并或更新，不要重复新建。HTML 清理、模板占位清理、空表格清理、重复 badge/image 清理通常应合成一个 README 模板噪声清理工具，除非有明确安全边界需要拆分。
5. 后处理字段归一化规则也应尽量集中在一个确定性属性规范化工具中。若保留多个相似工具，必须在 registry 中说明边界和不能合并的原因。
6. 写入或修改工具后，更新 tool_registry.json 的 purpose、evidence_cases、expected_behavior、safety_constraints、validation_criteria。
   update_tool_registry 会按工具 name 增量合并：未在 patch.tools 中提及的已注册工具必须保留；同名工具更新，新名称追加。更新后必须 read_tool_registry，确认最终活动工具集合符合预期，不得无意禁用其它阶段的工具。
7. managed tool 代码必须定义可导入口函数。优先定义 def run(payload: dict) -> dict；后处理工具也可以定义 def postprocess_extraction(extraction, resource_id, readme_content, payload) -> dict；预处理工具也可以定义 def preprocess_readme(readme_content, resource_id, payload) -> dict。禁止只写 main()、读取 sys.stdin 或只作为命令行脚本运行。
8. 写入 prompt 时必须使用可读 UTF-8 文本；中文或英文都可以，但必须避免编码损坏后的不可读文本。

五、工作方式
1. 先检查代表性失败样本和异常样本，再决定改 prompt 还是改工具；具体可用工具及其功能见下方 tools_json，不要在检查前预设根因。
2. 每轮只输出一个 EvolutionPlan JSON，可以包含多次 tool_calls；但每一代最多新建一个 managed tool。可以围绕这一个工具多次读取、写入、注册、测试或修改；如果还需要生成其它新工具，请留到后续代。
3. 如果已经没有把握继续改进，返回 done=true 且 tool_calls=[]。
4. 如果已经读取过 prompt、tool contract 或 registry，下一轮不要重复读取相同内容，除非有新的具体问题需要确认。应尽快进入 write_prompt、write_tool_file、update_tool_registry 或 run_managed_tool。

六、输出格式硬性要求
1. 你的每一次回复必须是单个 JSON object，不能有 Markdown、代码块、解释文字、项目符号或前后缀。
2. JSON 顶层必须且只能包含 analysis、strategy、tool_calls、done 四个字段。
3. 如果只是分析但不调用工具，必须返回 tool_calls=[] 且 done=true；不要输出自然语言分析段落。
4. analysis 和 strategy 中如需换行，必须使用 JSON 字符串中的 \\n，不能直接在字符串内部换真实换行。
5. 合法示例：{"analysis":"brief reason","strategy":"stop safely","tool_calls":[],"done":true}

可用工具如下：
{tools_json}
"""
# 3. 可以创建或更新 preprocess_readme 工具做 README 输入清洗，例如去 HTML 注释、模板占位、空表格、重复 badge/image、无信息 boilerplate；但必须保留 license、base model、dataset、architecture、usage、training/evaluation 等事实证据。
# 4. 可以创建或更新 postprocess_extraction 工具做 LLM 输出后的确定性规范化，例如删除无证据占位字段、清理 arXiv ID、转换参数量单位、规范布尔值或其它高频格式噪声。
# 5. 如果 meta.input_cost 中 readme_chars_reduction_rate 长期为 0，且样本观察显示存在大量 HTML 注释、badge-only 行、重复图片、模板占位、空表格、自动生成目录、无信息免责声明或安装模板，应优先考虑创建 preprocess_readme 工具。
# 6. README 清洗工具必须采用保守白名单思路：只删除明显无事实信息的模板噪声；不得删除包含 license、model id、base_model、dataset、architecture、AutoModel、pipeline、training、evaluation、intended use、limitations、citation、parameters、file format 等关键词的段落或代码块。
# 7. 如果 validation/evolve mismatch 中反复出现可程序化错配，应优先尝试创建或更新 postprocess_extraction 工具。尤其是：README 未明确该字段，或值为 unknown、not specified、not mentioned、N/A、None、空字符串时，应删除该属性；cited_papers 可清理为标准 arXiv ID；num_parameters / model_size 可做 B、M、K 等单位转换。


def _build_evolve_user_message(evolve_report: dict[str, Any], generation: int) -> str:
    meta = evolve_report.get("meta") or {}
    summary = _strip_relation_feedback(
        _slim_evolve_summary(evolve_report.get("summary", {}), meta)
    )
    mismatches = [
        _compact_mismatch_for_prompt(item)
        for item in _strip_relation_feedback(evolve_report.get("mismatches", []))
    ][:MAX_EVOLVE_MISMATCH_SAMPLES]
    lines = _strip_relation_feedback(evolve_report.get("compare_summary_lines", []))
    interesting_cases = _strip_relation_feedback(
        evolve_report.get("interesting_cases", [])
    )[:MAX_EVOLVE_INTERESTING_CASES]
    return (
        f"[GENERATION] {generation}\n"
        f"[EVOLVE_SPLIT] {meta.get('dataset_split', 'evolve')}\n"
        f"[COMPARE_LINES]\n" + "\n".join(lines) + "\n\n"
        f"[SUMMARY_JSON]\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"[INTERESTING_CASES]\n{json.dumps(interesting_cases, ensure_ascii=False, indent=2)}\n\n"
        f"[MISMATCH_SAMPLES]\n{json.dumps(mismatches, ensure_ascii=False, indent=2)}\n\n"
        "Return EvolutionPlan JSON:\n"
        '{"analysis":"...", "strategy":"...", "tool_calls":[{"tool":"...", "arguments":{}}], "done": false}'
    )


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if "```" not in cleaned:
        return cleaned
    return cleaned.replace("```json", "").replace("```JSON", "").replace("```", "").strip()


def _balanced_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    in_string = False
    escaped = False
    start: int | None = None
    depth = 0

    for idx, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = idx
                depth = 1
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start : idx + 1].strip())
                start = None

    return candidates


def _evolution_json_candidates(text: str) -> list[str]:
    cleaned = _strip_json_fence(text)
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)
    candidates.extend(_balanced_json_object_candidates(cleaned))
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        candidates.append(cleaned[first : last + 1].strip())
    uniq: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            uniq.append(candidate)
            seen.add(candidate)
    return uniq


def _normalize_evolution_payload(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    payload = dict(data)
    calls = payload.get("tool_calls")
    if isinstance(calls, list):
        normalized_calls = []
        for call in calls:
            if not isinstance(call, dict):
                normalized_calls.append(call)
                continue
            row = dict(call)
            if "tool" not in row and "name" in row:
                row["tool"] = row.get("name")
            function = row.get("function")
            if "tool" not in row and isinstance(function, dict):
                row["tool"] = function.get("name")
                if "arguments" not in row and isinstance(function.get("arguments"), dict):
                    row["arguments"] = function.get("arguments")
            normalized_calls.append(row)
        payload["tool_calls"] = normalized_calls
    return payload


def _escape_invalid_json_backslashes(text: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)


def _is_evolve_context_length_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "maximum context length" in text
        or "reduce the length of the input prompt" in text
        or "context length" in text
    )


def _shrink_evolve_prompt(prompt: str) -> str:
    # Preserve complete tool history and exact file contents. If the request is too
    # large, reduce the replaceable mismatch evidence instead of deleting memory.
    mismatch_marker = "\n[MISMATCH_SAMPLES]\n"
    mismatch_end_marker = "\n\nReturn EvolutionPlan JSON:\n"
    if mismatch_marker in prompt and mismatch_end_marker in prompt:
        prefix, remainder = prompt.split(mismatch_marker, 1)
        mismatch_text, suffix = remainder.split(mismatch_end_marker, 1)
        if len(mismatch_text) > 12000:
            keep = 12000
        elif len(mismatch_text) > 3000:
            keep = max(3000, len(mismatch_text) // 2)
        else:
            keep = len(mismatch_text)
        if keep < len(mismatch_text):
            mismatch_text = (
                mismatch_text[:keep]
                + '\n{"note":"Remaining mismatch samples omitted after a context-length error; complete tool history is preserved."}'
            )
        return prefix + mismatch_marker + mismatch_text + mismatch_end_marker + suffix
    # A context retry must never silently discard tool/file memory. If the prompt
    # has no shrinkable report section, retry unchanged and fail safely if needed.
    return prompt


def _parse_evolution_plan(raw_text: str) -> EvolutionPlan:
    last_err: Exception | None = None
    for candidate in _evolution_json_candidates(raw_text):
        for json_text in (candidate, _escape_invalid_json_backslashes(candidate)):
            try:
                data = json.loads(json_text)
                return EvolutionPlan.model_validate(_normalize_evolution_payload(data))
            except Exception as e:
                last_err = e
    if last_err is not None:
        raise last_err
    return parse_json_model(raw_text, EvolutionPlan)


def _require_actionable_plan(plan: EvolutionPlan) -> EvolutionPlan:
    if not plan.done and not plan.tool_calls:
        raise ValueError(
            "invalid EvolutionPlan: done=false requires at least one tool_call; "
            "return done=true when no further action is needed"
        )
    return plan


class EvolveAgent:
    def __init__(
        self,
        store: PromptStore,
        *,
        temperature: float = 0.2,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tool_rounds: int = 8,
    ):
        self.store = store
        self.tool_runner = EvolutionToolRunner(store)
        self.llm = create_role_llm(
            role="evolve",
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
        )
        self.max_tool_rounds = max(1, max_tool_rounds)

    def _invoke_plan_with_retries(
        self,
        prompt: str,
        *,
        round_i: int,
    ) -> tuple[EvolutionPlan | None, list[dict[str, Any]]]:
        errors: list[dict[str, Any]] = []
        current_prompt = prompt
        for attempt in range(MAX_EVOLVE_PLAN_PARSE_RETRIES + 1):
            raw_text = ""
            try:
                response = self.llm.invoke(current_prompt)
                content = response.content
                if isinstance(content, list):
                    raw_text = "\n".join(str(x) for x in content)
                else:
                    raw_text = str(content)
                plan = _parse_evolution_plan(raw_text)
                plan = _require_actionable_plan(plan)
                if attempt > 0:
                    logger.info(
                        f"EvolveAgent round {round_i + 1}: JSON parse recovered "
                        f"after retry {attempt}"
                    )
                return plan, errors
            except Exception as e:
                err = {
                    "attempt": attempt + 1,
                    "error": repr(e),
                    "raw_response": raw_text[:MAX_RAW_RESPONSE_LOG_CHARS],
                }
                errors.append(err)
                logger.warning(
                    f"EvolveAgent round {round_i + 1}: invalid EvolutionPlan JSON "
                    f"attempt {attempt + 1}/{MAX_EVOLVE_PLAN_PARSE_RETRIES + 1}: {e}"
                )
                if _is_evolve_context_length_error(e):
                    current_prompt = _shrink_evolve_prompt(current_prompt)
                    continue
                current_prompt = (
                    prompt
                    + "\n\n[JSON_REPAIR_INSTRUCTION]\n"
                    "Your previous response was not valid JSON for EvolutionPlan. "
                    "Return exactly one JSON object and no markdown, no prose, no code fence. "
                    "The object must match this schema: "
                    '{"analysis":"string","strategy":"string","tool_calls":[{"tool":"string","arguments":{}}],"done":false}. '
                    "If no further action is needed, return "
                    '{"analysis":"JSON repair fallback","strategy":"stop safely","tool_calls":[],"done":true}.\n'
                    "\n[INVALID_PREVIOUS_RESPONSE]\n"
                    + raw_text[:MAX_RAW_RESPONSE_LOG_CHARS]
                    + "\n\n[PARSE_ERROR]\n"
                    + repr(e)
                )
        return None, errors

    def run(
        self,
        evolve_report: dict[str, Any],
        *,
        generation: int,
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        log_entries: list[dict[str, Any]] = []
        self.tool_runner.set_observation_context(evolve_report)
        user_msg = _build_evolve_user_message(evolve_report, generation)
        system = EVOLVE_SYSTEM.replace("{tools_json}", tools_prompt_block()) + EVOLVE_ACTION_ENFORCEMENT
    
        for round_i in range(self.max_tool_rounds):
            prompt = (
                system
                + "\n\n[CURRENT_MANIFEST]\n"
                + json.dumps(self.store.load_manifest(), ensure_ascii=False, indent=2)
                + "\n\n[USER]\n"
                + user_msg
            )
            if log_entries:
                intra_iteration_facts = _build_intra_iteration_memory(log_entries, self.tool_runner)
                prompt += "\n\n[INTRA_ITERATION_STATE]\n" + _bounded_json_dumps(
                    intra_iteration_facts,
                    limit=MAX_INTRA_ITERATION_FACTS_CHARS,
                )
                prompt += "\n\n[COMPLETE_TOOL_HISTORY]\n" + json.dumps(
                    log_entries, ensure_ascii=False, indent=2, default=str
                )
    
            plan, parse_errors = self._invoke_plan_with_retries(prompt, round_i=round_i)
            if plan is None:
                entry = {
                    "round": round_i + 1,
                    "plan": {
                        "analysis": "EvolveAgent stopped because the model did not return valid EvolutionPlan JSON after retries.",
                        "strategy": "stop safely after JSON parse failures",
                        "tool_calls": [],
                        "done": True,
                    },
                    "tool_results": [],
                    "parse_errors": parse_errors,
                }
                log_entries.append(entry)
                logger.warning(
                    f"EvolveAgent round {round_i + 1}: stopping evolution after "
                    f"{len(parse_errors)} invalid JSON responses"
                )
                break
            entry: dict[str, Any] = {
                "round": round_i + 1,
                "plan": plan.model_dump(),
                "tool_results": [],
            }
            if parse_errors:
                entry["parse_errors"] = parse_errors
            logger.info(f"EvolveAgent round {round_i + 1}: {plan.strategy or plan.analysis[:120]}")
    
            if plan.done or not plan.tool_calls:
                log_entries.append(entry)
                break
    
            for tc in plan.tool_calls:
                result = self.tool_runner.execute(tc.tool, tc.arguments)
                entry["tool_results"].append(
                    {"tool": tc.tool, "arguments": tc.arguments, "result": result}
                )
                logger.info(f"  tool {tc.tool}: ok={result.get('ok')}")
    
            log_entries.append(entry)
            user_msg = (
                "上一轮工具已执行。若仍需改进，请继续给出具体 tool_calls；否则返回 done=true。\\n"
                + _build_evolve_user_message(evolve_report, generation)
            )
            user_msg = (
                "[NEXT_ROUND_ACTION_REQUIREMENT]\n"
                "Do not repeat read-only calls if the needed content was already read. "
                "Use write_prompt only after reading the prompt and deciding a complete prompt rewrite is safer; avoid repeatedly appending constraint sections. Use write_tool_file to create/update tools, "
                "update_tool_registry to register tools, and run_managed_tool to validate tools. "
                "There is no tool named write_tool_registry. "
                "If no safe write remains, return done=true with tool_calls=[]. "
                "Never return done=false with tool_calls=[].\n\n"
                + user_msg
            )
    
        payload = {
            "meta": {
                "kind": "self_evolve_evolution_log",
                "memory_mode": "complete_tool_history",
                "generation": generation,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rounds": len(log_entries),
            },
            "entries": log_entries,
            "final_manifest": self.store.load_manifest(),
        }
    
        if run_dir is not None:
            run_dir = Path(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / "evolution_log.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"EvolveAgent: 日志已写入 {path}")
    
        return payload

