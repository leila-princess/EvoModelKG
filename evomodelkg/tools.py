from __future__ import annotations

import json
from typing import Any

from evomodelkg.prompt_store import PromptStore
from evomodelkg.readme_observer import (
    profile_markdown_text,
    read_excerpt,
    repeated_markdown_patterns,
)
from evomodelkg.tool_store import ToolStore


def _content_arg(args: dict[str, Any]) -> str:
    if "content_lines" in args and args["content_lines"] is not None:
        lines = args["content_lines"]
        if not isinstance(lines, list):
            raise TypeError("content_lines must be a list of strings")
        return "\n".join(str(line) for line in lines)
    if "content" in args:
        return str(args["content"])
    raise KeyError("content or content_lines is required")


ALLOWED_MANIFEST_WORKFLOWS = {"unified", "split_attributes", "full_split", "split_relations"}
ALLOWED_MANIFEST_PATCH_KEYS = {"description", "workflow", "prompt_files", "notes"}


def _merge_tool_registry_patch(previous: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge a registry patch without dropping tools omitted from the patch.

    Registry tools are keyed by ``name``. A patched tool updates the existing
    entry with that name, a new name is appended, and unrelated tools retain
    their existing order and configuration.
    """
    merged = dict(previous)
    for key, value in patch.items():
        if key != "tools":
            merged[key] = value
            continue
        if not isinstance(value, (list, dict)):
            raise TypeError("patch.tools must be a list or object")

        previous_tools = ToolStore._normalize_registry(previous).get("tools", [])
        patch_tools = ToolStore._normalize_registry(
            {"version": previous.get("version", 1), "tools": value}
        ).get("tools", [])
        by_name = {
            str(entry["name"]): dict(entry)
            for entry in previous_tools
            if isinstance(entry, dict) and entry.get("name")
        }
        order = list(by_name)
        for entry in patch_tools:
            name = str(entry["name"])
            if name not in by_name:
                order.append(name)
                by_name[name] = {}
            by_name[name] = {**by_name[name], **entry}
        merged["tools"] = [by_name[name] for name in order]
    return merged


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_prompt_files",
        "description": "列出 prompts 目录下所有 .txt/.md 提示词文件名。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_prompt",
        "description": "读取指定提示词文件全文。",
        "parameters": {
            "type": "object",
            "properties": {"filename": {"type": "string", "description": "如 unified_extract.txt"}},
            "required": ["filename"],
        },
    },
    {
        "name": "write_prompt",
        "description": "覆盖写入提示词文件（用于优化 prompt 文案）。",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string", "description": "新的完整文件内容"},
                "content_lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "新的完整文件内容，按行提供；适合本地小模型输出多行文本。",
                },
            },
            "required": ["filename"],
        },
    },
    {
        "name": "read_manifest",
        "description": "读取 workflow 配置 manifest.json。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "update_manifest",
        "description": (
            "合并更新 manifest.json。可改 workflow、prompt_files；"
            "attribute_field_groups 属于固定实验协议，禁止修改。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "object",
                    "description": "要浅合并进 manifest 的 JSON 对象",
                }
            },
            "required": ["patch"],
        },
    },
    {
        "name": "sample_interesting_cases",
        "description": "List anomalous cases worth inspecting, such as long input with low extraction yield.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 5}},
            "required": [],
        },
    },
    {
        "name": "get_case_failure_detail",
        "description": "Inspect extraction counts, observation metadata, mismatches, and extraction output for one case.",
        "parameters": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
    },
    {
        "name": "read_readme_excerpt",
        "description": "Read a README excerpt for a case. mode may be head, middle, tail, or without_code.",
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {"type": "string"},
                "mode": {"type": "string", "default": "head"},
                "max_chars": {"type": "integer", "default": 4000},
            },
            "required": ["case_id"],
        },
    },
    {
        "name": "profile_markdown_structure",
        "description": "Profile generic Markdown structure: headings, section sizes, comments, code blocks, links, tables.",
        "parameters": {
            "type": "object",
            "properties": {"case_id": {"type": "string"}},
            "required": ["case_id"],
        },
    },
    {
        "name": "find_repeated_markdown_patterns",
        "description": "Find repeated headings, HTML comments, and lines across several README cases.",
        "parameters": {
            "type": "object",
            "properties": {
                "case_ids": {"type": "array", "items": {"type": "string"}},
                "min_count": {"type": "integer", "default": 2},
            },
            "required": [],
        },
    },
    {
        "name": "list_tool_files",
        "description": "List managed self-evolved tool files under tools/.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_tool_contract",
        "description": (
            "Read the managed tool API contract, including payload input, return format, "
            "extraction schema, and tool_registry schema. Use this before creating or modifying tools."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_tool_file",
        "description": "Read a managed self-evolved tool file.",
        "parameters": {
            "type": "object",
            "properties": {"filename": {"type": "string"}},
            "required": ["filename"],
        },
    },
    {
        "name": "write_tool_file",
        "description": "Create or overwrite a managed tool file under tools/.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
                "content_lines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Full file content as a list of lines; preferred for multi-line Python code.",
                },
            },
            "required": ["filename"],
        },
    },
    {
        "name": "read_tool_registry",
        "description": "Read the registry describing active managed tools.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "update_tool_registry",
        "description": (
            "Incrementally merge managed-tool registrations by tool name. Tools omitted "
            "from patch.tools remain registered; a same-name entry is updated and a new "
            "name is appended. Read the registry after updating to verify the active set."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {"type": "object"},
            },
            "required": ["patch"],
        },
    },
    {
        "name": "run_managed_tool",
        "description": "Run an active managed tool from tools/ with a JSON payload.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "payload": {"type": "object"},
            },
            "required": ["name", "payload"],
        },
    },
]


def tools_prompt_block() -> str:
    for spec in TOOL_SPECS:
        if spec.get("name") == "update_manifest":
            spec["description"] = (
                "Merge safe updates into manifest.json. Only description, workflow, "
                "prompt_files, and notes are editable. workflow must be one of: "
                "unified, split_attributes, full_split, split_relations. Managed "
                "preprocess/postprocess tools are enabled through tool_registry.json, "
                "not by adding manifest keys such as preprocess_tool or postprocess_tool. "
                "attribute_field_groups is locked."
            )
    return json.dumps(TOOL_SPECS, ensure_ascii=False, indent=2)


class EvolutionToolRunner:
    def __init__(self, store: PromptStore):
        self.store = store
        self.tool_store = ToolStore()
        self.observation_context: dict[str, Any] = {}
        self._handlers = {
            "list_prompt_files": self._list_prompt_files,
            "read_prompt": self._read_prompt,
            "write_prompt": self._write_prompt,
            "read_manifest": self._read_manifest,
            "update_manifest": self._update_manifest,
            "sample_interesting_cases": self._sample_interesting_cases,
            "get_case_failure_detail": self._get_case_failure_detail,
            "read_readme_excerpt": self._read_readme_excerpt,
            "profile_markdown_structure": self._profile_markdown_structure,
            "find_repeated_markdown_patterns": self._find_repeated_markdown_patterns,
            "list_tool_files": self._list_tool_files_managed,
            "read_tool_contract": self._read_tool_contract,
            "read_tool_file": self._read_tool_file,
            "write_tool_file": self._write_tool_file,
            "read_tool_registry": self._read_tool_registry,
            "update_tool_registry": self._update_tool_registry,
            "run_managed_tool": self._run_managed_tool,
        }

    def set_observation_context(self, report: dict[str, Any] | None) -> None:
        self.observation_context = report or {}

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self._handlers:
            return {"ok": False, "error": f"未知工具: {name}"}
        try:
            result = self._handlers[name](arguments or {})
            return {"ok": True, "result": result}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _list_prompt_files(self, _: dict) -> list[str]:
        return self.store.list_prompt_files()

    def _read_prompt(self, args: dict) -> str:
        return self.store.read_prompt(str(args["filename"]))

    def _write_prompt(self, args: dict) -> dict[str, str]:
        path = self.store.write_prompt(str(args["filename"]), _content_arg(args))
        return {"path": str(path)}

    def _read_manifest(self, _: dict) -> dict:
        return self.store.load_manifest()

    def _update_manifest(self, args: dict) -> dict:
        patch = args.get("patch") or {}
        if not isinstance(patch, dict):
            raise TypeError("patch 必须是 JSON 对象")
        forbidden = sorted(set(patch) - ALLOWED_MANIFEST_PATCH_KEYS)
        if forbidden:
            raise ValueError(
                "unsupported manifest patch keys: "
                + ", ".join(forbidden)
                + ". Managed tools are controlled through tool_registry.json; "
                + "do not add preprocess_tool/postprocess_tool or new workflow keys."
            )
        if "workflow" in patch:
            workflow = str(patch.get("workflow") or "").strip()
            if workflow not in ALLOWED_MANIFEST_WORKFLOWS:
                raise ValueError(
                    f"unsupported workflow {workflow!r}; allowed workflows are "
                    f"{sorted(ALLOWED_MANIFEST_WORKFLOWS)}"
                )
            patch = dict(patch)
            patch["workflow"] = workflow
        if "prompt_files" in patch and not isinstance(patch["prompt_files"], dict):
            raise TypeError("manifest prompt_files must be a JSON object")
        manifest = self.store.load_manifest()
        for k, v in patch.items():
            manifest[k] = v
        self.store.save_manifest(manifest)
        return manifest

    def _readmes_by_case_id(self) -> dict[str, str]:
        rows = self.observation_context.get("case_inputs") or []
        return {
            str(row.get("resource_id")): str(row.get("readme_content") or "")
            for row in rows
            if row.get("resource_id")
        }

    def _extractions_by_case_id(self) -> dict[str, dict[str, Any]]:
        rows = self.observation_context.get("extractions") or []
        return {
            str(row.get("resource_id")): {
                key: value
                for key, value in (row.get("extraction") or {}).items()
                if key != "relations"
            }
            for row in rows
            if row.get("resource_id")
        }

    def _observations_by_case_id(self) -> dict[str, dict[str, Any]]:
        rows = self.observation_context.get("case_observations") or []
        return {
            str(row.get("resource_id")): row
            for row in rows
            if row.get("resource_id")
        }

    def _sample_interesting_cases(self, args: dict) -> list[dict[str, Any]]:
        limit = max(1, min(int(args.get("limit") or 5), 20))
        rows = self.observation_context.get("interesting_cases") or []
        return list(rows)[:limit]

    def _get_case_failure_detail(self, args: dict) -> dict[str, Any]:
        case_id = str(args["case_id"])
        mismatches = [
            m
            for m in (self.observation_context.get("mismatches") or [])
            if str(m.get("resource_id") or m.get("case_id") or "") == case_id
        ][:10]
        mismatch_attrs = {
            str(m.get("attribute") or "").strip()
            for m in mismatches
            if str(m.get("attribute") or "").strip()
        }
        extraction = self._extractions_by_case_id().get(case_id) or {}
        attrs = extraction.get("attributes") or []
        compact_attrs = []
        for row in attrs:
            if not isinstance(row, dict):
                continue
            attr = str(row.get("attribute") or "").strip()
            if mismatch_attrs and attr not in mismatch_attrs:
                continue
            compact_attrs.append(
                {
                    "attribute": attr,
                    "value": row.get("value"),
                    "evidence_span": row.get("evidence_span") or row.get("evidence"),
                    "normalization_note": row.get("normalization_note"),
                }
            )
        return {
            "case_id": case_id,
            "observation": self._observations_by_case_id().get(case_id),
            "extraction": {
                "attribute_count": len(attrs),
                "mismatch_related_attributes": compact_attrs[:12],
                "_meta": extraction.get("_meta"),
            },
            "mismatches": mismatches,
        }

    def _read_readme_excerpt(self, args: dict) -> dict[str, Any]:
        case_id = str(args["case_id"])
        readme = self._readmes_by_case_id().get(case_id)
        if readme is None:
            raise KeyError(f"case_id not found in current observation context: {case_id}")
        mode = str(args.get("mode") or "head")
        max_chars = int(args.get("max_chars") or 4000)
        return {
            "case_id": case_id,
            "mode": mode,
            "chars": len(readme),
            "excerpt": read_excerpt(readme, mode=mode, max_chars=max_chars),
        }

    def _profile_markdown_structure(self, args: dict) -> dict[str, Any]:
        case_id = str(args["case_id"])
        readme = self._readmes_by_case_id().get(case_id)
        if readme is None:
            raise KeyError(f"case_id not found in current observation context: {case_id}")
        return {"case_id": case_id, **profile_markdown_text(readme)}

    def _find_repeated_markdown_patterns(self, args: dict) -> dict[str, Any]:
        case_ids = args.get("case_ids")
        if not case_ids:
            case_ids = [
                str(row.get("resource_id"))
                for row in (self.observation_context.get("interesting_cases") or [])
                if row.get("resource_id")
            ]
        return repeated_markdown_patterns(
            self._readmes_by_case_id(),
            case_ids=[str(x) for x in (case_ids or [])],
            min_count=int(args.get("min_count") or 2),
        )

    def _list_tool_files_managed(self, _: dict) -> list[str]:
        return self.tool_store.list_tool_files()

    def _read_tool_contract(self, _: dict) -> str:
        return self.tool_store.read_tool_contract()

    def _read_tool_file(self, args: dict) -> str:
        return self.tool_store.read_tool_file(str(args["filename"]))

    def _write_tool_file(self, args: dict) -> dict[str, str]:
        path = self.tool_store.write_tool_file(
            str(args["filename"]),
            _content_arg(args),
        )
        return {"path": str(path)}

    def _read_tool_registry(self, _: dict) -> dict[str, Any]:
        return self.tool_store.load_registry()

    def _update_tool_registry(self, args: dict) -> dict[str, Any]:
        patch = args.get("patch") or {}
        if not isinstance(patch, dict):
            raise TypeError("patch must be a JSON object")
        previous_registry = self.tool_store.load_registry()
        registry = _merge_tool_registry_patch(previous_registry, patch)
        saved, rejected = self.tool_store.save_registry_pruning_invalid(
            registry,
            previous_registry=previous_registry,
        )
        return {
            "registry": saved,
            "rejected_tool_registrations": rejected,
        }

    def _run_managed_tool(self, args: dict) -> dict[str, Any]:
        name = str(args["name"])
        payload = args.get("payload") or {}
        if not isinstance(payload, dict):
            raise TypeError("payload must be a JSON object")
        return self.tool_store.run_tool(name, payload)
