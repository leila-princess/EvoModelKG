#!/usr/bin/env python3
"""Standard-library-only integrity check for the public EvoModelKG release."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = (
    "run_evolution.py",
    "requirements.txt",
    "evomodelkg/__init__.py",
    "evomodelkg/evolve_agent.py",
    "evomodelkg/prompt_store.py",
    "evomodelkg/tool_store.py",
    "evomodelkg/tools.py",
    "prompts/manifest.json",
    "tools/TOOL_CONTRACT.md",
    "tools/tool_registry.json",
    "data/split_protocol.json",
    "data/splits/candidate_pool_15000.json",
    "data/splits/validation_ids.json",
    "data/splits/heldout_test_ids.json",
    "data/splits/evolution_pool_ids.json",
    "data/splits/evolution_generation_ids.json",
)

META_TOOLS = {
    "list_prompt_files",
    "read_prompt",
    "write_prompt",
    "read_tool_file",
    "write_tool_file",
    "read_tool_registry",
    "update_tool_registry",
    "run_managed_tool",
}

PROMPT_PLACEHOLDERS = {
    "{readme_content}",
    "{target_fields}",
    "{resource_id}",
    "{resource_type}",
}


def main() -> int:
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for path in sorted(ROOT.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        except Exception as exc:
            errors.append(
                f"invalid Python syntax: {path.relative_to(ROOT)}: {exc}"
            )

    tools_path = ROOT / "evomodelkg" / "tools.py"
    if tools_path.is_file():
        tools_source = tools_path.read_text(encoding="utf-8-sig")
        missing = sorted(
            name for name in META_TOOLS if f'"name": "{name}"' not in tools_source
        )
        if missing:
            errors.append(f"missing evolution meta-tools: {missing}")

    manifest_path = ROOT / "prompts" / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        for filename in (manifest.get("prompt_files") or {}).values():
            prompt_path = ROOT / "prompts" / str(filename)
            if not prompt_path.is_file():
                errors.append(f"manifest references missing prompt: {filename}")
                continue
            prompt = prompt_path.read_text(encoding="utf-8-sig")
            missing = sorted(token for token in PROMPT_PLACEHOLDERS if token not in prompt)
            if missing:
                errors.append(f"prompt {filename} is missing placeholders: {missing}")

    registry_path = ROOT / "tools" / "tool_registry.json"
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
        for entry in registry.get("tools", []):
            if not entry.get("enabled", True):
                continue
            filename = entry.get("filename") or entry.get("file")
            if not filename or not (ROOT / "tools" / str(filename)).is_file():
                errors.append(
                    f"enabled tool has no source file: {entry.get('name')}"
                )

    if errors:
        print("EvoModelKG public-release preflight: FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("EvoModelKG public-release preflight: OK")
    print(f"- fixed evolution meta-tools: {len(META_TOOLS)}")
    print("- Python syntax: OK")
    print("- prompts and required placeholders: OK")
    print("- enabled managed-tool files: OK")
    print("- published split files: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
