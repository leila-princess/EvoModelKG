from __future__ import annotations

import json
import shutil
import importlib.util
import ast
import re
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_TOOL_CONTRACT = """# Managed Tool Contract

Read this before creating or modifying managed tools.

## Payload Input Contract

Preprocess tools (`stage="preprocess_readme"`) receive:

```json
{
  "stage": "preprocess_readme",
  "resource_type": "model",
  "resource_id": "org/model-name",
  "readme_content": "README markdown after YAML front matter has been stripped",
  "readme": "alias of readme_content"
}
```

Postprocess tools (`stage="postprocess_extraction"`) receive:

```json
{
  "stage": "postprocess_extraction",
  "resource_type": "model",
  "resource_id": "org/model-name",
  "readme_content": "README markdown after YAML front matter has been stripped",
  "readme": "alias of readme_content",
  "extraction": {"attributes": [], "relations": [], "summary": "..."},
  "extraction_result": {"attributes": [], "relations": [], "summary": "..."}
}
```

Read postprocess input with:

```python
extraction = payload.get("extraction") or payload.get("extraction_result") or {}
```

## Return Format Contract

Preprocess tools return:

```json
{"readme_content": "cleaned markdown"}
```

Postprocess tools return one of:

```json
{"extraction": {"attributes": [], "relations": [], "summary": "..."}}
```

```json
{"extraction_result": {"attributes": [], "relations": [], "summary": "..."}}
```

```json
{"attributes": [], "relations": []}
```

Do not return an empty extraction when the input extraction is non-empty unless the registry entry sets `"allow_empty_output": true`.

## Extraction Schema

Attributes:

```json
{
  "entity_id": "org/model-name",
  "entity_type": "model",
  "attribute": "license",
  "value": "apache-2.0",
  "evidence_span": "README evidence",
  "confidence": 0.8,
  "source": "readme",
  "normalization_note": "optional"
}
```

Relations:

```json
{
  "source_id": "org/model-name",
  "target_id": "dataset-or-paper-id",
  "relation_type": "TRAINED_ON",
  "evidence_span": "README evidence",
  "confidence": 0.8,
  "source": "readme",
  "properties": {}
}
```

Preserve unknown existing fields unless there is a clear reason to remove them.

## Registry Schema

```json
{
  "version": 1,
  "tools": [
    {
      "name": "normalize_attributes",
      "filename": "normalize_attributes.py",
      "stage": "postprocess_extraction",
      "enabled": true,
      "description": "...",
      "purpose": "...",
      "evidence_cases": ["org/model-a"],
      "expected_behavior": "...",
      "safety_constraints": "...",
      "validation_criteria": "...",
      "allow_empty_output": false
    }
  ]
}
```

Required tool fields: `name`, `filename`, `stage`, `enabled`.
Allowed stages: `preprocess_readme`, `postprocess_extraction`.
`file` is accepted as an alias for `filename`, but `filename` is preferred.
"""


class ToolStore:
    """Managed workspace for self-evolved tools."""

    REGISTRY_NAME = "tool_registry.json"
    CONTRACT_NAME = "TOOL_CONTRACT.md"
    ALLOWED_STAGES = {"preprocess_readme", "postprocess_extraction"}

    def __init__(self, tools_dir: Path | None = None):
        self.tools_dir = (
            Path(tools_dir)
            if tools_dir is not None
            else Path(__file__).resolve().parents[1] / "tools"
        ).resolve()
        self._module_cache: dict[tuple[str, int], ModuleType] = {}
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.is_file():
            self.save_registry({"version": 1, "tools": [], "updated_at": None})
        self._ensure_contract_file()

    @property
    def registry_path(self) -> Path:
        return self.tools_dir / self.REGISTRY_NAME

    def list_tool_files(self) -> list[str]:
        return sorted(
            p.name
            for p in self.tools_dir.iterdir()
            if p.is_file() and p.suffix in {".py", ".json", ".md", ".txt"}
        )

    def load_registry(self) -> dict[str, Any]:
        with self.registry_path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)

    def save_registry(self, registry: dict[str, Any]) -> None:
        registry = self._normalize_registry(registry)
        registry["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self.registry_path.open("w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    def validate_enabled_tool_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Validate the file behind one enabled registry entry before activation."""
        valid = self._validate_tool_entry(entry)
        if not valid.get("enabled", True):
            return valid
        filename = str(valid["filename"])
        path = self._resolve_tool_path(filename)
        content = path.read_text(encoding="utf-8")
        self._validate_python_tool_source(filename, content)
        module = self._load_module(filename)
        legal_entrypoints = (
            "run",
            "preprocess_readme",
            "postprocess_extraction",
            "validate_attributes",
        )
        if not any(callable(getattr(module, name, None)) for name in legal_entrypoints):
            raise ValueError(
                f"managed tool {filename} has no callable managed-tool entrypoint"
            )
        return valid

    def save_registry_pruning_invalid(
        self,
        registry: dict[str, Any],
        *,
        previous_registry: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Save valid registry entries without rolling back unrelated tools.

        A bad new entry is omitted. If an invalid update targets an existing tool,
        its previously valid entry is retained. Disabled entries need no backing
        file because they cannot be executed.
        """
        candidate = self._normalize_registry(registry)
        previous = self._normalize_registry(previous_registry or {"version": 1, "tools": []})
        previous_by_name = {
            str(row["name"]): row
            for row in previous.get("tools", [])
            if isinstance(row, dict) and row.get("name")
        }
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        for entry in candidate.get("tools", []):
            name = str(entry.get("name") or "")
            try:
                accepted.append(self.validate_enabled_tool_entry(entry))
                continue
            except Exception as error:
                rejection = {
                    "name": name,
                    "filename": entry.get("filename"),
                    "error": str(error),
                    "action": "removed_invalid_registration",
                }

            prior = previous_by_name.get(name)
            if prior is not None and prior != entry:
                try:
                    accepted.append(self.validate_enabled_tool_entry(prior))
                    rejection["action"] = "restored_previous_registration"
                except Exception as prior_error:
                    rejection["previous_error"] = str(prior_error)
            rejected.append(rejection)

        sanitized = {
            "version": candidate.get("version", 1),
            "tools": accepted,
        }
        self.save_registry(sanitized)
        return self.load_registry(), rejected

    @staticmethod
    def _default_tool_filename(name: str) -> str:
        return name if name.endswith((".py", ".json", ".md", ".txt")) else f"{name}.py"

    @classmethod
    def _normalize_tool_entry(cls, name: str, entry: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(entry)
        normalized.setdefault("name", name)
        if "filename" not in normalized and "file" not in normalized:
            normalized["filename"] = cls._default_tool_filename(name)
        filename = normalized.get("filename") or normalized.get("file")
        if filename is not None:
            normalized["filename"] = str(filename)
            normalized.pop("file", None)
        return normalized

    @classmethod
    def _validate_tool_entry(cls, entry: dict[str, Any]) -> dict[str, Any]:
        name = str(entry.get("name") or "").strip()
        filename = str(entry.get("filename") or "").strip()
        stage = str(entry.get("stage") or "").strip()
        if not name:
            raise ValueError("tool registry entry missing required field: name")
        if not filename:
            raise ValueError(f"tool registry entry {name!r} missing required field: filename")
        if stage not in cls.ALLOWED_STAGES:
            raise ValueError(
                f"tool registry entry {name!r} has unsupported stage {stage!r}; "
                f"allowed stages are {sorted(cls.ALLOWED_STAGES)}"
            )
        if not filename.endswith(".py"):
            raise ValueError(f"tool registry entry {name!r} filename must be a .py file")
        out = dict(entry)
        out["name"] = name
        out["filename"] = filename
        out["stage"] = stage
        out["enabled"] = bool(out.get("enabled", True))
        return out

    @classmethod
    def _normalize_registry(cls, registry: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "version": registry.get("version", 1),
            "tools": [],
        }
        tools = registry.get("tools")
        collected: list[dict[str, Any]] = []
        if isinstance(tools, dict):
            for name, value in tools.items():
                if isinstance(value, dict):
                    collected.append(cls._normalize_tool_entry(str(name), value))
        elif isinstance(tools, list):
            for row in tools:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or row.get("filename") or row.get("file") or "").strip()
                if not name:
                    continue
                if name.endswith((".py", ".json", ".md", ".txt")):
                    name = Path(name).stem
                collected.append(cls._normalize_tool_entry(name, row))
        for key, value in registry.items():
            if key in {"version", "tools", "updated_at"}:
                continue
            if isinstance(value, dict):
                collected.append(cls._normalize_tool_entry(str(key), value))
        deduped: dict[str, dict[str, Any]] = {}
        for entry in collected:
            valid = cls._validate_tool_entry(entry)
            deduped[valid["name"]] = valid
        normalized["tools"] = list(deduped.values())
        return normalized

    def tool_entries(self) -> dict[str, dict[str, Any]]:
        registry = self._normalize_registry(self.load_registry())
        entries: dict[str, dict[str, Any]] = {}
        tools = registry.get("tools")
        if isinstance(tools, list):
            for row in tools:
                if isinstance(row, dict) and row.get("name"):
                    entries[str(row["name"])] = dict(row)
        return entries

    def active_tools(self, stage: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name, entry in self.tool_entries().items():
            if entry.get("enabled", True) is False:
                continue
            entry = dict(entry)
            entry.setdefault("name", name)
            if stage is None or entry.get("stage") == stage or (
                stage == "postprocess_extraction" and "stage" not in entry
            ):
                out.append(entry)
        return out

    def read_tool_file(self, filename: str) -> str:
        path = self._resolve_tool_path(filename)
        return path.read_text(encoding="utf-8")

    def read_tool_contract(self) -> str:
        self._ensure_contract_file()
        return self.read_tool_file(self.CONTRACT_NAME)

    def write_tool_file(self, filename: str, content: str) -> Path:
        if not filename.endswith((".py", ".json", ".md", ".txt")):
            filename = f"{filename}.py"
        path = (self.tools_dir / filename).resolve()
        if not str(path).startswith(str(self.tools_dir)):
            raise ValueError("Refusing to write outside generated_tools")
        if filename.endswith(".py"):
            self._validate_python_tool_source(filename, content)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _validate_python_tool_source(filename: str, content: str) -> None:
        try:
            tree = ast.parse(content, filename=filename)
        except SyntaxError as e:
            raise ValueError(f"managed tool {filename} is not valid Python: {e}") from e
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        if not (
            "run" in function_names
            or "preprocess_readme" in function_names
            or "postprocess_extraction" in function_names
            or "validate_attributes" in function_names
        ):
            raise ValueError(
                f"managed tool {filename} must define run(payload), "
                "preprocess_readme(...), postprocess_extraction(...), or validate_attributes(...)"
            )
        if re.search(r"\.startswith\(\s*(['\"])\1\s*\)", content):
            raise ValueError(
                f"managed tool {filename} contains startswith(''), which is always true"
            )
        if (
            "postprocess_extraction" in function_names
            and '.get("name"' in content
            and '.get("attribute"' not in content
        ):
            raise ValueError(
                f"postprocess tool {filename} appears to read attr.get('name') instead of "
                "the extraction schema field attr.get('attribute')"
            )

    def run_tool(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        entries = self.tool_entries()
        entry = entries.get(name)
        if not entry:
            raise KeyError(f"managed tool not found in registry: {name}")
        filename = entry.get("file") or entry.get("filename")
        if not filename:
            raise ValueError(f"managed tool {name} has no file field")
        module = self._load_module(str(filename))
        payload = dict(payload or {})
        # Backward/LLM-friendly aliases. Several generated tools naturally used
        # "extraction_result" for post-processing payloads; keep both names in
        # sync so a small schema wording mistake does not disable a tool.
        if "extraction" in payload and "extraction_result" not in payload:
            payload["extraction_result"] = payload["extraction"]
        if "extraction_result" in payload and "extraction" not in payload:
            payload["extraction"] = payload["extraction_result"]
        if "readme_content" in payload and "readme" not in payload:
            payload["readme"] = payload["readme_content"]
        if hasattr(module, "run"):
            result = module.run(payload)
        elif hasattr(module, "preprocess_readme"):
            result = module.preprocess_readme(
                payload.get("readme_content") or "",
                payload.get("resource_id") or "",
                payload,
            )
        elif hasattr(module, "postprocess_extraction"):
            result = module.postprocess_extraction(
                payload.get("extraction") or {},
                payload.get("resource_id") or "",
                payload.get("readme_content") or "",
                payload,
            )
        elif hasattr(module, "validate_attributes"):
            result = module.validate_attributes(
                payload.get("extraction") or {},
                payload.get("resource_id") or "",
                payload.get("readme_content") or "",
            )
        else:
            raise AttributeError(
                f"managed tool {name} must define run(payload), preprocess_readme(...), "
                "postprocess_extraction(...), or validate_attributes(...)"
            )
        normalized = self._normalize_result(result)
        if entry.get("stage") == "postprocess_extraction":
            normalized = self._guard_postprocess_result(entry, payload, normalized)
        return normalized

    @staticmethod
    def _extract_attr_count(extraction: Any) -> int:
        if not isinstance(extraction, dict):
            return 0
        attrs = extraction.get("attributes")
        return len(attrs) if isinstance(attrs, list) else 0

    def _guard_postprocess_result(
        self,
        entry: dict[str, Any],
        payload: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        if entry.get("allow_empty_output") is True:
            return result
        before = payload.get("extraction") or payload.get("extraction_result") or {}
        after = result.get("extraction") or result.get("extraction_result") or result
        if self._extract_attr_count(before) > 0 and self._extract_attr_count(after) == 0:
            raise ValueError(
                f"managed postprocess tool {entry.get('name')} returned an empty extraction "
                "from a non-empty input; set allow_empty_output=true only if this is intended"
            )
        return result

    def _load_module(self, filename: str):
        path = self._resolve_tool_path(filename)
        mtime_ns = path.stat().st_mtime_ns
        cache_key = (str(path), mtime_ns)
        cached = self._module_cache.get(cache_key)
        if cached is not None:
            return cached
        module_name = f"evomodelkg_generated_{path.stem}_{abs(hash(cache_key))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load managed tool: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._module_cache = {
            key: value for key, value in self._module_cache.items() if key[0] != str(path)
        }
        self._module_cache[cache_key] = module
        return module

    @staticmethod
    def _normalize_result(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            return {"readme_content": result}
        return {"result": result}

    def _resolve_tool_path(self, filename: str) -> Path:
        path = (self.tools_dir / filename).resolve()
        if not str(path).startswith(str(self.tools_dir)):
            raise ValueError("Refusing to access outside generated_tools")
        if not path.is_file():
            raise FileNotFoundError(filename)
        return path

    def _ensure_contract_file(self) -> None:
        contract_path = self.tools_dir / self.CONTRACT_NAME
        if contract_path.is_file():
            return
        source_path = Path(__file__).resolve().parents[1] / "tools" / self.CONTRACT_NAME
        if source_path.is_file() and source_path.resolve() != contract_path.resolve():
            contract_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            contract_path.write_text(DEFAULT_TOOL_CONTRACT, encoding="utf-8")

    def snapshot_to(self, dest_dir: Path) -> Path:
        dest = Path(dest_dir)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.tools_dir, dest)
        return dest

    def restore_from(self, src_dir: Path) -> None:
        src = Path(src_dir)
        if not src.is_dir():
            raise FileNotFoundError(src)
        if self.tools_dir.exists():
            shutil.rmtree(self.tools_dir)
        shutil.copytree(src, self.tools_dir)
