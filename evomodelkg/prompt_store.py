from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evomodelkg.attribute_schema import locked_attribute_field_groups


REQUIRED_EXTRACTION_PLACEHOLDERS = (
    "{readme_content}",
    "{target_fields}",
    "{resource_id}",
    "{resource_type}",
)


class PromptStore:
    """管理 prompts 目录下的 manifest 与模板文件。"""

    MANIFEST_NAME = "manifest.json"

    def __init__(self, prompts_dir: Path):
        self.prompts_dir = Path(prompts_dir).resolve()
        if not self.prompts_dir.is_dir():
            raise FileNotFoundError(f"prompts 目录不存在: {self.prompts_dir}")

    @property
    def manifest_path(self) -> Path:
        return self.prompts_dir / self.MANIFEST_NAME

    def load_manifest(self) -> dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8-sig") as f:
            manifest = json.load(f)
        # Attribute batching is part of the fixed experimental protocol.  It is
        # intentionally not evolvable, so historical/local manifest mutations
        # cannot silently change the scoring and LLM-call basis between runs.
        manifest["attribute_field_groups"] = locked_attribute_field_groups()
        return manifest

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        manifest = dict(manifest)
        manifest["attribute_field_groups"] = locked_attribute_field_groups()
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    def list_prompt_files(self) -> list[str]:
        return sorted(
            p.name
            for p in self.prompts_dir.iterdir()
            if p.is_file() and p.suffix in {".txt", ".md"}
        )

    def read_prompt(self, filename: str) -> str:
        path = self._resolve_prompt_path(filename)
        return path.read_text(encoding="utf-8")

    def write_prompt(self, filename: str, content: str) -> Path:
        if not filename.endswith((".txt", ".md")):
            filename = f"{filename}.txt"
        self.validate_prompt_template(filename, content)
        path = self.prompts_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def validate_prompt_template(self, filename: str, content: str) -> None:
        missing = [p for p in REQUIRED_EXTRACTION_PLACEHOLDERS if p not in content]
        if missing:
            raise ValueError(
                f"refuse to write prompt {filename}: missing required template "
                f"placeholders {missing}. Keep README extraction inputs wired by "
                "preserving {readme_content}, {target_fields}, {resource_id}, "
                "and {resource_type}."
            )

    def _resolve_prompt_path(self, filename: str) -> Path:
        path = (self.prompts_dir / filename).resolve()
        if not str(path).startswith(str(self.prompts_dir)):
            raise ValueError("禁止访问 prompts 目录外的路径")
        if not path.is_file():
            raise FileNotFoundError(f"提示词文件不存在: {filename}")
        return path

    def snapshot_to(self, dest_dir: Path) -> Path:
        """复制当前 prompts 目录到 run 子目录（版本快照）。"""
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        for item in self.prompts_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
        return dest

    def restore_from(self, src_dir: Path) -> None:
        """从快照目录覆盖当前 prompts。"""
        src = Path(src_dir)
        if not src.is_dir():
            raise FileNotFoundError(src)
        for item in self.prompts_dir.iterdir():
            if item.is_file():
                item.unlink()
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, self.prompts_dir / item.name)

    # 仅替换这些占位符；模板/示例中的 JSON 花括号与 README 正文里的 { } 不再走 str.format。
    _TEMPLATE_KEYS = (
        "resource_type",
        "resource_id",
        "readme_content",
        "relation_whitelist",
        "target_fields",
    )

    def render_template(
        self,
        filename: str,
        *,
        variables: dict[str, Any],
    ) -> str:
        text = self.read_prompt(filename)
        safe = {k: str(v) for k, v in variables.items()}
        missing = [k for k in self._TEMPLATE_KEYS if k in text and k not in safe]
        if missing:
            raise KeyError(f"模板 {filename} 缺少变量: {missing}")

        # readme_content 最后替换，避免正文中的字面量干扰其它占位符
        for key in self._TEMPLATE_KEYS:
            if key == "readme_content":
                continue
            placeholder = "{" + key + "}"
            if placeholder in text:
                text = text.replace(placeholder, safe[key])
        if "{readme_content}" in text:
            text = text.replace("{readme_content}", safe.get("readme_content", ""))
        return text
