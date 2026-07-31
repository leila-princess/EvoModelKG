from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

ReadmeFilenameStyle = Literal["auto", "underscore", "double_underscore"]


def _candidate_paths(readme_dir: Path, model_id: str, style: ReadmeFilenameStyle) -> list[Path]:
    base = readme_dir
    candidates: list[Path] = []
    if style in {"auto", "underscore"}:
        candidates.append(base / f"{model_id.replace('/', '_')}.md")
    if style in {"auto", "double_underscore"}:
        candidates.append(base / f"{model_id.replace('/', '__')}.md")
        candidates.append(base / f"{model_id.replace('/', '__')}.txt")
    # 去重且保持顺序
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered


def resolve_readme_content(
    model_id: str,
    row: dict[str, Any],
    *,
    readme_field: str = "readme_content",
    readme_dir: Path | None = None,
    readme_filename_style: ReadmeFilenameStyle = "auto",
) -> str:
    """
    解析评测用 README 正文。

    优先级：
    1) models.json 行内字段（默认 readme_content，可用 readme_field 覆盖）
    2) readme_dir 下本地文件（与 model_crawler.crawl_readme 命名习惯兼容）
    """
    inline = (row.get(readme_field) or row.get("readme") or "").strip()
    if inline:
        return inline

    if readme_dir is None:
        return ""

    readme_dir = Path(readme_dir)
    if not readme_dir.is_dir():
        return ""

    for path in _candidate_paths(readme_dir, model_id, readme_filename_style):
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
    return ""
