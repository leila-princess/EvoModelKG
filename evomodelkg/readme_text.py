from __future__ import annotations


def strip_readme_front_matter(readme: str) -> str:
    """
    去掉 README 顶部 Hugging Face 卡片 YAML front matter（--- ... ---）。
    该段与 HubStats/结构化字段重复，不应再送入 LLM 抽取。
    """
    text = (readme or "").replace("\r\n", "\n")
    if not text:
        return ""
    lines = text.split("\n")
    if not lines:
        return ""
    first = lines[0].lstrip("\ufeff").strip()
    if first != "---":
        return text
    end_idx = None
    for i in range(1, len(lines)):
        marker = lines[i].strip()
        if marker in {"---", "..."}:
            end_idx = i
            break
    if end_idx is None:
        return text
    return "\n".join(lines[end_idx + 1 :]).lstrip()
