from __future__ import annotations

import re
from collections import Counter
from typing import Any

from evomodelkg.benchmark import BenchmarkCase

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.M)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.S)
_URL_RE = re.compile(r"https?://\S+")


def _line_norm(line: str) -> str:
    line = line.strip().lower()
    line = re.sub(r"https?://\S+", "<url>", line)
    line = re.sub(r"\d+(?:\.\d+)?", "<num>", line)
    line = re.sub(r"\s+", " ", line)
    return line


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def profile_markdown_text(text: str) -> dict[str, Any]:
    headings: list[dict[str, Any]] = []
    matches = list(_HEADING_RE.finditer(text))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        headings.append(
            {
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "body_chars": len(body.strip()),
            }
        )

    lines = text.splitlines()
    return {
        "chars": len(text),
        "estimated_tokens": estimate_tokens(text),
        "line_count": len(lines),
        "heading_count": len(headings),
        "headings": headings[:80],
        "html_comment_count": len(_HTML_COMMENT_RE.findall(text)),
        "html_comment_chars": sum(len(x) for x in _HTML_COMMENT_RE.findall(text)),
        "code_block_count": len(_CODE_BLOCK_RE.findall(text)),
        "url_count": len(_URL_RE.findall(text)),
        "table_line_count": sum(1 for line in lines if line.strip().startswith("|")),
        "numeric_token_count": len(re.findall(r"\b\d+(?:\.\d+)?\b", text)),
    }


def read_excerpt(text: str, *, mode: str = "head", max_chars: int = 4000) -> str:
    max_chars = max(200, min(int(max_chars), 20000))
    mode = (mode or "head").lower()
    if mode == "tail":
        return text[-max_chars:]
    if mode == "middle":
        start = max(0, (len(text) - max_chars) // 2)
        return text[start : start + max_chars]
    if mode == "without_code":
        return _CODE_BLOCK_RE.sub("\n[CODE_BLOCK_REMOVED]\n", text)[:max_chars]
    return text[:max_chars]


def summarize_case_for_observation(
    case: BenchmarkCase,
    extraction: dict[str, Any],
) -> dict[str, Any]:
    attrs = extraction.get("attributes") or []
    profile = profile_markdown_text(case.readme_content)
    attr_count = len(attrs) if isinstance(attrs, list) else 0
    reasons: list[str] = []
    if profile["estimated_tokens"] >= 1200 and attr_count <= 2:
        reasons.append("high_input_tokens_low_extraction")
    if profile["chars"] >= 3000 and attr_count == 0:
        reasons.append("long_readme_zero_attributes")
    if profile["html_comment_count"] >= 3:
        reasons.append("many_html_comments")
    if profile["heading_count"] >= 20 and profile["chars"] >= 4000:
        reasons.append("many_markdown_sections")

    return {
        "resource_id": case.resource_id,
        "input_chars": profile["chars"],
        "estimated_input_tokens": profile["estimated_tokens"],
        "attribute_count": attr_count,
        "heading_count": profile["heading_count"],
        "html_comment_count": profile["html_comment_count"],
        "code_block_count": profile["code_block_count"],
        "url_count": profile["url_count"],
        "reasons": reasons,
    }


def select_interesting_cases(
    cases: list[BenchmarkCase],
    extractions: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows = [summarize_case_for_observation(c, e) for c, e in zip(cases, extractions)]

    def score(row: dict[str, Any]) -> tuple[int, int, int]:
        return (
            len(row.get("reasons") or []),
            int(row.get("estimated_input_tokens") or 0),
            -int(row.get("attribute_count") or 0),
        )

    return sorted(
        [row for row in rows if row.get("reasons")],
        key=score,
        reverse=True,
    )[:limit]


def repeated_markdown_patterns(
    readmes: dict[str, str],
    *,
    case_ids: list[str] | None = None,
    min_count: int = 2,
) -> dict[str, Any]:
    selected = case_ids or list(readmes.keys())
    heading_counter: Counter[str] = Counter()
    comment_counter: Counter[str] = Counter()
    line_counter: Counter[str] = Counter()
    used: list[str] = []
    for case_id in selected:
        text = readmes.get(case_id)
        if not text:
            continue
        used.append(case_id)
        heading_counter.update(m.group(2).strip() for m in _HEADING_RE.finditer(text))
        comment_counter.update(
            re.sub(r"\s+", " ", m.group(0)).strip()[:180]
            for m in _HTML_COMMENT_RE.finditer(text)
        )
        line_counter.update(
            _line_norm(line)
            for line in text.splitlines()
            if 25 <= len(line.strip()) <= 180 and not line.strip().startswith("---")
        )
    min_count = max(2, int(min_count))
    return {
        "case_ids": used,
        "common_headings": [
            {"text": text, "count": count}
            for text, count in heading_counter.most_common(30)
            if count >= min_count
        ],
        "common_html_comments": [
            {"text": text, "count": count}
            for text, count in comment_counter.most_common(20)
            if count >= min_count
        ],
        "common_lines": [
            {"text": text, "count": count}
            for text, count in line_counter.most_common(30)
            if count >= min_count
        ],
    }
