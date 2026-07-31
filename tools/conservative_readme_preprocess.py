import re


_EVIDENCE_KEYWORDS = re.compile(
    r"license|citation|bibtex|arxiv|doi|model|base[_ -]?model|dataset|"
    r"architecture|pipeline|train|evaluat|benchmark|parameter|safetensor|"
    r"gguf|gptq|awq|int4|int8|4-bit|8-bit",
    re.IGNORECASE,
)
_BADGE_ONLY = re.compile(
    r"^\s*(?:\[?\s*!\[[^\]]*\]\([^)]*\)\s*\]?\s*)+$"
)


def preprocess_readme(readme_content, resource_id, payload):
    """Remove only high-confidence noise while preserving possible evidence."""
    if not isinstance(readme_content, str) or not readme_content:
        return {"readme_content": readme_content}

    text = re.sub(r"<!--.*?-->", "", readme_content, flags=re.DOTALL)
    kept = []
    for line in text.splitlines():
        # Keep evidence-bearing badges such as license/model badges. Remove only
        # lines made entirely of unrelated images/badges.
        if _BADGE_ONLY.match(line) and not _EVIDENCE_KEYWORDS.search(line):
            continue
        kept.append(line.rstrip())

    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {"readme_content": text}


def run(payload):
    return preprocess_readme(
        payload.get("readme_content") or payload.get("readme") or "",
        payload.get("resource_id") or "",
        payload,
    )
