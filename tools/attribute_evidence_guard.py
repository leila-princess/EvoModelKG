import math


_BASE_PLACEHOLDERS = {
    "", "unknown", "not specified", "n/a", "na", "none", "null",
    "not mentioned", "not found", "unsupported", "empty", "not provided",
    "not available", "not stated", "not indicated", "not given", "not clear",
    "not explicitly stated", "not directly mentioned", "no information",
    "no mention", "no evidence", "no mention found", "no explicit mention",
    "no evidence found",
}
_SCOPES = {
    "readme", "the readme", "provided text", "the provided text", "input",
    "the input", "content", "the content", "document", "the document", "text",
    "the text", "readme content", "the readme content", "provided readme",
    "the provided readme", "model card", "the model card", "documentation",
    "the documentation", "repository", "the repository", "source", "the source",
    "code", "the code", "paper", "the paper", "description", "the description",
    "abstract", "the abstract", "section", "the section", "paragraph",
    "the paragraph", "sentence", "the sentence", "line", "the line", "file",
    "the file", "data", "the data", "metadata", "the metadata", "config",
    "the config", "configuration", "the configuration", "settings", "the settings",
}
_SCOPED_PATTERNS = {
    "not found in {}", "not specified in {}", "not available in {}",
    "not provided in {}", "not stated in {}", "not indicated in {}",
    "not given in {}", "not clear from {}", "not explicitly stated in {}",
    "not directly mentioned in {}", "no information in {}",
    "no mention of this in {}", "no evidence in {}", "no mention in {}",
    "not mentioned in {}", "no explicit mention in {}",
}
PLACEHOLDER_VALUES = _BASE_PLACEHOLDERS | {
    pattern.format(scope)
    for scope in _SCOPES
    for pattern in _SCOPED_PATTERNS
}

_INPUT_DERIVED_FIELDS = {"model_id", "model_name", "author"}


def _text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _is_placeholder(value):
    return _text(value).lower().rstrip(".。") in PLACEHOLDER_VALUES


def _rank(attr):
    try:
        confidence = float(attr.get("confidence", 0))
        if not math.isfinite(confidence):
            confidence = 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = _text(attr.get("evidence_span"))
    return (bool(evidence), confidence, min(len(evidence), 100))


def postprocess_extraction(extraction, resource_id, readme_content, payload):
    """Own all empty/placeholder, evidence, structure, and duplicate cleanup."""
    if not isinstance(extraction, dict):
        return extraction

    selected = {}
    order = []
    for attr in extraction.get("attributes") or []:
        if not isinstance(attr, dict):
            continue
        field = _text(attr.get("attribute"))
        value = _text(attr.get("value"))
        evidence = _text(attr.get("evidence_span"))
        if not field or not value or _is_placeholder(value):
            continue
        if field not in _INPUT_DERIVED_FIELDS and (
            not evidence or _is_placeholder(evidence)
        ):
            continue
        attr["attribute"] = field
        attr["value"] = value
        if field not in selected:
            selected[field] = attr
            order.append(field)
        elif _rank(attr) > _rank(selected[field]):
            selected[field] = attr

    extraction["attributes"] = [selected[field] for field in order]
    return extraction


def run(payload):
    extraction = payload.get("extraction") or payload.get("extraction_result") or {}
    result = postprocess_extraction(
        extraction,
        payload.get("resource_id") or "",
        payload.get("readme_content") or payload.get("readme") or "",
        payload,
    )
    return {"extraction": result}
