import json
import re


SAFETENSORS_POSITIVE = re.compile(
    r"\.safetensors\b|\bsafe_open\b|\bsafetensors\.(?:torch|numpy)\b|"
    r"\bfrom\s+safetensors\b|\b(?:safe_serialization|use_safetensors)\s*=\s*true\b",
    re.IGNORECASE,
)
SAFETENSORS_NEGATIVE = re.compile(
    r"\b(?:no|not|without|doesn.t|does not)\b[^\n.]{0,50}\bsafetensors\b|"
    r"\bsafetensors\b[^\n.]{0,50}\b(?:not provided|not used|unsupported|unavailable)\b|"
    r"\bonly\b[^\n.]{0,40}\.(?:bin|pt)\b",
    re.IGNORECASE,
)
PARAMETER_UNITS = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
QUANTIZATION_FORMATS = {
    "awq", "bnb", "bitsandbytes", "gptq", "int4", "int8", "qlora",
}
CONFIG_CLASS_SUFFIX = re.compile(
    r"(?:ForCausalLM|ForSequenceClassification|ForTokenClassification|"
    r"ForQuestionAnswering|ForMaskedLM|ForConditionalGeneration|"
    r"ForImageClassification|ForAudioClassification|ForCTC|ForSeq2SeqLM|"
    r"ForMultipleChoice|ForNextSentencePrediction|WithLMHead|Config|Model)$",
    re.IGNORECASE,
)
SPLIT_SUFFIX = re.compile(
    r"\s*\(\s*(?:test|validation|valid|dev|train)(?:\s+set|\s+split)?\s*\)\s*$",
    re.IGNORECASE,
)
ARXIV_ID = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", re.IGNORECASE)
HF_DATASET_URL = re.compile(
    r"(?:https?://)?huggingface\.co/datasets/"
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
LANGUAGE_CODES = {
    "arabic": "ar",
    "chinese": "zh",
    "czech": "cs",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "norwegian": "no",
    "norwegian bokmål": "nb",
    "norwegian bokmal": "nb",
    "norwegian nynorsk": "nn",
    "polish": "pl",
    "portuguese": "pt",
    "romanian": "ro",
    "russian": "ru",
    "spanish": "es",
    "swedish": "sv",
    "thai": "th",
    "turkish": "tr",
    "vietnamese": "vi",
    "arabic (egyptian dialect)": ("ar", "arz"),
}


def _text(value):
    return "" if value is None else str(value).strip()


def _note(attr, message):
    previous = _text(attr.get("normalization_note"))
    attr["normalization_note"] = f"{previous}; {message}" if previous else message


def _model_name_from_id(resource_id):
    if not isinstance(resource_id, str):
        return ""
    clean = resource_id.strip().rstrip("/")
    if "/" not in clean:
        return ""
    return clean.rsplit("/", 1)[-1].strip()


def _author_from_id(resource_id):
    if not isinstance(resource_id, str):
        return ""
    clean = resource_id.strip().strip("/")
    if "/" not in clean:
        return ""
    return clean.split("/", 1)[0].strip()


def _normalize_num_parameters(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = _text(value).replace(",", "")
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*([kmb])(?:\s*(?:parameters?|params?))?",
        text,
        re.IGNORECASE,
    )
    if match:
        return int(float(match.group(1)) * PARAMETER_UNITS[match.group(2).lower()])
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return int(float(text))
    return value


def _normalize_config_model_type(value):
    text = _text(value)
    if not text:
        return value
    normalized = CONFIG_CLASS_SUFFIX.sub("", text).strip()
    return normalized.lower() if normalized else value


def _as_list(value):
    if isinstance(value, (list, tuple, set)):
        return list(value), True
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return parsed, True
            except json.JSONDecodeError:
                pass
    return [value], False


def _normalize_file_formats(value):
    items, was_collection = _as_list(value)
    cleaned = []
    for item in items:
        text = _text(item)
        if text and text.lower().lstrip(".") not in QUANTIZATION_FORMATS:
            cleaned.append(text.lstrip(".").lower())
    if not cleaned:
        return None
    return cleaned if was_collection else cleaned[0]


def _normalize_evaluation_datasets(value):
    items, was_collection = _as_list(value)
    cleaned = []
    for item in items:
        text = SPLIT_SUFFIX.sub("", _text(item)).strip()
        if text:
            cleaned.append(text)
    if not cleaned:
        return None
    return cleaned if was_collection else cleaned[0]


def _dataset_key(value):
    return re.sub(r"[^a-z0-9]+", "", _text(value).lower())


def _normalize_dataset_ids(value, evidence):
    """Resolve an existing short dataset name from a unique explicit HF URL."""
    items, was_collection = _as_list(value)
    linked_ids = list(dict.fromkeys(HF_DATASET_URL.findall(evidence or "")))
    normalized = []
    for item in items:
        text = _text(item)
        if not text or "/" in text:
            normalized.append(item)
            continue
        item_key = _dataset_key(text)
        matches = []
        for dataset_id in linked_ids:
            basename_key = _dataset_key(dataset_id.rsplit("/", 1)[-1])
            if (
                len(item_key) >= 4
                and len(basename_key) >= 4
                and (item_key == basename_key or item_key in basename_key or basename_key in item_key)
            ):
                matches.append(dataset_id)
        normalized.append(matches[0] if len(matches) == 1 else item)
    return normalized if was_collection else normalized[0]


def _split_scalar_collection(value):
    items, was_collection = _as_list(value)
    if was_collection:
        return items
    text = _text(value)
    return re.split(r"\s*[,;]\s*", text) if re.search(r"[,;]", text) else [text]


def _normalize_languages(value):
    normalized = []
    for item in _split_scalar_collection(value):
        text = _text(item)
        if not text:
            continue
        lower = text.lower()
        # Preserve already canonical language codes and schema labels.
        canonical = (
            lower
            if re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]+)?", lower)
            else LANGUAGE_CODES.get(lower)
        )
        canonical_values = (
            list(canonical)
            if isinstance(canonical, tuple)
            else [canonical or text]
        )
        for canonical_value in canonical_values:
            if canonical_value not in normalized:
                normalized.append(canonical_value)
    return normalized or None


def _normalize_cited_papers(value):
    ids = []
    values, _ = _as_list(value)
    for item in values:
        for paper_id in ARXIV_ID.findall(_text(item)):
            if paper_id not in ids:
                ids.append(paper_id)
    return ids or None


def postprocess_extraction(extraction, resource_id, readme_content, payload):
    """Apply deterministic, evidence-safe field normalization."""
    if not isinstance(extraction, dict):
        return extraction

    filtered = []
    derived_model_name = _model_name_from_id(resource_id)
    derived_author = _author_from_id(resource_id)
    for original in extraction.get("attributes") or []:
        if not isinstance(original, dict):
            filtered.append(original)
            continue

        attr = dict(original)
        attr_name = _text(attr.get("attribute")).lower()
        value = attr.get("value")
        value_text = _text(value)
        value_lower = value_text.lower()

        if attr_name == "model_name" and derived_model_name:
            if value != derived_model_name:
                attr["value"] = derived_model_name
                _note(attr, f"derived model_name from resource_id {resource_id!r}")

        elif attr_name == "author" and derived_author:
            if value != derived_author:
                attr["value"] = derived_author
                _note(attr, f"derived author from resource_id {resource_id!r}")

        elif attr_name == "num_parameters":
            normalized = _normalize_num_parameters(value)
            if normalized != value:
                attr["value"] = normalized
                _note(attr, f"normalized parameter count from {value!r}")

        elif attr_name == "config_model_type":
            normalized = _normalize_config_model_type(value)
            if normalized != value:
                attr["value"] = normalized
                _note(attr, f"removed config/model class suffix from {value!r}")

        elif attr_name == "model_file_formats":
            normalized = _normalize_file_formats(value)
            if normalized is None:
                continue
            if normalized != value:
                attr["value"] = normalized
                _note(attr, f"removed quantization labels from {value!r}")

        elif attr_name in {"training_datasets", "evaluation_datasets"}:
            normalized = _normalize_evaluation_datasets(value)
            if normalized is None:
                continue
            evidence = "\n".join(
                part for part in (
                    _text(attr.get("evidence_span") or attr.get("evidence")),
                    readme_content or "",
                )
                if part
            )
            normalized = _normalize_dataset_ids(normalized, evidence)
            if normalized != value:
                attr["value"] = normalized
                _note(attr, f"normalized evidence-linked dataset identifier from {value!r}")

        elif attr_name == "languages":
            normalized = _normalize_languages(value)
            if normalized is None:
                continue
            if normalized != value:
                attr["value"] = normalized
                _note(attr, f"normalized language names to codes from {value!r}")

        elif attr_name == "cited_papers":
            normalized = _normalize_cited_papers(value)
            if normalized is None:
                continue
            if normalized != value:
                attr["value"] = normalized
                _note(attr, f"normalized arXiv identifiers from {value!r}")

        elif attr_name == "uses_safetensors":
            value_true = value is True or value_lower in {"true", "1", "yes"}
            value_false = value is False or value_lower in {"false", "0", "no"}
            has_positive = bool(SAFETENSORS_POSITIVE.search(readme_content or ""))
            has_negative = bool(SAFETENSORS_NEGATIVE.search(readme_content or ""))
            if value_true and not has_positive:
                continue
            if value_false and not has_negative:
                continue
            if value_true and value is not True:
                attr["value"] = True
                _note(attr, "normalized evidence-supported boolean")
            elif value_false and value is not False:
                attr["value"] = False
                _note(attr, "normalized evidence-supported boolean")

        # AutoModel* is a loader abstraction, not a concrete architecture.
        elif attr_name == "architecture" and value_text.lower().startswith("automodel"):
            continue

        filtered.append(attr)

    extraction["attributes"] = filtered
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
