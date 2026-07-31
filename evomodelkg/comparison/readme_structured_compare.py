"""
README 多智能体抽取 vs 结构化 baseline 的一致性对比（供管线与 scripts 共用）。
"""
from __future__ import annotations

import json
import ast
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

RELATION_KEY_MAP = {
    "TRAINED_ON": "model_trained_on",
    "SOURCE_DATASET": "dataset_source",
    "DERIVED_FROM": "model_derived_from",
    "EVALUATED_ON": "model_evaluated_on",
    "GENERATED": "model_generated",
    "ANNOTATED": "model_annotated",
    "MENTIONS_ARXIV": "mentions_arxiv",
    "USES_TOOL": "uses_tool",
    "LICENSED_UNDER": "licensed_under",
}

_LANGUAGE_EQUIVALENTS_BASE = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "chinese": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "fr": "fr",
    "fra": "fr",
    "french": "fr",
    "de": "de",
    "deu": "de",
    "german": "de",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "ja": "ja",
    "jpn": "ja",
    "japanese": "ja",
}


def _build_language_equivalents() -> dict[str, str]:
    """
    语言等价表：
    1) 先使用手工映射（兼容项目既有行为）
    2) 若可用，补充 pycountry 的 ISO639 三字母 -> 两字母映射
       （含 alpha_3 / bibliographic / terminology 变体）
    """
    out = dict(_LANGUAGE_EQUIVALENTS_BASE)
    try:
        import pycountry  # type: ignore

        for lang in pycountry.languages:
            alpha2 = getattr(lang, "alpha_2", None)
            if not alpha2:
                continue
            alpha2 = str(alpha2).strip().lower()
            if len(alpha2) != 2:
                continue

            for key_name in ("alpha_3", "bibliographic", "terminology"):
                code = getattr(lang, key_name, None)
                if not code:
                    continue
                code = str(code).strip().lower()
                if len(code) == 3 and code.isalpha():
                    out[code] = alpha2
    except Exception:
        # 运行环境未安装 pycountry 时，保持手工映射兜底。
        pass
    return out


LANGUAGE_EQUIVALENTS = _build_language_equivalents()

MODEL_EXCLUDE_FOR_BASELINE = frozenset({
    "readme_content", "config_json", "adapter_config", "tokenizer_config", "siblings",
    "eval_results", "linked_spaces",
})
DATASET_EXCLUDE_FOR_BASELINE = frozenset({
    "readme_content", "card_data_raw", "siblings_preview",
})

# 用于单独统计“相对可判定字段”的准确率（避免粗细粒度差异字段污染指标）。
# 仅统计 README 与 structured 都有值（overlap）的条目。
RELIABLE_ATTRIBUTE_FIELDS = frozenset({
    "library_name",
    "pipeline_tag",
    "language",
    "languages",
    "license",
    "license_name",
    "base_model",
    "training_datasets",
    "evaluation_datasets",
    "model_file_formats",
    "num_parameters"
})

DATASET_ATTRIBUTE_FIELDS = frozenset({
    "training_datasets",
    "evaluation_datasets",
})

NUM_PARAMETERS_REL_TOL = 0.20

# 完全排除出 README vs structured 属性比较主流程的字段（不计入 total/overlap/accuracy）。
EXCLUDED_ATTRIBUTE_FIELDS = frozenset({
    "datasets",
    "has_training_details",
    "has_evaluation_results",
    "tags",
})

# baseline 结构化侧已知噪声值：出现时跳过本条属性对比（不计入 total/overlap）。
# 键均为 _norm_noise_token 规范化后的小写串（空格折叠）。
_STRUCTURED_BASELINE_NOISE_BY_FIELD: dict[str, frozenset[str]] = {
    # README/管线侧常用 config_model_type；统计文件里也可能叫 model_type
    "config_model_type": frozenset({
        "hhemv2config",
        "vietnamese",
        "new",
        "bilingual",
        "multimodality",
        "videoautoencoderpipeline",
    }),
    "model_type": frozenset({
        "hhemv2config",
        "vietnamese",
        "new",
        "bilingual",
        "multimodality",
        "videoautoencoderpipeline",
    }),
    "auto_model": frozenset({
        "videoautoencoderpipeline",
        "safetychecker",
    }),
    "library_name": frozenset({
        "generic",
        "gguf",
    }),
    "architecture": frozenset({
        "automodel",
        "model",
        "newmodel",
        "newforsequenceclassification",
        "staticmodel",
        "titan",
        "safetychecker",
        "stablediffusionsafetychecker",
        "videoautoencoderpipeline",
        "vietnamesemodel",
        "bilingualmodel",
    }),
    "language": frozenset({
        "mul",
        "und",
        "mis",
        "multilingual",
        "zxx",
    }),
    "languages": frozenset({
        "mul",
        "und",
        "mis",
        "multilingual",
        "zxx",
    }),
}


def _norm_noise_token(s: str) -> str:
    """仅保留字母数字，统一大小写（匹配 multi_modality / multi modality / Bilingual）。"""
    t = unicodedata.normalize("NFKC", str(s)).strip().lower()
    t = "".join(ch for ch in t if unicodedata.category(ch) not in {"Cf", "Cc"})
    return re.sub(r"[^a-z0-9]", "", t)


def _structured_tokens_for_noise_check(attr: str, val: Any) -> list[str]:
    """抽出用于噪声判定的离散串（均已 _norm_noise_token）。"""
    out: list[str] = []

    def push(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, Enum):
            push(x.value)
            return
        if isinstance(x, dict):
            for v in x.values():
                push(v)
            return
        if isinstance(x, list):
            for item in x:
                push(item)
            return
        if isinstance(x, bool):
            return
        s = str(x).strip()
        if not s:
            return
        if attr in {"language", "languages"}:
            parts = [p.strip(" '\"") for p in re.split(r"[,;/|]+", s)]
            for p in parts:
                nk = _norm_noise_token(p)
                if nk:
                    out.append(nk)
            return

        # 非语言：整段 + 按 / 切段（如 Bilingual / bilingual）
        chunks = [s] + [c.strip() for c in re.split(r"[/]", s) if c.strip()]
        for chunk in chunks:
            nk = _norm_noise_token(chunk)
            if nk:
                out.append(nk)

    push(val)
    return out


def is_structured_baseline_noise(attr: str, structured_value: Any) -> bool:
    """
    baseline 为该字段噪声值时不做对比。
    language/languages：若拆分后**全部为**噪声码则整条跳过；若无有效 token 则不视为噪声。
    其它字段：任一 token（整段或拆分）落入噪声表即跳过。
    """
    attr_key = attr.strip().lower()
    noise = _STRUCTURED_BASELINE_NOISE_BY_FIELD.get(attr_key)
    if not noise:
        return False
    if is_empty(structured_value):
        return False

    tokens = _structured_tokens_for_noise_check(attr_key, structured_value)
    if not tokens:
        return False

    if attr_key in {"language", "languages"}:
        return all(t in noise for t in tokens)

    return any(t in noise for t in tokens)


@dataclass
class Counter:
    total: int = 0
    overlap: int = 0
    matched: int = 0

    def ratio(self, num: int, den: int) -> float:
        if den <= 0:
            return 0.0
        return num / den


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, (list, dict, tuple, set)):
        return len(v) == 0
    return False


def to_scalar_tokens(value: Any, attr: str | None = None) -> set[str]:
    tokens: set[str] = set()
    if value is None:
        return tokens
    if isinstance(value, Enum):
        # pipeline 内存快照可能保留 Enum；比较时统一取 value。
        return to_scalar_tokens(value.value, attr=attr)
    if isinstance(value, bool):
        tokens.add("true" if value else "false")
        return tokens
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            tokens.add(str(int(value)))
        else:
            tokens.add(str(value))
        return tokens
    if isinstance(value, list):
        for x in value:
            tokens |= to_scalar_tokens(x, attr=attr)
        return tokens
    if isinstance(value, dict):
        tokens.add(norm_text(json.dumps(value, ensure_ascii=False, sort_keys=True), attr=attr))
        return tokens

    s = str(value).strip()
    if not s:
        return tokens
    # 语言字段常见 README 写法是 "en, zh" 这类分隔字符串，需拆分后再比较。
    if attr in {"language", "languages"}:
        parts = [x.strip(" '\"") for x in re.split(r"[,;|/]+", s)]
        parts = [x for x in parts if x]
        if len(parts) > 1:
            for item in parts:
                tokens.add(norm_text(item, attr=attr))
            return tokens
    if attr in DATASET_ATTRIBUTE_FIELDS:
        parts = [x.strip(" '\"") for x in re.split(r"[,;|\n]+", s)]
        parts = [x for x in parts if x]
        if len(parts) > 1:
            for item in parts:
                tokens.add(norm_text(item, attr=attr))
            return tokens
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple, set)):
            return to_scalar_tokens(list(parsed), attr=attr)
        maybe_items = [x.strip(" '\"") for x in s[1:-1].split(",")]
        if maybe_items:
            for item in maybe_items:
                if item:
                    tokens.add(norm_text(item, attr=attr))
            return tokens
    tokens.add(norm_text(s, attr=attr))
    return tokens


def norm_text(text: str, attr: str | None = None) -> str:
    # 先做 Unicode 兼容归一化，再去除不可见控制字符，避免“看起来相同却不相等”。
    s = unicodedata.normalize("NFKC", text)
    s = "".join(ch for ch in s if unicodedata.category(ch) not in {"Cf", "Cc"})
    s = s.strip().lower()
    if attr in {"language", "languages"}:
        return LANGUAGE_EQUIVALENTS.get(s, s)
    if attr in {"license", "license_name"}:
        compact = re.sub(r"[^a-z0-9]", "", s)
        license_aliases = {
            "apache2": "apache-2.0",
            "apache20": "apache-2.0",
            "apachelicense2": "apache-2.0",
            "apachelicense20": "apache-2.0",
            "apacheversion2": "apache-2.0",
            "apacheversion20": "apache-2.0",
        }
        canonical = license_aliases.get(compact)
        if canonical:
            return canonical
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def _parse_numeric_value(value: Any) -> Optional[float]:
    """尽量把常见文本数值（含科学计数法写法）解析为 float。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if not s:
        return None
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.replace(",", "")
    s = s.replace("×", "x")

    # 兼容缩写量级：30b / 7.61b / 120m / 900k / 1.2t
    m_suffix = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([kmbt])", s)
    if m_suffix:
        try:
            base = float(m_suffix.group(1))
            unit = m_suffix.group(2)
            scale = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[unit]
            return base * scale
        except Exception:
            return None

    # 兼容 "3.97 x 10^9" / "3.97*10^9" / "3.97 x10^9"
    m = re.fullmatch(
        r"([+-]?\d+(?:\.\d+)?)\s*(?:x|\*)\s*10\s*\^?\s*([+-]?\d+)",
        s,
    )
    if m:
        try:
            base = float(m.group(1))
            exp = int(m.group(2))
            return base * (10 ** exp)
        except Exception:
            return None

    # 常规数字 / e 记法（如 3.97e9）
    m2 = re.fullmatch(r"[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?", s)
    if m2:
        try:
            return float(s)
        except Exception:
            return None
    return None


def _quantization_alnum_blob(s: str) -> str:
    """去掉分隔符后的连续小写字母数字串，便于子串命中（如 mxfp4 ⊆ mxfp4quantization）。"""
    t = unicodedata.normalize("NFKC", s).lower()
    t = "".join(ch for ch in t if unicodedata.category(ch) not in {"Cf", "Cc"})
    return re.sub(r"[^a-z0-9]", "", t)


def _quantization_tokens_from_readme(readme_val: Any) -> set[str]:
    """README 文本中的量化相关 token（单词级）。"""
    if readme_val is None:
        return set()
    if isinstance(readme_val, dict):
        return _quantization_tokens_from_struct(readme_val)
    s = str(readme_val).strip()
    if not s:
        return set()
    parts = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", s).lower())
    return {p for p in parts if len(p) >= 2}


def _quantization_tokens_from_struct(obj: Any) -> set[str]:
    """结构化 quantization_config dict/list 中叶节点字符串与小整数的 token。"""
    out: set[str] = set()

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for item in x:
                walk(item)
        elif isinstance(x, str):
            t = x.strip().lower()
            if not t:
                return
            for p in re.findall(r"[a-z0-9]+", t):
                if len(p) >= 2:
                    out.add(p)
        elif isinstance(x, bool):
            return
        elif isinstance(x, int):
            si = str(x)
            if len(si) >= 1:
                out.add(si)
        elif isinstance(x, float):
            if x.is_integer():
                si = str(int(x))
                if si:
                    out.add(si)

    walk(obj)
    return out


def _quantization_config_values_match(readme_val: Any, struct_val: Any) -> Optional[bool]:
    """
    quantization_config：README 常为自然语言描述，structured 常为 dict。
    若双方均有可比对内容则返回 True/False；否则返回 None 交由通用逻辑处理。
    """
    if isinstance(struct_val, dict) and struct_val:
        struct_tokens = _quantization_tokens_from_struct(struct_val)
        if not struct_tokens:
            return None
        readme_tokens = _quantization_tokens_from_readme(readme_val)
        if readme_tokens & struct_tokens:
            return True
        blob = _quantization_alnum_blob(str(readme_val) if readme_val is not None else "")
        if blob:
            for tok in struct_tokens:
                if len(tok) >= 2 and tok in blob:
                    return True
            return False
        return False
    if isinstance(readme_val, dict) and readme_val and not isinstance(struct_val, dict):
        r_tokens = _quantization_tokens_from_struct(readme_val)
        if not r_tokens:
            return None
        s_tokens = _quantization_tokens_from_readme(struct_val)
        if s_tokens & r_tokens:
            return True
        blob = _quantization_alnum_blob(str(struct_val) if struct_val is not None else "")
        if blob:
            for tok in r_tokens:
                if len(tok) >= 2 and tok in blob:
                    return True
        return False
    return None


def _base_model_variants(value: Any) -> set[str]:
    """
    base_model 专用归一化：
    - 保留原值归一化
    - 对类似 org/model 的 repo id，额外提取末段 model 名参与比对
    """
    out: set[str] = set()

    def add_one(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, Enum):
            add_one(x.value)
            return
        if isinstance(x, list):
            for item in x:
                add_one(item)
            return
        if isinstance(x, dict):
            for v in x.values():
                add_one(v)
            return
        s = str(x).strip()
        if not s:
            return
        out.add(norm_text(s, attr="base_model"))
        # huggingface repo id 常见形态：namespace/model_name
        if "/" in s:
            tail = s.rsplit("/", 1)[-1].strip()
            if tail:
                out.add(norm_text(tail, attr="base_model"))

    add_one(value)
    return {x for x in out if x}


def _base_model_values_match(readme_val: Any, struct_val: Any) -> Optional[bool]:
    rv = _base_model_variants(readme_val)
    sv = _base_model_variants(struct_val)
    if not rv or not sv:
        return None
    return len(rv & sv) > 0


def _numeric_values_match(a: Any, b: Any, *, rel_tol: float = 0.02, abs_tol: float = 1e-9) -> bool:
    """
    数值近似匹配：
    - 允许 README 的“近似值/科学计数法”与结构化精确值匹配
    - 默认允许 2% 相对误差
    """
    av = _parse_numeric_value(a)
    bv = _parse_numeric_value(b)
    if av is None or bv is None:
        return False
    diff = abs(av - bv)
    allowed = max(abs_tol, rel_tol * max(abs(av), abs(bv)))
    return diff <= allowed


PIPELINE_TAG_SYNONYM_GROUPS = tuple(
    frozenset(norm_text(x, attr="pipeline_tag") for x in group)
    for group in (
        # Hub task labels often use either a broad task or a more specific pipeline name.
        ("text-to-image", "image-generation"),
        ("image-to-image", "image-editing"),
        ("feature-extraction", "sentence-similarity", "image-feature-extraction"),
        ("image-segmentation", "mask-generation"),
        ("image-classification", "zero-shot-image-classification"),
        ("object-detection", "zero-shot-object-detection"),
    )
)

PIPELINE_TAG_PARENT_CHILD_GROUPS = tuple(
    (
        norm_text(parent, attr="pipeline_tag"),
        frozenset(norm_text(x, attr="pipeline_tag") for x in children),
    )
    for parent, children in (
        (
            "question-answering",
            (
                "table-question-answering",
                "document-question-answering",
                "visual-question-answering",
            ),
        ),
    )
)


def _values_match_synonym_group(readme_val: Any, struct_val: Any, attr: str | None) -> bool:
    if attr != "pipeline_tag":
        return False
    a = to_scalar_tokens(readme_val, attr=attr)
    b = to_scalar_tokens(struct_val, attr=attr)
    if not a or not b:
        return False
    if any((a & group) and (b & group) for group in PIPELINE_TAG_SYNONYM_GROUPS):
        return True
    for parent, children in PIPELINE_TAG_PARENT_CHILD_GROUPS:
        if (parent in a and b & children) or (parent in b and a & children):
            return True
    return False


def values_match(readme_val: Any, struct_val: Any, attr: str | None = None) -> bool:
    # 数值字段先做近似比较，避免“表达精度不同”被误判
    numeric_rel_tol = NUM_PARAMETERS_REL_TOL if attr == "num_parameters" else 0.02
    if _numeric_values_match(readme_val, struct_val, rel_tol=numeric_rel_tol):
        return True
    if _values_match_synonym_group(readme_val, struct_val, attr):
        return True
    if attr == "quantization_config":
        qm = _quantization_config_values_match(readme_val, struct_val)
        if qm is not None:
            return qm
    if attr == "base_model":
        bm = _base_model_values_match(readme_val, struct_val)
        if bm is not None:
            return bm
    a = to_scalar_tokens(readme_val, attr=attr)
    b = to_scalar_tokens(struct_val, attr=attr)
    if not a or not b:
        return False
    if attr in {"cited_papers", "model_file_formats", "model_sub_types"}:
        return a.issubset(b)
    return len(a & b) > 0


def index_entities(rows: list[dict], id_key: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        rid = row.get(id_key)
        if isinstance(rid, str) and rid.strip():
            out[rid] = row
    return out


def build_relation_index(relations_payload: dict[str, list[dict]]) -> dict[tuple[str, str, str], dict]:
    idx: dict[tuple[str, str, str], dict] = {}
    for rel_type, rel_key in RELATION_KEY_MAP.items():
        for rel in relations_payload.get(rel_key, []):
            sid = rel.get("source_id")
            tid = rel.get("target_id")
            if not isinstance(sid, str) or not isinstance(tid, str):
                continue
            idx[(rel_type, sid, tid)] = rel
    return idx


def readme_relation_rows(entry: dict) -> list[dict]:
    return list(entry.get("relations_readme_extracted") or [])


def readme_attribute_rows(entry: dict) -> list[dict]:
    return list(entry.get("attributes_readme_extracted") or [])


def entity_attribute_snapshot_from_dict(
    data: dict[str, Any], exclude: set[str]
) -> tuple[dict[str, Any], list[str]]:
    """与 pipeline.Pipeline._entity_attribute_snapshot 一致，对 dict 形态实体做快照。"""
    filled: dict[str, Any] = {}
    missing: list[str] = []
    for k, v in data.items():
        if k in exclude:
            continue
        is_empty = (
            v is None
            or (isinstance(v, (list, dict, str)) and len(v) == 0)
            or v is False and k not in ("private", "disabled", "gated")
        )
        if is_empty:
            missing.append(k)
        else:
            filled[k] = v
    return filled, missing


def collect_readme_not_found_baseline_stats(resources: list[tuple[str, Any]]) -> dict[str, Any]:
    """
    统计 README 缺失时，baseline 仍有值的字段分布（不参与 README 对比准确率）。
    resources: (\"model\"|\"dataset\", entity)
    """
    field_counts: dict[str, int] = {}
    resources_missing_readme = 0
    total_baseline_filled_attributes = 0
    for resource_type, ent in resources:
        readme = (getattr(ent, "readme_content", None) or "")
        if isinstance(readme, str) and readme.strip():
            continue
        resources_missing_readme += 1
        if resource_type == "model":
            ex = MODEL_EXCLUDE_FOR_BASELINE
        else:
            ex = DATASET_EXCLUDE_FOR_BASELINE
        try:
            data = ent.model_dump()
        except Exception:
            data = {}
        filled, _ = entity_attribute_snapshot_from_dict(data, ex)
        names = list(filled.keys())
        total_baseline_filled_attributes += len(names)
        for n in names:
            field_counts[n] = field_counts.get(n, 0) + 1
    top_fields = sorted(field_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:30]
    return {
        "resources_missing_readme": resources_missing_readme,
        "total_baseline_filled_attributes": total_baseline_filled_attributes,
        "unique_field_count": len(field_counts),
        "field_counts": field_counts,
        "top_fields": [{"field": k, "count": v} for k, v in top_fields],
    }


def compare(
    audit_entries: list[dict],
    model_idx: dict[str, dict],
    dataset_idx: dict[str, dict],
    relation_idx: dict[tuple[str, str, str], dict],
) -> dict[str, Any]:
    attr_counter = Counter()
    reliable_attr_counter = Counter()
    rel_counter = Counter()
    rel_prop_counter = Counter()
    attr_samples: list[dict[str, Any]] = []
    rel_samples: list[dict[str, Any]] = []
    skipped_noise_total = 0
    skipped_noise_by_field: dict[str, int] = {}
    dataset_attr_counters = {name: Counter() for name in DATASET_ATTRIBUTE_FIELDS}

    for entry in audit_entries:
        resource_type = entry.get("resource_type")
        resource_id = entry.get("resource_id")
        if resource_type == "model":
            entity = model_idx.get(resource_id, {})
        else:
            entity = dataset_idx.get(resource_id, {})

        for attr_row in readme_attribute_rows(entry):
            attr = str(attr_row.get("attribute", "")).strip()
            if not attr:
                continue
            if attr in EXCLUDED_ATTRIBUTE_FIELDS:
                continue
            struct_val = entity.get(attr)
            if (
                not is_empty(struct_val)
                and is_structured_baseline_noise(attr, struct_val)
            ):
                skipped_noise_total += 1
                skipped_noise_by_field[attr] = skipped_noise_by_field.get(attr, 0) + 1
                continue
            attr_counter.total += 1
            dataset_counter = dataset_attr_counters.get(attr)
            if dataset_counter is not None:
                dataset_counter.total += 1
            if is_empty(struct_val):
                continue
            attr_counter.overlap += 1
            if dataset_counter is not None:
                dataset_counter.overlap += 1
            matched = values_match(attr_row.get("value"), struct_val, attr=attr)
            if matched:
                attr_counter.matched += 1
                if dataset_counter is not None:
                    dataset_counter.matched += 1
            if attr in RELIABLE_ATTRIBUTE_FIELDS:
                reliable_attr_counter.total += 1
                reliable_attr_counter.overlap += 1
                if matched:
                    reliable_attr_counter.matched += 1
            if len(attr_samples) < 60:
                attr_samples.append(
                    {
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "attribute": attr,
                        "readme_value": attr_row.get("value"),
                        "structured_value": struct_val,
                        "matched": matched,
                    }
                )

        for rel_row in readme_relation_rows(entry):
            rel_type = rel_row.get("relation_type")
            sid = rel_row.get("source_id")
            tid = rel_row.get("target_id")
            if not (isinstance(rel_type, str) and isinstance(sid, str) and isinstance(tid, str)):
                continue
            rel_counter.total += 1
            struct_rel = relation_idx.get((rel_type, sid, tid))
            if not struct_rel:
                continue
            rel_counter.overlap += 1
            rel_counter.matched += 1

            readme_props = rel_row.get("properties") or {}
            for pk, pv in readme_props.items():
                if pk in {"resource_type"}:
                    continue
                if pk not in struct_rel or is_empty(struct_rel.get(pk)):
                    continue
                rel_prop_counter.total += 1
                rel_prop_counter.overlap += 1
                pmatch = values_match(pv, struct_rel.get(pk), attr=pk)
                if pmatch:
                    rel_prop_counter.matched += 1
                if len(rel_samples) < 60:
                    rel_samples.append(
                        {
                            "relation_type": rel_type,
                            "source_id": sid,
                            "target_id": tid,
                            "property": pk,
                            "readme_value": pv,
                            "structured_value": struct_rel.get(pk),
                            "matched": pmatch,
                        }
                    )

    return {
        "attribute_stats": {
            "readme_extracted_total": attr_counter.total,
            "overlap_with_structured": attr_counter.overlap,
            "matched_in_overlap": attr_counter.matched,
            "overlap_rate": attr_counter.ratio(attr_counter.overlap, attr_counter.total),
            "accuracy_on_overlap": attr_counter.ratio(attr_counter.matched, attr_counter.overlap),
        },
        "attribute_stats_reliable_fields": {
            "accuracy_on_overlap_in_whitelist": reliable_attr_counter.ratio(
                reliable_attr_counter.matched, reliable_attr_counter.overlap
            ),
        },
        "dataset_attribute_stats": {
            name: {
                "readme_extracted_total": counter.total,
                "overlap_with_structured": counter.overlap,
                "matched_in_overlap": counter.matched,
                "overlap_rate": counter.ratio(counter.overlap, counter.total),
                "accuracy_on_overlap": counter.ratio(counter.matched, counter.overlap),
            }
            for name, counter in dataset_attr_counters.items()
        },
        "relation_stats": {
            "readme_extracted_total": rel_counter.total,
            "overlap_with_structured": rel_counter.overlap,
            "matched_in_overlap": rel_counter.matched,
            "overlap_rate": rel_counter.ratio(rel_counter.overlap, rel_counter.total),
            "accuracy_on_overlap": rel_counter.ratio(rel_counter.matched, rel_counter.overlap),
        },
        "relation_property_stats": {
            "comparable_total": rel_prop_counter.total,
            "matched": rel_prop_counter.matched,
            "accuracy": rel_prop_counter.ratio(rel_prop_counter.matched, rel_prop_counter.total),
        },
        "sample_attribute_comparisons": attr_samples,
        "sample_relation_property_comparisons": rel_samples,
        "skipped_structured_noise_baseline": {
            "total": skipped_noise_total,
            "by_field": dict(sorted(skipped_noise_by_field.items(), key=lambda kv: kv[0])),
        },
    }


def run_full_compare_report(
    audit_entries: list[dict],
    model_idx: dict[str, dict],
    dataset_idx: dict[str, dict],
    relations_payload: dict[str, list[dict]],
    resources: Optional[list[tuple[str, Any]]] = None,
) -> dict[str, Any]:
    """对比 README 抽取与 baseline，并附带 README 缺失时的 baseline 字段统计。"""
    rel_idx = build_relation_index(relations_payload)
    base = compare(
        audit_entries=audit_entries,
        model_idx=model_idx,
        dataset_idx=dataset_idx,
        relation_idx=rel_idx,
    )
    if resources:
        base["readme_not_found_baseline_stats"] = collect_readme_not_found_baseline_stats(resources)
    return base


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def format_compare_summary_lines(report: dict[str, Any]) -> list[str]:
    a = report["attribute_stats"]
    aw = report.get("attribute_stats_reliable_fields", {})
    r = report["relation_stats"]
    p = report["relation_property_stats"]
    lines = [
        f"[属性] overlap={a['overlap_with_structured']}/{a['readme_extracted_total']} "
        f"({pct(a['overlap_rate'])}), overlap_accuracy={pct(a['accuracy_on_overlap'])}",
        f"[关系] overlap={r['overlap_with_structured']}/{r['readme_extracted_total']} "
        f"({pct(r['overlap_rate'])}), overlap_accuracy={pct(r['accuracy_on_overlap'])}",
        f"[关系属性] comparable={p['comparable_total']}, matched={p['matched']}, accuracy={pct(p['accuracy'])}",
    ]
    if aw:
        lines.append(
            f"[属性-可信字段] overlap_accuracy="
            f"{pct(float(aw.get('accuracy_on_overlap_in_whitelist', 0.0) or 0.0))}"
        )
    for field_name, stats in report.get("dataset_attribute_stats", {}).items():
        lines.append(
            f"[Dataset attribute: {field_name}] "
            f"overlap={stats.get('overlap_with_structured', 0)}/"
            f"{stats.get('readme_extracted_total', 0)}, "
            f"overlap_accuracy={pct(float(stats.get('accuracy_on_overlap', 0.0) or 0.0))}"
        )
    miss = report.get("readme_not_found_baseline_stats")
    if miss:
        lines.append(
            f"[README缺失-baseline字段统计] 资源数={miss.get('resources_missing_readme', 0)}, "
            f"baseline非空属性总数={miss.get('total_baseline_filled_attributes', 0)}, "
            f"字段种类数={miss.get('unique_field_count', 0)}"
        )
    sn = report.get("skipped_structured_noise_baseline")
    if isinstance(sn, dict) and int(sn.get("total", 0) or 0) > 0:
        lines.append(f"[baseline噪声跳过] count={sn.get('total', 0)}")
    return lines


def aggregate_compare_reports(batch_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """将多个 batch 对比报告聚合为总体报告（按计数重算比率）。"""
    attr_total = attr_overlap = attr_matched = 0
    reliable_acc_values: list[float] = []
    rel_total = rel_overlap = rel_matched = 0
    prop_total = prop_matched = 0
    miss_resources = miss_attr_total = miss_unique_fields = 0
    agg_field_counts: dict[str, int] = {}
    batch_summaries: list[dict[str, Any]] = []
    agg_skipped_noise_total = 0
    agg_skipped_noise_by_field: dict[str, int] = {}
    agg_dataset_attrs = {
        name: {"total": 0, "overlap": 0, "matched": 0}
        for name in DATASET_ATTRIBUTE_FIELDS
    }

    for i, rep in enumerate(batch_reports, start=1):
        a = rep.get("attribute_stats", {})
        aw = rep.get("attribute_stats_reliable_fields", {})
        r = rep.get("relation_stats", {})
        p = rep.get("relation_property_stats", {})
        for name, stats in rep.get("dataset_attribute_stats", {}).items():
            if name not in agg_dataset_attrs:
                continue
            agg_dataset_attrs[name]["total"] += int(stats.get("readme_extracted_total", 0) or 0)
            agg_dataset_attrs[name]["overlap"] += int(stats.get("overlap_with_structured", 0) or 0)
            agg_dataset_attrs[name]["matched"] += int(stats.get("matched_in_overlap", 0) or 0)
        m = rep.get("readme_not_found_baseline_stats", {})
        if not m:
            # 兼容历史报告键名
            legacy = rep.get("readme_missing_baseline", {})
            if legacy:
                lf = {}
                for item in list(legacy.get("resources", []) or []):
                    for name in list(item.get("baseline_filled_attribute_names", []) or []):
                        lf[name] = lf.get(name, 0) + 1
                m = {
                    "resources_missing_readme": legacy.get("resources_missing_readme", 0),
                    "total_baseline_filled_attributes": legacy.get("total_baseline_filled_attributes", 0),
                    "unique_field_count": len(lf),
                    "field_counts": lf,
                }

        attr_total += int(a.get("readme_extracted_total", 0) or 0)
        attr_overlap += int(a.get("overlap_with_structured", 0) or 0)
        attr_matched += int(a.get("matched_in_overlap", 0) or 0)
        if "accuracy_on_overlap_in_whitelist" in aw:
            try:
                reliable_acc_values.append(float(aw.get("accuracy_on_overlap_in_whitelist") or 0.0))
            except Exception:
                pass

        rel_total += int(r.get("readme_extracted_total", 0) or 0)
        rel_overlap += int(r.get("overlap_with_structured", 0) or 0)
        rel_matched += int(r.get("matched_in_overlap", 0) or 0)

        prop_total += int(p.get("comparable_total", 0) or 0)
        prop_matched += int(p.get("matched", 0) or 0)

        sn = rep.get("skipped_structured_noise_baseline") or {}
        agg_skipped_noise_total += int(sn.get("total", 0) or 0)
        for k, v in dict(sn.get("by_field") or {}).items():
            agg_skipped_noise_by_field[str(k)] = agg_skipped_noise_by_field.get(str(k), 0) + int(v or 0)

        miss_resources += int(m.get("resources_missing_readme", 0) or 0)
        miss_attr_total += int(m.get("total_baseline_filled_attributes", 0) or 0)
        miss_unique_fields += int(m.get("unique_field_count", 0) or 0)
        for k, v in dict(m.get("field_counts", {}) or {}).items():
            agg_field_counts[k] = agg_field_counts.get(k, 0) + int(v or 0)

        batch_summaries.append(
            {
                "batch_index": int(rep.get("meta", {}).get("batch_index", i)),
                "attribute_readme_extracted_total": int(a.get("readme_extracted_total", 0) or 0),
                "relation_readme_extracted_total": int(r.get("readme_extracted_total", 0) or 0),
                "missing_readme_resources": int(m.get("resources_missing_readme", 0) or 0),
                "skipped_structured_noise_baseline": int((rep.get("skipped_structured_noise_baseline") or {}).get("total", 0) or 0),
            }
        )

    def _safe_ratio(num: int, den: int) -> float:
        return (num / den) if den > 0 else 0.0

    top_fields = sorted(agg_field_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:50]
    return {
        "meta": {
            "batch_count": len(batch_reports),
            "kind": "readme_structured_compare_overall",
        },
        "attribute_stats": {
            "readme_extracted_total": attr_total,
            "overlap_with_structured": attr_overlap,
            "matched_in_overlap": attr_matched,
            "overlap_rate": _safe_ratio(attr_overlap, attr_total),
            "accuracy_on_overlap": _safe_ratio(attr_matched, attr_overlap),
        },
        "attribute_stats_reliable_fields": {
            "accuracy_on_overlap_in_whitelist": (
                sum(reliable_acc_values) / len(reliable_acc_values)
                if reliable_acc_values else 0.0
            ),
        },
        "dataset_attribute_stats": {
            name: {
                "readme_extracted_total": counts["total"],
                "overlap_with_structured": counts["overlap"],
                "matched_in_overlap": counts["matched"],
                "overlap_rate": _safe_ratio(counts["overlap"], counts["total"]),
                "accuracy_on_overlap": _safe_ratio(counts["matched"], counts["overlap"]),
            }
            for name, counts in agg_dataset_attrs.items()
        },
        "relation_stats": {
            "readme_extracted_total": rel_total,
            "overlap_with_structured": rel_overlap,
            "matched_in_overlap": rel_matched,
            "overlap_rate": _safe_ratio(rel_overlap, rel_total),
            "accuracy_on_overlap": _safe_ratio(rel_matched, rel_overlap),
        },
        "relation_property_stats": {
            "comparable_total": prop_total,
            "matched": prop_matched,
            "accuracy": _safe_ratio(prop_matched, prop_total),
        },
        "readme_not_found_baseline_stats": {
            "resources_missing_readme": miss_resources,
            "total_baseline_filled_attributes": miss_attr_total,
            "sum_unique_field_count_across_batches": miss_unique_fields,
            "unique_field_count": len(agg_field_counts),
            "field_counts": agg_field_counts,
            "top_fields": [{"field": k, "count": v} for k, v in top_fields],
        },
        "batches": batch_summaries,
        "skipped_structured_noise_baseline": {
            "total": agg_skipped_noise_total,
            "by_field": dict(sorted(agg_skipped_noise_by_field.items(), key=lambda kv: kv[0])),
        },
    }


def build_overall_report_from_batch_files(processed_dir: Path) -> dict[str, Any] | None:
    """从 readme_compare_batch_*.json 读取并聚合；无文件时返回 None。"""
    files = sorted(processed_dir.glob("readme_compare_batch_*.json"))
    if not files:
        return None
    reports: list[dict[str, Any]] = []
    for fp in files:
        try:
            data = load_json(fp)
            if isinstance(data, dict):
                reports.append(data)
        except Exception:
            continue
    if not reports:
        return None
    return aggregate_compare_reports(reports)



