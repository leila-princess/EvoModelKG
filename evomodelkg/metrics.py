from __future__ import annotations

import re
from typing import Any

import evomodelkg.comparison.readme_structured_compare as compare_module
from evomodelkg.comparison.readme_structured_compare import (
    build_relation_index,
    compare,
    values_match,
)
from evomodelkg.benchmark import BenchmarkCase, cases_to_audit_entries
from evomodelkg.candidate_grouping import grouped_values_match


ACCURACY_EXCLUDED_ATTRIBUTES = {"auto_model", "quantization_config"}

LANGUAGE_NAME_TO_CODE = {
    "arabic": "ar",
    "chinese": "zh",
    "english": "en",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "turkish": "tr",
}

MONTH_NAME_TO_NUM = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}


def _filter_accuracy_audit_entries(audit_entries: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    for entry in audit_entries:
        new_entry = dict(entry)
        new_entry["attributes_readme_extracted"] = [
            row
            for row in (entry.get("attributes_readme_extracted") or [])
            if str(row.get("attribute") or "").strip() not in ACCURACY_EXCLUDED_ATTRIBUTES
        ]
        filtered.append(new_entry)
    return filtered


def _relation_keys(rows: list[dict]) -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for r in rows:
        rt = str(r.get("relation_type", "")).upper().strip()
        sid = str(r.get("source_id", "")).strip()
        tid = str(r.get("target_id", "")).strip()
        if rt and sid and tid:
            out.add((rt, sid, tid))
    return out


def _attribute_keys(rows: list[dict]) -> set[str]:
    return {
        str(a.get("attribute", "")).strip().lower()
        for a in rows
        if str(a.get("attribute", "")).strip()
    }


def _relation_types(rows: list[dict]) -> set[str]:
    return {
        str(r.get("relation_type", "")).upper().strip()
        for r in rows
        if str(r.get("relation_type", "")).strip()
    }


def _flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_values(item))
        return out
    return [str(value).strip()] if str(value).strip() else []


def _simple_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _compact_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _language_codes(value: Any) -> set[str]:
    codes: set[str] = set()
    for raw in _flatten_values(value):
        text = raw.lower()
        parts = re.split(r"[,;/|&+]|\band\b", text)
        for part in parts:
            token = part.strip(" .()[]{}'\"")
            if not token:
                continue
            if re.fullmatch(r"[a-z]{2,3}", token):
                codes.add(token)
                continue
            normalized = _simple_text(token)
            if normalized in LANGUAGE_NAME_TO_CODE:
                codes.add(LANGUAGE_NAME_TO_CODE[normalized])
    return codes


def _date_granularities(value: Any) -> set[str]:
    out: set[str] = set()
    text = str(value or "").strip()
    if not text:
        return out
    m = re.search(r"\b(?:19|20)\d{2}-\d{1,2}-\d{1,2}\b", text)
    if m:
        y, mon, day = m.group(0).split("-")
        out.add(y)
        out.add(f"{y}-{int(mon):02d}")
        out.add(f"{y}-{int(mon):02d}-{int(day):02d}")
    m = re.search(r"\b((?:19|20)\d{2})-(\d{1,2})\b", text)
    if m:
        y = m.group(1)
        mon = int(m.group(2))
        out.add(y)
        out.add(f"{y}-{mon:02d}")
    for month_name, month_num in MONTH_NAME_TO_NUM.items():
        m = re.search(rf"\b{month_name}\.?\s+((?:19|20)\d{{2}})\b", text, flags=re.I)
        if m:
            y = m.group(1)
            out.add(y)
            out.add(f"{y}-{month_num}")
    for y in re.findall(r"\b(?:19|20)\d{2}\b", text):
        out.add(y)
    return out


def _dataset_aliases(value: Any) -> set[str]:
    aliases: set[str] = set()
    for raw in _flatten_values(value):
        candidates = [raw]
        candidates.extend(re.split(r"[,;|\n]+", raw))
        for candidate in candidates:
            item = candidate.strip(" `[](){}'\"")
            if not item:
                continue
            item = re.sub(r"https?://huggingface\.co/datasets/", "", item, flags=re.I)
            item = re.sub(r"https?://huggingface\.co/", "", item, flags=re.I)
            item = item.strip("/")
            base = item.rsplit("/", 1)[-1]
            for text in {item, base}:
                norm = _compact_text(text)
                if norm:
                    aliases.add(norm)
                    if norm.endswith("dataset") and len(norm) > len("dataset"):
                        aliases.add(norm[: -len("dataset")])
    return aliases


def _dataset_item_f1_match(readme_val: Any, struct_val: Any, threshold: float = 0.67) -> bool:
    pred = _dataset_aliases(readme_val)
    gold = _dataset_aliases(struct_val)
    if not pred or not gold:
        return False
    matched = len(pred & gold)
    if matched == 0:
        return False
    precision = matched / len(pred)
    recall = matched / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
    return f1 >= threshold


def _listish_tokens(value: Any, *, field: str) -> set[str]:
    if field in {"language", "languages"}:
        codes = _language_codes(value)
        if codes:
            return codes
    tokens: set[str] = set()
    for raw in _flatten_values(value):
        for part in re.split(r"[,;/|&+\n]|\band\b|\bor\b", raw.lower()):
            token = _compact_text(part)
            if token:
                tokens.add(token)
    return tokens


def _set_jaccard_match(
    readme_val: Any,
    struct_val: Any,
    *,
    field: str,
    threshold: float = 0.80,
) -> bool:
    pred = _listish_tokens(readme_val, field=field)
    gold = _listish_tokens(struct_val, field=field)
    if not pred or not gold:
        return False
    return len(pred & gold) / len(pred | gold) >= threshold


def _custom_values_match(readme_val: Any, struct_val: Any, attr: str | None) -> bool | None:
    field = str(attr or "").strip()
    if field in {"language", "languages", "model_file_formats"}:
        return _set_jaccard_match(readme_val, struct_val, field=field, threshold=0.80)
    if field == "created_at":
        pred = _date_granularities(readme_val)
        gold = _date_granularities(struct_val)
        if pred and gold:
            return bool(pred & gold)
    if field in {"training_datasets", "evaluation_datasets"}:
        return _dataset_item_f1_match(readme_val, struct_val)
    return None


def format_self_evolve_summary_lines(
    accuracy_report: dict[str, Any],
    completeness_report: dict[str, Any] | None = None,
) -> list[str]:
    """自进化报告摘要：准确性（对标结构化）+ 完整性（仅看 README 抽取量）。"""
    a = accuracy_report["attribute_stats"]

    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    lines = [
        f"[属性-准确性] overlap={a['overlap_with_structured']}/{a['readme_extracted_total']} "
        f"({pct(a['overlap_rate'])}), 双方有值时匹配率={pct(a['accuracy_on_overlap'])}",
    ]
    for field_name, stats in accuracy_report.get("dataset_attribute_stats", {}).items():
        lines.append(
            f"[Dataset attribute: {field_name}] "
            f"overlap={stats['overlap_with_structured']}/{stats['readme_extracted_total']} "
            f"({pct(stats['overlap_rate'])}), "
            f"accuracy={pct(stats['accuracy_on_overlap'])}"
        )
    if completeness_report:
        t = completeness_report.get("totals", {})
        d = completeness_report.get("diversity", {})
        z = completeness_report.get("zero_extraction_cases", {})
        lines.append(
            f"[完整性-抽取量] 属性={t.get('attributes_extracted', 0)} "
            f"(均 {t.get('avg_attributes_per_case', 0):.1f}/样本)"
        )
        lines.append(
            f"[完整性-多样性] 均 unique 属性名={d.get('avg_unique_attribute_names_per_case', 0):.1f}, "
            f"零属性样本={z.get('attributes', 0)}"
        )
    return lines


def group_aware_values_match(readme_val: Any, struct_val: Any, attr: str | None = None) -> bool:
    custom = _custom_values_match(readme_val, struct_val, attr)
    if custom is not None:
        return custom
    grouped = grouped_values_match(readme_val, struct_val, attr)
    if grouped is not None:
        return grouped
    return values_match(readme_val, struct_val, attr=attr)


def group_aware_compare(**kwargs: Any) -> dict[str, Any]:
    original = compare_module.values_match
    compare_module.values_match = group_aware_values_match
    try:
        return compare(**kwargs)
    finally:
        compare_module.values_match = original


def expected_attributes_for_readme(
    char_count: int,
    *,
    base: float = 4.0,
    chars_per_attr: float = 2500.0,
    max_attrs: float = 14.0,
) -> float:
    """按 README 长度估计「合适」的属性条数（启发式，用于完整度评分）。"""
    if char_count <= 0:
        return base
    return min(max_attrs, base + char_count / max(1.0, chars_per_attr))


def compute_length_adjusted_completeness(
    cases: list[BenchmarkCase],
    extractions: list[dict[str, Any]],
    *,
    base: float = 4.0,
    chars_per_attr: float = 2500.0,
    max_attrs: float = 14.0,
) -> dict[str, Any]:
    """
    完整度：实际抽取属性数 vs 按 README 长度估计的合理条数。
    过少/过多均扣分；返回 [0,1] 的 score。
    """
    per_case: list[dict[str, Any]] = []
    scores: list[float] = []
    for case, ext in zip(cases, extractions):
        n_chars = len(case.readme_content or "")
        expected = expected_attributes_for_readme(
            n_chars, base=base, chars_per_attr=chars_per_attr, max_attrs=max_attrs
        )
        actual = len(ext.get("attributes") or [])
        if expected <= 0:
            s = 1.0 if actual == 0 else 0.5
        else:
            ratio = actual / expected
            if ratio <= 1.0:
                s = ratio
            elif ratio <= 1.5:
                s = 1.0 - 0.3 * (ratio - 1.0) / 0.5
            else:
                s = max(0.0, 0.7 - 0.2 * (ratio - 1.5))
        scores.append(s)
        per_case.append(
            {
                "resource_id": case.resource_id,
                "readme_chars": n_chars,
                "expected_attributes": round(expected, 2),
                "actual_attributes": actual,
                "case_score": round(s, 4),
            }
        )
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "score": avg,
        "params": {
            "base": base,
            "chars_per_attr": chars_per_attr,
            "max_attrs": max_attrs,
        },
        "per_case": per_case,
    }


def estimate_llm_calls_per_case(manifest: dict[str, Any]) -> int:
    """按 workflow 估计每个样本的 README 抽取 LLM 调用次数。"""
    wf = str(manifest.get("workflow", "unified")).strip().lower()
    groups = manifest.get("attribute_field_groups") or []
    n_attr_passes = len([g for g in groups if g])
    if wf == "unified":
        return 1
    if wf == "split_attributes":
        return max(1, n_attr_passes)
    if wf == "full_split":
        return max(1, n_attr_passes)
    return 1


def compute_completeness(
    cases: list[BenchmarkCase],
    extractions: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    完整性：只统计 README 抽取结果本身的数量与多样性，不对比结构化 baseline。
    """
    n = len(cases)
    if n == 0:
        return {"case_count": 0}

    total_attrs = 0
    total_rels = 0
    unique_attr_sum = 0
    unique_rel_type_sum = 0
    zero_attr_cases = 0
    zero_rel_cases = 0
    per_case: list[dict[str, Any]] = []

    for case, ext in zip(cases, extractions):
        attrs = ext.get("attributes") or []
        rels = ext.get("relations") or []
        attr_names = _attribute_keys(attrs)
        rel_types = _relation_types(rels)
        rel_keys = _relation_keys(rels)

        na = len(attrs)
        nr = len(rels)
        total_attrs += na
        total_rels += nr
        unique_attr_sum += len(attr_names)
        unique_rel_type_sum += len(rel_types)
        if na == 0:
            zero_attr_cases += 1
        if nr == 0:
            zero_rel_cases += 1

        per_case.append(
            {
                "resource_id": case.resource_id,
                "attributes_extracted": na,
                "relations_extracted": nr,
                "unique_attribute_names": sorted(attr_names),
                "unique_relation_types": sorted(rel_types),
                "unique_relation_triples": len(rel_keys),
            }
        )

    return {
        "case_count": n,
        "note": "完整性仅基于 README 抽取输出条数/多样性，不涉及结构化标答。",
        "totals": {
            "attributes_extracted": total_attrs,
            "relations_extracted": total_rels,
            "avg_attributes_per_case": total_attrs / n,
            "avg_relations_per_case": total_rels / n,
        },
        "diversity": {
            "avg_unique_attribute_names_per_case": unique_attr_sum / n,
            "avg_unique_relation_types_per_case": unique_rel_type_sum / n,
        },
        "zero_extraction_cases": {
            "attributes": zero_attr_cases,
            "relations": zero_rel_cases,
        },
        "per_case": per_case,
    }


def evaluate_extractions(
    cases: list[BenchmarkCase],
    extractions: list[dict[str, Any]],
) -> dict[str, Any]:
    """准确性（对标结构化 baseline）+ 完整性（仅 README 抽取量）。"""
    model_idx: dict[str, dict[str, Any]] = {}
    for case in cases:
        if case.resource_type != "model":
            continue
        entity = dict(case.structured_entity)
        training_datasets: list[str] = []
        evaluation_datasets: list[str] = []
        cited_papers: list[str] = []
        for relation in case.baseline_relations:
            if str(relation.get("source_id") or "").strip() != case.resource_id:
                continue
            relation_type = str(relation.get("relation_type") or "").upper().strip()
            target_id = str(relation.get("target_id") or "").strip()
            if not target_id:
                continue
            if relation_type == "EVALUATED_ON":
                evaluation_datasets.append(target_id)
            elif relation_type == "TRAINED_ON":
                training_datasets.append(target_id)
            elif relation_type == "MENTIONS_ARXIV":
                cited_papers.append(target_id)

        # Older/cached baselines may not contain TRAINED_ON edges. Preserve their
        # model-level datasets as a backwards-compatible training baseline.
        if not training_datasets:
            legacy_datasets = entity.get("datasets") or []
            if isinstance(legacy_datasets, str):
                legacy_datasets = [legacy_datasets]
            training_datasets.extend(str(x).strip() for x in legacy_datasets if str(x).strip())

        entity["training_datasets"] = sorted(set(training_datasets))
        entity["evaluation_datasets"] = sorted(set(evaluation_datasets))
        entity["cited_papers"] = sorted(set(cited_papers))
        model_idx[case.resource_id] = entity
    dataset_idx = {
        c.resource_id: c.structured_entity for c in cases if c.resource_type == "dataset"
    }

    rel_lists: list[dict] = []
    for c in cases:
        rel_lists.extend(c.baseline_relations)
    relations_payload: dict[str, list[dict]] = {
        "model_trained_on": [],
        "dataset_source": [],
        "model_derived_from": [],
        "model_evaluated_on": [],
        "model_generated": [],
        "model_annotated": [],
        "mentions_arxiv": [],
        "uses_tool": [],
        "licensed_under": [],
    }
    key_map = {
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
    for r in rel_lists:
        rt = str(r.get("relation_type", "")).upper()
        bucket = key_map.get(rt)
        if bucket:
            relations_payload[bucket].append(r)

    audit_entries = cases_to_audit_entries(list(zip(cases, extractions)))
    accuracy_audit_entries = _filter_accuracy_audit_entries(audit_entries)
    rel_idx = build_relation_index(relations_payload)
    accuracy_report = group_aware_compare(
        audit_entries=accuracy_audit_entries,
        model_idx=model_idx,
        dataset_idx=dataset_idx,
        relation_idx=rel_idx,
    )
    completeness_report = compute_completeness(cases, extractions)
    completeness_fit = compute_length_adjusted_completeness(cases, extractions)

    a = accuracy_report["attribute_stats"]
    comp = completeness_report
    completeness_totals = {
        key: value
        for key, value in comp.get("totals", {}).items()
        if "relation" not in key
    }
    completeness_diversity = {
        key: value
        for key, value in comp.get("diversity", {}).items()
        if "relation" not in key
    }

    summary = {
        "accuracy": {
            "attribute_overlap_accuracy": a.get("accuracy_on_overlap"),
            "attribute_overlap_rate": a.get("overlap_rate"),
        },
        "dataset_accuracy": accuracy_report.get("dataset_attribute_stats", {}),
        "completeness": {
            **completeness_totals,
            **completeness_diversity,
            "zero_attribute_cases": comp.get("zero_extraction_cases", {}).get("attributes", 0),
        },
        "completeness_fit": completeness_fit,
        "counts": {
            "readme_attributes_extracted": a.get("readme_extracted_total", 0),
        },
        "glossary": {
            "accuracy": (
                "准确性：README 抽取 vs Neo4j 结构化标答；"
                "双方同字段都有值时，比较值是否匹配。"
            ),
            "completeness": (
                "完整性：README 抽取条数/多样性；"
                "completeness_fit 按 README 长度估计合理属性条数并评分。"
            ),
            "mismatches": (
                "仅含 attribute_mismatch；"
                "有值时含 evidence_span、normalization_note（空键省略）。"
            ),
        },
    }

    mismatches = _collect_mismatches(
        accuracy_audit_entries, model_idx, dataset_idx, limit=120
    )

    return {
        "summary": summary,
        "accuracy_report": accuracy_report,
        "completeness_report": completeness_report,
        "completeness_fit_report": completeness_fit,
        "mismatches": mismatches,
        "compare_summary_lines": format_self_evolve_summary_lines(
            accuracy_report, completeness_report
        ),
    }


def _extraction_evidence_fields(row: dict[str, Any]) -> dict[str, str]:
    """从抽取行带出非空证据字段，供进化 agent 对照原文改 prompt。

    仅输出 evidence_span（README 原文短片段；无 span 时用 evidence 回填）
    与 normalization_note（归一化说明）。二者无内容时不写入键。
    """
    out: dict[str, str] = {}
    span = str(row.get("evidence_span") or "").strip()
    if not span:
        span = str(row.get("evidence") or "").strip()
    if span:
        out["evidence_span"] = span
    note = str(row.get("normalization_note") or "").strip()
    if note:
        out["normalization_note"] = note
    return out


def _collect_mismatches(
    audit_entries: list[dict],
    model_idx: dict,
    dataset_idx: dict,
    *,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """仅收集准确性错误（attribute_mismatch），供进化 agent 改 prompt。"""
    from evomodelkg.comparison.readme_structured_compare import (
        EXCLUDED_ATTRIBUTE_FIELDS,
        is_empty,
        is_structured_baseline_noise,
        readme_attribute_rows,
    )

    out: list[dict[str, Any]] = []

    for entry in audit_entries:
        resource_type = entry.get("resource_type")
        resource_id = entry.get("resource_id")
        entity = model_idx.get(resource_id, {}) if resource_type == "model" else dataset_idx.get(
            resource_id, {}
        )

        for attr_row in readme_attribute_rows(entry):
            attr = str(attr_row.get("attribute", "")).strip()
            if (
                not attr
                or attr in EXCLUDED_ATTRIBUTE_FIELDS
                or attr in ACCURACY_EXCLUDED_ATTRIBUTES
            ):
                continue
            struct_val = entity.get(attr)
            if is_empty(struct_val) or is_structured_baseline_noise(attr, struct_val):
                continue
            matched = group_aware_values_match(attr_row.get("value"), struct_val, attr=attr)
            if not matched:
                item: dict[str, Any] = {
                    "kind": "attribute_mismatch",
                    "resource_id": resource_id,
                    "attribute": attr,
                    "readme_value": attr_row.get("value"),
                    "structured_value": struct_val,
                }
                item.update(_extraction_evidence_fields(attr_row))
                out.append(item)

    return out[:limit]
