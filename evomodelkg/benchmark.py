from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from evomodelkg.comparison.readme_structured_compare import (
    MODEL_EXCLUDE_FOR_BASELINE,
    build_relation_index,
    entity_attribute_snapshot_from_dict,
    load_json,
)
from evomodelkg.readme_loader import ReadmeFilenameStyle, resolve_readme_content
from evomodelkg.neo4j_baseline import fetch_neo4j_baseline


@dataclass
class BenchmarkCase:
    resource_type: str
    resource_id: str
    readme_content: str
    structured_entity: dict[str, Any]
    baseline_relations: list[dict[str, Any]] = field(default_factory=list)


def _load_relations_payload(benchmark_dir: Path, relations_json: str | None) -> dict[str, list[dict]]:
    if relations_json:
        path = benchmark_dir / relations_json
        if path.is_file():
            return load_json(path)

    batch_files = sorted(benchmark_dir.glob("relations_batch_*.json"))
    if batch_files:
        merged: dict[str, list[dict]] = {}
        for bf in batch_files:
            payload = load_json(bf)
            if not isinstance(payload, dict):
                continue
            for k, rows in payload.items():
                if isinstance(rows, list):
                    merged.setdefault(k, []).extend(rows)
        if merged:
            return merged

    default = benchmark_dir / "relations.json"
    if default.is_file():
        return load_json(default)
    return {}


def _relations_for_resource(
    relation_idx: dict[tuple[str, str, str], dict],
    resource_type: str,
    resource_id: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (rtype, sid, tid), rel in relation_idx.items():
        if sid == resource_id or tid == resource_id:
            out.append(
                {
                    "relation_type": rtype,
                    "source_id": sid,
                    "target_id": tid,
                    **{k: v for k, v in rel.items() if k not in ("source_id", "target_id")},
                }
            )
    return out


def load_resource_ids_from_file(path: Path) -> list[str]:
    """从 JSON 加载 model_id 列表。支持 ["id", ...] 或 {"resource_ids": [...]}。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到用例列表文件: {path}")
    data = load_json(path)
    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    if isinstance(data, dict):
        ids = data.get("resource_ids") or data.get("model_ids") or data.get("cases")
        if isinstance(ids, list):
            return [str(x).strip() for x in ids if str(x).strip()]
    raise ValueError(f"{path} 格式无效，需为 model_id 数组或 {{\"resource_ids\": [...]}}")


def sample_model_ids_from_hubstats(
    hubstats_dir: Path,
    *,
    max_cases: int,
    seed: int,
    resource_ids: list[str] | None = None,
    start_index: int = 0,
) -> list[str]:
    """从 models.parquet 顺序读取 model_id（仅作 neo4j 无 hit 文件时的回退）。"""
    from crawlers.hubstats_model_crawler import HubStatsModelCrawler
    from config import CrawlerConfig, HuggingFaceConfig

    crawler = HubStatsModelCrawler(
        HuggingFaceConfig(),
        CrawlerConfig(),
        str(hubstats_dir),
    )
    if resource_ids:
        return list(resource_ids)[:max_cases]

    pool: list[str] = []
    skip = max(0, int(start_index))
    for row in crawler._iter_rows():
        mid = row.get("id")
        if isinstance(mid, str) and mid.strip():
            if skip > 0:
                skip -= 1
                continue
            pool.append(mid.strip())
            if len(pool) >= max(1, max_cases):
                break
    return pool[: max(1, max_cases)]


def load_benchmark_cases(
    benchmark_dir: Path,
    *,
    models_json: str = "models.json",
    relations_json: str | None = None,
    max_cases: int = 15,
    seed: int = 42,
    resource_ids: list[str] | None = None,
    readme_dir: Path | None = None,
    readme_field: str = "readme_content",
    readme_filename_style: ReadmeFilenameStyle = "auto",
    hubstats_dir: Path | None = None,
    start_index: int = 0,
    read_batch_size: int = 20,
    baseline_source: str = "neo4j",
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str = "neo4j",
    readme_hit_ids_file: Path | None = None,
    require_readme_hit_ids: bool = False,
) -> list[BenchmarkCase]:
    """加载评测用例。标答默认从 Neo4j 读取；cached 从 processed JSON 读取。"""
    if baseline_source == "neo4j":
        if not neo4j_uri or not neo4j_user:
            raise ValueError("baseline_source=neo4j 需要提供 neo4j_uri 与 neo4j_user")
        required = max(1, max_cases)
        read_batch_size = max(1, int(read_batch_size))
        cases: list[BenchmarkCase] = []
        using_hit_file = False
        if resource_ids:
            ids_source = list(resource_ids)
        else:
            ids_source = []
            hit_file = Path(readme_hit_ids_file) if readme_hit_ids_file else None
            if hit_file is not None and hit_file.is_file():
                using_hit_file = True
                payload = load_json(hit_file)
                if not isinstance(payload, list):
                    raise ValueError(f"{hit_file} 需为 JSON 数组")
                all_ids = [str(x).strip() for x in payload if str(x).strip()]
                start = max(0, int(start_index))
                ids_source = all_ids[start:]
                logger.info(
                    f"[baseline][neo4j] 使用预筛选 id 文件: {hit_file}, "
                    f"total={len(all_ids)}, start={start}, remain={len(ids_source)}"
                )
                if not ids_source:
                    raise ValueError(
                        f"{hit_file} 在 start_index={start} 后无可用 id，"
                        "请降低 start_index 或更新 readme_hit_ids_file"
                    )
            else:
                if require_readme_hit_ids:
                    raise FileNotFoundError(
                        f"未找到 readme_hit_ids 文件: {hit_file}；"
                        "请提供 data/splits/candidate_pool_15000.json 或使用 --allow-parquet-fallback"
                    )
                cursor = max(0, int(start_index))
                logger.warning(
                    "[baseline][neo4j] 未使用 hit 文件，回退 parquet 顺序扫描（可能与 README 不匹配）"
                )
        logger.info(
            f"[baseline][neo4j] 开始: required={required}, start_index={start_index}, "
            f"batch_size={read_batch_size}"
        )
        cursor = None if (resource_ids or ids_source) else max(0, int(start_index))
        while len(cases) < required:
            if resource_ids or ids_source:
                batch_ids = ids_source[:read_batch_size]
                ids_source = ids_source[read_batch_size:]
            else:
                batch_ids = sample_model_ids_from_hubstats(
                    Path(hubstats_dir) if hubstats_dir else Path("dataset_hub_stats"),
                    max_cases=read_batch_size,
                    seed=seed,
                    start_index=int(cursor or 0),
                )
                cursor = (cursor or 0) + len(batch_ids)
            if not batch_ids:
                break
            model_props, rel_map = fetch_neo4j_baseline(
                model_ids=batch_ids,
                uri=str(neo4j_uri),
                user=str(neo4j_user),
                password=str(neo4j_password or ""),
                database=str(neo4j_database),
            )
            kept = 0
            for mid in batch_ids:
                props = model_props.get(mid)
                if not props:
                    continue
                readme = resolve_readme_content(
                    mid,
                    {"model_id": mid},
                    readme_field=readme_field,
                    readme_dir=readme_dir,
                    readme_filename_style=readme_filename_style,
                )
                if not readme:
                    continue
                cases.append(
                    BenchmarkCase(
                        resource_type="model",
                        resource_id=mid,
                        readme_content=readme,
                        structured_entity=props,
                        baseline_relations=rel_map.get(mid, []),
                    )
                )
                kept += 1
                if len(cases) >= required:
                    break
            logger.info(
                f"[baseline][neo4j] 批次完成: ids={len(batch_ids)}, "
                f"with_node={len(model_props)}, kept={kept}, cases_now={len(cases)}/{required}"
            )
            if (resource_ids or using_hit_file) and not ids_source:
                break
            if not resource_ids and not using_hit_file and len(batch_ids) < read_batch_size:
                break
        if not cases:
            raise ValueError("Neo4j baseline 未命中可用模型，或本地 README 全缺失")
        return cases[:required]

    if baseline_source != "cached":
        raise ValueError(
            f"不支持的 baseline_source={baseline_source!r}；请使用 neo4j 或 cached。"
            "已移除 live 当场结构化抽取路径。"
        )

    benchmark_dir = Path(benchmark_dir)
    models_path = benchmark_dir / models_json
    if not models_path.is_file():
        raise FileNotFoundError(f"未找到模型文件: {models_path}")

    models = load_json(models_path)
    rel_payload = _load_relations_payload(benchmark_dir, relations_json)
    relation_idx = build_relation_index(rel_payload)

    candidates: list[BenchmarkCase] = []
    for row in models:
        mid = row.get("model_id")
        if not isinstance(mid, str) or not mid.strip():
            continue
        readme = resolve_readme_content(
            mid,
            row,
            readme_field=readme_field,
            readme_dir=readme_dir,
            readme_filename_style=readme_filename_style,
        )
        if not readme:
            continue
        if resource_ids and mid not in resource_ids:
            continue
        baseline_rels = _relations_for_resource(relation_idx, "model", mid)
        candidates.append(
            BenchmarkCase(
                resource_type="model",
                resource_id=mid,
                readme_content=readme,
                structured_entity=row,
                baseline_relations=baseline_rels,
            )
        )

    if not candidates:
        hint = (
            f"models.json 字段 `{readme_field}`"
            + (f" 或 readme_dir={readme_dir}" if readme_dir else "")
        )
        raise ValueError(f"{models_path} 中没有可用于评测的 README（请检查 {hint}）")

    if resource_ids:
        return candidates[:max_cases]

    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[: max(1, max_cases)]


def cases_to_audit_entries(
    extractions: list[tuple[BenchmarkCase, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """将抽取结果转为 compare 所需的 audit entry 格式。"""
    entries: list[dict[str, Any]] = []
    for case, extracted in extractions:
        entries.append(
            {
                "resource_type": case.resource_type,
                "resource_id": case.resource_id,
                "relations_readme_extracted": extracted.get("relations") or [],
                "attributes_readme_extracted": extracted.get("attributes") or [],
            }
        )
    return entries


def baseline_filled_attribute_names(entity: dict[str, Any], resource_type: str) -> set[str]:
    exclude = MODEL_EXCLUDE_FOR_BASELINE if resource_type == "model" else frozenset(
        {"readme_content", "card_data_raw", "siblings_preview"}
    )
    filled, _ = entity_attribute_snapshot_from_dict(entity, set(exclude))
    return set(filled.keys())
