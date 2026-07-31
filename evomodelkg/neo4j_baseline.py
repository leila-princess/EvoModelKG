from __future__ import annotations

from typing import Any

from loguru import logger

from evomodelkg.graph.neo4j_manager import Neo4jManager


RELATION_TYPES = [
    "DERIVED_FROM",
    "TRAINED_ON",
    "EVALUATED_ON",
    "GENERATED",
    "ANNOTATED",
    "SOURCE_DATASET",
    "MENTIONS_ARXIV",
    "USES_TOOL",
    "LICENSED_UNDER",
]


def fetch_neo4j_baseline(
    *,
    model_ids: list[str],
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """
    从 Neo4j 读取模型属性与关系，作为 README 评测 baseline。
    返回 (model_props_map, relations_by_source)。
    """
    if not model_ids:
        return {}, {}

    db = Neo4jManager(uri=uri, user=user, password=password, database=database)
    try:
        node_query = """
        MATCH (m:Model)
        WHERE m.model_id IN $ids
        RETURN m.model_id AS model_id, properties(m) AS props
        """
        rel_query = """
        MATCH (m:Model)-[r]->(t)
        WHERE m.model_id IN $ids AND type(r) IN $rel_types
        RETURN
            m.model_id AS source_id,
            type(r) AS relation_type,
            CASE
              WHEN t.model_id IS NOT NULL THEN t.model_id
              WHEN t.dataset_id IS NOT NULL THEN t.dataset_id
              WHEN t.arxiv_id IS NOT NULL THEN t.arxiv_id
              WHEN t.tool_name IS NOT NULL THEN t.tool_name
              WHEN t.license_id IS NOT NULL THEN t.license_id
              ELSE ''
            END AS target_id,
            properties(r) AS props
        """
        node_rows = db.run_query(node_query, {"ids": model_ids})
        rel_rows = db.run_query(rel_query, {"ids": model_ids, "rel_types": RELATION_TYPES})
    finally:
        db.close()

    model_props: dict[str, dict[str, Any]] = {}
    for row in node_rows:
        mid = str(row.get("model_id") or "").strip()
        props = row.get("props") or {}
        if not mid or not isinstance(props, dict):
            continue
        props = dict(props)
        props["model_id"] = mid
        model_props[mid] = props

    rel_map: dict[str, list[dict[str, Any]]] = {}
    for row in rel_rows:
        sid = str(row.get("source_id") or "").strip()
        tid = str(row.get("target_id") or "").strip()
        rt = str(row.get("relation_type") or "").upper().strip()
        if not sid or not tid or not rt:
            continue
        props = row.get("props") or {}
        rel = {
            "relation_type": rt,
            "source_id": sid,
            "target_id": tid,
        }
        if isinstance(props, dict):
            rel.update(props)
        rel_map.setdefault(sid, []).append(rel)

    logger.info(
        f"[baseline][neo4j] 读取完成: models={len(model_props)}, "
        f"relations={sum(len(v) for v in rel_map.values())}"
    )
    return model_props, rel_map

