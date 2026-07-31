"""
Neo4j数据库管理模块
负责连接管理、Schema创建和CRUD操作
"""
from typing import Any, Optional

from loguru import logger
from neo4j import GraphDatabase, Session

# Neo4j Bolt / PackStream 仅支持 64 位有符号整数；超出会 OverflowError
NEO4J_INT64_MIN = -(2**63)
NEO4J_INT64_MAX = 2**63 - 1

class Neo4jManager:
    """Neo4j数据库管理器"""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"连接Neo4j: {uri}")

    def close(self):
        self.driver.close()

    def verify_connectivity(self):
        """验证连接"""
        self.driver.verify_connectivity()
        logger.info("Neo4j连接验证成功")

    def run_query(self, query: str, parameters: Optional[dict] = None) -> list[dict]:
        """执行Cypher查询"""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def run_write(self, query: str, parameters: Optional[dict] = None):
        """执行写入操作"""
        with self.driver.session(database=self.database) as session:
            session.execute_write(
                lambda tx: tx.run(query, parameters or {})
            )

    # ============================================================
    # Schema 创建
    # ============================================================

    def create_schema(self):
        """创建图谱Schema：约束和索引"""
        logger.info("创建Neo4j Schema...")

        constraints = [
            # 实体唯一性约束
            "CREATE CONSTRAINT model_id IF NOT EXISTS FOR (m:Model) REQUIRE m.model_id IS UNIQUE",
            "CREATE CONSTRAINT dataset_id IF NOT EXISTS FOR (d:Dataset) REQUIRE d.dataset_id IS UNIQUE",
            "CREATE CONSTRAINT org_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.org_id IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
            "CREATE CONSTRAINT arxiv_id IF NOT EXISTS FOR (a:Arxiv) REQUIRE a.arxiv_id IS UNIQUE",
            "CREATE CONSTRAINT license_id IF NOT EXISTS FOR (l:License) REQUIRE l.license_id IS UNIQUE",
            "CREATE CONSTRAINT tool_name IF NOT EXISTS FOR (t:Tool) REQUIRE t.tool_name IS UNIQUE",
            "CREATE CONSTRAINT space_id IF NOT EXISTS FOR (s:Space) REQUIRE s.space_id IS UNIQUE",
        ]

        indexes = [
            # 性能索引
            "CREATE INDEX model_author IF NOT EXISTS FOR (m:Model) ON (m.author)",
            "CREATE INDEX model_downloads IF NOT EXISTS FOR (m:Model) ON (m.downloads)",
            "CREATE INDEX model_sub_types IF NOT EXISTS FOR (m:Model) ON (m.model_sub_types)",
            "CREATE INDEX model_pipeline_tag IF NOT EXISTS FOR (m:Model) ON (m.pipeline_tag)",
            "CREATE INDEX model_architecture IF NOT EXISTS FOR (m:Model) ON (m.architecture)",
            "CREATE INDEX dataset_author IF NOT EXISTS FOR (d:Dataset) ON (d.author)",
            "CREATE INDEX dataset_downloads IF NOT EXISTS FOR (d:Dataset) ON (d.downloads)",
            "CREATE INDEX dataset_sub_types IF NOT EXISTS FOR (d:Dataset) ON (d.dataset_sub_types)",
            "CREATE INDEX org_type IF NOT EXISTS FOR (o:Organization) ON (o.org_type)",
        ]

        for stmt in constraints + indexes:
            try:
                self.run_write(stmt)
            except Exception as e:
                logger.warning(f"创建约束/索引失败: {stmt[:80]}... - {e}")

        logger.info("Schema创建完成")

    def clear_database(self):
        """清空数据库（慎用）"""
        logger.warning("正在清空Neo4j数据库...")
        self.run_write("MATCH (n) DETACH DELETE n")
        logger.info("数据库已清空")

    # ============================================================
    # 实体 CRUD
    # ============================================================

    def upsert_model(self, props: dict):
        """创建或更新模型节点；model_sub_types 为列表时可附加多个 Neo4j 标签"""
        sub_types = props.get("model_sub_types")
        if sub_types is None:
            legacy = props.get("model_sub_type")
            sub_types = [legacy] if legacy else ["Unknown"]
        if isinstance(sub_types, str):
            sub_types = [sub_types]
        label_names: list[str] = []
        for s in sub_types:
            s = str(s).strip()
            if not s or s == "Unknown":
                continue
            if all(c.isalnum() or c == "_" for c in s):
                label_names.append(s)
        extra_labels = ":".join(dict.fromkeys(label_names))
        query = """
        MERGE (m:Model {model_id: $model_id})
        SET m += $props
        """
        if extra_labels:
            query += f" SET m:{extra_labels}"
        clean_props = self._clean_props(props)
        self.run_write(query, {"model_id": props["model_id"], "props": clean_props})

    def upsert_dataset(self, props: dict):
        """创建或更新数据集节点；dataset_sub_types 为列表时可附加多个 Neo4j 标签"""
        sub_types = props.get("dataset_sub_types")
        if sub_types is None:
            legacy = props.get("dataset_sub_type")
            sub_types = [legacy] if legacy else ["Unknown"]
        if isinstance(sub_types, str):
            sub_types = [sub_types]
        label_names: list[str] = []
        for s in sub_types:
            s = str(s).strip()
            if not s or s == "Unknown":
                continue
            if all(c.isalnum() or c == "_" for c in s):
                label_names.append(s)
        extra_labels = ":".join(dict.fromkeys(label_names))
        query = """
        MERGE (d:Dataset {dataset_id: $dataset_id})
        SET d += $props
        """
        if extra_labels:
            query += f" SET d:{extra_labels}"
        clean_props = self._clean_props(props)
        self.run_write(query, {"dataset_id": props["dataset_id"], "props": clean_props})

    def upsert_organization(self, props: dict):
        """创建或更新组织节点"""
        org_type = props.get("org_type", "Unknown")
        query = f"""
        MERGE (o:Organization {{org_id: $org_id}})
        SET o += $props
        SET o:{org_type}
        """
        clean_props = self._clean_props(props)
        self.run_write(query, {"org_id": props["org_id"], "props": clean_props})

    def upsert_user(self, props: dict):
        """创建或更新用户节点"""
        query = """
        MERGE (u:User {user_id: $user_id})
        SET u += $props
        """
        clean_props = self._clean_props(props)
        self.run_write(query, {"user_id": props["user_id"], "props": clean_props})

    def upsert_arxiv(self, props: dict):
        """创建或更新论文节点"""
        query = """
        MERGE (a:Arxiv {arxiv_id: $arxiv_id})
        SET a += $props
        """
        clean_props = self._clean_props(props)
        self.run_write(query, {"arxiv_id": props["arxiv_id"], "props": clean_props})

    def upsert_license(self, props: dict):
        """创建或更新许可证节点"""
        query = """
        MERGE (l:License {license_id: $license_id})
        SET l += $props
        """
        clean_props = self._clean_props(props)
        self.run_write(query, {"license_id": props["license_id"], "props": clean_props})

    def upsert_simple_entity(self, label: str, id_field: str, props: dict):
        """创建或更新简单实体（Language, Architecture, ModelType等）"""
        query = f"""
        MERGE (n:{label} {{{id_field}: $id_val}})
        SET n += $props
        """
        clean_props = self._clean_props(props)
        self.run_write(query, {"id_val": props[id_field], "props": clean_props})

    # ============================================================
    # 关系 CRUD
    # ============================================================

    def create_relationship(
        self,
        source_label: str,
        source_id_field: str,
        source_id: str,
        target_label: str,
        target_id_field: str,
        target_id: str,
        rel_type: str,
        rel_props: Optional[dict] = None,
    ):
        """创建关系"""
        props_clause = ""
        if rel_props:
            clean_props = self._clean_props(rel_props)
            props_clause = "SET r += $rel_props"
        else:
            clean_props = {}

        query = f"""
        MATCH (a:{source_label} {{{source_id_field}: $source_id}})
        MATCH (b:{target_label} {{{target_id_field}: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        {props_clause}
        """
        params = {
            "source_id": source_id,
            "target_id": target_id,
        }
        if rel_props:
            params["rel_props"] = clean_props
        self.run_write(query, params)

    def create_relationship_with_auto_node(
        self,
        source_label: str,
        source_id_field: str,
        source_id: str,
        target_label: str,
        target_id_field: str,
        target_id: str,
        rel_type: str,
        rel_props: Optional[dict] = None,
    ):
        """
        创建关系，如果目标节点不存在则自动创建
        用于处理引用了HuggingFace上不存在的模型/数据集的情况
        """
        props_clause = ""
        if rel_props:
            clean_props = self._clean_props(rel_props)
            props_clause = "SET r += $rel_props"
        else:
            clean_props = {}

        query = f"""
        MERGE (a:{source_label} {{{source_id_field}: $source_id}})
        MERGE (b:{target_label} {{{target_id_field}: $target_id}})
        MERGE (a)-[r:{rel_type}]->(b)
        {props_clause}
        """
        params = {
            "source_id": source_id,
            "target_id": target_id,
        }
        if rel_props:
            params["rel_props"] = clean_props
        self.run_write(query, params)

    # ============================================================
    # 批量操作
    # ============================================================

    def batch_upsert_nodes(self, label: str, id_field: str, nodes: list[dict], batch_size: int = 500):
        """批量创建/更新节点"""
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            query = f"""
            UNWIND $batch AS props
            MERGE (n:{label} {{{id_field}: props.{id_field}}})
            SET n += props
            """
            clean_batch = [self._clean_props(n) for n in batch]
            self.run_write(query, {"batch": clean_batch})

    def batch_create_relationships(
        self,
        source_label: str,
        source_id_field: str,
        target_label: str,
        target_id_field: str,
        rel_type: str,
        relationships: list[dict],
        batch_size: int = 500,
    ):
        """批量创建关系"""
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            query = f"""
            UNWIND $batch AS rel
            MATCH (a:{source_label} {{{source_id_field}: rel.source_id}})
            MATCH (b:{target_label} {{{target_id_field}: rel.target_id}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += rel.props
            """
            formatted_batch = []
            for r in batch:
                props = {k: v for k, v in r.items()
                         if k not in ("source_id", "target_id") and v is not None}
                props = self._clean_props(props)
                formatted_batch.append({
                    "source_id": r["source_id"],
                    "target_id": r["target_id"],
                    "props": props,
                })
            self.run_write(query, {"batch": formatted_batch})

    # ============================================================
    # 统计查询
    # ============================================================

    def get_stats(self) -> dict:
        """获取图谱统计信息"""
        stats = {}

        # 节点统计
        node_query = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
        """
        results = self.run_query(node_query)
        stats["nodes"] = {r["label"]: r["count"] for r in results}

        # 关系统计
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
        """
        results = self.run_query(rel_query)
        stats["relationships"] = {r["rel_type"]: r["count"] for r in results}

        # 总计
        stats["total_nodes"] = sum(stats["nodes"].values())
        stats["total_relationships"] = sum(stats["relationships"].values())

        return stats

    # ============================================================
    # 辅助方法
    # ============================================================
    def _sanitize_scalar_for_bolt(self, key: str, v: Any) -> Any:
        """将标量转为 Bolt 可序列化形式；bool 须在 int 之前判断。"""
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            if NEO4J_INT64_MIN <= v <= NEO4J_INT64_MAX:
                return v
            logger.warning(
                f"属性 {key!r} 的整数值 {v} 超出 Neo4j int64 范围 "
                f"[{NEO4J_INT64_MIN}, {NEO4J_INT64_MAX}]，已钳位（常见于 JSON/浮点精度）"
            )
            return NEO4J_INT64_MAX if v > 0 else NEO4J_INT64_MIN
        if isinstance(v, float):
            return v
        if isinstance(v, str):
            return v
        return v

    def _clean_props(self, props: dict) -> dict:
        """
        清理属性字典，使其适合Neo4j存储
        - 移除None值
        - 将复杂类型转为字符串
        - 移除过大的字段
        - 整型钳位到 int64，避免 PackStream OverflowError
        """
        import json

        clean = {}
        skip_fields = {
            "config_json", "adapter_config", "tokenizer_config",
            "readme_content", "siblings", "card_data_raw",
            "liked_repos", "eval_results", "evidence_sources",
            "model_sub_type",
            "dataset_sub_type",
            "linked_spaces",
        }

        for k, v in props.items():
            if k in skip_fields:
                continue
            if v is None:
                continue
            if isinstance(v, bool):
                clean[k] = v
            elif isinstance(v, int):
                clean[k] = self._sanitize_scalar_for_bolt(k, v)
            elif isinstance(v, (str, float)):
                clean[k] = v
            elif isinstance(v, (list, tuple)):
                # Neo4j支持简单类型的列表
                if all(isinstance(item, (str, int, float, bool)) for item in v):
                    out_list = []
                    for i, item in enumerate(v):
                        if isinstance(item, bool):
                            out_list.append(item)
                        elif isinstance(item, int):
                            out_list.append(
                                self._sanitize_scalar_for_bolt(f"{k}[{i}]", item)
                            )
                        else:
                            out_list.append(item)
                    clean[k] = out_list
                else:
                    clean[k] = json.dumps(v, ensure_ascii=False, default=str)
            elif isinstance(v, dict):
                clean[k] = json.dumps(v, ensure_ascii=False, default=str)
            else:
                clean[k] = str(v)

        return clean