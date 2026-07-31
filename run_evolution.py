#!/usr/bin/env python3
"""
README 抽取自进化实验入口。

示例：
  py -3 run_evolution.py --benchmark-dir data/processed_hubstats --generations 3 \\
    --test-max-cases 10 --test-seed 42 --run-max-cases 15 --run-seed 1042 \\
    --final-test-max-cases 10

仅验证集评测（不进化）：
  py -3 run_evolution.py --test-only --test-max-cases 10

指定用例列表文件（JSON 数组或 {"resource_ids": [...]}）：
  py -3 run_evolution.py --test-cases-file data/splits/validation_ids.json \\
    --run-cases-file data/splits/evolution_pool_ids.json --generations 2

仅进化（基于已有 test_report.json）：
  py -3 run_evolution.py --evolve-only --test-report path/to/test_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# 项目根目录
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")
os.environ.setdefault("MULTI_AGENT_LLM_TIMEOUT_SEC", "480")

from evomodelkg.config import EvolutionConfig
from evomodelkg.evolve_agent import EvolveAgent
from evomodelkg.orchestrator import EvolutionOrchestrator
from evomodelkg.prompt_store import PromptStore
from evomodelkg.test_agent import TestAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="README 抽取自进化（验证/训练/最终测试三分）")
    p.add_argument(
        "--prompts-dir",
        type=str,
        default=None,
        help="提示词目录（默认 prompts）",
    )
    p.add_argument(
        "--benchmark-dir",
        type=str,
        default="data/processed_hubstats",
        help="含 models.json 与 relations 的 processed 目录",
    )
    p.add_argument("--models-json", type=str, default="models.json")
    p.add_argument(
        "--relations-json",
        type=str,
        default=None,
        help="relations 文件名；默认自动合并 relations_batch_*.json 或 relations.json",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=15,
        help="未指定 --test-max-cases/--run-max-cases 时，验证集与训练/进化集共用的默认上限",
    )
    p.add_argument("--test-max-cases", type=int, default=None, help="验证集样本数（每代固定评分/版本选择；兼容旧参数名）")
    p.add_argument("--run-max-cases", type=int, default=None, help="训练/进化集样本数（每代反馈，可轮换）")
    p.add_argument("--final-test-max-cases", type=int, default=None, help="最终测试集样本数（只在选出最优版本后评测）")
    p.add_argument("--test-seed", type=int, default=None, help="验证集随机抽样种子，默认 --seed")
    p.add_argument("--run-seed", type=int, default=None, help="训练/进化集随机抽样种子，默认 seed+1000")
    p.add_argument("--final-test-seed", type=int, default=None, help="最终测试集随机抽样种子，默认 seed+2000")
    p.add_argument("--test-resource-ids", type=str, default=None, help="验证集 model_id，逗号分隔（兼容旧参数名）")
    p.add_argument("--run-resource-ids", type=str, default=None, help="训练/进化集 model_id，逗号分隔")
    p.add_argument("--final-test-resource-ids", type=str, default=None, help="最终测试集 model_id，逗号分隔")
    p.add_argument("--test-cases-file", type=str, default=None, help="验证集 model_id 列表 JSON（兼容旧参数名）")
    p.add_argument("--run-cases-file", type=str, default=None, help="训练/进化集 model_id 列表 JSON")
    p.add_argument("--final-test-cases-file", type=str, default=None, help="最终测试集 model_id 列表 JSON")
    p.add_argument("--generations", type=int, default=3, help="最大进化代数；未开启 early stopping 时就是固定轮数")
    p.add_argument("--cases-per-generation", type=int, default=None, help="仅覆盖运行集样本数")
    p.add_argument("--seed", type=int, default=42, help="验证集默认抽样种子")
    p.add_argument("--runs-dir", type=str, default=None)
    p.add_argument("--test-only", action="store_true", help="只跑验证集评测（兼容旧参数名）")
    p.add_argument("--evolve-only", action="store_true", help="只跑进化 agent")
    p.add_argument("--test-report", type=str, default=None, help="evolve-only 时使用的旧测试/验证报告")
    p.add_argument("--generation", type=int, default=1, help="evolve-only 时标记的代数")
    p.add_argument(
        "--resource-ids",
        type=str,
        default=None,
        help="逗号分隔 model_id（等同 --test-resource-ids，兼容旧参数）",
    )
    p.add_argument("--extract-model", type=str, default=None, help="抽取用 LLM 模型名")
    p.add_argument("--evolve-model", type=str, default=None, help="进化用 LLM 模型名")
    p.add_argument("--extract-api-key", type=str, default=None, help="抽取 agent 使用的 OpenAI-compatible API key")
    p.add_argument("--extract-base-url", type=str, default=None, help="抽取 agent 使用的 OpenAI-compatible base URL")
    p.add_argument("--evolve-api-key", type=str, default=None, help="进化 agent 使用的 OpenAI-compatible API key")
    p.add_argument("--evolve-base-url", type=str, default=None, help="进化 agent 使用的 OpenAI-compatible base URL")
    p.add_argument("--resume-run-dir", type=str, default=None, help="从已有 runs/run_xxx 目录断点续跑")
    p.add_argument("--max-evolve-tool-rounds", type=int, default=8)
    p.add_argument("--concurrency", type=int, default=2, help="并行抽取 worker 数")
    p.add_argument(
        "--evaluation-checkpoint-size",
        type=int,
        default=10,
        help="评估时每完成多少个样本就重算累计指标并写入断点（默认 10）",
    )
    p.add_argument("--extract-context-window", type=int, default=None, help="抽取模型最大上下文 token；README 超出时自动分块")
    p.add_argument("--extract-output-token-budget", type=int, default=3072, help="为抽取 JSON 输出预留的 token 数")
    p.add_argument("--extract-context-safety-tokens", type=int, default=768, help="上下文预算安全余量 token")
    p.add_argument("--extract-chunk-overlap-tokens", type=int, default=256, help="README 分块之间的重叠 token 数")
    p.add_argument(
        "--sample-timeout-sec",
        type=float,
        default=None,
        help="单个 README 样本抽取的硬超时；默认 SELF_EVOLVE_SAMPLE_TIMEOUT_SEC 或 600 秒",
    )
    p.add_argument(
        "--early-stopping",
        action="store_true",
        help="启用基于验证集 composite 的早停；--generations 作为最大代数",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="验证集 composite 连续多少代未明显提升后停止",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
        help="认为验证集 composite 有提升所需的最小增量",
    )
    p.add_argument(
        "--readme-dir",
        type=str,
        default=None,
        help="本地 README 目录；models.json 无正文时按 org_repo.md / org__repo.md 回退",
    )
    p.add_argument(
        "--readme-field",
        type=str,
        default="readme_content",
        help="models.json 中 README 字段名（默认 readme_content）",
    )
    p.add_argument(
        "--readme-filename-style",
        type=str,
        choices=["auto", "underscore", "double_underscore"],
        default="auto",
        help="readme-dir 下文件名风格（与 model_crawler readme_local_dir 一致可用 underscore）",
    )
    p.add_argument(
        "--hubstats-dir",
        type=str,
        default="dataset_hub_stats",
        help="HubStats parquet 目录（live 结构化标答，需含 models.parquet）",
    )
    p.add_argument(
        "--use-cached-relations",
        action="store_true",
        help="使用 processed 目录现成 relations（等同 --baseline-source cached）",
    )
    p.add_argument(
        "--baseline-source",
        type=str,
        choices=["neo4j", "cached"],
        default="neo4j",
        help="评测基线来源：neo4j（默认）或 cached(processed 现成文件)",
    )
    p.add_argument("--neo4j-uri", type=str, default=None, help="Neo4j bolt URI，例如 bolt://192.168.15.101:17687")
    p.add_argument("--neo4j-user", type=str, default=None, help="Neo4j 用户名")
    p.add_argument("--neo4j-password", type=str, default=None, help="Neo4j 密码")
    p.add_argument("--neo4j-database", type=str, default=None, help="Neo4j database 名称")
    p.add_argument(
        "--readme-hit-ids-file",
        type=str,
        default="data/splits/candidate_pool_15000.json",
        help="预筛选的 README 命中 model_id 列表（JSON 数组）；neo4j 模式优先使用",
    )
    p.add_argument(
        "--readme-local-only",
        action="store_true",
        help="README 仅从 --readme-dir 读取，不联网下载",
    )
    p.add_argument(
        "--test-start-index",
        type=int,
        default=0,
        help="验证集起始偏移（按 readme_hit_ids/models.parquet 顺序，兼容旧参数名）",
    )
    p.add_argument(
        "--run-start-index",
        type=int,
        default=20,
        help="训练/进化集起始偏移（按 readme_hit_ids/models.parquet 顺序，默认与验证集错开）",
    )
    p.add_argument(
        "--final-test-start-index",
        type=int,
        default=None,
        help="最终测试集起始偏移；默认=test-start-index + 验证集样本数",
    )
    p.add_argument(
        "--parquet-read-batch-size",
        type=int,
        default=20,
        help="按顺序读取 parquet 时，每批读取并处理的模型数",
    )
    p.add_argument(
        "--no-rotate-evolve-cases",
        action="store_true",
        help="禁用每代轮换进化集（固定 run_start_index）",
    )
    p.add_argument(
        "--evolve-case-stride",
        type=int,
        default=None,
        help="每代进化集起始偏移步长，默认等于 run-max-cases",
    )
    p.add_argument(
        "--no-version-selector",
        action="store_true",
        help="禁用周期性/末代按综合分选最优 prompt（始终保留最后一代）",
    )
    p.add_argument(
        "--best-reload-interval",
        type=int,
        default=3,
        help="每 N 代后回灌截至当前综合分最高的 prompt 作为下一代初始；0 表示只在末代选择",
    )
    p.add_argument(
        "--allow-parquet-fallback",
        action="store_true",
        help="无 readme_hit_ids 时允许回退扫 parquet（默认必须使用 hit 文件）",
    )
    p.add_argument(
        "--evolve-report",
        type=str,
        default=None,
        help="evolve-only 时使用的进化集报告（默认 evolve_report.json）",
    )
    return p.parse_args()


def _split_ids(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    return ids or None


def build_config(args: argparse.Namespace) -> EvolutionConfig:
    test_ids = _split_ids(args.test_resource_ids) or _split_ids(args.resource_ids)
    run_ids = _split_ids(args.run_resource_ids)
    final_test_ids = _split_ids(args.final_test_resource_ids)
    cfg = EvolutionConfig(
        benchmark_dir=Path(args.benchmark_dir),
        models_json=args.models_json,
        relations_json=args.relations_json,
        max_cases=args.max_cases,
        test_max_cases=args.test_max_cases,
        run_max_cases=args.run_max_cases,
        final_test_max_cases=args.final_test_max_cases,
        test_seed=args.test_seed,
        run_seed=args.run_seed,
        final_test_seed=args.final_test_seed,
        test_resource_ids=test_ids,
        run_resource_ids=run_ids,
        final_test_resource_ids=final_test_ids,
        test_cases_file=Path(args.test_cases_file) if args.test_cases_file else None,
        run_cases_file=Path(args.run_cases_file) if args.run_cases_file else None,
        final_test_cases_file=(
            Path(args.final_test_cases_file) if args.final_test_cases_file else None
        ),
        generations=args.generations,
        cases_per_generation=args.cases_per_generation,
        seed=args.seed,
        max_evolve_tool_rounds=args.max_evolve_tool_rounds,
        concurrency=max(1, int(args.concurrency)),
        sample_timeout_sec=args.sample_timeout_sec,
        extract_context_window=args.extract_context_window,
        extract_output_token_budget=args.extract_output_token_budget,
        extract_context_safety_tokens=args.extract_context_safety_tokens,
        extract_chunk_overlap_tokens=args.extract_chunk_overlap_tokens,
        early_stopping=bool(args.early_stopping),
        early_stopping_patience=max(1, int(args.early_stopping_patience)),
        early_stopping_min_delta=max(0.0, float(args.early_stopping_min_delta)),
    )
    if args.prompts_dir:
        cfg.prompts_dir = Path(args.prompts_dir)
    if args.runs_dir:
        cfg.runs_dir = Path(args.runs_dir)
    if args.evolve_model:
        cfg.evolve_model = args.evolve_model
    if args.extract_model:
        cfg.extract_model = args.extract_model
    if args.extract_api_key:
        cfg.extract_api_key = args.extract_api_key
    if args.extract_base_url:
        cfg.extract_base_url = args.extract_base_url
    if args.evolve_api_key:
        cfg.evolve_api_key = args.evolve_api_key
    if args.evolve_base_url:
        cfg.evolve_base_url = args.evolve_base_url
    if args.readme_dir:
        cfg.readme_dir = Path(args.readme_dir)
    if args.readme_field:
        cfg.readme_field = args.readme_field
    cfg.readme_filename_style = args.readme_filename_style
    cfg.hubstats_dir = Path(args.hubstats_dir)
    if args.use_cached_relations:
        cfg.baseline_source = "cached"
    else:
        cfg.baseline_source = args.baseline_source
    cfg.readme_local_only = bool(args.readme_local_only)
    cfg.rotate_evolve_cases = not args.no_rotate_evolve_cases
    cfg.use_version_selector = not args.no_version_selector
    cfg.best_reload_interval = max(0, int(args.best_reload_interval))
    cfg.require_readme_hit_ids = not args.allow_parquet_fallback
    if args.evolve_case_stride is not None:
        cfg.evolve_case_stride = max(1, int(args.evolve_case_stride))
    if args.neo4j_uri:
        cfg.neo4j_uri = args.neo4j_uri
    if args.neo4j_user:
        cfg.neo4j_user = args.neo4j_user
    if args.neo4j_password:
        cfg.neo4j_password = args.neo4j_password
    if args.neo4j_database:
        cfg.neo4j_database = args.neo4j_database
    if args.readme_hit_ids_file:
        cfg.readme_hit_ids_file = Path(args.readme_hit_ids_file)
    cfg.test_start_index = max(0, int(args.test_start_index))
    cfg.run_start_index = max(0, int(args.run_start_index))
    if args.final_test_start_index is not None:
        cfg.final_test_start_index = max(0, int(args.final_test_start_index))
    cfg.parquet_read_batch_size = max(1, int(args.parquet_read_batch_size))
    cfg.evaluation_checkpoint_size = max(1, int(args.evaluation_checkpoint_size))
    cfg.resolve_paths()
    return cfg


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    args = parse_args()
    cfg = build_config(args)
    cfg.resolve_paths()

    llm = cfg.llm_env_summary()
    logger.info(
        f"LLM: extract={llm['extract_model']}, evolve={llm['evolve_model']}, "
        f"extract_base_url={llm['extract_base_url']}, evolve_base_url={llm['evolve_base_url']}, "
        f"extract_api_key={llm['extract_api_key']}, evolve_api_key={llm['evolve_api_key']}"
    )
    logger.info(
        f"Baseline: {cfg.baseline_source}, neo4j={cfg.neo4j_uri}/{cfg.neo4j_database}, "
        f"rotate_evolve={cfg.rotate_evolve_cases}, version_selector={cfg.use_version_selector}, "
        f"best_reload_interval={cfg.resolved_best_reload_interval()}"
    )

    test_resource_ids = _split_ids(args.test_resource_ids) or _split_ids(args.resource_ids)
    run_resource_ids = _split_ids(args.run_resource_ids)

    if args.evolve_only:
        report_path = args.evolve_report or args.test_report
        if not report_path:
            raise SystemExit("--evolve-only 需要 --evolve-report 或 --test-report")
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        store = PromptStore(cfg.prompts_dir)
        agent = EvolveAgent(
            store,
            temperature=cfg.evolve_temperature,
            model=cfg.resolved_evolve_model(),
            api_key=cfg.evolve_api_key,
            base_url=cfg.evolve_base_url,
            max_tool_rounds=cfg.max_evolve_tool_rounds,
        )
        out_dir = Path(report_path).parent
        agent.run(report, generation=args.generation, run_dir=out_dir)
        return

    if args.test_only:
        agent = TestAgent(cfg)
        cases = agent.load_test_cases(resource_ids=test_resource_ids)
        run_dir = cfg.runs_dir / "test_only_latest"
        agent.run(
            cases,
            run_dir=run_dir,
            report_name="validation_report.json",
            dataset_split="validation",
            concurrency=cfg.concurrency,
        )
        return

    orch = EvolutionOrchestrator(cfg)
    if test_resource_ids:
        orch.config.test_resource_ids = test_resource_ids
        orch.config.test_max_cases = len(test_resource_ids)
    if run_resource_ids:
        orch.config.run_resource_ids = run_resource_ids
        orch.config.run_max_cases = len(run_resource_ids)
    orch.run(resume_run_dir=Path(args.resume_run_dir) if args.resume_run_dir else None)


if __name__ == "__main__":
    main()
