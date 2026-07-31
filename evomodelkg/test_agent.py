from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from loguru import logger

from evomodelkg.benchmark import (
    BenchmarkCase,
    load_benchmark_cases,
    load_resource_ids_from_file,
)
from evomodelkg.config import EvolutionConfig
from evomodelkg.extractor import ReadmeExtractor
from evomodelkg.metrics import evaluate_extractions, estimate_llm_calls_per_case
from evomodelkg.prompt_store import PromptStore
from evomodelkg.readme_observer import (
    select_interesting_cases,
    summarize_case_for_observation,
)


def _summarize_input_cost(extractions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        e.get("_meta")
        for e in extractions
        if isinstance(e.get("_meta"), dict)
    ]
    before_chars = sum(int(r.get("readme_chars_before_preprocess") or 0) for r in rows)
    after_chars = sum(int(r.get("readme_chars_after_preprocess") or 0) for r in rows)
    before_tokens = sum(
        int(r.get("estimated_readme_tokens_before_preprocess") or 0) for r in rows
    )
    after_tokens = sum(
        int(r.get("estimated_readme_tokens_after_preprocess") or 0) for r in rows
    )
    chunk_counts = [int(r.get("readme_chunk_count") or 1) for r in rows]
    chunked_cases = sum(1 for count in chunk_counts if count > 1)
    reduction = (
        max(0, before_chars - after_chars) / before_chars
        if before_chars > 0
        else 0.0
    )
    return {
        "case_count_with_input_cost": len(rows),
        "readme_chars_before_preprocess": before_chars,
        "readme_chars_after_preprocess": after_chars,
        "readme_chars_reduction_rate": round(reduction, 6),
        "estimated_readme_tokens_before_preprocess": before_tokens,
        "estimated_readme_tokens_after_preprocess": after_tokens,
        "estimated_readme_tokens_saved": max(0, before_tokens - after_tokens),
        "readme_chunked_case_count": chunked_cases,
        "max_readme_chunk_count": max(chunk_counts) if chunk_counts else 0,
    }


def _is_failed_extraction(extraction: dict[str, Any]) -> bool:
    """请求失败、超时等空结果不参与评分，但仍保留在产物中供重试和审计。"""
    meta = extraction.get("_meta")
    return bool(
        isinstance(meta, dict)
        and (
            meta.get("extraction_error_type")
            or meta.get("extraction_error")
        )
    )


def _build_mismatch_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    mismatches = payload.get("mismatches") or []
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in mismatches:
        rid = str(row.get("resource_id") or "unknown")
        by_case.setdefault(rid, []).append(row)
    return {
        "meta": payload.get("meta") or {},
        "summary": payload.get("summary") or {},
        "compare_summary_lines": payload.get("compare_summary_lines") or [],
        "mismatch_count": len(mismatches),
        "case_count_with_mismatches": len(by_case),
        "mismatches_by_case": by_case,
        "mismatches": mismatches,
        "interesting_cases": payload.get("interesting_cases") or [],
    }


def _md_value(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").strip()


def _format_mismatch_markdown(artifact: dict[str, Any]) -> str:
    meta = artifact.get("meta") or {}
    lines = [
        f"# Mismatches - {meta.get('dataset_split', 'unknown')}",
        "",
        f"- created_at: `{meta.get('created_at', '')}`",
        f"- case_count: `{meta.get('case_count', 0)}`",
        f"- mismatch_count: `{artifact.get('mismatch_count', 0)}`",
        f"- case_count_with_mismatches: `{artifact.get('case_count_with_mismatches', 0)}`",
        "",
        "## Compare Summary",
        "",
    ]
    compare_lines = artifact.get("compare_summary_lines") or []
    if compare_lines:
        lines.extend(f"- {_md_value(row)}" for row in compare_lines)
    else:
        lines.append("- (none)")
    lines.extend(["", "## Mismatches", ""])
    by_case = artifact.get("mismatches_by_case") or {}
    if not by_case:
        lines.append("No mismatches collected.")
        return "\n".join(lines) + "\n"
    for rid, rows in by_case.items():
        lines.extend([f"### {rid}", ""])
        for i, row in enumerate(rows, start=1):
            lines.append(f"{i}. `{_md_value(row.get('kind') or row.get('type'))}`")
            for key in (
                "attribute",
                "readme_value",
                "structured_value",
                "evidence_span",
                "normalization_note",
            ):
                value = row.get(key)
                if value not in (None, ""):
                    lines.append(f"   - {key}: {_md_value(value)}")
            lines.append("")
    return "\n".join(lines)


class TestAgent:
    """在固定提示词/工作流下跑评测集，输出准确性+完整性报告。"""

    def __init__(self, config: EvolutionConfig):
        config.resolve_paths()
        self.config = config
        self.store = PromptStore(config.prompts_dir)
        self.extractor = ReadmeExtractor(
            self.store,
            temperature=config.extract_temperature,
            model=config.resolved_extract_model(),
            api_key=config.extract_api_key,
            base_url=config.extract_base_url,
            context_window=config.resolved_extract_context_window(),
            output_token_budget=config.resolved_extract_output_token_budget(),
            context_safety_tokens=config.resolved_extract_context_safety_tokens(),
            chunk_overlap_tokens=config.resolved_extract_chunk_overlap_tokens(),
        )

    def _resolve_resource_ids(
        self,
        *,
        split: str,
        override_ids: list[str] | None,
    ) -> list[str] | None:
        if override_ids:
            return override_ids
        if split in {"test", "validation"}:
            ids = self.config.resolved_test_resource_ids()
            path = self.config.test_cases_file
        elif split == "final_test":
            ids = self.config.resolved_final_test_resource_ids()
            path = self.config.final_test_cases_file
        else:
            ids = self.config.resolved_run_resource_ids()
            path = self.config.run_cases_file
        if ids:
            return ids
        if path is not None:
            return load_resource_ids_from_file(path)
        return None

    def _benchmark_kwargs(self, *, start_index: int) -> dict:
        return {
            "readme_dir": self.config.readme_dir,
            "readme_field": self.config.readme_field,
            "readme_filename_style": self.config.readme_filename_style,
            "baseline_source": self.config.baseline_source,
            "hubstats_dir": self.config.hubstats_dir,
            "neo4j_uri": self.config.neo4j_uri,
            "neo4j_user": self.config.neo4j_user,
            "neo4j_password": self.config.neo4j_password,
            "neo4j_database": self.config.neo4j_database,
            "readme_hit_ids_file": self.config.readme_hit_ids_file,
            "require_readme_hit_ids": self.config.require_readme_hit_ids,
            "start_index": start_index,
            "read_batch_size": max(1, int(self.config.parquet_read_batch_size)),
        }

    def load_test_cases(self, resource_ids: list[str] | None = None) -> list[BenchmarkCase]:
        logger.info(
            f"评测标答: Neo4j（uri={self.config.neo4j_uri}, db={self.config.neo4j_database}）"
            if self.config.baseline_source == "neo4j"
            else f"评测标答: cached（{self.config.benchmark_dir}）"
        )
        return load_benchmark_cases(
            self.config.benchmark_dir,
            models_json=self.config.models_json,
            relations_json=self.config.relations_json,
            max_cases=self.config.resolved_test_max_cases(),
            seed=self.config.resolved_test_seed(),
            resource_ids=self._resolve_resource_ids(split="test", override_ids=resource_ids),
            **self._benchmark_kwargs(start_index=max(0, int(self.config.test_start_index))),
        )

    def load_validation_cases(self, resource_ids: list[str] | None = None) -> list[BenchmarkCase]:
        """固定验证集：每代评分和版本选择使用。"""
        return self.load_test_cases(resource_ids=resource_ids)

    def load_final_test_cases(self, resource_ids: list[str] | None = None) -> list[BenchmarkCase]:
        """最终测试集：只在自进化结束、选出最优版本后评测一次。"""
        return load_benchmark_cases(
            self.config.benchmark_dir,
            models_json=self.config.models_json,
            relations_json=self.config.relations_json,
            max_cases=self.config.resolved_final_test_max_cases(),
            seed=self.config.resolved_final_test_seed(),
            resource_ids=self._resolve_resource_ids(
                split="final_test",
                override_ids=resource_ids,
            ),
            **self._benchmark_kwargs(
                start_index=self.config.resolved_final_test_start_index()
            ),
        )

    def load_evolve_cases(
        self,
        *,
        start_index: int,
        resource_ids: list[str] | None = None,
    ) -> list[BenchmarkCase]:
        """进化反馈用例（每代可轮换 start_index）。"""
        return load_benchmark_cases(
            self.config.benchmark_dir,
            models_json=self.config.models_json,
            relations_json=self.config.relations_json,
            max_cases=self.config.resolved_run_max_cases(),
            seed=self.config.resolved_run_seed(),
            resource_ids=self._resolve_resource_ids(split="run", override_ids=resource_ids),
            **self._benchmark_kwargs(start_index=max(0, int(start_index))),
        )

    def load_run_cases(self, resource_ids: list[str] | None = None) -> list[BenchmarkCase]:
        """兼容旧接口：等同 load_evolve_cases（固定 run_start_index）。"""
        return self.load_evolve_cases(
            start_index=max(0, int(self.config.run_start_index)),
            resource_ids=resource_ids,
        )

    def load_cases(self, resource_ids: list[str] | None = None) -> list[BenchmarkCase]:
        return self.load_test_cases(resource_ids=resource_ids)

    def run(
        self,
        cases: list[BenchmarkCase],
        *,
        run_dir: Path | None = None,
        report_name: str = "test_report.json",
        dataset_split: str = "test",
        concurrency: int = 2,
    ) -> dict[str, Any]:
        logger.info(f"TestAgent [{dataset_split}]: 评测 {len(cases)} 个样本")
        manifest = self.store.load_manifest()
        llm_per_case = estimate_llm_calls_per_case(manifest)
        sample_timeout_sec = self.config.resolved_sample_timeout_sec()
        checkpoint_size = max(1, int(self.config.evaluation_checkpoint_size))
        run_dir = Path(run_dir) if run_dir is not None else None
        extractions_path = (
            run_dir / f"{dataset_split}_extractions.json" if run_dir is not None else None
        )
        completed_by_id: dict[str, dict[str, Any]] = {}
        extract_seconds = 0.0

        if extractions_path is not None and extractions_path.is_file():
            try:
                saved = json.loads(extractions_path.read_text(encoding="utf-8"))
                saved_meta = saved.get("meta") or {}
                if (
                    saved_meta.get("dataset_split") == dataset_split
                    and saved_meta.get("manifest") == manifest
                ):
                    valid_ids = {c.resource_id for c in cases}
                    completed_by_id = {
                        str(row["resource_id"]): row["extraction"]
                        for row in saved.get("extractions") or []
                        if row.get("resource_id") in valid_ids
                        and isinstance(row.get("extraction"), dict)
                    }
                    extract_seconds = float(
                        (saved_meta.get("timing") or {}).get("extract_seconds") or 0.0
                    )
                    if completed_by_id:
                        logger.info(
                            f"TestAgent [{dataset_split}]: 从断点恢复 "
                            f"{len(completed_by_id)}/{len(cases)} 条"
                        )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                logger.warning(f"TestAgent: 忽略无法读取的评估断点 {extractions_path}: {exc}")

        def build_payload() -> tuple[dict[str, Any], dict[str, Any]]:
            completed_cases = [c for c in cases if c.resource_id in completed_by_id]
            extractions = [completed_by_id[c.resource_id] for c in completed_cases]
            scored_pairs = [
                (case, extraction)
                for case, extraction in zip(completed_cases, extractions)
                if not _is_failed_extraction(extraction)
            ]
            scored_cases = [case for case, _ in scored_pairs]
            scored_extractions = [extraction for _, extraction in scored_pairs]
            failed_extractions = [
                extraction
                for extraction in extractions
                if _is_failed_extraction(extraction)
            ]
            failed_case_ids = [
                case.resource_id
                for case, extraction in zip(completed_cases, extractions)
                if _is_failed_extraction(extraction)
            ]
            failure_types: dict[str, int] = {}
            for extraction in failed_extractions:
                error_type = str(
                    (extraction.get("_meta") or {}).get("extraction_error_type")
                    or "unknown"
                )
                failure_types[error_type] = failure_types.get(error_type, 0) + 1
            report = evaluate_extractions(scored_cases, scored_extractions)
            failed_case_count = len(failed_extractions)
            attempted_case_count = len(completed_cases)
            report["summary"]["evaluation_population"] = {
                "attempted_case_count": attempted_case_count,
                "scored_case_count": len(scored_cases),
                "failed_case_count": failed_case_count,
                "failure_rate": (
                    failed_case_count / attempted_case_count
                    if attempted_case_count
                    else 0.0
                ),
                "failure_types": failure_types,
                "failed_samples_excluded_from_scores": True,
            }
            timing = {
                "extract_seconds": round(extract_seconds, 3),
                "llm_calls_estimated": llm_per_case * len(completed_cases),
                "llm_calls_per_case": llm_per_case,
                "sample_timeout_sec": sample_timeout_sec,
            }
            input_cost = _summarize_input_cost(extractions)
            payload = {
                "meta": {
                    "kind": "self_evolve_test_report",
                    "dataset_split": dataset_split,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "case_count": len(completed_cases),
                    "total_case_count": len(cases),
                    "completed_case_count": len(completed_cases),
                    "scored_case_count": len(scored_cases),
                    "failed_case_count": failed_case_count,
                    "failure_rate": (
                        failed_case_count / attempted_case_count
                        if attempted_case_count
                        else 0.0
                    ),
                    "failed_case_ids": failed_case_ids,
                    "failure_types": failure_types,
                    "failed_samples_excluded_from_scores": True,
                    "is_complete": len(completed_cases) == len(cases),
                    "checkpoint_size": checkpoint_size,
                    "case_ids": [c.resource_id for c in completed_cases],
                    "all_case_ids": [c.resource_id for c in cases],
                    "manifest": manifest,
                    "timing": timing,
                    "input_cost": input_cost,
                },
                "summary": report["summary"],
                "compare_summary_lines": report["compare_summary_lines"],
                "accuracy_report": report["accuracy_report"],
                "completeness_report": report["completeness_report"],
                "completeness_fit_report": report.get("completeness_fit_report"),
                "mismatches": report["mismatches"],
                "case_observations": [
                    summarize_case_for_observation(c, e)
                    for c, e in zip(completed_cases, extractions)
                ],
                "interesting_cases": select_interesting_cases(
                    completed_cases, extractions
                ),
                "case_inputs": [
                    {"resource_id": c.resource_id, "readme_content": c.readme_content}
                    for c in completed_cases
                ],
                "extractions": [
                    {"resource_id": c.resource_id, "extraction": e}
                    for c, e in zip(completed_cases, extractions)
                ],
            }
            return payload, report

        def write_payload(payload: dict[str, Any]) -> None:
            if run_dir is None or extractions_path is None:
                return
            run_dir.mkdir(parents=True, exist_ok=True)
            out_path = run_dir / report_name
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            with extractions_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "meta": payload["meta"],
                        "case_inputs": payload["case_inputs"],
                        "extractions": payload["extractions"],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            mismatch_artifact = _build_mismatch_artifact(payload)
            mismatch_md_path = run_dir / f"{dataset_split}_mismatches.md"
            mismatch_md_path.write_text(
                _format_mismatch_markdown(mismatch_artifact),
                encoding="utf-8",
            )
            logger.info(
                f"TestAgent [{dataset_split}]: 检查点 "
                f"{payload['meta']['completed_case_count']}/"
                f"{payload['meta']['total_case_count']} 已写入 {out_path}"
            )

        remaining = [c for c in cases if c.resource_id not in completed_by_id]
        payload: dict[str, Any] | None = None
        report: dict[str, Any] | None = None
        for start in range(0, len(remaining), checkpoint_size):
            batch = remaining[start : start + checkpoint_size]
            triples = [(c.resource_type, c.resource_id, c.readme_content) for c in batch]
            t0 = perf_counter()
            batch_extractions = asyncio.run(
                self.extractor.extract_many_async(
                    triples,
                    concurrency=concurrency,
                    sample_timeout_sec=sample_timeout_sec,
                )
            )
            extract_seconds += perf_counter() - t0
            completed_by_id.update(
                {c.resource_id: e for c, e in zip(batch, batch_extractions)}
            )
            payload, report = build_payload()
            write_payload(payload)

        if payload is None or report is None:
            payload, report = build_payload()
            write_payload(payload)

        timing = payload["meta"]["timing"]
        input_cost = payload["meta"]["input_cost"]

        for line in report["compare_summary_lines"]:
            logger.info(f"  {line}")
        s = report["summary"]
        c = s.get("completeness", {})
        fit = s.get("completeness_fit") or {}
        logger.info(
            f"  [汇总-准确性] attr_acc={s['accuracy'].get('attribute_overlap_accuracy')}"
        )
        logger.info(
            f"  [汇总-完整度-fit] score={fit.get('score')}, "
            f"avg_attrs={c.get('avg_attributes_per_case')}, "
            f"耗时={timing['extract_seconds']}s, llm≈{timing['llm_calls_estimated']}, "
            f"readme压缩={input_cost['readme_chars_reduction_rate']:.1%}"
        )
        return payload
