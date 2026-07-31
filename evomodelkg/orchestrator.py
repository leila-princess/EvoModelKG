from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from evomodelkg.case_rotation import evolve_generation_label, evolve_start_index
from evomodelkg.config import EvolutionConfig
from evomodelkg.evolve_agent import EvolveAgent
from evomodelkg.prompt_store import PromptStore
from evomodelkg.test_agent import TestAgent
from evomodelkg.tool_store import ToolStore
from evomodelkg.version_registry import (
    load_registry,
    register_version,
    restore_best_prompts,
    select_best_version,
)


class EvolutionOrchestrator:
    """测试 → 进化 → 再测试；周期性按综合分回灌最优 prompt 版本。"""

    def __init__(self, config: EvolutionConfig):
        config.resolve_paths()
        self.config = config
        self.store = PromptStore(config.prompts_dir)
        self.tool_store = ToolStore()

    def _new_run_dir(self) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.runs_dir / f"run_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def _is_complete_report(cls, path: Path) -> bool:
        if not path.is_file():
            return False
        # 旧版报告只会在整批完成后写入，缺少该字段时按完整报告处理。
        return bool((cls._load_json(path).get("meta") or {}).get("is_complete", True))

    @staticmethod
    def _latest_generation_with_after(run_dir: Path) -> int:
        latest = 0
        for gen_dir in Path(run_dir).glob("gen[0-9][0-9][0-9]"):
            try:
                gen = int(gen_dir.name[3:])
            except ValueError:
                continue
            if (gen_dir / "prompts_after").is_dir() and (gen_dir / "tools_after").is_dir():
                latest = max(latest, gen)
        return latest

    @staticmethod
    def _early_state_from_versions(run_dir: Path, *, min_delta: float) -> tuple[float, int, int]:
        best = float("-inf")
        best_gen = 0
        wait = 0
        versions = sorted(load_registry(run_dir), key=lambda v: int(v.get("generation") or 0))
        for version in versions:
            gen = int(version.get("generation") or 0)
            composite = float(version.get("composite") or float("-inf"))
            if composite > best + min_delta:
                best = composite
                best_gen = gen
                wait = 0
            else:
                wait += 1
        return best, best_gen, wait

    def run(self, *, resume_run_dir: Path | None = None) -> dict[str, Any]:
        run_dir = Path(resume_run_dir).resolve() if resume_run_dir is not None else self._new_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        resume = resume_run_dir is not None
        logger.info(f"自进化 run 目录: {run_dir}" + (" (resume)" if resume else ""))
        llm = self.config.llm_env_summary()
        logger.info(
            f"LLM 配置: extract={llm['extract_model']}, evolve={llm['evolve_model']}, "
            f"extract_base_url={llm['extract_base_url']}, evolve_base_url={llm['evolve_base_url']}, "
            f"extract_api_key={llm['extract_api_key']}, evolve_api_key={llm['evolve_api_key']}"
        )

        gen0_prompts = run_dir / "gen000_prompts"
        if not gen0_prompts.exists():
            self.store.snapshot_to(gen0_prompts)
        gen0_tools = run_dir / "gen000_tools"
        if not gen0_tools.exists():
            self.tool_store.snapshot_to(gen0_tools)

        latest_after = self._latest_generation_with_after(run_dir) if resume else 0
        if latest_after > 0:
            latest_dir = run_dir / f"gen{latest_after:03d}"
            self.store.restore_from(latest_dir / "prompts_after")
            self.tool_store.restore_from(latest_dir / "tools_after")
            logger.info(f"Resume: 已恢复 gen{latest_after:03d} 的 prompts_after/tools_after")
        elif resume:
            if gen0_prompts.is_dir():
                self.store.restore_from(gen0_prompts)
            if gen0_tools.is_dir():
                self.tool_store.restore_from(gen0_tools)
            logger.info("Resume: 未发现已完成代，已恢复 gen000 初始 prompts/tools")

        test_agent = TestAgent(self.config)
        evolve_agent = EvolveAgent(
            self.store,
            temperature=self.config.evolve_temperature,
            model=self.config.resolved_evolve_model(),
            api_key=self.config.evolve_api_key,
            base_url=self.config.evolve_base_url,
            max_tool_rounds=self.config.max_evolve_tool_rounds,
        )

        validation_cases = test_agent.load_validation_cases()
        validation_ids = {c.resource_id for c in validation_cases}
        logger.info(
            f"固定验证集: {len(validation_cases)} 条 (start={self.config.test_start_index}, "
            f"用于每代版本评分); 训练/进化集每代轮换; 最终测试集只在选出最优版本后评测"
        )
        if self.config.early_stopping and not self.config.use_version_selector:
            logger.warning("已启用 early_stopping，但 --no-version-selector 会禁用 composite 版本评分；本次将按固定 generations 运行")

        summary_path = run_dir / "run_summary.json"
        previous_summary = self._load_json(summary_path) if resume and summary_path.is_file() else {}
        history: list[dict[str, Any]] = list(previous_summary.get("history") or [])
        last_validation_report: dict[str, Any] | None = None
        final_test_report: dict[str, Any] | None = None
        best_version: dict[str, Any] | None = None
        early_best_composite, early_best_generation, early_wait = self._early_state_from_versions(
            run_dir,
            min_delta=self.config.resolved_early_stopping_min_delta(),
        ) if resume else (float("-inf"), 0, 0)
        stopped_early = bool(
            resume
            and self.config.early_stopping
            and self.config.use_version_selector
            and early_wait >= self.config.resolved_early_stopping_patience()
        )
        if stopped_early:
            logger.info(
                "Resume: 已有版本历史满足 EarlyStopping "
                f"(wait={early_wait}/{self.config.resolved_early_stopping_patience()}, "
                f"best={early_best_composite:.4f}@gen{early_best_generation:03d})，"
                "跳过后续进化，直接进入最终测试"
            )

        generation_range = (
            range(0) if stopped_early
            else range(max(1, latest_after + 1), self.config.generations + 1)
        )
        for gen in generation_range:
            gen_dir = run_dir / f"gen{gen:03d}"
            gen_dir.mkdir(parents=True, exist_ok=True)
            prompts_before_dir = gen_dir / "prompts_before"
            tools_before_dir = gen_dir / "tools_before"
            if not prompts_before_dir.is_dir():
                self.store.snapshot_to(prompts_before_dir)
            if not tools_before_dir.is_dir():
                self.tool_store.snapshot_to(tools_before_dir)

            logger.info(f"========== Generation {gen}/{self.config.generations} ==========")

            validation_report_path = gen_dir / "validation_report.json"
            if resume and self._is_complete_report(validation_report_path):
                validation_report = self._load_json(validation_report_path)
                logger.info(f"Resume: 复用 {validation_report_path}")
            else:
                validation_report = test_agent.run(
                    validation_cases,
                    run_dir=gen_dir,
                    report_name="validation_report.json",
                    dataset_split="validation",
                    concurrency=self.config.concurrency,
                )
            last_validation_report = validation_report
            history.append(
                {
                    "generation": gen,
                    "phase": "validation",
                    "summary": validation_report.get("summary"),
                    "timing": validation_report.get("meta", {}).get("timing"),
                }
            )

            version_entry: dict[str, Any] | None = None
            if self.config.use_version_selector:
                version_id = f"gen{gen:03d}_before"
                version_entry = register_version(
                    run_dir,
                    version_id=version_id,
                    prompts_src=prompts_before_dir,
                    tools_src=tools_before_dir,
                    test_report=validation_report,
                    generation=gen,
                    manifest=self.store.load_manifest(),
                    extract_seconds=float(
                        (validation_report.get("meta") or {})
                        .get("timing", {})
                        .get("extract_seconds")
                        or 0
                    ),
                    weights=self.config.score_weights,
                    extra_meta={"phase": "before_evolve", "selection_split": "validation"},
                )

            if self.config.early_stopping and version_entry is not None:
                current_composite = float(version_entry.get("composite") or float("-inf"))
                min_delta = self.config.resolved_early_stopping_min_delta()
                if current_composite > early_best_composite + min_delta:
                    early_best_composite = current_composite
                    early_best_generation = gen
                    early_wait = 0
                    improved = True
                else:
                    early_wait += 1
                    improved = False
                patience = self.config.resolved_early_stopping_patience()
                history.append(
                    {
                        "generation": gen,
                        "phase": "early_stopping_check",
                        "current_composite": current_composite,
                        "best_composite": early_best_composite,
                        "best_generation": early_best_generation,
                        "wait": early_wait,
                        "patience": patience,
                        "min_delta": min_delta,
                        "improved": improved,
                    }
                )
                logger.info(
                    f"EarlyStopping: composite={current_composite:.4f}, "
                    f"best={early_best_composite:.4f}@gen{early_best_generation:03d}, "
                    f"wait={early_wait}/{patience}"
                )
                if early_wait >= patience:
                    stopped_early = True
                    if self.config.use_version_selector:
                        checkpoint_best = restore_best_prompts(
                            run_dir,
                            self.store.prompts_dir,
                            self.tool_store.tools_dir,
                        )
                        self.store.snapshot_to(gen_dir / "prompts_after")
                        self.tool_store.snapshot_to(gen_dir / "tools_after")
                        history.append(
                            {
                                "generation": gen,
                                "phase": "early_stop",
                                "selected_version_id": checkpoint_best.get("version_id"),
                                "composite": checkpoint_best.get("composite"),
                                "reason": (
                                    f"validation composite did not improve by "
                                    f"{min_delta} for {patience} consecutive generations"
                                ),
                            }
                        )
                    logger.info(
                        f"EarlyStopping 触发：验证集 composite 连续 {patience} 代未明显提升，停止于 gen{gen:03d}"
                    )
                    break

            evolve_index = evolve_start_index(self.config, gen)
            evolve_label = evolve_generation_label(gen, evolve_index)
            evolve_cases = test_agent.load_evolve_cases(start_index=evolve_index)
            evolve_ids = {c.resource_id for c in evolve_cases}
            overlap = validation_ids & evolve_ids
            if overlap:
                logger.warning(
                    f"训练/进化集与验证集重叠 {len(overlap)} 条，已从训练/进化集剔除: {sorted(overlap)[:5]}..."
                )
                evolve_cases = [c for c in evolve_cases if c.resource_id not in overlap]

            evolve_report_path = gen_dir / "evolve_report.json"
            if resume and self._is_complete_report(evolve_report_path):
                evolve_report = self._load_json(evolve_report_path)
                logger.info(f"Resume: 复用 {evolve_report_path}")
            else:
                evolve_report = test_agent.run(
                    evolve_cases,
                    run_dir=gen_dir,
                    report_name="evolve_report.json",
                    dataset_split="evolve",
                    concurrency=self.config.concurrency,
                )
            history.append(
                {
                    "generation": gen,
                    "phase": "evolve_eval",
                    "evolve_label": evolve_label,
                    "evolve_start_index": evolve_index,
                    "case_ids": [c.resource_id for c in evolve_cases],
                    "summary": evolve_report.get("summary"),
                }
            )

            if gen >= self.config.generations:
                break

            best_reload_interval = self.config.resolved_best_reload_interval()
            if (
                self.config.use_version_selector
                and best_reload_interval > 0
                and gen % best_reload_interval == 0
            ):
                checkpoint_best = restore_best_prompts(
                    run_dir,
                    self.store.prompts_dir,
                    self.tool_store.tools_dir,
                )
                self.store.snapshot_to(gen_dir / "prompts_after")
                self.tool_store.snapshot_to(gen_dir / "tools_after")
                history.append(
                    {
                        "generation": gen,
                        "phase": "best_reload",
                        "interval": best_reload_interval,
                        "selected_version_id": checkpoint_best.get("version_id"),
                        "composite": checkpoint_best.get("composite"),
                        "manifest": self.store.load_manifest(),
                    }
                )
                logger.info(
                    f"Generation {gen} 后回灌当前最优版本作为下一代初始: "
                    f"{checkpoint_best.get('version_id')} "
                    f"(composite={checkpoint_best.get('composite'):.4f})"
                )
                continue

            evolve_agent.run(
                evolve_report,
                generation=gen,
                run_dir=gen_dir,
            )
            self.store.snapshot_to(gen_dir / "prompts_after")
            self.tool_store.snapshot_to(gen_dir / "tools_after")
            history.append(
                {
                    "generation": gen,
                    "phase": "evolve",
                    "manifest": self.store.load_manifest(),
                }
            )

        if self.config.use_version_selector:
            best_version = select_best_version(run_dir)
            if best_version:
                restore_best_prompts(run_dir, self.store.prompts_dir, self.tool_store.tools_dir)
                best_snap = run_dir / "selected_best_prompts"
                self.store.snapshot_to(best_snap)
                self.tool_store.snapshot_to(run_dir / "selected_best_tools")
                logger.info(
                    f"已应用最优版本: {best_version.get('version_id')} "
                    f"(composite={best_version.get('composite'):.4f})"
                )

        final_test_cases = test_agent.load_final_test_cases()
        final_ids = {c.resource_id for c in final_test_cases}
        if final_ids & validation_ids:
            logger.warning(
                f"最终测试集与验证集重叠 {len(final_ids & validation_ids)} 条: "
                f"{sorted(final_ids & validation_ids)[:5]}..."
            )
        final_dir = run_dir / "final_test"
        final_report_path = final_dir / "final_test_report.json"
        if resume and self._is_complete_report(final_report_path):
            final_test_report = self._load_json(final_report_path)
            logger.info(f"Resume: 复用 {final_report_path}")
        else:
            final_test_report = test_agent.run(
                final_test_cases,
                run_dir=final_dir,
                report_name="final_test_report.json",
                dataset_split="final_test",
                concurrency=self.config.concurrency,
            )
        history.append(
            {
                "phase": "final_test",
                "case_ids": [c.resource_id for c in final_test_cases],
                "summary": final_test_report.get("summary"),
                "timing": final_test_report.get("meta", {}).get("timing"),
            }
        )

        versions = load_registry(run_dir)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_dir": str(run_dir),
                    "generations": self.config.generations,
                    "stopped_early": stopped_early,
                    "early_stopping": {
                        "enabled": self.config.early_stopping,
                        "patience": self.config.resolved_early_stopping_patience(),
                        "min_delta": self.config.resolved_early_stopping_min_delta(),
                        "best_generation": early_best_generation or None,
                        "best_composite": (
                            early_best_composite
                            if early_best_composite != float("-inf")
                            else None
                        ),
                    },
                    "validation_cases": [c.resource_id for c in validation_cases],
                    "final_test_cases": [c.resource_id for c in final_test_cases],
                    "rotate_evolve_cases": self.config.rotate_evolve_cases,
                    "evolve_case_stride": self.config.resolved_evolve_case_stride(),
                    "best_reload_interval": self.config.resolved_best_reload_interval(),
                    "history": history,
                    "version_registry": versions,
                    "best_version": best_version,
                    "final_validation_summary": (last_validation_report or {}).get("summary"),
                    "final_test_summary": (final_test_report or {}).get("summary"),
                    "final_manifest": self.store.load_manifest(),
                    "final_tool_registry": self.tool_store.load_registry(),
                },
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        logger.info(f"自进化完成，汇总: {summary_path}")
        return {
            "run_dir": str(run_dir),
            "history": history,
            "best_version": best_version,
        }

    def restore_prompts_from_run(self, run_dir: Path, generation: int, *, when: str = "after") -> None:
        run_dir = Path(run_dir)
        sub = run_dir / f"gen{generation:03d}" / f"prompts_{when}"
        if not sub.is_dir():
            raise FileNotFoundError(sub)
        for item in sub.iterdir():
            if item.is_file():
                shutil.copy2(item, self.store.prompts_dir / item.name)

    def restore_best_from_run(self, run_dir: Path) -> dict[str, Any]:
        return restore_best_prompts(
            Path(run_dir),
            self.store.prompts_dir,
            self.tool_store.tools_dir,
        )
