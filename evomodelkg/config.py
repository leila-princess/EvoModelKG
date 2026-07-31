from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from evomodelkg.attribute_schema import locked_attribute_field_groups
from evomodelkg.llm_config import mask_secret, role_api_key, role_base_url
from evomodelkg.version_registry import ScoreWeights

ReadmeFilenameStyle = Literal["auto", "underscore", "double_underscore"]
BaselineSource = Literal["neo4j", "cached"]


DEFAULT_RELATION_WHITELIST = [
    "TRAINED_ON",
    "SOURCE_DATASET",
    "DERIVED_FROM",
    "EVALUATED_ON",
    "GENERATED",
    "ANNOTATED",
    "MENTIONS_ARXIV",
    "USES_TOOL",
    "LICENSED_UNDER",
]

DEFAULT_MODEL_ATTR_GROUPS = locked_attribute_field_groups()


def default_extract_model() -> str:
    return (
        os.getenv("SELF_EVOLVE_EXTRACT_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-chat"
    )


def default_evolve_model() -> str:
    return (
        os.getenv("SELF_EVOLVE_EVOLVE_MODEL")
        or os.getenv("DEEPSEEK_MODEL")
        or "deepseek-chat"
    )


@dataclass
class EvolutionConfig:
    """自进化实验配置。"""

    prompts_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "prompts")
    runs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "runs")
    benchmark_dir: Path = field(default_factory=lambda: Path("data/processed_hubstats"))
    models_json: str = "models.json"
    relations_json: str | None = None
    max_cases: int = 15
    test_max_cases: int | None = None
    run_max_cases: int | None = None
    test_seed: int | None = None
    run_seed: int | None = None
    test_resource_ids: list[str] | None = None
    run_resource_ids: list[str] | None = None
    final_test_resource_ids: list[str] | None = None
    test_cases_file: Path | None = None
    run_cases_file: Path | None = None
    final_test_cases_file: Path | None = None
    generations: int = 3
    cases_per_generation: int | None = None
    final_test_max_cases: int | None = None
    final_test_seed: int | None = None
    extract_model: str | None = None
    evolve_model: str | None = None
    extract_api_key: str | None = None
    extract_base_url: str | None = None
    evolve_api_key: str | None = None
    evolve_base_url: str | None = None
    extract_temperature: float = 0.0
    evolve_temperature: float = 0.2
    extract_context_window: int | None = None
    extract_output_token_budget: int = 3072
    extract_context_safety_tokens: int = 768
    extract_chunk_overlap_tokens: int = 256
    max_evolve_tool_rounds: int = 8
    concurrency: int = 2
    evaluation_checkpoint_size: int = 10
    sample_timeout_sec: float | None = None
    seed: int = 42
    readme_dir: Path | None = field(
        default_factory=lambda: (
            Path(os.environ["SELF_EVOLVE_README_DIR"])
            if os.getenv("SELF_EVOLVE_README_DIR")
            else None
        )
    )
    readme_field: str = "readme_content"
    readme_filename_style: ReadmeFilenameStyle = "auto"
    baseline_source: BaselineSource = "neo4j"
    hubstats_dir: Path | None = field(default_factory=lambda: Path("dataset_hub_stats"))
    readme_local_only: bool = False
    # Public release: supply the connection through NEO4J_URI or a CLI option.
    neo4j_uri: str = os.getenv("NEO4J_URI", "")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    readme_hit_ids_file: Path | None = None
    require_readme_hit_ids: bool = True
    test_start_index: int = 0
    run_start_index: int = 20
    final_test_start_index: int | None = None
    parquet_read_batch_size: int = 20
    rotate_evolve_cases: bool = True
    evolve_case_stride: int | None = None
    use_version_selector: bool = True
    best_reload_interval: int = 3
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)

    def resolve_paths(self) -> None:
        self.prompts_dir = Path(self.prompts_dir).resolve()
        self.runs_dir = Path(self.runs_dir).resolve()
        self.benchmark_dir = Path(self.benchmark_dir).resolve()
        if self.test_cases_file is not None:
            self.test_cases_file = Path(self.test_cases_file).resolve()
        if self.run_cases_file is not None:
            self.run_cases_file = Path(self.run_cases_file).resolve()
        if self.final_test_cases_file is not None:
            self.final_test_cases_file = Path(self.final_test_cases_file).resolve()
        if self.readme_dir is not None:
            self.readme_dir = Path(self.readme_dir).resolve()
        if self.hubstats_dir is not None:
            self.hubstats_dir = Path(self.hubstats_dir).resolve()
        if self.readme_hit_ids_file is not None:
            self.readme_hit_ids_file = Path(self.readme_hit_ids_file).resolve()

    def resolved_test_max_cases(self) -> int:
        return self.test_max_cases if self.test_max_cases is not None else self.max_cases

    def resolved_run_max_cases(self) -> int:
        if self.cases_per_generation is not None:
            return self.cases_per_generation
        if self.run_max_cases is not None:
            return self.run_max_cases
        return self.max_cases

    def resolved_final_test_max_cases(self) -> int:
        return (
            self.final_test_max_cases
            if self.final_test_max_cases is not None
            else self.resolved_test_max_cases()
        )

    def resolved_final_test_start_index(self) -> int:
        if self.final_test_start_index is not None:
            return max(0, int(self.final_test_start_index))
        return max(0, int(self.test_start_index) + self.resolved_test_max_cases())

    def resolved_evolve_case_stride(self) -> int:
        if self.evolve_case_stride is not None:
            return max(1, int(self.evolve_case_stride))
        return max(1, self.resolved_run_max_cases())

    def resolved_best_reload_interval(self) -> int:
        return max(0, int(self.best_reload_interval))

    def resolved_early_stopping_patience(self) -> int:
        return max(1, int(self.early_stopping_patience))

    def resolved_early_stopping_min_delta(self) -> float:
        return max(0.0, float(self.early_stopping_min_delta))

    def resolved_sample_timeout_sec(self) -> float:
        raw = (
            self.sample_timeout_sec
            if self.sample_timeout_sec is not None
            else os.getenv("SELF_EVOLVE_SAMPLE_TIMEOUT_SEC", "600")
        )
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return 600.0
        return val if val > 0 else 600.0

    def resolved_extract_context_window(self) -> int | None:
        raw = (
            self.extract_context_window
            if self.extract_context_window is not None
            else os.getenv("SELF_EVOLVE_EXTRACT_CONTEXT_WINDOW")
        )
        if raw in (None, ""):
            return None
        try:
            val = int(raw)
        except (TypeError, ValueError):
            return None
        return val if val > 0 else None

    def resolved_extract_output_token_budget(self) -> int:
        raw = os.getenv("SELF_EVOLVE_EXTRACT_OUTPUT_TOKEN_BUDGET")
        if raw not in (None, "") and self.extract_output_token_budget == 3072:
            try:
                return max(256, int(raw))
            except (TypeError, ValueError):
                pass
        return max(256, int(self.extract_output_token_budget))

    def resolved_extract_context_safety_tokens(self) -> int:
        raw = os.getenv("SELF_EVOLVE_EXTRACT_CONTEXT_SAFETY_TOKENS")
        if raw not in (None, "") and self.extract_context_safety_tokens == 768:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
        return max(0, int(self.extract_context_safety_tokens))

    def resolved_extract_chunk_overlap_tokens(self) -> int:
        raw = os.getenv("SELF_EVOLVE_EXTRACT_CHUNK_OVERLAP_TOKENS")
        if raw not in (None, "") and self.extract_chunk_overlap_tokens == 256:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
        return max(0, int(self.extract_chunk_overlap_tokens))

    def resolved_test_seed(self) -> int:
        return self.test_seed if self.test_seed is not None else self.seed

    def resolved_run_seed(self) -> int:
        return self.run_seed if self.run_seed is not None else (self.seed + 1000)

    def resolved_final_test_seed(self) -> int:
        return self.final_test_seed if self.final_test_seed is not None else (self.seed + 2000)

    def resolved_test_resource_ids(self) -> list[str] | None:
        return self.test_resource_ids

    def resolved_run_resource_ids(self) -> list[str] | None:
        return self.run_resource_ids

    def resolved_final_test_resource_ids(self) -> list[str] | None:
        return self.final_test_resource_ids

    def resolved_extract_model(self) -> str:
        return self.extract_model or default_extract_model()

    def resolved_evolve_model(self) -> str:
        return self.evolve_model or default_evolve_model()

    def llm_env_summary(self) -> dict[str, str]:
        extract_key = self.extract_api_key or role_api_key("extract") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_COMPAT_API_KEY") or ""
        evolve_key = self.evolve_api_key or role_api_key("evolve") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_COMPAT_API_KEY") or ""
        return {
            "extract_model": self.resolved_extract_model(),
            "evolve_model": self.resolved_evolve_model(),
            "extract_base_url": (
                self.extract_base_url
                or role_base_url("extract")
                or os.getenv("OPENAI_COMPAT_BASE_URL")
                or os.getenv("DEEPSEEK_BASE_URL")
                or "https://api.deepseek.com/v1"
            ),
            "evolve_base_url": (
                self.evolve_base_url
                or role_base_url("evolve")
                or os.getenv("OPENAI_COMPAT_BASE_URL")
                or os.getenv("DEEPSEEK_BASE_URL")
                or "https://api.deepseek.com/v1"
            ),
            "extract_api_key": mask_secret(extract_key),
            "evolve_api_key": mask_secret(evolve_key),
        }
