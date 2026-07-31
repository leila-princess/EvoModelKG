from __future__ import annotations

from evomodelkg.config import EvolutionConfig


def evolve_start_index(config: EvolutionConfig, generation: int) -> int:
    """
    每代进化反馈用例在 readme_hit_ids / parquet 顺序上的起始偏移。
    generation 从 1 开始；不轮换时固定为 run_start_index。
    """
    base = max(0, int(config.run_start_index))
    if not config.rotate_evolve_cases:
        return base
    stride = config.resolved_evolve_case_stride()
    return base + max(0, generation - 1) * stride


def evolve_generation_label(generation: int, start_index: int) -> str:
    return f"gen{generation:03d}@offset{start_index}"
