from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvolutionToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class EvolutionPlan(BaseModel):
    """进化 agent 单轮输出：分析 + 工具调用列表。"""

    analysis: str = ""
    strategy: str = ""
    tool_calls: list[EvolutionToolCall] = Field(default_factory=list)
    done: bool = False


class GenerationRecord(BaseModel):
    generation: int
    manifest_snapshot: str
    test_report_path: str
    evolution_log_path: str
    summary: dict[str, Any] = Field(default_factory=dict)
