from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from evomodelkg.metrics import (
    compute_length_adjusted_completeness,
    estimate_llm_calls_per_case,
)


@dataclass
class ScoreWeights:
    """综合评分权重：越高越重视该项；cost 为惩罚项。准确性仅看属性匹配率。"""

    accuracy: float = 0.55
    completeness: float = 0.30
    cost: float = 0.15
    input_compression_reward: float = 0.05
    time_baseline_seconds: float = 600.0
    llm_calls_baseline: int = 240
    time_cost_share: float = 0.5
    llm_cost_share: float = 0.5
    compression_accuracy_tolerance: float = 0.02
    compression_completeness_tolerance: float = 0.03


def _accuracy_score(summary: dict[str, Any], _weights: ScoreWeights) -> float:
    """属性 overlap 内匹配率（attribute_overlap_accuracy），不含关系。"""
    acc = summary.get("accuracy") or {}
    return float(acc.get("attribute_overlap_accuracy") or 0.0)


def _cost_penalty(
    *,
    extract_seconds: float,
    llm_calls: int,
    weights: ScoreWeights,
) -> float:
    time_p = min(1.0, extract_seconds / max(1.0, weights.time_baseline_seconds))
    call_p = min(1.0, llm_calls / max(1, weights.llm_calls_baseline))
    return weights.time_cost_share * time_p + weights.llm_cost_share * call_p


def _input_compression_score(report: dict[str, Any]) -> float:
    meta = report.get("meta") or {}
    input_cost = meta.get("input_cost") or {}
    return max(0.0, min(1.0, float(input_cost.get("readme_chars_reduction_rate") or 0.0)))


def _quality_preserved_for_compression_reward(
    *,
    accuracy_score: float,
    completeness_score: float,
    quality_baseline: dict[str, float] | None,
    weights: ScoreWeights,
) -> bool:
    if not quality_baseline:
        return False
    baseline_acc = float(quality_baseline.get("accuracy_score") or 0.0)
    baseline_comp = float(quality_baseline.get("completeness_score") or 0.0)
    return (
        accuracy_score + weights.compression_accuracy_tolerance >= baseline_acc
        and completeness_score + weights.compression_completeness_tolerance >= baseline_comp
    )


def compute_composite_score(
    report: dict[str, Any],
    *,
    extract_seconds: float = 0.0,
    llm_calls: int = 0,
    weights: ScoreWeights | None = None,
    quality_baseline: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    综合分 = 准确性 + 完整度 - 成本惩罚 + 无损压缩奖励。
    压缩奖励只在准确性和完整度相对历史质量基线基本不下降时启用，避免鼓励乱删 README。
    """
    w = weights or ScoreWeights()
    summary = report.get("summary") or {}
    acc_s = _accuracy_score(summary, w)
    comp_s = float((summary.get("completeness_fit") or {}).get("score") or 0.0)
    cost_p = _cost_penalty(
        extract_seconds=extract_seconds,
        llm_calls=llm_calls,
        weights=w,
    )
    compression_s = _input_compression_score(report)
    compression_reward_applied = _quality_preserved_for_compression_reward(
        accuracy_score=acc_s,
        completeness_score=comp_s,
        quality_baseline=quality_baseline,
        weights=w,
    )
    compression_reward = compression_s if compression_reward_applied else 0.0
    composite = (
        w.accuracy * acc_s
        + w.completeness * comp_s
        - w.cost * cost_p
        + w.input_compression_reward * compression_reward
    )
    return {
        "composite": composite,
        "accuracy_score": acc_s,
        "completeness_score": comp_s,
        "cost_penalty": cost_p,
        "input_compression_score": compression_s,
        "input_compression_reward": compression_reward,
        "input_compression_reward_applied": compression_reward_applied,
        "quality_baseline_for_compression": quality_baseline or {},
        "extract_seconds": extract_seconds,
        "llm_calls": llm_calls,
        "weights": asdict(w),
    }


def archive_prompt_version(
    run_dir: Path,
    *,
    version_id: str,
    prompts_src: Path,
    tools_src: Path | None = None,
    test_report: dict[str, Any],
    score_detail: dict[str, Any],
    generation: int,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """将 prompt 快照与 test 表现写入 run_dir/versions/<version_id>/。"""
    dest = Path(run_dir) / "versions" / version_id
    dest.mkdir(parents=True, exist_ok=True)
    snap = dest / "prompts"
    if snap.exists():
        shutil.rmtree(snap)
    shutil.copytree(prompts_src, snap)
    tools_snap = None
    if tools_src is not None and Path(tools_src).is_dir():
        tools_snap = dest / "tools"
        if tools_snap.exists():
            shutil.rmtree(tools_snap)
        shutil.copytree(tools_src, tools_snap)
    payload = {
        "version_id": version_id,
        "generation": generation,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "score": score_detail,
        "prompts_dir": str(snap),
        "tools_dir": str(tools_snap) if tools_snap is not None else None,
        "test_summary": test_report.get("summary"),
        "case_ids": (test_report.get("meta") or {}).get("case_ids"),
        "meta": extra_meta or {},
    }
    with (dest / "version_meta.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    with (dest / "test_report.json").open("w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2, default=str)
    return dest


def load_registry(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "version_registry.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return list(data.get("versions") or [])


def save_registry(run_dir: Path, versions: list[dict[str, Any]]) -> None:
    path = Path(run_dir) / "version_registry.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "kind": "self_evolve_version_registry",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "versions": versions,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def register_version(
    run_dir: Path,
    *,
    version_id: str,
    prompts_src: Path,
    tools_src: Path | None = None,
    test_report: dict[str, Any],
    generation: int,
    manifest: dict[str, Any],
    extract_seconds: float,
    weights: ScoreWeights | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    n_cases = max(1, int((test_report.get("meta") or {}).get("case_count") or 1))
    llm_per = estimate_llm_calls_per_case(manifest)
    llm_calls = llm_per * n_cases
    existing_versions = load_registry(run_dir)
    quality_baseline = None
    if existing_versions:
        quality_baseline = {
            "accuracy_score": max(float(v.get("accuracy_score") or 0.0) for v in existing_versions),
            "completeness_score": max(
                float(v.get("completeness_score") or 0.0) for v in existing_versions
            ),
        }
    score_detail = compute_composite_score(
        test_report,
        extract_seconds=extract_seconds,
        llm_calls=llm_calls,
        weights=weights,
        quality_baseline=quality_baseline,
    )
    archive_prompt_version(
        run_dir,
        version_id=version_id,
        prompts_src=prompts_src,
        tools_src=tools_src,
        test_report=test_report,
        score_detail=score_detail,
        generation=generation,
        extra_meta=extra_meta,
    )
    entry = {
        "version_id": version_id,
        "generation": generation,
        "prompts_dir": str(Path(run_dir) / "versions" / version_id / "prompts"),
        "tools_dir": (
            str(Path(run_dir) / "versions" / version_id / "tools")
            if tools_src is not None
            else None
        ),
        **score_detail,
        "extra_meta": extra_meta or {},
    }
    versions = existing_versions
    versions = [v for v in versions if v.get("version_id") != version_id]
    versions.append(entry)
    save_registry(run_dir, versions)
    logger.info(
        f"版本归档 {version_id}: composite={score_detail['composite']:.4f} "
        f"(acc={score_detail['accuracy_score']:.3f}, comp={score_detail['completeness_score']:.3f}, "
        f"cost_pen={score_detail['cost_penalty']:.3f})"
    )
    return entry


def select_best_version(
    run_dir: Path,
) -> dict[str, Any] | None:
    versions = load_registry(run_dir)
    if not versions:
        return None
    best = max(versions, key=lambda v: float(v.get("composite") or -1e9))
    best["selected"] = True
    return best


def restore_best_prompts(
    run_dir: Path,
    prompts_dir: Path,
    tools_dir: Path | None = None,
) -> dict[str, Any]:
    best = select_best_version(run_dir)
    if not best:
        raise ValueError(f"{run_dir} 中无版本记录，无法恢复最优 prompt")
    src = Path(best["prompts_dir"])
    if not src.is_dir():
        raise FileNotFoundError(f"最优版本 prompt 目录不存在: {src}")
    dest = Path(prompts_dir)
    for item in dest.iterdir():
        if item.is_file():
            item.unlink()
    for item in src.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)
    if tools_dir is not None and best.get("tools_dir"):
        tools_src = Path(str(best["tools_dir"]))
        if not tools_src.is_dir():
            raise FileNotFoundError(f"best version tools dir not found: {tools_src}")
        tools_dest = Path(tools_dir)
        if tools_dest.exists():
            shutil.rmtree(tools_dest)
        shutil.copytree(tools_src, tools_dest)
    logger.info(
        f"已恢复最优 prompt: version_id={best.get('version_id')}, "
        f"composite={best.get('composite'):.4f}"
    )
    return best
