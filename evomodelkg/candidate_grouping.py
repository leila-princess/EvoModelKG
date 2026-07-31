from __future__ import annotations

import re
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


GROUPED_FIELDS = {"architecture", "config_model_type"}
GROUPS_PATH = Path(__file__).resolve().parent / "generated_tools" / "candidate_groups.json"

AUTO_MODEL_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("causal_lm", ("causallm",)),
    ("masked_lm", ("maskedlm",)),
    ("seq2seq_lm", ("seq2seqlm",)),
    ("sequence_classification", ("sequenceclassification",)),
    ("token_classification", ("tokenclassification",)),
    ("question_answering", ("questionanswering",)),
    ("image_text_to_text", ("imagetexttotext", "vision2seq")),
    ("image_classification", ("imageclassification",)),
    ("object_detection", ("objectdetection",)),
    ("speech_seq2seq", ("speechseq2seq",)),
    ("ctc", ("ctc",)),
    ("audio_classification", ("audioclassification",)),
    ("text_to_waveform", ("texttowaveform",)),
    ("text_to_spectrogram", ("texttospectrogram",)),
]

FAMILY_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    ("qwen2_5_vl", ("qwen25vl", "qwen2vl", "qwen2_5_vl")),
    ("qwen3_moe", ("qwen3moe",)),
    ("qwen3", ("qwen3",)),
    ("qwen2", ("qwen25", "qwen2", "qwen")),
    ("llama", ("llama", "mllama", "tinyllama")),
    ("olmo", ("olmo",)),
    ("mistral", ("mistral",)),
    ("mixtral", ("mixtral",)),
    ("bert", ("bert",)),
    ("camembert", ("camembert",)),
    ("distilbert", ("distilbert",)),
    ("roberta", ("roberta",)),
    ("xlm_roberta", ("xlmroberta", "xlmroberta")),
    ("gpt2", ("gpt2",)),
    ("gpt_neox", ("gptneox",)),
    ("t5", ("flant5", "mt5", "t5")),
    ("whisper", ("whisper",)),
    ("vit", ("vit",)),
    ("nougat", ("nougat",)),
    ("vision_encoder_decoder", ("visionencoderdecoder",)),
    ("yolos", ("yolos",)),
    ("phi3_v", ("phi3v",)),
    ("phi3", ("phi3",)),
    ("phi", ("phi",)),
    ("gemma3", ("gemma3",)),
    ("gemma2", ("gemma2",)),
    ("gemma", ("gemma",)),
]

TASK_ALIASES: list[tuple[str, tuple[str, ...]]] = [
    ("causal_lm", ("forcausallm", "lmheadmodel")),
    ("conditional_generation", ("forconditionalgeneration",)),
    ("seq2seq_lm", ("forseq2seqlm",)),
    ("masked_lm", ("formaskedlm",)),
    ("sequence_classification", ("forsequenceclassification",)),
    ("token_classification", ("fortokenclassification",)),
    ("question_answering", ("forquestionanswering",)),
    ("image_text_to_text", ("forimagetexttotext",)),
    ("image_classification", ("forimageclassification",)),
    ("object_detection", ("forobjectdetection",)),
    ("speech_seq2seq", ("forspeechseq2seq",)),
    ("ctc", ("forctc",)),
    ("audio_classification", ("foraudioclassification",)),
    ("base_model", ("model",)),
]


def compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def split_tokens(value: Any) -> list[str]:
    text = str(value or "")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ").replace(".", " ")
    return [x.lower() for x in re.findall(r"[A-Za-z0-9]+", text) if x]


def _first_match(blob: str, rules: list[tuple[str, tuple[str, ...]]]) -> str | None:
    for group, needles in rules:
        if any(needle in blob for needle in needles):
            return group
    return None


def auto_model_group(value: Any) -> str:
    blob = compact(value)
    if blob == "automodel":
        return "auto_model_base"
    task = _first_match(blob, AUTO_MODEL_GROUPS)
    return f"auto_model::{task or blob}" if blob else ""


def architecture_group(value: Any) -> str:
    blob = compact(value)
    if not blob:
        return ""
    family = _first_match(blob, FAMILY_ALIASES)
    task = _first_match(blob, TASK_ALIASES)
    if family:
        return f"architecture::{family}"
    return f"architecture::{blob}"


def config_model_type_group(value: Any) -> str:
    blob = compact(value)
    if not blob:
        return ""
    family = _first_match(blob, FAMILY_ALIASES)
    return f"config_model_type::{family or blob}"


def candidate_group(attr: str | None, value: Any) -> str:
    field = str(attr or "").strip()
    if field == "auto_model":
        return auto_model_group(value)
    if field == "architecture":
        return architecture_group(value)
    if field == "config_model_type":
        return config_model_type_group(value)
    return ""


@lru_cache(maxsize=1)
def _multi_value_groups() -> dict[str, set[str]]:
    if not GROUPS_PATH.exists():
        return {}
    try:
        data = json.loads(GROUPS_PATH.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    groups = data.get("groups") if isinstance(data, dict) else {}
    if not isinstance(groups, dict):
        return {}
    out: dict[str, set[str]] = {}
    for field, field_groups in groups.items():
        if not isinstance(field_groups, dict):
            continue
        out[str(field)] = {
            str(group_id)
            for group_id, members in field_groups.items()
            if isinstance(members, list) and len(members) > 1
        }
    return out


def grouped_values_match(readme_val: Any, struct_val: Any, attr: str | None) -> bool | None:
    if attr not in GROUPED_FIELDS:
        return None
    pred_group = candidate_group(attr, readme_val)
    gold_group = candidate_group(attr, struct_val)
    if not pred_group or not gold_group:
        return None
    multi_groups = _multi_value_groups().get(str(attr), set())
    if pred_group not in multi_groups or gold_group not in multi_groups:
        return None
    return pred_group == gold_group
