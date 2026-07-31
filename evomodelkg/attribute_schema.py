"""Immutable attribute batching protocol used by every evolution generation."""

from __future__ import annotations


LOCKED_ATTRIBUTE_FIELD_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "model_id",
        "model_name",
        "model_sub_types",
        "author",
        "version",
        "library_name",
        "pipeline_tag",
        "license",
        "license_name",
        "license_link",
        "base_model",
        "languages",
        "auto_model",
        "architecture",
        "config_model_type",
        "num_parameters",
        "model_size",
        "model_file_formats",
        "context_length",
        "max_position_embeddings",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "uses_safetensors",
        "training_datasets",
        "evaluation_datasets",
        "code_repository",
        "demo_url",
        "direct_use",
        "out_of_scope_use",
        "training_time",
        "inference_latency",
        "compute_infrastructure",
        "co2_emission",
        "risks_and_biases",
        "cited_papers",
        "citation_bibtex",
        "created_at",
    ),
)


def locked_attribute_field_groups() -> list[list[str]]:
    """Return a mutable copy without exposing the protocol constant."""
    return [list(group) for group in LOCKED_ATTRIBUTE_FIELD_GROUPS]
