# Managed Tool Contract

Read this before creating or modifying managed tools.

## Payload Input Contract

Preprocess tools (`stage="preprocess_readme"`) receive:

```json
{
  "stage": "preprocess_readme",
  "resource_type": "model",
  "resource_id": "org/model-name",
  "readme_content": "README markdown after YAML front matter has been stripped",
  "readme": "alias of readme_content"
}
```

Postprocess tools (`stage="postprocess_extraction"`) receive:

```json
{
  "stage": "postprocess_extraction",
  "resource_type": "model",
  "resource_id": "org/model-name",
  "readme_content": "README markdown after YAML front matter has been stripped",
  "readme": "alias of readme_content",
  "extraction": {"attributes": [], "relations": [], "summary": "..."},
  "extraction_result": {"attributes": [], "relations": [], "summary": "..."}
}
```

Read postprocess input with:

```python
extraction = payload.get("extraction") or payload.get("extraction_result") or {}
```

## Return Format Contract

Preprocess tools return:

```json
{"readme_content": "cleaned markdown"}
```

Postprocess tools return one of:

```json
{"extraction": {"attributes": [], "relations": [], "summary": "..."}}
```

```json
{"extraction_result": {"attributes": [], "relations": [], "summary": "..."}}
```

```json
{"attributes": [], "relations": []}
```

Do not return an empty extraction when the input extraction is non-empty unless the registry entry sets `"allow_empty_output": true`.

## Extraction Schema

Attributes:

```json
{
  "entity_id": "org/model-name",
  "entity_type": "model",
  "attribute": "license",
  "value": "apache-2.0",
  "evidence_span": "README evidence",
  "confidence": 0.8,
  "source": "readme",
  "normalization_note": "optional"
}
```

Relations:

```json
{
  "source_id": "org/model-name",
  "target_id": "dataset-or-paper-id",
  "relation_type": "TRAINED_ON",
  "evidence_span": "README evidence",
  "confidence": 0.8,
  "source": "readme",
  "properties": {}
}
```

Preserve unknown existing fields unless there is a clear reason to remove them.

## Registry Schema

```json
{
  "version": 1,
  "tools": [
    {
      "name": "normalize_attributes",
      "filename": "normalize_attributes.py",
      "stage": "postprocess_extraction",
      "enabled": true,
      "description": "...",
      "purpose": "...",
      "evidence_cases": ["org/model-a"],
      "expected_behavior": "...",
      "safety_constraints": "...",
      "validation_criteria": "...",
      "allow_empty_output": false
    }
  ]
}
```

Required tool fields: `name`, `filename`, `stage`, `enabled`.
Allowed stages: `preprocess_readme`, `postprocess_extraction`.
`file` is accepted as an alias for `filename`, but `filename` is preferred.
