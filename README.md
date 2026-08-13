# EvoModelKG

EvoModelKG is an archive-guided self-evolving pipeline for recovering structured, evidence-grounded attributes from Hugging Face model-card READMEs and materializing facts missing from an initial model supply-chain knowledge graph.

## What this release contains

- The extraction, deterministic evaluation, prompt/tool evolution, version archive, and periodic best-version restoration implementation.
- The final extraction prompt and three finalized deterministic tools.
- Frozen 15,000-ID candidate, validation, held-out-test, and 18-generation evolution identifiers, plus a corrected gold-annotation export.

## Install

```bash
python -m venv .venv
.venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
copy .env.example .env
```

Set the required endpoint and credentials in `.env`. Do not commit `.env`.

## Reproduce

Run the evolutionary pipeline with a locally prepared README corpus and a structured baseline:

```bash
python run_evolution.py --help
```

The original full evaluation relied on a private Neo4j baseline and a point-in-time Hugging Face README collection. This release therefore provides split identifiers and derived, non-redistributable result snapshots. Recreating the full run requires obtaining the source records under their respective terms, then configuring `SELF_EVOLVE_README_DIR` and the baseline connection or cached baseline files.



## Repository layout

- `run_evolution.py`: main entry point for the self-evolution experiment.
- `evomodelkg/`: all core implementation and internal support modules.
- `prompts/`: the fixed extraction prompt and its workflow manifest.
- `tools/`: the three final deterministic tools and their registry.
- `data/`: immutable split IDs, the split protocol, and the currently located corrected annotation export.

The published split protocol uses candidate IDs 0--299 for validation, 300--2,299 for held-out testing, and 2,300--7,699 for the 18 fixed, non-overlapping evolution batches (300 READMEs per generation and stride 300). The full evolution pool contains IDs 2,300--14,999.

## Limitations and responsible use

The system extracts what model-card authors document; absent documentation is not evidence that a capability, risk, restriction, or provenance fact is absent. Structured records used as evolutionary feedback are incomplete weak references, not exhaustive ground truth. External LLM services can change over time, so exact generation-level results require the recorded model version and endpoint behavior.

# EvoModelKG Knowledge Graph

This directory provides anonymous access to the EvoModelKG knowledge graph
artifact introduced in the accompanying manuscript:

https://figshare.com/s/c2a26f3e5f5f71e99249

EvoModelKG covers 172,549 Hugging Face models with more than 1,000 accumulated
downloads as of the July 2026 snapshot. The graph is initialized from structured
metadata using an ontology spanning models, datasets, authors, tools, licences,
and arXiv papers. It is further completed with schema-aligned facts extracted
from the model cards available for 130,396 models.

## Files

- `neo4j.dump`: Neo4j database dump of the EvoModelKG knowledge graph.
- `nodes.csv`: exported graph nodes, including labels, identifiers, and node
  properties.
- `relationships.csv`: exported graph relationships, including endpoints,
  relationship types, and relationship properties.

These files correspond specifically to the EvoModelKG knowledge graph
contribution.

The files can be accessed through the anonymous link without signing in.
A permanent public dataset record and DOI will be provided after completion
of peer review.

## Citation

If you use this code, please cite the associated EvoModelKG paper. A `CITATION.cff` file will be finalized when author, venue, and archival DOI metadata are available.

## License

This project is licensed under the [MIT License](LICENSE).
