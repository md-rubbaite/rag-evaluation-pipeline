# ADR 0001: Lexical, deterministic metrics as the first quality gate

- **Status:** Accepted
- **Date:** 2026-11-20

## Context

RAG systems fail in two places: retrieval misses the right source, or generation states things
the sources do not support. Teams need a check that runs on every pull request, costs nothing,
gives the same answer twice and can be explained to a reviewer.

Model-based judges (an LLM grading answers) catch paraphrase and reasoning errors that word
overlap cannot, but they add cost, latency, non-determinism and their own failure modes, and
they must themselves be calibrated against human labels.

## Options

| Option | Strengths | Weaknesses |
|---|---|---|
| A. Lexical metrics only | Free, fast, deterministic, explainable, no data leaves the build | Misses paraphrase; can be fooled by copying words |
| B. LLM judge only | Understands meaning | Cost per run, non-deterministic, needs calibration, sends data to a model |
| C. Lexical gate in CI, model judge offline on a schedule | Cheap fast feedback plus deeper periodic review | Two systems to maintain |

## Decision

Option C. This repository implements the lexical layer: document-level retrieval metrics
(recall@k, precision@k, MRR, nDCG@k) and sentence-level groundedness against the retrieved
contexts. A sentence counts as supported only when one single context covers at least
`support_threshold` of its content words, so evidence cannot be stitched together from several
unrelated chunks.

The `Retriever` protocol and the JSONL answer format let a team plug in its own retriever or
add a model-based judge later without changing the gate.

## Consequences

- Gates are reproducible: the same commit gives the same numbers.
- Groundedness is a **screen**, not proof. Flagged sentences are candidates for human review;
  unflagged ones can still be wrong when they reuse source words with a different meaning.
- Thresholds must be set from a labelled baseline for each corpus, not copied from the sample.

## Revisit when

- Paraphrase-heavy answers cause more than ~10% false flags on a labelled sample, or
- a calibrated model judge agrees with human labels well enough to justify its cost.
