# An Ensemble Event Extraction Prototype on News (Paper-inspired)

This repository provides a small prototype inspired by:
"An Ensemble Event Extraction Method on News" (Springer, 2025)

## What this repo does
- Extracts events from news texts using 3 extractors:
  - Model B: rule-based triggers
  - Model C: heuristic pattern-based extractor
  - Model A: optional HuggingFace NER-based extractor (auto-downloads model)
- Computes Confidence Score (CS) using:
  - prior confidence per model
  - pairwise consistency between model outputs
  - iterative CS update
- Produces JSONL outputs and evaluates event_type classification.

## Setup
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
