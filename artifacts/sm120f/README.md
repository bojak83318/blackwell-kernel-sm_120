# SM120f Artifacts Convention

This directory contains evidence and patches for the SM120f MoE enablement effort.

## Structure
- `artifacts/sm120f/<slice>/`: Evidence scaffolding (logs, smoke check outputs).
- `artifacts/sm120f/patches/`: Exported git patches for the upstream vLLM repository.
- `artifacts/sm120f/patches/<slice>.patch`: Patch file for each slice.
- `artifacts/sm120f/logs/`: Durable session logs, branch-state summaries, and verification notes.

## Patch Export
When a slice that edits vLLM is completed, use:
`git -C $VLLM_SM120F_ROOT format-patch -1 HEAD --stdout > artifacts/sm120f/patches/<slice>.patch`

For operational runs, keep one dated note under `artifacts/sm120f/logs/` and append new evidence to that note instead of creating ad hoc log files in the repo root.
