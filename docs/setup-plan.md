# Setup plan: Safe Latent Diffusion baseline

## Guardrails

- Reproduce the published SLD behaviour before proposing any change.
- Separate baseline and future experimental configurations so comparisons are fair.
- Use fixed seeds, versioned prompt manifests, and machine-readable run metadata.
- Keep safety components active for demos and avoid publishing sensitive prompt material in a public repository.

## Phase 1 - Establish a reproducible environment

1. Choose one supported Python version (3.10 or 3.11) and record OS, GPU, CUDA, driver, and package versions.
2. Install a PyTorch build that matches the available CUDA environment, then install `requirements.txt`.
3. Obtain only the permitted model access and dataset materials; store credentials in `.env` and downloaded assets outside Git.
4. Record the exact model revision and the commit/release of the baseline implementation in run metadata.

Acceptance check: `python -c "import torch, diffusers, transformers"` succeeds and GPU availability is recorded.

## Phase 2 - Reproduce the unmodified baseline

1. Run a benign-prompt smoke test using `configs/baseline/sld-medium.yaml`.
2. Confirm that output image, seed, model identifier, configuration, runtime, and library versions are captured together.
3. Repeat the same seed and verify deterministic behaviour as far as the chosen GPU/runtime permits.
4. Compare the pipeline parameters with the published SLD configuration; document any unavoidable compatibility differences.

Acceptance check: a single baseline run completes, stores metadata, and can be repeated from a fresh environment.

## Phase 3 - Define evaluation before modification

1. Select an appropriately licensed, permitted prompt benchmark and record its provenance and filtering policy.
2. Freeze a held-out evaluation manifest with prompt IDs, categories, and seeds before testing any proposed improvement.
3. Define metrics for safety, prompt adherence, image quality proxy, generation success, runtime, and GPU memory.
4. Run the original SLD baseline once across that manifest and store aggregate metrics plus per-prompt records.

Acceptance check: the baseline results table is complete and the evaluation command has no manually edited parameters.

## Phase 4 - Analyse, then propose an improvement

1. Inspect baseline failures and trade-offs using the pre-registered evaluation evidence.
2. State one narrow, defensible weakness and its threat model.
3. Write a proposal with a clear hypothesis, changed component, ablations, and expected failure modes.
4. Add the method only in a new `configs/experiments/<name>.yaml` and a separate source module; never overwrite baseline results.

This phase is deliberately not implemented yet.

## Phase 5 - Coursework packaging

1. Create a narrative notebook in `notebooks/` that installs dependencies, runs the baseline, loads saved outputs, and explains results.
2. Pre-render all required notebook outputs for Milestone 3.
3. Prepare a concise Milestone 1 challenge that tests a stated property of the defence without exposing unnecessary unsafe content.
4. Maintain a weekly decision and experiment log for the individual Milestone 4 report.
