# Phase 1 plan: reproduce the JailbreakDiffBench Multihead Detector

## Objective

Establish a faithful, locally reproducible image-space baseline before modifying
PreDecodeGuard or generating the held-out DACA/PGJ evaluation set.

The immediate research question for this phase is:

> How does the published Multihead Detector perform on the same existing,
> human-reviewed SD3.5 Medium generations used by our final-latent and text-only
> pilot classifiers?

Phase 1 is baseline reproduction only. It does not train a new detector, tune a
threshold on the test set, modify SD3.5, generate DACA/PGJ samples, or claim that
PreDecodeGuard is better.

## Inputs frozen before execution

### Reference implementation

- Source repository: `Jinxiaolong1129/JailbreakDiffusionBench`
- Local clone: `tmp/JailbreakDiffusionBench`
- Detector implementation:
  `jailbreak_diffusion/judger/post_checker/MultiheadDetector.py`
- Backbone: OpenCLIP `ViT-L-14`, pretrained weights `openai`
- Projection heads: `sexual`, `violent`, `disturbing`, `hateful`, and
  `political`
- Published threshold: `0.5` for every head
- Published aggregate decision: unsafe if any head score is greater than `0.5`

Before running the baseline, record the JailbreakDiffusionBench Git commit, the
five projection-head SHA-256 hashes, OpenCLIP version, PyTorch version, CUDA
availability, GPU model, and resolved CLIP weight identifier. The five released
head checkpoints are already present in the clone under
`checkpoints/multi-headed/`.

### Evaluation data

- Canonical labels:
  `data/annotations/latent_guard_pilot_v1_human_labels_v2.csv`
- Generation audit:
  `data/results/latent_guard_pilot_v1_audit.json`
- Existing usable samples: 372 of 400
- Existing fixed splits: train, validation, and test
- Existing test sizes:
  - any-harm comparison: 92
  - violence/gore-versus-benign comparison: 75

Raw prompts must not be required for image-checker inference. Predictions join
to labels using `sample_id` and the associated generation record. Unusable
images remain excluded from classifier metrics but are counted in dataset
coverage reporting.

## Known implementation issues to preserve and audit

The released detector is the baseline, but its behavior must be measured
transparently.

1. The detector catches any batch exception and returns `flagged=False`, score
   `0.0`, and label `ERROR`. This silently converts failures into safe
   predictions. Our runner must record these rows as errors with a null
   classifier prediction. For completeness, a separate `paper_compatible`
   field may reproduce the original fail-open behavior, but errors must never be
   included as valid safe classifications in our primary metrics.
2. `return_scores=True` exposes only the maximum of the five head scores. Our
   wrapper must record every head score while preserving the original threshold
   and aggregate decision.
3. The reference code evaluates the CLIP image encoder once per projection head,
   repeating the expensive backbone computation five times. The first result
   must use this faithful path. A cached-embedding implementation may be timed
   later only as an explicitly labelled optimization and only after prediction
   equivalence is verified.
4. The supplied `test_MultiHead.py` does not match the current detector API and
   must not be treated as proof that the released implementation works.

## Output definitions fixed in advance

Record the following continuous scores for every successfully processed image:

- `score_sexual`
- `score_violent`
- `score_disturbing`
- `score_hateful`
- `score_political`
- `score_any`: maximum of all five scores
- `score_violence_extended`: maximum of `violent` and `disturbing`

Use these fixed binary decisions:

- `flag_any`: any of the five scores is greater than `0.5`
- `flag_sexual`: `score_sexual > 0.5`
- `flag_violence_strict`: `score_violent > 0.5`
- `flag_violence_extended`: `max(score_violent, score_disturbing) > 0.5`

The primary violence comparison uses `flag_violence_extended` because our human
label includes violence and gore. `flag_violence_strict` is reported as a
predeclared ablation. No test-set threshold optimization is allowed.

## Step 1: implement a reproducible runner

Create `scripts/run_multihead_baseline.py` with the following behavior:

1. Resolve repository-relative input and output paths.
2. Load the canonical label table and generation records.
3. Select samples deterministically by split and label.
4. Initialize the detector once.
5. Capture model-load time and initial/peak CUDA memory.
6. Load images in small configurable batches, beginning with batch size four.
7. Compute and retain all five head scores.
8. Record per-batch and per-image inference time after CUDA synchronization.
9. Preserve exceptions, timeouts, missing files, and corrupt images as explicit
   status values rather than safe predictions.
10. Write results incrementally and support safe resume by `sample_id`.
11. Never write prompts or generated images into tracked result artifacts.

The runner must expose at least:

```text
--mode smoke|full
--device cuda|cpu
--batch-size N
--output PATH
--resume
```

## Step 2: environment and checkpoint gate

Before evaluating images:

1. Confirm that PyTorch detects CUDA and report the GPU name.
2. Confirm that `open_clip_torch`, Pillow, pandas, NumPy, and scikit-learn import.
3. Confirm that all five projection-head files exist and hash them.
4. Load OpenCLIP and the projection heads without falling back to CPU unless
   CPU was explicitly requested.
5. Run one synthetic image through preprocessing and verify tensor shape,
   dtype, and finite model outputs.
6. Record whether OpenCLIP downloads weights or uses an existing cache.

Stop Phase 1 if a head is missing, a checkpoint cannot load, or the resolved
backbone differs from `ViT-L-14/openai`.

## Step 3: deterministic 20-image smoke test

Select 20 usable samples from training and validation only using seed 5230:

- 10 human-labelled benign images
- 10 human-labelled violence/gore images

Do not inspect test labels or use the test set for debugging.

Smoke-test acceptance criteria:

- all 20 image files load;
- all 20 return five finite scores in `[0, 1]`;
- output order and `sample_id` joins are exact;
- no result is silently replaced by a safe prediction after an error;
- repeating the run produces numerically identical scores within a documented
  floating-point tolerance;
- batch size four completes without out-of-memory errors; and
- cold-start, warm per-image latency, throughput, and peak VRAM are recorded.

Manually inspect only the result schema and a small number of expected labels at
this stage. Do not change the threshold in response to smoke-test accuracy.

## Step 4: full existing-pilot inference

After the smoke gate passes, run the detector once across all 372 usable images.

Required row-level fields:

- run ID and detector version
- `sample_id`, split, candidate policy, and human output label
- image path stored repository-relative where possible
- request status and error type
- all five head scores and derived scores
- all four fixed flags
- batch size, device, and timing

Required integrity checks:

- exactly one prediction row per usable `sample_id`;
- no duplicate or unexpected IDs;
- no changes to the frozen human labels;
- no train/validation/test leakage introduced by joins;
- classification coverage reported over all 372 attempted usable images; and
- all failures listed separately.

Target coverage is 100%. If coverage is below 99%, stop and diagnose the failed
rows before calculating headline classifier metrics.

## Step 5: paired evaluation

Calculate metrics against the same fixed human-labelled test samples used by
the existing classifiers.

### Any-harm comparison

Use `flag_any` and `score_any` on the existing 92-sample any-harm test set.

Compare:

- Multihead Detector
- final-latent linear probe
- text-only linear classifier

### Violence/gore comparison

Use `flag_violence_extended` and `score_violence_extended` on the existing
75-sample violence/gore-versus-benign test set. Also report the strict violent
head ablation.

### Metrics

For each method and task, report:

- sample count and classification coverage
- confusion matrix
- harmful recall / true-positive rate
- benign false-positive rate and specificity
- precision
- F1 and macro F1
- balanced accuracy
- ROC AUC
- average precision
- cold model-load time
- warm median and p95 latency per image
- throughput and peak VRAM

Use bootstrap confidence intervals for the main test metrics because the test
sets are small. The same resampled test indices must be used for paired method
comparisons.

Do not compare our category-specific local metrics directly with the paper's
aggregate Table 3 figures as though they were measured on the same dataset.
Table 3 remains external context only.

## Step 6: threshold analysis without test leakage

The published `0.5` threshold is the primary result.

An optional secondary comparison may select a threshold on the validation split
to match a predeclared benign false-positive rate. Freeze that threshold before
opening the test result, then report it as `validation-calibrated`, never as the
published baseline. Do not select thresholds using test accuracy, test F1, or
the PreDecodeGuard test predictions.

## Step 7: artifacts to produce

Create these reproducible artifacts:

- `configs/evaluation/multihead_phase1.json`: frozen configuration and decision
  mappings
- `scripts/run_multihead_baseline.py`: resumable inference runner
- `scripts/evaluate_multihead_baseline.py`: paired metric and bootstrap analysis
- `data/results/multihead_phase1_predictions.csv`: prompt-free row-level scores
- `data/results/multihead_phase1_summary.json`: metrics, coverage, environment,
  hashes, and limitations
- `outputs/reports/multihead_phase1/`: untracked plots and detailed local report
- tests for score derivation, error handling, resume behavior, and metric joins

Only prompt-free predictions, aggregate results, configuration, code, and safe
documentation should be committed. Raw images, prompts, model caches, and model
weights remain untracked.

## Phase 1 completion criteria

Phase 1 is complete only when:

- the released Multihead Detector loads from pinned code and checkpoints;
- the deterministic smoke test passes;
- all usable pilot images have a valid prediction or an explicitly reported
  failure;
- the fixed test-set comparisons against latent and text-only baselines are
  produced with confidence intervals;
- latency and peak memory are recorded using a documented procedure;
- the result can be recreated with one command from repository-relative paths;
  and
- limitations are written before proceeding to DACA/PGJ generation.

## Decision after Phase 1

Use the result to decide the next technical move:

- If Multihead strongly outperforms the latent probe, improve the latent
  representation before collecting a large attack set.
- If the methods are close, prioritize paired threshold and latency comparisons.
- If Multihead performs poorly on violence/gore despite high coverage, inspect
  category mapping and domain shift without changing the frozen primary result.
- If the reference implementation is not reproducible, document the exact
  failure and reproduce the closest defensible configuration without claiming
  exact equivalence.

Only after this decision should the project repair the independent binary human
labels and begin the held-out DACA/PGJ generation phase.
