# FIT5230 SD3.5 Medium Jailbreak Defence

Reproducible coursework repository for FIT5230 Malicious AI (2026).

- Theme: Text-to-Image
- Side: Light (defence)
- Backbone: Stable Diffusion 3.5 Medium
- Baseline paper: JailbreakDiffBench (ICCV 2025)
- Topic: develop PreDecodeGuard-SD3.5, a lightweight safety classifier that operates on SD3.5
  Medium's final pre-decode latent representation, focusing on nudity/sexual
  content and physical violence generated from direct, DACA, and PGJ prompts.
- Evaluation: compare the latent guard with the published Multihead Detector and
  a text-only baseline on the same human-labelled generations, including
  detection quality, benign false positives, latency, and refusal-aware results.

## Why this baseline

JailbreakDiffBench provides a published attack and evaluation framework covering
modern text-to-image systems, including SD3.5 Medium. For a Light-side project,
an attack-focused paper is an appropriate baseline because the assignment permits
developing a defence against the selected attack. SD3.5 Medium is the fixed local
generation backbone used for reproducible experiments.

Primary references:

- Paper: [JailbreakDiffBench: A Comprehensive Benchmark for Jailbreaking Diffusion Models](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Jin_JailbreakDiffBench_A_Comprehensive_Benchmark_for_Jailbreaking_Diffusion_Models_ICCV_2025_paper.pdf)
- Reference code: [Jinxiaolong1129/JailbreakDiffusionBench](https://github.com/Jinxiaolong1129/JailbreakDiffusionBench)
- Backbone: [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)

## Repository map

```text
configs/       Reproducible run and evaluation configurations
data/          Dataset manifests and metadata only; generated data is ignored
docs/          Assignment notes, protocol, and setup plan
notebooks/     Colab-facing narrative notebooks
outputs/       Generated images, metrics, and figures (ignored except .gitkeep)
scripts/       Thin command-line entry points for baseline runs and evaluation
src/           Reusable project package (baseline, data, evaluation, utilities)
tests/         Smoke and regression tests
```

Start with the assignment-facing
[Milestone 2 notebook](notebooks/milestone2_show_of_force.ipynb). The older
Milestone 1 and residual notebooks are retained as development history. See
[docs/assignment-brief.md](docs/assignment-brief.md) for the marking requirements
and [docs/research-branch-counterfactual-early-safety.md](docs/research-branch-counterfactual-early-safety.md)
for the separately parked publication-oriented branch.

The latent capture and dataset protocol is documented in
[docs/latent-guard-dataset-pipeline.md](docs/latent-guard-dataset-pipeline.md).
ShieldGemma 2 remains a preliminary teacher/comparator rather than the proposed
defence itself. The refusal-handling issue found in the reference benchmark is
documented in
[docs/jailbreakdiffbench-gpt4o-audit.md](docs/jailbreakdiffbench-gpt4o-audit.md).

## Environment

Use Python 3.10, 3.11, or 3.12 for project tooling. SD3.5 Medium image
generation currently runs through the local ComfyUI workflow; the Milestone 1
Colab is a lightweight public entry point and does not download model weights.

```bash
pip install -r requirements.txt
pip install -e .
```

Store any model-access token locally in `.env`; never commit it.

### Phase 1: Multihead Detector baseline

The Phase 1 protocol is frozen in
[docs/phase1-multihead-baseline-plan.md](docs/phase1-multihead-baseline-plan.md).
With a CUDA-compatible PyTorch installation and the released checkpoints in
`tmp/JailbreakDiffusionBench`, run the deterministic smoke gate before the full
pilot inference:

```powershell
.\.venv\Scripts\python.exe scripts/run_multihead_baseline.py --mode smoke --device cuda --batch-size 4 --clip-weights C:\path\to\ViT-L-14.pt --output data/results/multihead_phase1_smoke_predictions.csv
```

After it passes, run the full prompt-free inference and evaluation:

```powershell
.\.venv\Scripts\python.exe scripts/run_multihead_baseline.py --mode full --device cuda --batch-size 4 --resume
.\.venv\Scripts\python.exe scripts/evaluate_multihead_baseline.py
```

The first run may download the OpenCLIP `ViT-L-14/openai` weights; record this
in the generated run metadata. The runner rejects a requested CUDA run when it
cannot see a GPU and records image or inference failures explicitly rather than
silently treating them as safe.

The official OpenAI `ViT-L-14.pt` file is a TorchScript archive. When using it
locally with PyTorch 2.6 or newer, the runner verifies its published SHA-256
before applying the required TorchScript compatibility load path.

The released detector retains OpenCLIP's training preprocessing transform,
which uses randomized crops. The runner preserves that transform but seeds all
relevant random-number generators from `--seed` (default `5230`) so the frozen
baseline can be repeated exactly.

Pinned text-only prompt sources can be downloaded or integrity-checked with:

```powershell
python scripts/prepare_prompt_sources.py
python scripts/prepare_prompt_sources.py --verify-only
```

See [data/manifests/prompt-sources.md](data/manifests/prompt-sources.md) for the source inventory,
observed record counts, dataset quirks, and intended train/test roles.

Create the normalized local prompt table and deterministic 400-sample latent-guard feasibility
manifest with:

```powershell
python scripts/build_latent_guard_manifest.py
```

After ComfyUI is running with the FIT5230 custom nodes installed, start or resume final-latent
capture and ShieldGemma teacher labelling with:

```powershell
python scripts/run_latent_guard_batch.py
```

Monitor a running collection with:

```powershell
python scripts/report_latent_guard_progress.py --run-id latent_guard_pilot_v1
```

See [docs/latent-guard-dataset-pipeline.md](docs/latent-guard-dataset-pipeline.md) for the fixed
configuration, artifact schema, resumability rules, and validated smoke result.

## Blind human image review

Completed generations can be labelled with a local two-button desktop reviewer. The interface
hides prompts, source categories, teacher decisions, and sample IDs to reduce reviewer bias. Each
decision is immediately appended to a resumable CSV audit log.

```powershell
.\.venv\Scripts\python.exe scripts\review_latent_guard_images.py --run-id latent_guard_pilot_v1
```

Press `B` for benign, `S` for sexual, `V` for violence/gore, `D` for dangerous content, or `I`
for an unusable/invalid generation. Closing the window is safe; launching the same command later
skips samples already labelled. Labels are written to
`data/annotations/latent_guard_pilot_v1_human_labels.csv`.

## Current status

The SD3.5 Medium workflow, fixed-seed runner, final-latent capture, resumable
dataset pipeline, blind human-review interface, compact spatial CNN, faithful
Multihead baseline, and assignment notebook are functioning.

On the frozen 75-image violence test set, the mixed-expansion spatial CNN reaches
0.8524 balanced accuracy, 0.7692 harmful recall, 0.0645 benign FPR, and 0.9045
AUROC. Multihead remains more accurate overall (0.9504 AUROC and 0.0161 benign
FPR) while matching the same 0.7692 harmful recall.

Warm batch-one median inference on the same RTX 3060 Laptop GPU is 1.8965 ms for
the latent CNN and 183.1275 ms for Multihead. The pipeline's tiled VAE decode is
2492.7486 ms, making the measured blocked-output post-denoising decision path
1.8965 ms for PreDecodeGuard versus 2675.8761 ms for decode plus Multihead.
Denoising is common to both pipelines and excluded from that comparison.

The immediate assignment task is peer engagement: evaluate Dark.Chaos modified
prompts with the frozen detector and preserve the public exchange for Milestone 2.
