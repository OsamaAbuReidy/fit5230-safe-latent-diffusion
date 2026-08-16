# FIT5230 SD3.5 Medium Jailbreak Defence

Reproducible coursework repository for FIT5230 Malicious AI (2026).

- Theme: Text-to-Image
- Side: Light (defence)
- Backbone: Stable Diffusion 3.5 Medium
- Baseline paper: JailbreakDiffBench (ICCV 2025)
- Scope now: reproduce DACA/PGJ-style attack evaluation on a fixed SD3.5 Medium
  workflow and establish defensible text-only and in-generation safety baselines.
  No final research modification has been selected.

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

Start with the
[Milestone 1 notebook](notebooks/milestone1_baseline_challenge.ipynb). See
[docs/assignment-brief.md](docs/assignment-brief.md) for the marking requirements
and [docs/research-branch-counterfactual-early-safety.md](docs/research-branch-counterfactual-early-safety.md)
for the separately parked publication-oriented branch.

## Environment

Use Python 3.10, 3.11, or 3.12 for project tooling. SD3.5 Medium image
generation currently runs through the local ComfyUI workflow; the Milestone 1
Colab is a lightweight public entry point and does not download model weights.

```bash
pip install -r requirements.txt
pip install -e .
```

Store any model-access token locally in `.env`; never commit it.

## Current status

The SD3.5 Medium workflow, fixed-seed runner, early-feature logger, and
exploratory detector comparison are functioning. The current detector is not a
successful defence because it produces excessive benign false positives. The
next task is to freeze the Milestone 1 challenge and then expand the baseline
evaluation before selecting one separate, configurable defence modification.
